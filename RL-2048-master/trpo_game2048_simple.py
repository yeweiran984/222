"""
简化版TRPO算法用于2048游戏
专为快速测试和演示设计
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import game2048
import time
import tqdm
from itertools import count


class SimplePolicy(nn.Module):
    """简化的策略网络 - 使用卷积层"""
    def __init__(self, hidden_dim=256, hidden_layers=1):
        super(SimplePolicy, self).__init__()
        # 输入: 4x4的游戏板
        # 卷积层提取特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(hidden_dim, 4))
        self.fc_layers.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化输入并reshape为卷积格式
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x = x.reshape(batch_size, 4, 4).float()
        x = torch.log2(x + 1) / 11.0  # 归一化到[0,1]范围
        x = x.unsqueeze(1)  # 添加通道维度: (batch, 1, 4, 4)
        
        # 通过卷积层
        x = self.conv_layers(x)
        x = x.reshape(batch_size, -1)  # 展平
        
        # 通过全连接层
        probs = self.fc_layers(x)
        return probs


class SimpleValue(nn.Module):
    """简化的价值网络 - 使用卷积层"""
    def __init__(self, hidden_dim=256, hidden_layers=1):
        super(SimpleValue, self).__init__()
        
        # 卷积层提取特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
        self.fc_layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化输入并reshape为卷积格式
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        x = x.reshape(batch_size, 4, 4).float()
        x = torch.log2(x + 1) / 11.0
        x = x.unsqueeze(1)  # 添加通道维度: (batch, 1, 4, 4)
        
        # 通过卷积层
        x = self.conv_layers(x)
        x = x.reshape(batch_size, -1)  # 展平
        
        # 通过全连接层
        value = self.fc_layers(x)
        return value.squeeze(-1)


class SimpleTRPO:
    """简化的TRPO算法"""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.policy = SimplePolicy().to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value = SimpleValue().to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.lam = 0.95
        self.max_kl = 0.01
        
        print(f"使用设备: {device}")
        
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(state_tensor)
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def compute_advantages(self, rewards, values, dones):
        """计算GAE优势"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def collect_batch(self, env, batch_size=1024):
        """收集一批数据"""
        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        
        state, _ = env.reset()
        episode_reward = 0
        episode_rewards = []
        
        for _ in range(batch_size):
            action, log_prob = self.select_action(state)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self.value(state_tensor).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)
            values.append(value)
            
            episode_reward += reward
            state = next_state
            
            if done:
                episode_rewards.append(episode_reward)
                state, _ = env.reset()
                episode_reward = 0
        
        return states, actions, rewards, log_probs, dones, values, episode_rewards
    
    def update(self, states, actions, advantages, old_log_probs, returns):
        """更新策略和价值网络"""
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages = advantages.detach()
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        # 计算当前策略的log概率
        probs = self.policy(states_tensor)
        dist = Categorical(probs=probs)
        new_log_probs = dist.log_prob(actions_tensor)
        
        # 计算KL散度
        kl = (old_log_probs_tensor - new_log_probs).mean()
        
        # 如果KL散度太大,不更新策略
        if kl > self.max_kl * 1.5:
            policy_updated = False
        else:
            # 计算策略损失
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            policy_loss = -(ratio * advantages).mean()
            
            # 使用简单的梯度下降更新策略(简化版TRPO)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            policy_updated = True
        
        # 更新价值网络
        value_loss_value = 0.0
        for _ in range(5):
            values_pred = self.value(states_tensor)
            value_loss = F.mse_loss(values_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            value_loss_value = value_loss.item()
        
        return policy_updated, value_loss_value, kl.item()
    
    def train(self, num_iterations=100, batch_size=1024, eval_step=100, save_path='trpo_simple_model.pth'):
        """训练算法"""
        env : game2048.Game2048Env = gym.make("Game2048-v0") # type: ignore
        
        pbar = tqdm.tqdm(range(num_iterations) if num_iterations > 0 else count(0), desc="Iteration")
        try:
            for iteration in pbar:
                # 收集数据
                states, actions, rewards, log_probs, dones, values, episode_rewards = \
                    self.collect_batch(env, batch_size)
                
                # 计算优势和回报
                advantages = self.compute_advantages(rewards, values, dones)
                returns = advantages + torch.FloatTensor(values).to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 更新网络
                policy_updated, value_loss, kl = self.update(
                    states, actions, advantages, log_probs, returns
                )
                
                # 统计信息
                pbar.set_postfix({'Rewards(m)': np.mean(rewards),
                                  'max': np.max(rewards),
                                'Rounds': len(episode_rewards)})
                # if len(episode_rewards) > 0:
                #     avg_reward = np.mean(episode_rewards)
                #     max_reward = np.max(episode_rewards)
                    
                #     elapsed_time = time.time() - start_time
                    
                #     print(f"迭代 {iteration+1:3d}/{num_iterations} | "
                #           f"平均奖励: {avg_reward:8.1f} | 最大: {max_reward:8.1f} | "
                #           f"回合数: {len(episode_rewards):2d} | "
                #           f"价值损失: {value_loss:.4f} | KL: {kl:.6f} | "
                #           f"策略更新: {'✓' if policy_updated else '✗'} | "
                #           f"时间: {elapsed_time:.1f}s")

                # 每10次迭代评估一次
                if (iteration + 1) % eval_step == 0:
                    self.evaluate(env, num_episodes=5)
        finally:
            env.close()
            self.save_model(save_path)
        print("=" * 70)
        print("训练完成!")
        
    def evaluate(self, env : game2048.Game2048Env, num_episodes=5):
        """评估策略"""
        print("\n" + "-" * 70)
        print("评估中...")
        rewards = []
        max_tiles = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            max_tile = np.max(state)
            rewards.append(episode_reward)
            max_tiles.append(max_tile)
            print(f"  回合 {ep+1}: 奖励={episode_reward:8.1f}, 最大方块={int(max_tile)}")
        
        print(f"平均奖励: {np.mean(rewards):8.1f}, 平均最大方块: {np.mean(max_tiles):.1f}")
        print("-" * 70 + "\n")
    
    def save_model(self, path="trpo_simple_model.pth"):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path="trpo_simple_model.pth"):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        print(f"模型已加载: {path}")


def main():
    """主函数"""
    # 创建智能体
    agent = SimpleTRPO()
    
    agent.load_model("model_after_bc.pth")
    
    # 训练
    agent.train(num_iterations=100, batch_size=1024)
    
    # 最终评估
    env : game2048.Game2048Env = gym.make("Game2048-v0") # type: ignore
    agent.evaluate(env, num_episodes=10)
    env.close()


if __name__ == "__main__":
    main()
