import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import game2048
import time
import tqdm
from typing import List, Tuple
from trpo_game2048_simple import SimplePolicy, SimpleValue
from training_monitor import TrainingMonitor
from itertools import count


class VectorizedEnv:
    """向量化环境包装器 - 并行运行多个环境"""
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.envs = [gym.make("Game2048-v0") for _ in range(num_envs)]
        
    def reset(self):
        """重置所有环境"""
        states = []
        for env in self.envs:
            state, _ = env.reset()
            states.append(state)
        return np.array(states)
    
    def step(self, actions):
        """执行动作"""
        states = []
        rewards = []
        dones = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if done:
                state, _ = env.reset()
            
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        
        return np.array(states), np.array(rewards), np.array(dones)
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()


class FastTRPO:
    """优化采样速度的TRPO"""
    
    def __init__(self, num_envs=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_envs = num_envs
        
        self.policy = SimplePolicy().to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value = SimpleValue().to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        # 添加学习率调度器
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=30, gamma=0.5
        )
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=30, gamma=0.5
        )

        self.gamma = 0.99
        self.lam = 0.95
        self.max_kl = 0.01
        
        # 添加性能跟踪
        self.best_reward = -float('inf')
        self.best_model_state = None
        self.no_improvement_count = 0
        
        print(f"使用设备: {device}")
        print(f"并行环境数: {num_envs}")
        
    def select_actions_batch(self, states):
        """批量选择动作 - 关键优化点"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            probs = self.policy(states_tensor)
            dist = Categorical(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()
    
    def get_values_batch(self, states):
        """批量获取价值 - 关键优化点"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            values = self.value(states_tensor)
        return values.cpu().numpy()
    
    def compute_advantages(self, rewards, values, dones, last_values):
        """
        计算GAE优势 - 支持多环境的批量计算
        
        Args:
            rewards: 所有环境的奖励列表
            values: 所有环境的价值列表
            dones: 所有环境的done标志列表
            last_values: 每个环境的最后一个价值 (长度为num_envs的列表)
        
        Returns:
            advantages: torch.Tensor
        """
        # 将数据重组为每个环境的轨迹
        # rewards, values, dones 的长度应该是 num_envs 的倍数
        steps_per_env = len(rewards) // self.num_envs
        
        advantages = []
        all_env_advantages = []
        
        # 为每个环境分别计算优势
        for env_idx in range(self.num_envs):
            # 提取该环境的数据 (交错存储)
            env_rewards = rewards[env_idx::self.num_envs]
            env_values = values[env_idx::self.num_envs]
            env_dones = dones[env_idx::self.num_envs]
            env_last_value = last_values[env_idx]
            
            # 计算该环境的GAE
            env_advantages = []
            gae = 0
            
            for t in reversed(range(len(env_rewards))):
                if t == len(env_rewards) - 1:
                    next_value = env_last_value
                else:
                    next_value = env_values[t + 1]
                
                delta = env_rewards[t] + self.gamma * next_value * (1 - env_dones[t]) - env_values[t]
                gae = delta + self.gamma * self.lam * (1 - env_dones[t]) * gae
                env_advantages.insert(0, gae)
            
            all_env_advantages.append(env_advantages)
        
        for step_idx in range(steps_per_env):
            for env_idx in range(self.num_envs):
                advantages.append(all_env_advantages[env_idx][step_idx])
        
        return torch.FloatTensor(advantages).to(self.device)

    
    def collect_batch_vectorized(self, vec_env: VectorizedEnv, batch_size=1024):
        """向量化采样 - 最大优化点
        保证每个环境至少采样一局完整游戏
        """
        # 预分配内存
        total_steps = batch_size
        states_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        dones_list = []
        values_list = []
        
        episode_rewards = []
        episode_reward_trackers = [0.0] * self.num_envs
        
        # 跟踪每个环境是否完成了至少一局游戏
        env_completed_episodes = [0] * self.num_envs
        min_episodes_per_env = 1  # 每个环境至少完成1局
        
        # 重置环境
        current_states = vec_env.reset()
        
        steps = 0
        # 修改停止条件: 需要同时满足两个条件
        # 1. 达到最小步数 batch_size
        # 2. 每个环境都至少完成了min_episodes_per_env局游戏
        while True:
            # 批量选择动作
            actions, log_probs = self.select_actions_batch(current_states)
            
            # 批量获取价值
            values = self.get_values_batch(current_states)
            
            # 执行动作
            next_states, rewards, dones = vec_env.step(actions)
            
            # 存储数据
            for i in range(self.num_envs):
                states_list.append(current_states[i])
                actions_list.append(actions[i])
                rewards_list.append(rewards[i])
                log_probs_list.append(log_probs[i])
                dones_list.append(dones[i])
                values_list.append(values[i])
                
                episode_reward_trackers[i] += rewards[i]
                
                if dones[i]:
                    episode_rewards.append(episode_reward_trackers[i])
                    episode_reward_trackers[i] = 0.0
                    env_completed_episodes[i] += 1
                
                steps += 1
            
            current_states = next_states
            
            # 检查停止条件
            if steps >= batch_size:
                # 检查是否所有环境都完成了至少min_episodes_per_env局
                if all(count >= min_episodes_per_env for count in env_completed_episodes):
                    break
        last_values = []
        for i in range(self.num_envs):
            if not dones[i]:
                last_value = self.get_values_batch(np.array([current_states[i]]))[0]
            else:
                last_value = 0.0
            last_values.append(last_value)
        return (states_list, 
                actions_list, 
                rewards_list, 
                log_probs_list, 
                dones_list, 
                values_list, 
                episode_rewards,
                last_values)
    
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
            policy_loss_value = 0.0
        else:
            # 计算策略损失
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            policy_loss = -(ratio * advantages).mean()
            
            # 使用简单的梯度下降更新策略
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.3)
            self.policy_optimizer.step()
            policy_updated = True
            policy_loss_value = policy_loss.item()
        
        # 更新价值网络
        value_loss_value = 0.0
        for i in range(1):
            values_pred = self.value(states_tensor)
            value_loss = F.mse_loss(values_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.3)
            self.value_optimizer.step()
            value_loss_value = value_loss.item()
        
        return policy_updated, value_loss_value, kl.item(), policy_loss_value if policy_updated else 0.0
    
    def train(self, num_iterations=100, batch_size=1024, eval_step=100, save_path='trpo_fast_model.pth', enable_monitor=True):
        """训练算法"""
        vec_env = VectorizedEnv(num_envs=self.num_envs)
        
        # 创建监控器
        monitor = TrainingMonitor(window_size=10) if enable_monitor else None
        
        print("=" * 70)
        print("开始训练优化版TRPO (并行采样 + 稳定性增强)")
        print(f"迭代次数: {num_iterations}, 每次批量大小: {batch_size}")
        print(f"初始学习率: {self.policy_optimizer.param_groups[0]['lr']:.2e}")
        print(f"监控: {'✓ 启用' if enable_monitor else '✗ 禁用'}")
        print("=" * 70)
        
        pbar = tqdm.tqdm(range(num_iterations) if num_iterations > 0 else count(0), desc="Iteration")
        
        try:
            for iteration in pbar:
                start_time = time.time()
                
                # 向量化采样
                states, actions, rewards, log_probs, dones, values, episode_rewards, last_values = \
                    self.collect_batch_vectorized(vec_env, batch_size)
                
                sampling_time = time.time() - start_time
                actual_samples = len(states)
                
                # 计算优势和回报
                advantages = self.compute_advantages(rewards, values, dones, last_values)
                returns = advantages + torch.FloatTensor(values).to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 更新网络
                update_start = time.time()
                policy_updated, value_loss, kl, policy_loss = self.update(
                    states, actions, advantages, log_probs, returns
                )
                update_time = time.time() - update_start
                
                # # 更新学习率
                self.policy_scheduler.step()
                self.value_scheduler.step()
                
                # 统计信息
                avg_reward = np.mean(episode_rewards)
                current_lr = self.policy_optimizer.param_groups[0]['lr']
                
                # 更新监控器
                if monitor:
                    monitor.update(
                        iteration=iteration,
                        reward=avg_reward,
                        value_loss=value_loss,
                        policy_loss=policy_loss,
                        kl=kl,
                        lr=current_lr,
                        episode_count=len(episode_rewards)
                    )
                
                # 检查是否是最佳模型
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    # 保存最佳模型状态
                    self.best_model_state = {
                        'policy': self.policy.state_dict(),
                        'value': self.value.state_dict(),
                        'iteration': iteration,
                        'reward': avg_reward
                    }
                    self.save_model(save_path)
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.1f}',
                    'Best': f'{self.best_reward:.1f}',
                    'Episodes': len(episode_rewards),
                    'Samples': actual_samples,
                    'Max': np.max(states),
                    'KL': kl
                })
                
                # 如果连续多次没有改善，降低学习率或恢复最佳模型
                if self.no_improvement_count >= 100:
                    print(f"\n⚠️ 性能连续 {self.no_improvement_count} 次迭代未改善，恢复最佳模型...")
                    vec_env = VectorizedEnv(num_envs=self.num_envs)
                    if self.best_model_state is not None:
                        self.policy.load_state_dict(self.best_model_state['policy'])
                        self.value.load_state_dict(self.best_model_state['value'])
                        print(f"✓ 已恢复到第 {self.best_model_state['iteration']} 次迭代的模型 (奖励: {self.best_model_state['reward']:.1f})")
                    self.no_improvement_count = 0
                
                # 每10次迭代评估一次
                if (iteration + 1) % eval_step == 0:
                    self.evaluate(num_episodes=3)
        finally:
            vec_env.close()
            self.save_model(save_path)
        print("=" * 70)
        print("训练完成!")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            print(f"\n✓ 恢复最佳模型 (第 {self.best_model_state['iteration']} 次迭代, 奖励: {self.best_model_state['reward']:.1f})")
            self.policy.load_state_dict(self.best_model_state['policy'])
            self.value.load_state_dict(self.best_model_state['value'])
        
        # 保存监控数据
        if monitor:
            print(monitor.get_summary())
            monitor.save_data("training_data.json")
            monitor.plot("training_monitor.png")
        
    def evaluate(self, num_episodes=5):
        """评估策略"""
        env = gym.make("Game2048-v0")
        
        print("\n" + "-" * 70)
        print("评估中...")
        rewards = []
        max_tiles = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = self.select_actions_batch(np.array([state]))
                state, reward, terminated, truncated, _ = env.step(action[0])
                done = terminated or truncated
                episode_reward += float(reward)
            
            max_tile = np.max(state)
            rewards.append(episode_reward)
            max_tiles.append(max_tile)
            print(f"  回合 {ep+1}: 奖励={episode_reward:8.1f}, 最大方块={int(max_tile)}")
        
        print(f"平均奖励: {np.mean(rewards):8.1f}, 平均最大方块: {np.mean(max_tiles):.1f}")
        print("-" * 70 + "\n")
        
        env.close()
    
    def save_model(self, path="fast_trpo_model.pth"):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path="fast_trpo_model.pth"):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        print(f"模型已加载: {path}")


def compare_sampling_speed():
    """对比采样速度"""
    print("\n" + "=" * 70)
    print("采样速度对比测试")
    print("=" * 70)
    
    from trpo_game2048_simple import SimpleTRPO
    
    batch_size = 1024
    
    # 测试原始版本
    print("\n测试1: 原始单环境采样")
    agent_single = SimpleTRPO()
    env = gym.make("Game2048-v0")
    
    start = time.time()
    agent_single.collect_batch(env, batch_size)
    single_time = time.time() - start
    env.close()
    
    print(f"时间: {single_time:.2f}秒")
    print(f"采样速度: {batch_size / single_time:.1f} 样本/秒")
    
    # 测试向量化版本
    for num_envs in [2, 4, 8, 16, 32, 64]:
        print(f"\n测试{num_envs+1}: {num_envs}个并行环境")
        agent_vec = FastTRPO(num_envs=num_envs)
        vec_env = VectorizedEnv(num_envs=num_envs)
        
        start = time.time()
        states, actions, rewards, log_probs, dones, values, episode_rewards, _ = \
            agent_vec.collect_batch_vectorized(vec_env, batch_size)
        vec_time = time.time() - start
        vec_env.close()
        
        print(f"时间: {vec_time:.2f}秒")
        print(f"采样速度: {len(actions) / vec_time:.1f} 样本/秒")
        # print(f"加速比: {single_time / vec_time:.2f}x")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # 对比采样速度
        compare_sampling_speed()
    else:
        # 训练
        print("\n" + "🚀" * 35)
        print("优化采样速度的TRPO训练")
        print("🚀" * 35)
        
        agent = FastTRPO(num_envs=64)
        
        # 可选: 加载预训练模型
        # agent.load_model("trpo_fast_model_314.7.pth")
        
        # 训练
        agent.train(num_iterations=-1, batch_size=2048)
        
        # 保存模型
        agent.save_model("fast_trpo_model.pth")
        
        # 最终评估
        agent.evaluate(num_episodes=10)


if __name__ == "__main__":
    main()
