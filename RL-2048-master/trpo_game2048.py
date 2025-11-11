import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import game2048
import os
import re
from collections import deque
import time
import tqdm
import argparse
from itertools import count
from trpo_game2048_simple import SimplePolicy, SimpleValue
from vectorized_env_mp import VectorizedEnv as MPVectorizedEnv


class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 2048游戏的观察空间是4x4的网格,需要展平
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(-1)
        x = torch.log2(x.float() + 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_action(self, state):
        """获取动作和对数概率"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def evaluate(self, states, actions):
        """评估状态-动作对"""
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """价值网络"""
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        else:
            x = x.reshape(-1)
        x = torch.log2(x.float() + 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)

class VectorizedEnv:
    """向量化环境包装器 - 并行运行多个环境"""
    def __init__(self, num_envs: int = 4, target=2048):
        self.num_envs = num_envs
        self.envs = [gym.make("Game2048-v0", target=target) for _ in range(num_envs)]
    
    def reset(self):
        states = []
        for env in self.envs:
            state, _ = env.reset()
            states.append(state)
        return np.array(states)
    
    def step(self, actions):
        states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            state, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            if done:
                state, _ = env.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        return np.array(states), np.array(rewards), np.array(dones)
    
    def close(self):
        for env in self.envs:
            env.close()

    def reset_one(self, idx: int):
        """仅重置一个子环境并返回其初始观测。"""
        i = int(idx)
        if i < 0 or i >= self.num_envs:
            raise IndexError("env index out of range")
        obs, _ = self.envs[i].reset()
        return obs

class TRPO:
    """TRPO算法实现"""
    def __init__(
        self,
        env,
        hidden_dim=256,
        gamma=0.99,
        lam=0.95,
        max_kl=0.01,
        damping=1e-2,
        entropy_coeff = 0.01,
        value_lr=1e-3,
        train_value_iters=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_envs=1,
        vec_impl: str = 'auto',  # 'auto' | 'inproc' | 'mp'
        env_target: int = 2048,
    ):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping
        self.entropy_coeff = entropy_coeff
        self.value_lr = value_lr
        self.train_value_iters = train_value_iters
        self.device = device
        self.num_envs = int(max(1, num_envs))
        self.vec_impl = vec_impl
        self.env_target = int(env_target)
        self.start_iter = 0  # 将在加载模型时尝试从文件名解析
        
        # 获取状态和动作空间维度
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.n
        
        # 初始化网络
        # self.policy = PolicyNetwork(self.state_dim, hidden_dim, self.action_dim).to(device)
        # self.value_net = ValueNetwork(self.state_dim, hidden_dim).to(device)

        self.policy = SimplePolicy().to(device)
        self.value_net = SimpleValue().to(device)
        
        # 价值网络优化器
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        print(f"使用设备: {device}")
        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"并行环境数: {self.num_envs}")
    
    def select_action(self, state):
        """选择动作"""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 2:  # 4x4 -> [1,4,4]
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            probs = self.policy(state_t)            # [1, A]
            dist = Categorical(probs=probs.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def select_actions_batch(self, states):
        """批量选择动作"""
        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.policy(states_tensor)      # [B, A]
            dist = Categorical(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_values_batch(self, states):
        """批量获取价值"""
        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.value_net(states_tensor)  # 形如 [B] 或 [B,1]，SimpleValue 中已处理
        return values.detach().cpu().numpy()
    
    def compute_advantages(self, rewards, values, dones, last_value):
        """使用GAE计算优势函数 - 单环境"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def compute_advantages_vectorized(self, rewards, values, dones, last_values):
        """使用GAE计算优势函数 - 多环境批量"""
        steps_per_env = len(rewards) // self.num_envs
        all_env_adv = []
        for env_idx in range(self.num_envs):
            env_rewards = rewards[env_idx::self.num_envs]
            env_values = values[env_idx::self.num_envs]
            env_dones = dones[env_idx::self.num_envs]
            env_last_value = last_values[env_idx]
            env_adv = []
            gae = 0.0
            for t in reversed(range(len(env_rewards))):
                next_value = env_last_value if t == len(env_rewards) - 1 else env_values[t + 1]
                delta = env_rewards[t] + self.gamma * next_value * (1 - env_dones[t]) - env_values[t]
                gae = delta + self.gamma * self.lam * (1 - env_dones[t]) * gae
                env_adv.insert(0, gae)
            all_env_adv.append(env_adv)
        # 交错回到原顺序
        advantages = []
        for step_idx in range(steps_per_env):
            for env_idx in range(self.num_envs):
                advantages.append(all_env_adv[env_idx][step_idx])
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def collect_trajectories(self, num_steps):
        """收集轨迹数据 - 单环境"""
        states = []
        actions = []
        rewards = []
        log_probs = []
        dones = []
        values = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_rewards = []
        
        for step in range(num_steps):
            # 选择动作
            action, log_prob = self.select_action(state)
            
            # 获取价值估计
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0)  # [1,4,4]
            with torch.no_grad():
                value = self.value_net(state_tensor).squeeze(0).cpu().item()
            
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 存储数据
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
                state, _ = self.env.reset()
                episode_reward = 0
        
        # 计算最后一个状态的价值（如果最后一步是done，则为0）
        if len(dones) > 0 and dones[-1]:
            last_value = 0.0
        else:
            st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if st.dim() == 2:
                st = st.unsqueeze(0)  # [1,4,4]
            last_value = self.value_net(st).squeeze(0).cpu().item()
        
        return states, actions, rewards, log_probs, dones, values, episode_rewards, last_value

    def collect_batch_vectorized(self, vec_env, batch_size=1024, min_episodes_per_env=1):
        """向量化批量采样"""
        states_list, actions_list, rewards_list = [], [], []
        log_probs_list, dones_list, values_list = [], [], []
        episode_rewards, episode_trackers = [], [0.0] * vec_env.num_envs
        env_completed = [0] * vec_env.num_envs
        env_steps = [0] * vec_env.num_envs  # 每个环境自上次完成以来的步数

        current_states = vec_env.reset()
        steps = 0
        while True:
            actions, log_probs = self.select_actions_batch(current_states)
            values = self.get_values_batch(current_states)
            next_states, rewards, dones = vec_env.step(actions)

            for i in range(vec_env.num_envs):
                states_list.append(current_states[i])
                actions_list.append(int(actions[i]))
                rewards_list.append(float(rewards[i]))
                log_probs_list.append(float(log_probs[i]))
                dones_list.append(bool(dones[i]))
                values_list.append(float(values[i]))
                episode_trackers[i] += float(rewards[i])
                env_steps[i] += 1
                if dones[i]:
                    episode_rewards.append(episode_trackers[i])
                    episode_trackers[i] = 0.0
                    env_completed[i] += 1
                    env_steps[i] = 0
                elif env_steps[i] >= int(batch_size):
                    # 若单个环境已产生 batch_size 次交互但仍未结束，则强制重置并计为完成一次
                    # 尝试仅重置该环境
                    if hasattr(vec_env, 'reset_one'):
                        try:
                            forced_obs = vec_env.reset_one(i)
                            next_states[i] = forced_obs
                        except Exception:
                            # 失败时退化为整体 reset（代价较大，但保证正确）
                            next_states = vec_env.reset()
                    else:
                        # 无单环境重置能力，退化为整体 reset
                        next_states = vec_env.reset()

                    episode_rewards.append(episode_trackers[i])
                    episode_trackers[i] = 0.0
                    env_completed[i] += 1
                    env_steps[i] = 0
                steps += 1

            current_states = next_states
            if steps >= batch_size * 10:
                break
            if steps >= batch_size and all(c >= min_episodes_per_env for c in env_completed):
                break

        # 计算每个env的最后价值
        last_values = []
        for i in range(vec_env.num_envs):
            if not dones[i]:
                v = self.get_values_batch(current_states[i][None, ...])
                last_values.append(float(np.asarray(v).reshape(-1)[0]))
            else:
                last_values.append(0.0)

        return (states_list, actions_list, rewards_list, log_probs_list,
                dones_list, values_list, episode_rewards, last_values)

    def compute_policy_loss(self, states, actions, advantages, old_log_probs):
        """计算策略损失（基于 probs 的 Categorical）"""
        probs = self.policy(states)                  # [B, A]
        dist = Categorical(probs=probs)
        new_log_probs = dist.log_prob(actions)       # [B]
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean() - self.entropy_coeff * entropy
        return policy_loss, new_log_probs
    
    # def compute_kl_divergence(self, states, actions, old_log_probs):
    #     """计算KL散度"""
    #     new_log_probs, _ = self.policy.evaluate(states, actions)
    #     # KL(old||new) = E[log(old) - log(new)]
    #     kl = (old_log_probs - new_log_probs).mean()
    #     return kl
    
    def flat_grad(self, grads, params):
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(g.contiguous().view(-1))
        return torch.cat(flat)
    
    def flat_params(self, model):
        """获取展平的参数"""
        return torch.cat([param.data.reshape(-1) for param in model.parameters()])
    
    def set_flat_params(self, model, flat_params):
        """设置展平的参数"""
        idx = 0
        for param in model.parameters():
            param_length = param.numel()
            param.data.copy_(flat_params[idx:idx + param_length].view(param.shape))
            idx += param_length
    
    def conjugate_gradient(self, Avp_func, b, num_steps=10, tol=1e-10):
        """共轭梯度法"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(num_steps):
            Avp = Avp_func(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < tol:
                break
            
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def compute_fisher_vector_product(self, states, actions, vector):
        """
        计算Fisher信息矩阵与向量的乘积
        使用 KL(old || new) 的 Hessian（基于 probs）
        """
        probs = self.policy(states)                  # 当前分布
        with torch.no_grad():
            old_probs = probs.detach()               # 固定旧分布

        # KL(old||new) 的与参数相关部分：-sum(old * log(new))
        kl = -(old_probs * torch.log(probs + 1e-10)).sum(-1).mean()

        kl_grad = torch.autograd.grad(kl, list(self.policy.parameters()),
                                      create_graph=True, retain_graph=True)
        flat_kl_grad = self.flat_grad(kl_grad, self.policy.parameters())

        grad_vector_product = torch.sum(flat_kl_grad * vector)
        hvp = torch.autograd.grad(grad_vector_product, list(self.policy.parameters()),
                                  retain_graph=True)
        flat_hvp = self.flat_grad(hvp, self.policy.parameters())
        return flat_hvp + self.damping * vector
    
    def line_search(self, states, actions, advantages, old_log_probs, 
                    full_step, expected_improve, max_backtracks=10):
        """线搜索找到满足KL约束的步长"""
        old_params = self.flat_params(self.policy)
        old_loss, _ = self.compute_policy_loss(states, actions, advantages, old_log_probs)
        
        for step_frac in [0.5**x for x in range(max_backtracks)]:
            new_params = old_params + step_frac * full_step
            self.set_flat_params(self.policy, new_params)

            with torch.no_grad():
                probs = self.policy(states)
                dist = Categorical(probs=probs)
                new_log_probs = dist.log_prob(actions)
                kl = (old_log_probs - new_log_probs).mean()

            new_loss, _ = self.compute_policy_loss(states, actions, advantages, old_log_probs)
            actual_improve = old_loss - new_loss
            expected_improve_frac = expected_improve * step_frac
            improvement_ratio = actual_improve / (expected_improve_frac + 1e-8)
            
            if kl <= self.max_kl and actual_improve > 0 and improvement_ratio > 0.1:
                return True
        
        self.set_flat_params(self.policy, old_params)
        return False
    
    def update_policy(self, states, actions, advantages, old_log_probs):
        """使用TRPO更新策略"""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = advantages.detach()
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        # 计算策略损失和梯度
        loss, _ = self.compute_policy_loss(states, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters(), retain_graph=True) # type: ignore
        loss_grad = self.flat_grad(grads, self.policy.parameters())
        
        # 使用共轭梯度法求解搜索方向
        def Avp_func(v):
            return self.compute_fisher_vector_product(states, actions, v)
        
        step_dir = self.conjugate_gradient(Avp_func, -loss_grad, num_steps=10)
        
        # 计算步长
        shs = 0.5 * torch.dot(step_dir, Avp_func(step_dir))
        
        # 检查数值稳定性
        if shs < 0:
            print(f"  警告: shs={shs:.6f} < 0, 使用梯度下降方向")
            # 使用简单的梯度下降方向
            step_dir = -loss_grad
            shs = 0.5 * torch.dot(step_dir, Avp_func(step_dir))
            
            # 如果还是负数，跳过更新
            if shs < 0:
                print(f"  警告: 梯度下降方向也失败, 跳过策略更新")
                return False
        
        lm = torch.sqrt(shs / self.max_kl)
        full_step = step_dir / (lm + 1e-8)
        
        # 线搜索
        expected_improve = -torch.dot(loss_grad, full_step)
        success = self.line_search(states, actions, advantages, old_log_probs,
                                   full_step, expected_improve)
        
        return success
    
    def update_value(self, states, returns):
        """更新价值网络"""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        returns = returns.detach()
        
        for _ in range(self.train_value_iters):
            values = self.value_net(states)
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        return value_loss.item() # type: ignore

    def train(self, total_steps=100000, steps_per_update=2048, save_path=None):
        """训练TRPO算法
        
        Args:
            total_steps: 总训练步数，-1表示无限训练直到Ctrl+C中断
            steps_per_update: 每次更新的步数
        """
        print("=" * 70)
        print("开始训练TRPO算法")
        print(f"总步数: {'∞ (无限训练)' if total_steps == -1 else total_steps}, 每次更新步数: {steps_per_update}")
        if total_steps != -1:
            print(f"预计更新次数: {total_steps // steps_per_update}")
        else:
            print("按 Ctrl+C 停止训练")
        print("=" * 70)
        
        episode_rewards = deque(maxlen=100)
        step_count = 0
        update_count = 0
        
        start_time = time.time()
        
        # 创建进度条
        if total_steps == -1:
            # 无限训练模式，从 start_iter 开始计数
            pbar = tqdm.tqdm(count(self.start_iter), desc="训练进度")
            total_updates = None
        else:
            # 固定步数模式：以“更新次数”为单位的迭代器，从 start_iter 到 total_updates-1
            total_updates = max(1, total_steps // steps_per_update)
            pbar = tqdm.tqdm(range(self.start_iter, total_updates), desc="训练进度")

        vec_env = None
        if self.num_envs > 1:
            impl = self.vec_impl
            if impl == 'auto':
                impl = 'mp'  # 默认在多环境时使用多进程版
            if impl == 'mp':
                vec_env = MPVectorizedEnv(num_envs=self.num_envs, target=self.env_target)
            else:
                vec_env = VectorizedEnv(num_envs=self.num_envs, target=self.env_target)
        
        try:
            for _ in pbar:
                if total_steps != -1 and step_count >= total_steps:
                    break
                
                t0 = time.time()
                if vec_env is None:
                    # 单环境采样
                    states, actions, rewards, log_probs, dones, values, ep_rewards, last_value = \
                        self.collect_trajectories(steps_per_update)
                    t_collect = time.time()
                    # 计算优势和回报
                    advantages, returns = self.compute_advantages(rewards, values, dones, last_value)
                else:
                    # 向量化批量采样
                    states, actions, rewards, log_probs, dones, values, ep_rewards, last_values = \
                        self.collect_batch_vectorized(vec_env, steps_per_update, 1)
                    t_collect = time.time()
                    advantages, returns = self.compute_advantages_vectorized(
                        rewards, values, dones, last_values
                    )
                    
                t_sample = time.time()

                step_count += len(states)
                episode_rewards.extend(ep_rewards)
                
                t_stat = time.time()
                    
                # 更新策略
                success = self.update_policy(states, actions, advantages, log_probs)
                
                # 更新价值网络
                value_loss = self.update_value(states, returns)
                
                update_count += 1
                
                t_update = time.time()
                
                # 更新进度条
                # with torch.no_grad():
                #     st_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                #     probs_new = self.policy(st_tensor)
                #     dist_new = Categorical(probs=probs_new)
                #     new_log_probs = dist_new.log_prob(torch.LongTensor(actions).to(self.device))
                #     kl = (torch.FloatTensor(log_probs).to(self.device) - new_log_probs).mean().item()
                
                avg_reward = np.mean(rewards)
                max_tile = int(np.max(states))
                
                states_arr = np.array(states)
                dones_arr = np.array(dones, dtype=bool)
                end_states = states_arr[dones_arr]
                avg_max_tile = np.mean(np.max(end_states, axis=(1,2)))
                
                pbar.set_postfix({
                    '平均奖励': f'{avg_reward:.1f}',
                    '最大块': max_tile,
                    '平均最大块': f'{avg_max_tile:.1f}',
                    '无效动作': f'{np.sum(np.array(rewards) == -10.0) / len(rewards) * 100:.1f}%',
                    '价值损失': f'{value_loss:.1f}',
                    '策略': '✓' if success else '✗'
                })

                # pbar.write(f"采样时间：{t_collect - t0:.2f}, 样本数：{len(actions)}, 采样速度：{len(actions)/ (t_collect - t0):.2f} 样本/s, 计算优势时间：{t_sample - t_collect:.2f}, 统计时间：{t_stat - t_sample:.2f}, 更新时间：{t_update - t_stat:.2f}, 价值损失：{value_loss:.6f}")
        finally:
            # 结束时保存模型，文件名包含累计迭代次数与batch大小
            try:
                curr_iter = self.start_iter + update_count
                final_path = self._format_save_path(save_path, curr_iter, steps_per_update)
                self.save_model(final_path)
            except Exception as e:
                print(f"保存模型失败: {e}")
            
            pbar.close()
            if vec_env is not None:
                vec_env.close()
            
            print("=" * 70)
            print("训练完成!")
            total_time = time.time() - start_time
            print(f"总时间: {total_time:.2f}秒")
            print(f"总步数: {step_count}")
            print(f"总更新次数: {update_count}")
            print(f"平均每次更新耗时: {total_time/update_count:.2f}秒")
            
            if len(episode_rewards) > 0:
                print(f"最终平均奖励: {np.mean(episode_rewards):.2f}")
            print("=" * 70)
        
        return episode_rewards
    
    def evaluate(self, num_episodes=10, render=False):
        """评估策略"""
        print(f"\n开始评估 {num_episodes} 个回合...")
        episode_rewards = []
        max_tiles = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                action, _ = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            max_tile = np.max(state)
            episode_rewards.append(episode_reward)
            max_tiles.append(max_tile)
            
            print(f"回合 {ep+1}/{num_episodes}: 奖励 = {episode_reward:.2f}, 最大方块 = {max_tile}")
        
        avg_reward = np.mean(episode_rewards)
        avg_max_tile = np.mean(max_tiles)
        
        print(f"\n评估结果:")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均最大方块: {avg_max_tile:.2f}")
        print(f"奖励标准差: {np.std(episode_rewards):.2f}")
        
        return episode_rewards, max_tiles
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        # 尝试从文件名解析迭代次数，如 *_<iters>it_*.pth
        try:
            fname = os.path.basename(str(path))
            m = re.search(r"_(\d+)it(?:_|\.pth$)", fname)
            if m:
                self.start_iter = int(m.group(1))
                print(f"模型已从 {path} 加载 (起始迭代={self.start_iter})")
                return
        except Exception:
            pass
        print(f"模型已从 {path} 加载")

    def _format_save_path(self, save_path, iters: int, batch_size: int) -> str:
        """根据给定基路径/文件名，生成携带迭代与batch信息的保存路径。"""
        base_default = "trpo_game2048_model"
        ext = ".pth"
        if not save_path:
            directory = "."
            base = base_default
        else:
            directory, name = os.path.split(save_path)
            if not directory:
                directory = "."
            root, ext_in = os.path.splitext(name)
            if ext_in:
                base = root
            else:
                # 传入的是不带扩展名的基名
                base = name if name else base_default
        filename = f"{base}_{iters}it_{batch_size}batch{ext}"
        return os.path.join(directory, filename)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TRPO训练用于2048游戏')
    
    # 训练参数
    parser.add_argument('--total-steps', type=int, default=100000,
                        help='总训练步数 (-1表示无限训练)')
    parser.add_argument('--steps-per-update', type=int, default=2048,
                        help='每次更新的步数')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='并行环境数量(>1启用批量采样)')
    parser.add_argument('--vec-impl', type=str, choices=['auto', 'inproc', 'mp'], default='auto',
                        help='向量化环境实现：auto(默认>1时用mp)、inproc(单进程多环境)、mp(多进程)')
    parser.add_argument('--target', type=int, default=2048,
                        help='环境的目标数值（例如 2048/1024/4096）')
    
    # 网络参数
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='隐藏层维度')
    
    # TRPO参数
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE参数lambda')
    parser.add_argument('--max-kl', type=float, default=0.01,
                        help='最大KL散度约束')
    parser.add_argument('--damping', type=float, default=1e-2,
                        help='Fisher矩阵阻尼系数')
    parser.add_argument('--entropy-coeff', type=float, default=0.01,
                        help='熵正则化系数')
    parser.add_argument('--value-lr', type=float, default=1e-3,
                        help='价值网络学习率')
    parser.add_argument('--train-value-iters', type=int, default=10,
                        help='价值网络训练迭代次数')
    
    # 模型保存/加载
    parser.add_argument('--save-path', type=str, default='trpo_game2048_model.pth',
                        help='模型保存路径')
    parser.add_argument('--load-path', type=str, default=None,
                        help='加载预训练模型路径')
    
    # 评估参数
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='评估回合数')
    parser.add_argument('--no-eval', action='store_true',
                        help='训练后不进行评估')
    
    # 设备选择：允许 'auto', 'cpu', 'cuda' 或 'cuda:<gpu_id>'
    def device_type(s):
        s = s.strip()
        if s in ('auto', 'cpu', 'cuda'):
            return s
        if s.startswith('cuda:'):
            try:
                idx = int(s.split(':', 1)[1])
                if idx < 0:
                    raise ValueError
                return f'cuda:{idx}'
            except Exception:
                raise argparse.ArgumentTypeError(
                    "--device must be 'auto', 'cpu', 'cuda' or 'cuda:<non-negative-int>'"
                )
        raise argparse.ArgumentTypeError(
            "--device must be 'auto', 'cpu', 'cuda' or 'cuda:<non-negative-int>'"
        )

    parser.add_argument('--device', type=device_type, default='auto',
                        help="训练设备。可用: 'auto', 'cpu', 'cuda', 或 'cuda:<gpu_id>'")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 打印配置
    print("\n" + "🎮" * 35)
    print("TRPO 2048游戏训练")
    print("🎮" * 35)
    print("\n配置参数:")
    print(f"  训练步数: {'∞ (无限)' if args.total_steps == -1 else args.total_steps}")
    print(f"  每次更新步数: {args.steps_per_update}")
    print(f"  隐藏层维度: {args.hidden_dim}")
    print(f"  最大KL散度: {args.max_kl}")
    print(f"  阻尼系数: {args.damping}")
    print(f"  并行环境数: {args.num_envs}")
    print(f"  向量化实现: {args.vec_impl}")
    print(f"  设备: {device}")
    print(f"  目标数值: {args.target}")
    print(f"  模型保存路径: {args.save_path}")
    if args.load_path:
        print(f"  加载模型: {args.load_path}")
    print()
    
    # 创建环境
    env = gym.make("Game2048-v0", target=args.target, debug=True)
    
    # 创建TRPO智能体
    agent = TRPO(
        env=env,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lam=args.lam,
        max_kl=args.max_kl,
        damping=args.damping,
        entropy_coeff=args.entropy_coeff,
        value_lr=args.value_lr,
        train_value_iters=args.train_value_iters,
        device=device,
        num_envs=args.num_envs,
        vec_impl=args.vec_impl,
        env_target=args.target,
    )
    
    # 加载预训练模型（如果指定）
    if args.load_path:
        agent.load_model(args.load_path)
    
    # 训练
    agent.train(total_steps=args.total_steps, steps_per_update=args.steps_per_update, save_path=args.save_path)
    
    # 训练阶段已根据进度自动保存（文件名包含 it 与 batch 信息），此处无需重复保存
    
    # 评估
    if not args.no_eval:
        agent.evaluate(num_episodes=args.eval_episodes, render=False)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
