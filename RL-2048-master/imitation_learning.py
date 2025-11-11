"""
模仿学习(Imitation Learning)
使用人工示例进行行为克隆(Behavioral Cloning)
然后继续用TRPO进行强化学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import game2048
import pickle
import argparse
from trpo_game2048_simple import SimpleTRPO, SimplePolicy, SimpleValue
from fast_trpo import FastTRPO


class ExpertDataset(Dataset):
    """专家示例数据集"""
    
    def __init__(self, trajectories):
        self.states = []
        self.actions = []
        
        # 从轨迹中提取所有状态-动作对
        for traj in trajectories:
            for state, action in zip(traj['states'], traj['actions']):
                self.states.append(state)
                self.actions.append(action)
        
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        
        print(f"数据集大小: {len(self.states)} 个状态-动作对")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = torch.FloatTensor(self.states[idx])
        action = torch.LongTensor([self.actions[idx]])
        return state, action


class ImitationLearning:
    """模仿学习 + TRPO强化学习"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', impl='fast'):
        self.device = device
        self.impl = impl
        if impl == 'fast':
            self.agent = FastTRPO(num_envs=64, device=device)
        elif impl == 'basic':
            self.agent = SimpleTRPO(device=device)
        else:
            raise ValueError("未知的TRPO实现方式,请选择 'fast' 或 'basic'")
        
    def load_expert_data(self, filename):
        """加载专家示例数据"""
        print(f"加载专家数据: {filename}")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        trajectories = data['trajectories']
        print(f"加载了 {len(trajectories)} 局游戏的轨迹")
        
        # 统计信息
        rewards = [t['total_reward'] for t in trajectories]
        steps = [t['steps'] for t in trajectories]
        max_tiles = [t['max_tile'] for t in trajectories]
        
        print(f"专家数据统计:")
        print(f"  平均奖励: {np.mean(rewards):.1f}")
        print(f"  平均步数: {np.mean(steps):.1f}")
        print(f"  最大方块: {np.max(max_tiles)}")
        
        return trajectories
    
    def behavioral_cloning(self, trajectories, epochs=50, batch_size=4096, lr=1e-3):
        """行为克隆训练"""
        print("\n" + "=" * 70)
        print("阶段 1: 行为克隆 (Behavioral Cloning)")
        print("=" * 70)
        
        # 创建数据集和数据加载器
        dataset = ExpertDataset(trajectories)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 创建优化器(只训练策略网络)
        optimizer = torch.optim.Adam(self.agent.policy.parameters(), lr=lr)
        
        # 训练
        best_loss = float('inf')
        best_epoch = 1
        avg_accuracy = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0
            num_batches = 0
            
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.squeeze().to(self.device)
                
                # 前向传播
                probs = self.agent.policy(states)
                
                # 计算交叉熵损失
                loss = F.cross_entropy(probs, actions)
                
                # 计算准确率
                pred_actions = torch.argmax(probs, dim=1)
                accuracy = (pred_actions == actions).float().mean()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_accuracy = total_accuracy / num_batches
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"损失: {avg_loss:.4f} | "
                      f"准确率: {avg_accuracy*100:.2f}% | "
                      f"最佳: {best_loss:.4f} (Epoch {best_epoch})")
        
        print(f"\n行为克隆完成!")
        print(f"最终准确率: {avg_accuracy*100:.2f}%")
        
        # # 评估模仿学习后的策略
        # print("\n评估行为克隆后的策略:")
        # env = gym.make("Game2048-v0")
        # self.agent.evaluate(env, num_episodes=5)
        # env.close()
    
    def continue_with_rl(self, save_path, num_iterations=50, batch_size=1024):
        """继续使用TRPO强化学习"""
        print("\n" + "=" * 70)
        print("阶段 2: TRPO强化学习")
        print("=" * 70)
        print("从模仿学习的策略开始,继续用TRPO优化...\n")
        
        # 使用TRPO继续训练
        self.agent.train(num_iterations=num_iterations, batch_size=batch_size, save_path=save_path)

    def full_pipeline(self, expert_data_file, save_path,
                      bc_epochs=50, bc_batch_size=64, bc_lr=1e-3,
                      rl_iterations=50, rl_batch_size=1024):
        """完整的训练流程: 模仿学习 -> 强化学习"""
        print("\n" + "🚀" * 35)
        print("模仿学习 + 强化学习 完整训练流程")
        print("🚀" * 35)
        
        # 阶段1: 加载数据
        print("\n" + "=" * 70)
        print("准备阶段: 加载专家数据")
        print("=" * 70)
        trajectories = self.load_expert_data(expert_data_file)
        
        # 阶段2: 行为克隆
        self.behavioral_cloning(
            trajectories, 
            epochs=bc_epochs, 
            batch_size=bc_batch_size, 
            lr=bc_lr
        )
        
        # 保存行为克隆后的模型
        print("\n保存行为克隆后的模型...")
        self.agent.save_model("model_after_bc.pth")
        
        # 阶段3: 强化学习
        self.continue_with_rl(
            save_path=save_path,
            num_iterations=rl_iterations, 
            batch_size=rl_batch_size
        )
        
        # # 最终评估
        # print("\n" + "=" * 70)
        # print("最终评估")
        # print("=" * 70)
        # env: game2048.Game2048Env = gym.make("Game2048-v0") # type: ignore
        # self.agent.evaluate(env, num_episodes=10)
        # env.close()
        
        print("\n" + "=" * 70)
        print("✅ 训练流程完成!")
        print("=" * 70)
        print("\n生成的模型:")
        print("  1. model_after_bc.pth - 行为克隆后的模型")
        print("  2. model_after_bc_and_rl.pth - 强化学习后的最终模型")
        print("\n使用 play_trpo.py 可以观看训练效果")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模仿学习 + TRPO强化学习')
    parser.add_argument('--data', type=str, help='专家数据文件路径')
    parser.add_argument('--save-path', type=str, default='model_after_bc_and_rl.pth', help='模型保存路径')
    parser.add_argument('--impl', type=str, default='fast', choices=['fast', 'basic'], help='TRPO实现方式')
    parser.add_argument('--bc-epochs', type=int, default=25, help='行为克隆训练轮数')
    parser.add_argument('--bc-batch-size', type=int, default=4096, help='行为克隆批量大小')
    parser.add_argument('--bc-lr', type=float, default=1e-4, help='行为克隆学习率')
    parser.add_argument('--rl-iterations', type=int, default=100, help='强化学习迭代次数')
    parser.add_argument('--rl-batch-size', type=int, default=2048, help='强化学习批量大小')
    parser.add_argument('--compare', action='store_true', help='对比不同模型')
    parser.add_argument('--bc-only', action='store_true', help='仅进行行为克隆')
    
    args = parser.parse_args()
    
    if args.data:
        # 训练
        il = ImitationLearning()
        
        if args.bc_only:
            # 仅行为克隆
            trajectories = il.load_expert_data(args.data)
            il.behavioral_cloning(
                trajectories,
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
                lr=args.bc_lr
            )
            il.agent.save_model("model_after_bc.pth")
        else:
            # 完整流程
            il.full_pipeline(
                expert_data_file=args.data,
                save_path=args.save_path,
                bc_epochs=args.bc_epochs,
                bc_batch_size=args.bc_batch_size,
                bc_lr=args.bc_lr,
                rl_iterations=args.rl_iterations,
                rl_batch_size=args.rl_batch_size
            )
    else:
        print(parser.usage)


if __name__ == "__main__":
    main()
