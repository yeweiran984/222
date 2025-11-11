"""
训练监控工具
用于诊断和可视化训练过程中的问题
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import json


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        
        # 历史记录
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []
        self.kls = []
        self.learning_rates = []
        self.max_tiles = []
        self.episode_counts = []
        
        # 滑动窗口
        self.reward_window = deque(maxlen=window_size)
        
    def update(self, iteration, reward, value_loss, policy_loss, kl, lr, max_tile=None, episode_count=None):
        """更新监控数据"""
        self.rewards.append(reward)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
        self.kls.append(kl)
        self.learning_rates.append(lr)
        
        if max_tile is not None:
            self.max_tiles.append(max_tile)
        if episode_count is not None:
            self.episode_counts.append(episode_count)
        
        self.reward_window.append(reward)
        
    def detect_collapse(self):
        """检测训练崩溃"""
        if len(self.rewards) < self.window_size * 2:
            return False, ""
        
        recent_avg = np.mean(list(self.reward_window))
        previous_avg = np.mean(self.rewards[-self.window_size*2:-self.window_size])
        
        # 如果最近性能下降超过30%
        if recent_avg < previous_avg * 0.7:
            return True, f"性能下降 {((previous_avg - recent_avg) / previous_avg * 100):.1f}%"
        
        return False, ""
    
    def plot(self, save_path="training_monitor.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Monitoring Dashboard', fontsize=16)
        
        # 1. 奖励曲线
        ax = axes[0, 0]
        ax.plot(self.rewards, label='Episode Reward', alpha=0.6)
        if len(self.rewards) >= self.window_size:
            smoothed = np.convolve(self.rewards, 
                                  np.ones(self.window_size)/self.window_size, 
                                  mode='valid')
            ax.plot(range(self.window_size-1, len(self.rewards)), 
                   smoothed, label='Smoothed', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 价值损失
        ax = axes[0, 1]
        ax.plot(self.value_losses, label='Value Loss', color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Value Network Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. KL散度
        ax = axes[0, 2]
        ax.plot(self.kls, label='KL Divergence', color='red')
        ax.axhline(y=0.01, color='green', linestyle='--', label='Target KL')
        ax.axhline(y=0.015, color='orange', linestyle='--', label='Max KL')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Policy KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 学习率
        ax = axes[1, 0]
        ax.plot(self.learning_rates, label='Learning Rate', color='purple')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 5. 策略损失
        ax = axes[1, 1]
        if self.policy_losses:
            ax.plot(self.policy_losses, label='Policy Loss', color='green')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. 回合数
        ax = axes[1, 2]
        if self.episode_counts:
            ax.plot(self.episode_counts, label='Episodes per Iteration', color='brown')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Count')
            ax.set_title('Episodes per Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练监控图已保存: {save_path}")
        plt.close()
    
    def save_data(self, path="training_data.json"):
        """保存训练数据"""
        data = {
            'rewards': self.rewards,
            'value_losses': self.value_losses,
            'policy_losses': self.policy_losses,
            'kls': self.kls,
            'learning_rates': self.learning_rates,
            'max_tiles': self.max_tiles,
            'episode_counts': self.episode_counts
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"训练数据已保存: {path}")
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.rewards:
            return "暂无数据"
        
        recent_rewards = list(self.reward_window) if self.reward_window else self.rewards[-10:]
        
        summary = f"""
{'='*70}
训练摘要
{'='*70}
总迭代数: {len(self.rewards)}
最佳奖励: {max(self.rewards):.1f} (第 {np.argmax(self.rewards)+1} 次迭代)
最差奖励: {min(self.rewards):.1f} (第 {np.argmin(self.rewards)+1} 次迭代)
平均奖励: {np.mean(self.rewards):.1f}
最近{len(recent_rewards)}次平均: {np.mean(recent_rewards):.1f}

价值损失: 当前={self.value_losses[-1]:.4f}, 平均={np.mean(self.value_losses):.4f}
KL散度: 当前={self.kls[-1]:.6f}, 平均={np.mean(self.kls):.6f}
学习率: 当前={self.learning_rates[-1]:.2e}

趋势分析:
"""
        
        # 检测崩溃
        collapsed, msg = self.detect_collapse()
        if collapsed:
            summary += f"⚠️  警告: {msg}\n"
        else:
            summary += "✓ 训练稳定\n"
        
        summary += "=" * 70
        return summary
