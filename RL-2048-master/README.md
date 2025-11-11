# RL-2048: 基于强化学习的2048游戏AI

这是一个使用TRPO（Trust Region Policy Optimization）算法训练2048游戏AI的项目。

## 快速开始

### 训练模型

推荐使用 `fast_trpo.py` 进行训练：

```bash
python fast_trpo.py
```

该脚本包含了优化的TRPO实现，适合快速训练和实验。

### 其他文件说明

- `fast_trpo.py` - **推荐使用**：优化的TRPO训练脚本
- `trpo_game2048_simple.py` - 简化版TRPO实现（使用卷积网络）
- `compare_policies.py` - 策略对比工具
- `imitation_learning.py` - 模仿学习实现
- `training_monitor.py` - 训练监控工具
- `test_sampling.py` - 采样测试工具

## 重要提示：奖励函数设置

本项目的训练效果高度依赖于 `game2048` 库中的奖励函数设置。

### 当前推荐的奖励函数

本项目当前将 `game2048` 库中的奖励函数修改为以下实现：

```python
def _calculate_reward(self, prev_grid, reward_merge, terminated):
    reward = 0.0
    # 结束惩罚
    reward -= 100.0 if terminated else 0.0
    # 最大值奖励
    reward += np.log2(np.max(self.grid)) if np.max(self.grid) > np.max(prev_grid) else 0.0
    # 合并奖励: reward_merge = sum(log2(tile)) for each merged tile
    reward += reward_merge
    return float(reward)
```

### 奖励函数说明

该奖励函数包含三个部分：

1. **结束惩罚** (`-100.0`)：当游戏结束时给予负奖励，鼓励智能体尽可能延长游戏时间
2. **最大值奖励** (`log2(max_tile)`)：当棋盘上出现更大的方块时给予奖励，鼓励合成更大的数字
3. **合并奖励** (`reward_merge`)：每次合并方块时累加被合并方块的log2值，鼓励频繁合并

### 如何修改奖励函数

1. 找到 `game2048` 库安装位置（通常在Python的site-packages目录下）
2. 找到环境实现文件（`game2048/envs/game2048v.py`）
3. 修改 `_calculate_reward` 方法为上述实现
4. 保存后重新运行训练脚本
