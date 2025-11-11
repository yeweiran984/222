"""
对比测试: 随机策略 vs 训练的TRPO智能体
"""

import numpy as np
import torch
import gymnasium as gym
import game2048
from trpo_game2048_simple import SimpleTRPO
from trpo_game2048 import TRPO
import time
import argparse


def test_random_policy(env, num_episodes=10):
    """测试随机策略"""
    print("=" * 70)
    print("测试随机策略 (完全随机选择动作)")
    print("=" * 70)
    
    episode_rewards = []
    max_tiles = []
    steps_list = []
    
    per_episode = []  # collect for sorting by reward desc
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()  # 随机动作
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += float(reward)
            steps += 1
        
        max_tile = np.max(state)
        episode_rewards.append(episode_reward)
        max_tiles.append(max_tile)
        steps_list.append(steps)
        per_episode.append((episode_reward, int(max_tile), steps, ep + 1))
    
    # 按奖励降序打印
    print("\n按奖励降序 (随机策略):")
    for rank, (r, tile, st, original_ep) in enumerate(sorted(per_episode, key=lambda x: x[0], reverse=True), 1):
        print(f"#{rank:2d} 回合(原{original_ep:2d}): 奖励={r:8.1f}, 最大方块={tile:4d}, 步数={st:4d}")
    
    print(f"\n随机策略统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):8.1f} (± {np.std(episode_rewards):.1f})")
    print(f"  平均最大方块: {np.mean(max_tiles):6.1f} (± {np.std(max_tiles):.1f})")
    print(f"  平均步数: {np.mean(steps_list):6.1f} (± {np.std(steps_list):.1f})")
    
    return episode_rewards, max_tiles, steps_list


def test_trpo_policy(env, model_path, num_episodes=10):
    """测试TRPO策略"""
    print("\n" + "=" * 70)
    print("测试TRPO训练的策略")
    print("=" * 70)
    
    # 加载智能体
    # env = gym.make("Game2048-v0", debug=True)
    agent = TRPO(env)
    try:
        agent.load_model(model_path)
        print(f"成功加载模型: {model_path}\n")
    except Exception as e:
        print(f"警告: 无法加载模型 {model_path}")
        raise e
    
    episode_rewards = []
    max_tiles = []
    steps_list = []
    
    per_episode = []  # collect for sorting by reward desc
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += float(reward)
            steps += 1
        
        max_tile = np.max(state)
        episode_rewards.append(episode_reward)
        max_tiles.append(max_tile)
        steps_list.append(steps)
        per_episode.append((episode_reward, int(max_tile), steps, ep + 1))
    
    # 按奖励降序打印
    print("\n按奖励降序 (TRPO策略):")
    for rank, (r, tile, st, original_ep) in enumerate(sorted(per_episode, key=lambda x: x[0], reverse=True), 1):
        print(f"#{rank:2d} 回合(原{original_ep:2d}): 奖励={r:8.1f}, 最大方块={tile:4d}, 步数={st:4d}")
    
    print(f"\nTRPO策略统计:")
    print(f"  平均奖励: {np.mean(episode_rewards):8.1f} (± {np.std(episode_rewards):.1f})")
    print(f"  平均最大方块: {np.mean(max_tiles):6.1f} (± {np.std(max_tiles):.1f})")
    print(f"  平均步数: {np.mean(steps_list):6.1f} (± {np.std(steps_list):.1f})")
    
    return episode_rewards, max_tiles, steps_list


def compare_policies(num_episodes=20, model_path="trpo_game2048_simple.pth", debug: bool = False):
    """对比两种策略"""
    print("\n" + "🎮" * 35)
    print("2048游戏策略对比测试")
    print("🎮" * 35 + "\n")
    
    # 创建环境
    env_random = gym.make("Game2048-v0")  # 随机策略环境不传 debug
    
    # 测试随机策略
    random_rewards, random_tiles, random_steps = test_random_policy(env_random, num_episodes)
    
    # 测试TRPO策略
    env_agent = gym.make("Game2048-v0", debug=debug)  # 仅传给智能体的环境
    trpo_rewards, trpo_tiles, trpo_steps = test_trpo_policy(
        env_agent, model_path, num_episodes
    )
    
    # 对比结果
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    
    print("\n平均奖励对比:")
    print(f"  随机策略: {np.mean(random_rewards):8.1f}")
    print(f"  TRPO策略:  {np.mean(trpo_rewards):8.1f}")
    improvement_reward = (np.mean(trpo_rewards) - np.mean(random_rewards)) / np.mean(random_rewards) * 100
    print(f"  提升:      {improvement_reward:7.1f}% {'✅' if improvement_reward > 0 else '❌'}")
    
    print("\n平均最大方块对比:")
    print(f"  随机策略: {np.mean(random_tiles):6.1f}")
    print(f"  TRPO策略:  {np.mean(trpo_tiles):6.1f}")
    improvement_tile = (np.mean(trpo_tiles) - np.mean(random_tiles)) / np.mean(random_tiles) * 100
    print(f"  提升:      {improvement_tile:6.1f}% {'✅' if improvement_tile > 0 else '❌'}")
    
    print("\n平均步数对比:")
    print(f"  随机策略: {np.mean(random_steps):6.1f}")
    print(f"  TRPO策略:  {np.mean(trpo_steps):6.1f}")
    improvement_steps = (np.mean(trpo_steps) - np.mean(random_steps)) / np.mean(random_steps) * 100
    print(f"  提升:      {improvement_steps:6.1f}% {'✅' if improvement_steps > 0 else '❌'}")
    
    # 方块分布对比
    print("\n最大方块分布对比:")
    print("-" * 70)
    print(f"{'方块':^10s} | {'随机策略':^15s} | {'TRPO策略':^15s}")
    print("-" * 70)
    
    all_tiles = sorted(set(list(random_tiles) + list(trpo_tiles)), reverse=True)
    for tile in all_tiles:
        random_count = random_tiles.count(tile)
        trpo_count = trpo_tiles.count(tile)
        random_pct = random_count / len(random_tiles) * 100
        trpo_pct = trpo_count / len(trpo_tiles) * 100
        print(f"{int(tile):^10d} | {random_count:3d} ({random_pct:5.1f}%) | {trpo_count:3d} ({trpo_pct:5.1f}%)")
    
    print("-" * 70)
    
    # 总结
    print("\n" + "=" * 70)
    print("💡 结论")
    print("=" * 70)
    
    if improvement_reward > 50:
        print("✅ TRPO智能体表现优异,显著超过随机策略!")
    elif improvement_reward > 0:
        print("✅ TRPO智能体有所改进,但还有提升空间")
        print("💡 建议: 继续训练或调整超参数")
    else:
        print("❌ TRPO智能体未能超过随机策略")
        print("💡 建议: 检查训练过程,可能需要重新训练")
    
    print("\n如果结果不理想,可以:")
    print("  1. 增加训练迭代次数 (num_iterations)")
    print("  2. 调整学习率和其他超参数")
    print("  3. 使用更大的网络 (hidden_dim)")
    print("  4. 检查奖励函数设计")
    
    print("\n" + "=" * 70)
    
    env_random.close()
    env_agent.close()


def quick_compare(debug: bool = False):
    """快速对比 (少量回合)"""
    print("\n快速对比测试 (5个回合)\n")
    compare_policies(num_episodes=5, debug=debug)


def full_compare(model_path="trpo_game2048_simple.pth", debug: bool = False):
    """完整对比 (更多回合,更准确)"""
    print("\n完整对比测试 (20个回合)\n")
    compare_policies(num_episodes=20, model_path=model_path, debug=debug)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="对比测试: 随机策略 vs 训练的TRPO智能体")
    args.add_argument("--path", type=str, default="trpo_game2048_simple.pth", help="TRPO模型文件路径")
    args.add_argument("--debug", action="store_true", help="仅传给TRPO智能体环境的debug标志")
    parsed_args = args.parse_args()
    full_compare(model_path=parsed_args.path, debug=parsed_args.debug)
