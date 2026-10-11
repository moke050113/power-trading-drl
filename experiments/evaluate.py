"""按完整episode统计原始成本和Reward，固定合成曲线不代表泛化效果。"""
import argparse
import json
from pathlib import Path
import numpy as np
from agents.baseline import BaselinePolicy
from env.power_trading_env import PowerTradingEnv


def evaluate_policy(policy, episodes=3, seed=0, env_factory=PowerTradingEnv):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError('episodes必须是正整数')
    records = []
    for episode in range(episodes):
        env = env_factory()
        try:
            state, _ = env.reset(seed=seed + episode)
            cost = reward_sum = 0.0
            steps = 0
            while True:
                action, _ = policy.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                cost += float(info['total_cost'])  # total_cost是本小时，不重复累加episode_cost
                reward_sum += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            records.append({'episode': episode, 'steps': steps, 'cost_yuan': cost, 'reward': reward_sum})
        finally:
            env.close()
    costs = [row['cost_yuan'] for row in records]
    rewards = [row['reward'] for row in records]
    return {'episodes': records, 'cost_mean_yuan': float(np.mean(costs)),
            'cost_std_yuan': float(np.std(costs)), 'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=Path, help='可选：另行评估PPO模型zip文件')
    parser.add_argument('--output', type=Path, default=Path('outputs/baseline_evaluation.json'))
    args = parser.parse_args()
    results = {name: evaluate_policy(BaselinePolicy(name, args.seed), args.episodes, args.seed)
               for name in ('spot', 'fixed70', 'random')}
    if args.model:
        from stable_baselines3 import PPO
        from agents.train_ppo import make_training_env
        # 模型动作单位是比例，必须通过训练时同样的包装器评估。
        results['ppo'] = evaluate_policy(PPO.load(args.model, device='cpu'), args.episodes,
                                         args.seed, make_training_env)
    for name, result in results.items():
        print(f"{name}: 成本均值 {result['cost_mean_yuan']:.2f} 元，标准差 {result['cost_std_yuan']:.2f} 元；"
              f"Reward均值 {result['reward_mean']:.2f}，标准差 {result['reward_std']:.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'固定合成曲线运行统计，非学习效果证明。已保存：{args.output.resolve()}')


if __name__ == '__main__':
    main()
