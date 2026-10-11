"""PPO短训练框架：只验证运行，不证明策略有效；A组环境保持不变。"""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from env.power_trading_env import PowerTradingEnv


class ContractRatioAction(gym.ActionWrapper):
    """训练动作[0,1]映射为当小时合同MWh；结算info和Reward保持原单位。"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        ratio = np.asarray(action, dtype=float)
        if ratio.shape != (1,) or not np.all(np.isfinite(ratio)):
            raise ValueError('训练动作必须为一个有限比例值')
        env = self.env.unwrapped
        if env.state is None or env.hour >= env.horizon:
            raise RuntimeError('请先reset；episode结束后再次reset')
        load = float(env.load_profile[env.hour])
        return np.array([float(np.clip(ratio[0], 0, 1)) * load], dtype=np.float64)


def make_training_env():
    return ContractRatioAction(PowerTradingEnv())


def train_ppo(total_timesteps=96, seed=0, output_dir=Path('outputs/ppo_smoke')):
    if isinstance(total_timesteps, bool) or not isinstance(total_timesteps, int) or total_timesteps < 1:
        raise ValueError('total_timesteps必须是正整数')
    try:
        import torch
        import stable_baselines3 as sb3
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise RuntimeError('缺少PPO依赖，请按照README安装requirements-ppo.txt并使用PPO虚拟环境') from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = make_training_env()
    try:
        check_env(env, warn=True)
    finally:
        env.close()
    # 仅缩放训练Reward，不缩放观察值；保存模型即可在原始Reward上评估。
    vector_env = VecNormalize(DummyVecEnv([lambda: Monitor(make_training_env())]),
                              norm_obs=False, norm_reward=True)
    try:
        model = PPO('MlpPolicy', vector_env, n_steps=24, batch_size=24, n_epochs=2,
                    seed=seed, device='cpu', verbose=0, policy_kwargs={'net_arch': [32, 32]})
        model.learn(total_timesteps=total_timesteps)
        model_path = output_dir / 'ppo_power_trading.zip'
        model.save(model_path)
        vector_env.save(str(output_dir / 'reward_normalization.pkl'))
        # 保存再加载，验证预测与完整episode结算链路。
        from experiments.evaluate import evaluate_policy
        result = evaluate_policy(PPO.load(model_path, device='cpu'), episodes=2,
                                 seed=seed, env_factory=make_training_env)
        report = {'purpose': '运行验证，非训练效果证明', 'seed': seed,
                  'requested_timesteps': total_timesteps, 'actual_timesteps': model.num_timesteps,
                  'stable_baselines3': sb3.__version__, 'torch': torch.__version__,
                  'gymnasium': gym.__version__, 'evaluation': result}
        (output_dir / 'smoke_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
    finally:
        vector_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=96)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/ppo_smoke'))
    args = parser.parse_args()
    report = train_ppo(args.timesteps, args.seed, args.output_dir)
    print(f"PPO运行验证完成：请求{report['requested_timesteps']}步，实际{report['actual_timesteps']}步。")
    print(f"加载后评估原始成本均值：{report['evaluation']['cost_mean_yuan']:.2f} 元；非学习效果证明。")
    print(f'模型和报告目录：{args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
