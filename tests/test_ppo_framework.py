import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from agents.train_ppo import ContractRatioAction, make_training_env, train_ppo
from env.power_trading_env import PowerTradingEnv


class PPOFrameworkTests(unittest.TestCase):
    def test_ratio_mapping_cost_and_lifecycle(self):
        env = make_training_env()
        with self.assertRaises(RuntimeError):
            env.step([0.7])
        env.reset()
        for hour in range(24):
            _, reward, done, _, info = env.step([0.7])
            self.assertAlmostEqual(info['long_term_quantity'], info['load'] * 0.7)
            self.assertEqual(reward, -info['total_cost'])
            self.assertEqual(done, hour == 23)
        self.assertAlmostEqual(info['episode_cost'], 1287425)
        with self.assertRaises(RuntimeError):
            env.step([0])
        env.reset()
        self.assertEqual(env.step([2])[4]['long_term_quantity'], 100)
        env.reset()
        self.assertEqual(env.step([-1])[4]['long_term_quantity'], 0)
        env.reset()
        for action in ([np.nan], [1, 2]):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.unwrapped.hour, 0)

    def test_single_step_compatibility_and_argument_check(self):
        env = ContractRatioAction(PowerTradingEnv(horizon=1))
        env.reset()
        self.assertTrue(env.step([0.7])[2])
        with self.assertRaises(ValueError):
            train_ppo(total_timesteps=0)

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), '基础环境未安装可选PPO依赖')
    def test_real_training_save_load_and_evaluate(self):
        # 此项在PPO虚拟环境运行真实SB3训练，基础环境仅跳过这项。
        output_root = Path(__file__).resolve().parents[1] / 'outputs'
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as directory:
            report = train_ppo(total_timesteps=24, output_dir=directory)
            self.assertEqual(report['actual_timesteps'], 24)
            self.assertTrue((Path(directory) / 'ppo_power_trading.zip').exists())
            for row in report['evaluation']['episodes']:
                self.assertEqual(row['steps'], 24)
                self.assertAlmostEqual(row['cost_yuan'], -row['reward'])
