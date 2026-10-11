"""本次检查补充：需求边界、独立统计公式及A/B/C离线接口。"""
import math
import unittest

import numpy as np

from agents.baseline import BaselinePolicy
from agents.train_ppo import ContractRatioAction
from env.power_trading_env import PowerTradingEnv
from experiments.evaluate import evaluate_policy
from load_models.ev_load import EVLoad
from load_models.industrial_load import IndustrialLoad


class ModuleIntegrationTests(unittest.TestCase):
    def test_ev_soc_zero_full_and_satisfied_demand(self):
        for initial, target in ((0, 1), (1, 1), (0.5, 0.5)):
            with self.subTest(initial=initial, target=target):
                ev = EVLoad(capacity_mwh=0.06, initial_soc=initial, target_soc=target,
                            max_power_mw=0.1, start_hour=23, end_hour=24, efficiency=0.9)
                rows = [ev.step(hour, 1) for hour in range(24)]
                self.assertTrue(all(row['load_mwh'] == 0 for row in rows[:23]))
                self.assertTrue(all(0 <= row['soc'] <= 1 for row in rows))
                self.assertTrue(all(0 <= row['power_mw'] <= 0.1 for row in rows))
                self.assertAlmostEqual(sum(row['load_mwh'] for row in rows), (target - initial) * 0.06 / 0.9)
                self.assertAlmostEqual(ev.soc, target)
                self.assertTrue(rows[-1]['target_reached'])

    def test_ev_default_demand_shortfall_and_zero_power(self):
        ev = EVLoad()
        rows = [ev.step(hour, 1) for hour in range(24)]
        self.assertAlmostEqual(sum(row['load_mwh'] for row in rows), 0.042)
        self.assertAlmostEqual(ev.soc, 0.83)
        self.assertFalse(rows[-1]['target_reached'])
        self.assertAlmostEqual((ev.target_soc - ev.soc) * ev.capacity_mwh, 0.0102)
        for hour, row in enumerate(rows):
            self.assertAlmostEqual(row['load_mwh'], 0.007 if 18 <= hour < 24 else 0)
        ev = EVLoad(max_power_mw=0)
        self.assertEqual(sum(ev.step(hour, 1)['load_mwh'] for hour in range(24)), 0)
        self.assertEqual(ev.soc, ev.initial_soc)

    def test_industrial_random_feasible_tasks(self):
        rng = np.random.default_rng(11)
        for case in range(100):
            start = int(rng.integers(0, 24))
            end = int(rng.integers(start + 1, 25))
            maximum = float(rng.uniform(0.01, 10))
            energy = float(rng.uniform(0, maximum * (end - start)))
            model = IndustrialLoad(energy, maximum, start, end)
            with self.subTest(case=case):
                rows = [model.step(hour, float(rng.uniform(0, maximum * 2))) for hour in range(24)]
                self.assertAlmostEqual(sum(row['load_mwh'] for row in rows), energy, places=10)
                self.assertLessEqual(model.remaining_energy_mwh, 1e-10)
                for row in rows:
                    self.assertGreaterEqual(row['power_mw'], 0)
                    self.assertLessEqual(row['power_mw'], maximum + 1e-10)
                    if not start <= row['hour'] < end:
                        self.assertEqual(row['load_mwh'], 0)

    def test_statistics_against_known_three_episode_costs(self):
        class ThreeActions:
            def __init__(self):
                self.actions = iter((0, 50, 100))

            def predict(self, state, deterministic=True):
                return [next(self.actions)], None

        result = evaluate_policy(ThreeActions(), episodes=3,
                                 env_factory=lambda: PowerTradingEnv(horizon=1))
        self.assertEqual([row['cost_yuan'] for row in result['episodes']], [60000, 50000, 40000])
        self.assertEqual(result['cost_mean_yuan'], 50000)
        self.assertEqual(result['reward_mean'], -50000)
        expected_std = math.sqrt((10000 ** 2 + 0 + 10000 ** 2) / 3)
        self.assertAlmostEqual(result['cost_std_yuan'], expected_std)
        self.assertAlmostEqual(result['reward_std'], expected_std)

    def test_abc_offline_load_curve_and_baselines(self):
        source = PowerTradingEnv()
        ev = EVLoad(capacity_mwh=1, initial_soc=0, max_power_mw=0.5,
                    start_hour=0, end_hour=4, efficiency=1)
        industry = IndustrialLoad(12, 4, 8, 16)
        additions = [ev.step(hour, 0.5)['load_mwh'] + industry.step(hour, 0)['load_mwh']
                     for hour in range(24)]
        self.assertAlmostEqual(sum(additions), 13)
        loads = source.load_profile + additions
        prices = source.spot_price_profile
        factory = lambda: PowerTradingEnv(load_profile=loads, spot_price_profile=prices)
        for strategy in ('spot', 'fixed70', 'random'):
            result = evaluate_policy(BaselinePolicy(strategy, 11), episodes=1, env_factory=factory)
            row = result['episodes'][0]
            self.assertEqual(row['steps'], 24)
            self.assertAlmostEqual(row['reward'], -row['cost_yuan'])
            if strategy != 'random':
                ratio = 0 if strategy == 'spot' else 0.7
                expected = sum(load * (ratio * 400 + (1 - ratio) * price)
                               for load, price in zip(loads, prices))
                # 基准动作按float32返回，小数MWh会产生亚分级成本舍入差。
                self.assertAlmostEqual(row['cost_yuan'], expected, delta=0.01)

    def test_ppo_adapter_with_custom_load_curve(self):
        loads = [0, 0.1, 200, 999] * 6
        env = ContractRatioAction(PowerTradingEnv(load_profile=loads))
        env.reset()
        for hour, load in enumerate(loads):
            state, reward, done, _, info = env.step([0.7])
            self.assertAlmostEqual(info['long_term_quantity'], load * 0.7)
            self.assertAlmostEqual(info['long_term_quantity'] + info['spot_quantity'], load)
            self.assertEqual(reward, -info['total_cost'])
            self.assertEqual(done, hour == 23)
            self.assertTrue(env.observation_space.contains(state))
        env.close()
