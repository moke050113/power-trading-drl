import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from env.power_trading_env import PowerTradingEnv


class DailyEnvironmentTests(unittest.TestCase):
    def test_original_first_hour_settlement(self):
        env = PowerTradingEnv()
        env.reset()
        _, reward, done, truncated, info = env.step([70])
        self.assertEqual(info['long_term_cost'], 28000)
        self.assertEqual(info['spot_cost'], 18000)
        self.assertEqual(info['total_cost'], 46000)
        self.assertEqual(reward, -46000)
        self.assertFalse(done)
        self.assertFalse(truncated)

    def test_full_day_balance_cost_and_termination(self):
        loads = np.arange(1, 25, dtype=float)
        prices = np.arange(300, 540, 10, dtype=float)
        env = PowerTradingEnv(load_profile=loads, spot_price_profile=prices)
        state, _ = env.reset()
        rewards = []
        for hour in range(24):
            self.assertEqual(state[0], loads[hour])
            self.assertEqual(state[2], prices[hour])
            self.assertEqual(state[3], hour)
            state, reward, done, truncated, info = env.step([loads[hour] / 2])
            rewards.append(reward)
            self.assertEqual(info['hour'], hour)
            self.assertEqual(info['long_term_quantity'] + info['spot_quantity'], loads[hour])
            self.assertEqual(done, hour == 23)
            self.assertFalse(truncated)
            self.assertTrue(env.observation_space.contains(state))
        expected = np.sum(loads / 2 * (400 + prices))
        self.assertEqual(info['episode_cost'], expected)
        self.assertEqual(-sum(rewards), expected)
        self.assertEqual(state[3], 24)
        with self.assertRaises(RuntimeError):
            env.step([0])
        state, _ = env.reset()
        self.assertEqual(state[3], 0)
        self.assertEqual(env.episode_cost, 0)

    def test_clipping_zero_load_and_single_step(self):
        for load, action, expected in [(10, -5, 0), (10, 50, 10), (0, 20, 0)]:
            with self.subTest(load=load, action=action):
                env = PowerTradingEnv(horizon=1, load_profile=[load], spot_price_profile=[600])
                env.reset()
                _, _, done, _, info = env.step([action])
                self.assertTrue(done)
                self.assertEqual(info['long_term_quantity'], expected)
                self.assertEqual(info['spot_quantity'], load - expected)

    def test_invalid_actions_do_not_advance(self):
        env = PowerTradingEnv()
        with self.assertRaises(RuntimeError):
            env.step([0])
        env.reset()
        for action in [[], [1, 2], [[1]], 1, [np.nan], [np.inf]]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.hour, 0)
        self.assertEqual(env.episode_cost, 0)

    def test_invalid_configuration(self):
        for kwargs in [dict(horizon=0), dict(horizon=1.5), dict(horizon=True),
                       dict(load_profile=[100]), dict(load_profile=[-1] * 24),
                       dict(load_profile=[1001] * 24), dict(spot_price_profile=[np.nan] * 24),
                       dict(spot_price_profile=[2001] * 24), dict(long_term_price=np.inf),
                       dict(long_term_price=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PowerTradingEnv(**kwargs)

    def test_observations_and_profiles_are_copied(self):
        loads = [100] * 24
        env = PowerTradingEnv(load_profile=loads)
        loads[0] = 500
        state, _ = env.reset(seed=42)
        state[:] = -1
        self.assertEqual(env.state[0], 100)
        _, _, _, _, info = env.step([70])
        self.assertEqual(info['total_cost'], 46000)
        original, _ = env.reset(seed=42)
        repeated, _ = env.reset(seed=42)
        np.testing.assert_array_equal(original, repeated)

    def test_gymnasium_contract(self):
        check_env(PowerTradingEnv(), skip_render_check=True)


if __name__ == '__main__':
    unittest.main()
