import unittest
from agents.baseline import BaselinePolicy
from env.power_trading_env import PowerTradingEnv
from experiments.evaluate import evaluate_policy


class EvaluationTests(unittest.TestCase):
    def test_daily_cost_reward_and_fixed_repeatability(self):
        for strategy, expected in (('spot', 1514750), ('fixed70', 1287425)):
            result = evaluate_policy(BaselinePolicy(strategy), episodes=2)
            self.assertEqual(result['cost_mean_yuan'], expected)
            self.assertEqual(result['reward_mean'], -expected)
            self.assertEqual(result['cost_std_yuan'], 0)
            self.assertEqual([r['steps'] for r in result['episodes']], [24, 24])

    def test_random_statistics_reproducible(self):
        first = evaluate_policy(BaselinePolicy('random', 8))
        self.assertEqual(first, evaluate_policy(BaselinePolicy('random', 8)))
        self.assertGreater(first['cost_std_yuan'], 0)
        for row in first['episodes']:
            self.assertAlmostEqual(row['cost_yuan'], -row['reward'])

    def test_single_step_and_invalid_episode_count(self):
        factory = lambda: PowerTradingEnv(horizon=1)
        row = evaluate_policy(BaselinePolicy('fixed70'), env_factory=factory)['episodes'][0]
        self.assertEqual(row['cost_yuan'], 46000)
        self.assertEqual(row['steps'], 1)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluate_policy(BaselinePolicy(), episodes=count)
