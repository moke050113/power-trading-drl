import unittest
import numpy as np
from agents.baseline import BaselinePolicy


class BaselineTests(unittest.TestCase):
    def test_fixed_actions_and_zero_load(self):
        for strategy, expected in (('spot', 0), ('fixed70', 70)):
            policy = BaselinePolicy(strategy)
            action, _ = policy.predict([100, 400, 600, 0])
            self.assertEqual(action.shape, (1,))
            self.assertEqual(action[0], expected)
            self.assertEqual(policy.predict([0, 400, 600])[0][0], 0)

    def test_random_reproducibility_bounds_and_stream(self):
        policies = [BaselinePolicy('random', seed=42) for _ in range(2)]
        sequences = [[float(policy.predict([100, 400, 600])[0][0]) for _ in range(24)] for policy in policies]
        self.assertEqual(sequences[0], sequences[1])
        self.assertGreater(len(set(sequences[0])), 1)
        self.assertTrue(all(0 <= value <= 100 for value in sequences[0]))
        self.assertEqual(policies[0].predict([0, 400, 600])[0][0], 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            BaselinePolicy('unknown')
        for state in ([], [-1, 400, 600], [np.nan, 400, 600]):
            with self.assertRaises(ValueError):
                BaselinePolicy().predict(state)
