import unittest

from compare_strategies import compare_strategies


class StrategyComparisonTests(unittest.TestCase):
    def test_shared_curves_balance_and_daily_costs(self):
        results = compare_strategies()
        for key, rows in results.items():
            with self.subTest(strategy=key):
                self.assertEqual([row['hour'] for row in rows], list(range(24)))
                cumulative = 0.0
                for row, source in zip(rows, results['spot']):
                    for field in ('load', 'spot_price', 'long_term_price'):
                        self.assertEqual(row[field], source[field])
                    self.assertAlmostEqual(row['long_term_quantity'] + row['spot_quantity'], row['load'])
                    cumulative += row['total_cost']
                    self.assertAlmostEqual(row['episode_cost'], cumulative)
        self.assertEqual(results['fixed70'][-1]['episode_cost'], 1287425.0)

    def test_reference_switches_markets_and_handles_ties_and_zero_load(self):
        # 高于/低于/等于合同价，以及零负荷；用独立成本公式核对结算。
        loads = [10, 20, 30, 0] * 6
        prices = [600, 300, 400, 800] * 6
        results = compare_strategies(loads, prices)
        self.assertEqual([row['long_term_quantity'] for row in results['cheapest'][:4]], [10, 0, 0, 0])
        expected = {
            'spot': sum(load * price for load, price in zip(loads, prices)),
            'fixed70': sum(load * (0.7 * 400 + 0.3 * price) for load, price in zip(loads, prices)),
            'cheapest': sum(load * min(400, price) for load, price in zip(loads, prices)),
        }
        for key, cost in expected.items():
            self.assertAlmostEqual(results[key][-1]['episode_cost'], cost)
        for hour in range(24):
            reference = results['cheapest'][hour]['total_cost']
            self.assertLessEqual(reference, results['spot'][hour]['total_cost'])
            self.assertLessEqual(reference, results['fixed70'][hour]['total_cost'])


if __name__ == '__main__':
    unittest.main()
