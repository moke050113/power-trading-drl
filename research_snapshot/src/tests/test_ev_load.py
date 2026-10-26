import unittest
from load_models.ev_load import EVLoad


class EVLoadTests(unittest.TestCase):
    def test_soc_power_window_and_grid_energy(self):
        model = EVLoad(capacity_mwh=10, initial_soc=0.2, max_power_mw=3,
                       start_hour=1, end_hour=4, efficiency=0.8, target_soc=0.7)
        rows = [model.step(hour, 100) for hour in range(24)]
        for row, expected in zip(rows[:5], [0, 3, 3, 0.25, 0]):
            self.assertAlmostEqual(row['load_mwh'], expected)
        self.assertAlmostEqual(sum(r['load_mwh'] for r in rows), 6.25)
        self.assertAlmostEqual(sum(r['battery_energy_mwh'] for r in rows), 5)
        self.assertAlmostEqual(model.soc, 0.7)
        self.assertTrue(rows[-1]['target_reached'])

    def test_reset_and_unmet_target(self):
        model = EVLoad(start_hour=0, end_hour=1)
        for hour in range(24):
            row = model.step(hour, 0)
        self.assertFalse(row['target_reached'])
        with self.assertRaises(ValueError):
            model.step(24, 1)
        self.assertEqual(model.reset(), 0.2)
        with self.assertRaises(ValueError):
            model.step(1, 1)
        for power in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                model.step(0, power)
        self.assertEqual(model.hour, 0)

    def test_invalid_configuration(self):
        for kwargs in ({'capacity_mwh': 0}, {'initial_soc': 2}, {'efficiency': 0},
                       {'max_power_mw': -1}, {'start_hour': 24}, {'end_hour': 1.5},
                       {'target_soc': 0.1}, {'capacity_mwh': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EVLoad(**kwargs)
