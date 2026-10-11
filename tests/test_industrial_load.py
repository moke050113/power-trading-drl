import unittest
from load_models.industrial_load import IndustrialLoad


class IndustrialLoadTests(unittest.TestCase):
    def test_conservation_and_window_for_different_requests(self):
        for request in (0, 1.7, 100):
            with self.subTest(request=request):
                model = IndustrialLoad(task_energy_mwh=7.5, max_power_mw=3, start_hour=2, end_hour=5)
                rows = [model.step(hour, request) for hour in range(24)]
                self.assertAlmostEqual(sum(r['load_mwh'] for r in rows), 7.5)
                self.assertEqual(model.remaining_energy_mwh, 0)
                for row in rows:
                    self.assertLessEqual(row['power_mw'], 3)
                    if not 2 <= row['hour'] < 5:
                        self.assertEqual(row['load_mwh'], 0)

    def test_deadline_forcing_reset_and_invalid_requests(self):
        model = IndustrialLoad(task_energy_mwh=6, max_power_mw=3, start_hour=0, end_hour=2)
        self.assertEqual(model.step(0, 0)['load_mwh'], 3)
        self.assertEqual(model.step(1, 0)['load_mwh'], 3)
        self.assertEqual(model.reset(), 6)
        for hour, power in ((1, 0), (0, -1), (0, float('nan'))):
            with self.assertRaises(ValueError):
                model.step(hour, power)
        self.assertEqual(model.hour, 0)

    def test_zero_task_and_infeasible_configuration(self):
        model = IndustrialLoad(task_energy_mwh=0, max_power_mw=0)
        self.assertEqual(sum(model.step(h, 2)['load_mwh'] for h in range(24)), 0)
        for kwargs in ({'task_energy_mwh': 100}, {'max_power_mw': -1}, {'start_hour': 16},
                       {'end_hour': 25}, {'task_energy_mwh': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IndustrialLoad(**kwargs)
