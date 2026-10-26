"""V2.0内核独立验收；不替代原V1测试，不需要Gym适配器或PPO。"""
import unittest
import numpy as np

from env.multiscale_trading_day import MultiscaleTradingDay
from env.power_trading_env import PowerTradingEnv


def scenario(**changes):
    result = {'forecast_load': np.full(24, 100.0), 'forecast_spot_price': np.full(24, 550.0),
              'realized_base_load': np.full(24, 100.0), 'realized_spot_price': np.full(24, 600.0),
              'contract_price': 400.0}
    result.update(changes)
    return result


def initialized(**changes):
    day = MultiscaleTradingDay()
    day.reset(**scenario(**changes))
    return day


def finish(day):
    return [day.settle_next_hour() for _ in range(24)]


class MultiscaleTradingDayTests(unittest.TestCase):
    def test_alpha_bounds_and_forecast_hourly_plan(self):
        forecast = np.arange(1, 25, dtype=np.float64)
        for alpha in (0, 0.7, 1):
            with self.subTest(alpha=alpha):
                day = initialized(forecast_load=forecast)
                returned = day.lock_contract([alpha])
                np.testing.assert_array_equal(returned, alpha * forecast)
                self.assertEqual(day.contract_ratio, alpha)
                self.assertEqual(day.status, 'LOCKED')
                self.assertEqual(day.hour, 0)
                self.assertEqual(day.episode_cost, 0)
                self.assertEqual(day.contract_plan.dtype, np.float64)

    def test_four_independent_hand_calculations(self):
        for load, used, buy, surplus, spot, cost in (
            (100, 70, 30, 0, 18000, 46000), (120, 70, 50, 0, 30000, 58000),
            (50, 50, 0, 20, 0, 28000), (0, 0, 0, 70, 0, 28000),
        ):
            with self.subTest(load=load):
                day = initialized(realized_base_load=np.full(24, load))
                day.lock_contract([0.7])
                record = day.settle_next_hour()
                expected = {'contract_quantity': 70, 'contract_used': used, 'spot_buy': buy,
                            'contract_surplus': surplus, 'contract_cost': 28000,
                            'spot_cost': spot, 'total_cost': cost, 'episode_cost': cost,
                            'hour_reward': -cost}
                for key, value in expected.items():
                    self.assertEqual(record[key], value, key)

    def test_prediction_error_keeps_plan_and_pays_surplus(self):
        actual = np.array([120, 50, 0] * 8, dtype=np.float64)
        day = initialized(realized_base_load=actual)
        day.lock_contract([0.7])
        expected_plan = day.contract_plan
        rows = finish(day)
        np.testing.assert_array_equal(day.contract_plan, expected_plan)
        self.assertEqual(sum(r['contract_cost'] for r in rows), 24 * 28000)
        self.assertEqual(sum(r['spot_buy'] for r in rows), 8 * 50)
        self.assertEqual(sum(r['contract_surplus'] for r in rows), 8 * (20 + 70))
        self.assertEqual(day.episode_cost, 8 * (58000 + 28000 + 28000))

    def test_balances_and_cost_against_fractional_formula(self):
        forecast = np.linspace(0.123456789, 120.87654321, 24)
        actual = forecast[::-1].copy()
        prices = np.linspace(0, 987.654321, 24)
        ratio, contract_price = 0.73123456789, 412.3456789
        day = initialized(forecast_load=forecast, realized_base_load=actual,
                          realized_spot_price=prices, contract_price=contract_price)
        day.lock_contract([ratio])
        rows = finish(day)
        expected_total = 0.0
        for h, row in enumerate(rows):
            q = float(forecast[h]) * ratio
            expected = q * contract_price + max(float(actual[h]) - q, 0) * float(prices[h])
            expected_total += expected
            self.assertAlmostEqual(row['total_cost'], expected, places=8)
            self.assertAlmostEqual(row['episode_cost'], expected_total, places=8)
            self.assertAlmostEqual(row['contract_used'] + row['spot_buy'], actual[h], places=12)
            self.assertAlmostEqual(row['contract_quantity'] + row['spot_buy'],
                                   actual[h] + row['contract_surplus'], places=12)
            for key in ('contract_quantity', 'contract_cost', 'total_cost', 'episode_cost'):
                self.assertIsInstance(row[key], np.float64)
        self.assertAlmostEqual(day.episode_cost, expected_total, places=8)
        self.assertAlmostEqual(sum(r['hour_reward'] for r in rows), -expected_total, places=8)

    def test_zero_forecast_is_all_spot_even_when_alpha_one(self):
        day = initialized(forecast_load=np.zeros(24))
        day.lock_contract([1])
        rows = finish(day)
        self.assertEqual(sum(r['contract_quantity'] for r in rows), 0)
        self.assertEqual(day.episode_cost, 24 * 60000)

    def test_zero_load_and_zero_prices(self):
        day = initialized(forecast_load=np.zeros(24), realized_base_load=np.zeros(24),
                          forecast_spot_price=np.zeros(24), realized_spot_price=np.zeros(24),
                          contract_price=0)
        day.lock_contract([1])
        self.assertEqual(sum(r['total_cost'] for r in finish(day)), 0)

    def test_full_lifecycle_and_new_day_reset(self):
        day = initialized()
        day.lock_contract([0.7])
        for h in range(24):
            row = day.settle_next_hour()
            self.assertEqual(row['hour'], h)
            self.assertEqual(day.hour, h + 1)
            self.assertEqual(day.status, 'FINISHED' if h == 23 else 'LOCKED')
        self.assertEqual(len(day.hourly_records), 24)
        with self.assertRaises(RuntimeError):
            day.settle_next_hour()
        day.reset(**scenario(realized_base_load=np.full(24, 120.0)))
        self.assertEqual((day.status, day.hour, day.episode_cost), ('UNLOCKED', 0, 0))
        with self.assertRaises(RuntimeError):
            _ = day.contract_plan
        day.lock_contract([0])
        self.assertEqual(sum(r['total_cost'] for r in finish(day)), 24 * 72000)

    def test_illegal_call_order_and_midday_reset(self):
        day = MultiscaleTradingDay()
        for operation in (lambda: day.lock_contract([0.7]), day.settle_next_hour,
                          day.decision_view, lambda: day.contract_plan, lambda: day.contract_ratio):
            with self.assertRaises(RuntimeError):
                operation()
        day.reset(**scenario())
        with self.assertRaises(RuntimeError):
            day.settle_next_hour()
        day.lock_contract([0.7])
        for h in (0, 1):
            with self.subTest(hour=h):
                with self.assertRaises(RuntimeError):
                    day.lock_contract([0])
                with self.assertRaises(RuntimeError):
                    day.reset(**scenario())
                self.assertEqual(day.hour, h)
                self.assertEqual(day.contract_ratio, 0.7)
                day.settle_next_hour()
        for _ in range(22):
            day.settle_next_hour()
        with self.assertRaises(RuntimeError):
            day.lock_contract([0])

    def test_invalid_actions_do_not_lock_or_advance(self):
        day = initialized()
        for action in ([], [1, 2], [[0.7]], 0.7, [-0.01], [1.01], [np.nan], [np.inf],
                       [-np.inf], [True], ['0.7'], [1j], [None]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                day.lock_contract(action)
            self.assertEqual((day.status, day.hour, day.episode_cost), ('UNLOCKED', 0, 0))
        day.lock_contract([0.7])
        self.assertEqual(day.contract_ratio, 0.7)

    def test_invalid_profiles_and_prices_rejected_atomically(self):
        day = initialized()
        view = day.decision_view()
        for name, upper in (('forecast_load', 1000), ('realized_base_load', 1000),
                            ('forecast_spot_price', 2000), ('realized_spot_price', 2000)):
            for bad in ([], [1] * 23, [1] * 25, [[1] * 24], [np.nan] * 24,
                        [np.inf] * 24, [-1] * 24, [upper + 1] * 24, [True] * 24,
                        ['1'] * 24, [1j] * 24):
                with self.subTest(field=name, value=repr(bad)[:25]), self.assertRaises(ValueError):
                    day.reset(**scenario(**{name: bad}))
                np.testing.assert_array_equal(day.decision_view()['forecast_load'], view['forecast_load'])
                self.assertEqual(day.status, 'UNLOCKED')
        for price in ([-1], [400], -1, 2001, np.nan, np.inf, True, '400', 400 + 1j):
            with self.subTest(price=price), self.assertRaises(ValueError):
                day.reset(**scenario(contract_price=price))
        day.lock_contract([0.7])
        self.assertEqual(day.settle_next_hour()['total_cost'], 46000)

    def test_negative_price_not_clipped(self):
        for field in ('forecast_spot_price', 'realized_spot_price'):
            source = np.full(24, -100.0)
            day = MultiscaleTradingDay()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '禁止裁剪'):
                day.reset(**scenario(**{field: source}))
            self.assertEqual(day.status, 'UNINITIALIZED')
            np.testing.assert_array_equal(source, np.full(24, -100.0))

    def test_inclusive_upper_bounds_and_full_contract(self):
        day = initialized(forecast_load=np.full(24, 1000), realized_base_load=np.full(24, 1000),
                          forecast_spot_price=np.full(24, 2000), realized_spot_price=np.full(24, 2000),
                          contract_price=2000)
        day.lock_contract([1])
        rows = finish(day)
        self.assertEqual(sum(r['spot_buy'] for r in rows), 0)
        self.assertEqual(day.episode_cost, 48000000)

    def test_invalid_reset_preserves_completed_day(self):
        day = initialized()
        day.lock_contract([0.7])
        finish(day)
        rows, plan, cost = day.hourly_records, day.contract_plan, day.episode_cost
        with self.assertRaises(ValueError):
            day.reset(**scenario(realized_spot_price=np.full(24, -1.0)))
        self.assertEqual((day.status, day.hour, day.episode_cost), ('FINISHED', 24, cost))
        self.assertEqual(day.hourly_records, rows)
        np.testing.assert_array_equal(day.contract_plan, plan)

    def test_input_plan_observation_and_record_copies(self):
        source = scenario()
        day = MultiscaleTradingDay()
        view = day.reset(**source)
        for values in source.values():
            if isinstance(values, np.ndarray):
                values[:] = 999
        for values in view.values():
            values[:] = 0
        returned = day.lock_contract([0.7])
        returned[:] = 0
        plan = day.contract_plan
        plan[:] = 1
        first = day.settle_next_hour()
        self.assertEqual(first['total_cost'], 46000)
        first['total_cost'] = -1
        for _ in range(23):
            day.settle_next_hour()
        logs = day.hourly_records
        self.assertEqual(logs[0]['total_cost'], 46000)
        logs[0]['total_cost'] = 0
        logs.clear()
        self.assertEqual(len(day.hourly_records), 24)
        self.assertEqual(day.hourly_records[0]['total_cost'], 46000)
        np.testing.assert_array_equal(day.contract_plan, np.full(24, 70.0))
        for property_name in ('hour', 'status', 'episode_cost', 'contract_ratio', 'contract_plan'):
            with self.subTest(property=property_name), self.assertRaises(AttributeError):
                setattr(day, property_name, 0)

    def test_realized_values_do_not_enter_decision_or_plan(self):
        first = initialized()
        second = initialized(realized_base_load=np.full(24, 20.0), realized_spot_price=np.full(24, 900.0))
        self.assertEqual(set(first.decision_view()), {'forecast_load', 'forecast_spot_price', 'contract_price'})
        for key in first.decision_view():
            np.testing.assert_array_equal(first.decision_view()[key], second.decision_view()[key])
        first.lock_contract([0.7])
        second.lock_contract([0.7])
        np.testing.assert_array_equal(first.contract_plan, second.contract_plan)
        self.assertNotEqual(sum(r['total_cost'] for r in finish(first)), sum(r['total_cost'] for r in finish(second)))

    def test_forecast_prices_are_not_settlement_prices(self):
        day = initialized(forecast_spot_price=np.full(24, 1.0))
        day.lock_contract([0.7])
        row = day.settle_next_hour()
        self.assertEqual(row['forecast_spot_price'], 1)
        self.assertEqual(row['spot_cost'], 18000)
        self.assertEqual(set(row).intersection({'realized_base_load', 'realized_spot_price'}),
                         {'realized_base_load', 'realized_spot_price'})
        for key in ('realized_base_load', 'realized_spot_price'):
            self.assertEqual(np.asarray(row[key]).shape, ())

    def test_all_day_records_unavailable_until_finished(self):
        day = initialized()
        with self.assertRaises(RuntimeError):
            _ = day.hourly_records
        day.lock_contract([0.7])
        for _ in range(23):
            day.settle_next_hour()
            with self.assertRaises(RuntimeError):
                _ = day.hourly_records
        day.settle_next_hour()
        self.assertEqual(len(day.hourly_records), 24)

    def test_v1_equal_forecast_compatibility(self):
        v1 = PowerTradingEnv()
        v1.reset()
        day = initialized(forecast_load=v1.load_profile, realized_base_load=v1.load_profile,
                          forecast_spot_price=v1.spot_price_profile, realized_spot_price=v1.spot_price_profile)
        day.lock_contract([0.7])
        for _ in range(24):
            row = day.settle_next_hour()
            _, reward, _, _, info = v1.step([v1.load_profile[v1.hour] * 0.7])
            self.assertAlmostEqual(row['total_cost'], info['total_cost'], places=8)
            self.assertAlmostEqual(row['hour_reward'], reward, places=8)
        self.assertEqual(day.episode_cost, 1287425)


if __name__ == '__main__':
    unittest.main()
