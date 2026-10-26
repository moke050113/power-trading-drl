"""S2a设备级验证；不读市场数据，不接交易，不训练Agent。"""
from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from load_models.ev_fleet import EVFleetConfig, EVFleetExecutor
from load_models.ev_load import EVLoad


class EVFleetTests(unittest.TestCase):
    def ready(self, config=None):
        fleet = EVFleetExecutor(config)
        fleet.reset()
        return fleet

    @staticmethod
    def advance(fleet, end, request=0):
        return [fleet.step(h, request) for h in range(fleet.hour, end)]

    def test_approved_units_and_deadline_hand_calculation(self):
        cfg = EVFleetConfig()
        self.assertEqual((cfg.vehicle_count, cfg.capacity_mwh, cfg.max_power_mw), (1000, 60, 7))
        self.assertAlmostEqual(cfg.required_grid_mwh, 100/3)
        self.assertEqual(cfg.max_power_mw*(cfg.end_hour-cfg.start_hour), 42)
        self.assertAlmostEqual(cfg.required_grid_mwh/cfg.max_power_mw, 100/21)
        self.assertGreater(42, cfg.required_grid_mwh)

    def test_zero_requests_forced_schedule_and_deadline_satisfied(self):
        fleet = self.ready();rows = self.advance(fleet, 24)
        expected = [0]*19 + [16/3] + [7]*4
        np.testing.assert_allclose([r['grid_energy_mwh'] for r in rows], expected, atol=1e-12)
        self.assertAlmostEqual(rows[19]['minimum_power_mw'], 16/3)
        self.assertAlmostEqual(rows[19]['soc'], .28)
        self.assertAlmostEqual(fleet.soc, .7)
        self.assertTrue(rows[23]['target_reached'])
        self.assertAlmostEqual(sum(r['grid_energy_mwh'] for r in rows), 100/3)
        self.assertAlmostEqual(sum(r['battery_energy_mwh'] for r in rows), 30)
        self.assertAlmostEqual(sum(r['forced_charging_mwh'] for r in rows), 100/3)
        self.assertAlmostEqual(rows[-1]['remaining_grid_mwh'], 0)

    def test_full_requests_finish_early_and_target_caps_final_hour(self):
        fleet = self.ready();rows = self.advance(fleet, 24, 7)
        np.testing.assert_allclose([r['grid_energy_mwh'] for r in rows], [0]*18+[7]*4+[16/3,0], atol=1e-12)
        self.assertTrue(rows[22]['target_reached'])
        self.assertEqual(rows[23]['grid_energy_mwh'], 0)
        self.assertEqual(rows[23]['protected_power_mw'], 0)
        self.assertEqual(sum(r['forced_charging_mwh'] for r in rows), 0)

    def test_soc_and_grid_battery_conservation_for_mixed_requests(self):
        fleet = self.ready();previous=.2;grid=battery=forced=0.
        for hour in range(24):
            request = (0, 2, 4, 7)[hour % 4]
            row=fleet.step(hour,request)
            grid += row['grid_energy_mwh'];battery += row['battery_energy_mwh'];forced += row['forced_charging_mwh']
            self.assertAlmostEqual(row['battery_energy_mwh'], row['grid_energy_mwh']*.9)
            self.assertAlmostEqual((row['soc']-previous)*60,row['battery_energy_mwh'])
            self.assertAlmostEqual(row['total_grid_mwh'],grid)
            self.assertAlmostEqual(row['total_battery_mwh'],battery)
            self.assertAlmostEqual(row['total_forced_mwh'],forced)
            self.assertLessEqual(row['actual_power_mw'],7)
            self.assertLessEqual(row['soc'],.7)
            self.assertGreaterEqual(row['soc'],previous)
            if not 18<=hour<24:
                self.assertEqual((row['grid_energy_mwh'],row['protected_power_mw']), (0,0))
            previous=row['soc']
        self.assertAlmostEqual(battery,30);self.assertAlmostEqual(grid,100/3)

    def test_single_car_homogeneous_scaling_each_hour(self):
        fleet = self.ready();single = self.ready(replace(EVFleetConfig(),vehicle_count=1))
        for h in range(24):
            a=fleet.step(h,(0,1.3,7)[h%3]);b=single.step(h,(0,.0013,.007)[h%3])
            self.assertAlmostEqual(a['soc'],b['soc'],places=12)
            self.assertEqual(a['target_reached'],b['target_reached'])
            for field in ('minimum_power_mw','protected_power_mw','actual_power_mw',
                          'grid_energy_mwh','battery_energy_mwh','remaining_grid_mwh',
                          'forced_charging_mwh','total_grid_mwh','total_battery_mwh'):
                self.assertAlmostEqual(a[field],1000*b[field],places=10,msg=field)

    def test_raw_evload_semantics_one_call_and_no_duplicate_energy(self):
        fleet=self.ready();raw=EVLoad(capacity_mwh=60,max_power_mw=7,initial_soc=.2,
                                    target_soc=.7,efficiency=.9,start_hour=18,end_hour=24)
        original=EVLoad.step;calls=[]
        def tracked(model,h,request):
            calls.append((h,request));return original(model,h,request)
        with patch.object(EVLoad,'step',tracked):
            rows=self.advance(fleet,24)
        self.assertEqual(len(calls),24)
        self.assertEqual([h for h,_ in calls],list(range(24)))
        for h,row in enumerate(rows):
            other=raw.step(h,row['protected_power_mw'])
            self.assertEqual(row['grid_energy_mwh'],other['load_mwh'])
            self.assertEqual(row['battery_energy_mwh'],other['battery_energy_mwh'])
            self.assertEqual(row['soc'],other['soc'])

    def test_unreachable_soc_rejected_on_reset_without_lowering_target(self):
        config=replace(EVFleetConfig(),target_soc=1)
        fleet=EVFleetExecutor(config)
        with self.assertRaisesRegex(ValueError,'不可达'):fleet.reset()
        self.assertEqual(fleet.status,'UNINITIALIZED')
        self.assertEqual(fleet.config.target_soc,1)
        for cfg in (replace(EVFleetConfig(),max_charging_power_kw=0),
                    replace(EVFleetConfig(),end_hour=19),
                    replace(EVFleetConfig(),efficiency=.5)):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):EVFleetExecutor(cfg).reset()

    def test_invalid_config_types_bounds_and_overflow(self):
        cases=[{'vehicle_count':0},{'vehicle_count':1.5},{'vehicle_count':True},
               {'battery_capacity_kwh':0},{'max_charging_power_kw':-1},
               {'initial_soc':-1},{'target_soc':1.1},{'target_soc':.1},
               {'efficiency':0},{'efficiency':1.1},{'efficiency':True},
               {'start_hour':24},{'end_hour':18},{'end_hour':23.5},
               {'battery_capacity_kwh':float('inf')},{'initial_soc':float('nan')},
               {'battery_capacity_kwh':'60'},{'max_charging_power_kw':[7]},
               {'vehicle_count':10**400},{'battery_capacity_kwh':1e308}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):EVFleetConfig(**kwargs)
        with self.assertRaises(TypeError):EVFleetExecutor({})

    def test_config_frozen_and_state_views_and_records_are_copies(self):
        fleet=self.ready()
        with self.assertRaises(FrozenInstanceError):fleet.config.target_soc=1
        view=fleet.state_view();view['soc']=0;view['hour']=99
        row=fleet.step(0,0);row['soc']=99
        records=fleet.hourly_records;records[0]['soc']=99
        self.assertEqual(fleet.soc,.2);self.assertEqual(fleet.hour,1)
        self.assertEqual(fleet.hourly_records[0]['soc'],.2)

    def test_invalid_request_does_not_execute_device_and_can_retry(self):
        fleet=self.ready();old=fleet.state_view()
        with patch.object(EVLoad,'step',side_effect=AssertionError('不得执行')):
            for request in (-1,7.0001,True,np.bool_(False),'1',[1],np.array([1]),float('nan'),float('inf'),1j):
                with self.subTest(request=request),self.assertRaises(ValueError):fleet.step(0,request)
                self.assertEqual(fleet.state_view(),old)
        self.assertEqual(fleet.step(0,0)['hour'],0)

    def test_invalid_hour_order_and_after_day_finished(self):
        fleet=EVFleetExecutor()
        with self.assertRaises(RuntimeError):fleet.step(0,0)
        fleet.reset()
        for hour in (-1,1,True,.0,24):
            with self.subTest(hour=hour),self.assertRaises(ValueError):fleet.step(hour,0)
        fleet.step(0,0)
        with self.assertRaises(ValueError):fleet.step(0,0)
        self.advance(fleet,24)
        self.assertEqual((fleet.hour,fleet.status),(24,'FINISHED'))
        with self.assertRaises(RuntimeError):fleet.step(24,0)

    def test_candidate_exception_after_soc_change_rolls_back_and_retry_once(self):
        fleet=self.ready();self.advance(fleet,19);old=fleet.state_view();records=fleet.hourly_records
        original=EVLoad.step
        def fail(model,h,request):
            original(model,h,request)
            raise RuntimeError('候选已充电后异常')
        with patch.object(EVLoad,'step',fail),self.assertRaises(RuntimeError):fleet.step(19,0)
        self.assertEqual(fleet.state_view(),old);self.assertEqual(fleet.hourly_records,records)
        row=fleet.step(19,0)
        self.assertAlmostEqual(row['grid_energy_mwh'],16/3)
        self.assertEqual(fleet.hour,20)
        self.assertAlmostEqual(fleet.state_view()['total_grid_mwh'],16/3)

    def test_corrupt_candidate_outputs_never_commit(self):
        original=EVLoad.step
        cases=[('load_mwh',1),('power_mw',1),('battery_energy_mwh',1),('soc',.6),
               ('soc',float('nan')),('hour',True),('hour',2),('target_reached',True),
               ('target_reached',1)]
        fleet=self.ready();old=fleet.state_view()
        for key,value in cases:
            def corrupt(model,h,request):
                row=original(model,h,request);row[key]=value;return row
            with patch.object(EVLoad,'step',corrupt),self.assertRaises((ValueError,RuntimeError)):
                fleet.step(0,0)
            self.assertEqual(fleet.state_view(),old)
        self.assertEqual(fleet.step(0,0)['grid_energy_mwh'],0)

    def test_candidate_internal_clock_soc_or_config_corruption_rejected(self):
        fleet=self.ready();old=fleet.state_view();original=EVLoad.step
        for name,value in [('hour',3),('soc',.5),('max_power_mw',8),('target_soc',.8)]:
            def corrupt(model,h,request):
                row=original(model,h,request);setattr(model,name,value);return row
            with patch.object(EVLoad,'step',corrupt),self.assertRaises(RuntimeError):fleet.step(0,0)
            self.assertEqual(fleet.state_view(),old)

    def test_reset_failure_preserves_existing_valid_state(self):
        fleet=self.ready();old=fleet.state_view();cfg=fleet.config
        with self.assertRaises(ValueError):fleet.reset(config=replace(cfg,target_soc=1))
        self.assertEqual(fleet.config,cfg);self.assertEqual(fleet.state_view(),old)
        original=EVLoad.reset
        def fail(model):
            original(model);model.soc=.9
            raise RuntimeError('候选初始化失败')
        with patch.object(EVLoad,'reset',fail),self.assertRaises(RuntimeError):fleet.reset()
        self.assertEqual(fleet.state_view(),old)

    def test_no_midday_reset_and_new_day_does_not_carry_soc(self):
        fleet=self.ready();fleet.step(0,0);old=fleet.state_view()
        with self.assertRaises(RuntimeError):fleet.reset()
        self.assertEqual(fleet.state_view(),old)
        self.advance(fleet,24)
        view=fleet.reset()
        self.assertEqual((view['hour'],view['soc'],view['total_grid_mwh']), (0,.2,0))
        self.assertEqual(fleet.hourly_records,[])

    def test_instances_do_not_share_device_state_or_history(self):
        cfg=EVFleetConfig();first=self.ready(cfg);second=self.ready(cfg)
        self.advance(first,19,7)
        self.assertEqual(second.hour,0);self.assertEqual(second.soc,.2)
        self.assertEqual(second.hourly_records,[])
        self.assertGreater(first.soc,second.soc)

    def test_already_satisfied_target_and_zero_power_feasible(self):
        cfg=replace(EVFleetConfig(),target_soc=.2,max_charging_power_kw=0)
        fleet=self.ready(cfg);rows=self.advance(fleet,24)
        self.assertTrue(all(r['target_reached'] for r in rows))
        self.assertEqual(sum(r['grid_energy_mwh'] for r in rows),0)

    def test_exact_capacity_boundary_and_short_nonmidnight_window(self):
        cfg=EVFleetConfig(vehicle_count=1,battery_capacity_kwh=10000,max_charging_power_kw=1000,
                          initial_soc=0,target_soc=.6,efficiency=1,start_hour=1,end_hour=7)
        fleet=self.ready(cfg);rows=self.advance(fleet,24)
        np.testing.assert_allclose([r['grid_energy_mwh'] for r in rows],[0]+[1]*6+[0]*17,
                                   rtol=0,atol=1e-12)
        self.assertAlmostEqual(fleet.soc,.6)

    def test_reproducible_identical_call_order_without_rng_or_prices(self):
        first=self.ready();second=self.ready()
        for h in range(24):
            self.assertEqual(first.step(h,2.5),second.step(h,2.5))
        self.assertTrue(all(math.isfinite(x) for r in first.hourly_records for x in r.values()))


if __name__ == '__main__':
    unittest.main()
