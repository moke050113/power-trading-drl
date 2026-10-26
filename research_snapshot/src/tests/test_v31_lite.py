"""Lite boundaries with isolated fixtures; never loads old validation or any test."""
import csv,json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from v31.lite_data import LiteStore
from v31.lite_engine import LiteAnnualDriver
from v31.lite_rules import RULES
from v31.lite_gym import WorldCache,AnnualDecisionEnv,encode


class LiteTests(unittest.TestCase):
    def test_test_gate_before_history_or_arrays(self):
        with patch('v31.lite_data.HistoryContext',side_effect=AssertionError('must not construct')):
            with self.assertRaises(PermissionError):LiteStore(split='test')

    def test_first_forecast_is_actual_p1_p2_legal_lineage(self):
        s=LiteStore();rows=s.context.records['op:0']
        for r in rows:
            self.assertLess(r.published_at,r.grid.hours[0]);self.assertTrue(r.input_vintages)
            for ref in r.input_vintages:s.context.history.resolve(ref,r.as_of)
        with self.assertRaises(RuntimeError):LiteAnnualDriver(s).c.world.day_forecast(0,s.start)

    def test_no_future_realization_in_upper_or_lower(self):
        d=LiteAnnualDriver(LiteStore(group='G01'))
        raw=d.advance(stop='upper',lower=RULES['B0'][1])
        before={k:v.copy() for k,v in raw.items()}
        d.c.world._arrays=dict(d.c.world._arrays)
        for key in ('realized_load','balancing_price'):d.c.world._arrays[key]=d.c.world._arrays[key]+123
        self.assertTrue(all(np.array_equal(v,d.upper_observation()[k]) for k,v in before.items()))
        self.assertTrue(np.all(raw['device_state']==0))
        d.submit_upper(*RULES['B0'][0](raw));o=d.advance(stop='lower',upper=RULES['B0'][0])
        before={k:v.copy() for k,v in o.items()}
        d.c.world._arrays['realized_load']=d.c.world._arrays['realized_load']+33
        self.assertTrue(all(np.array_equal(v,d.lower_observation()[k]) for k,v in before.items()))
        self.assertEqual(o['devices'][0],0);self.assertTrue(np.array_equal(o['actual_state_mask'],[0,0]))

    def test_next_day_is_planned_epoch_not_old_actual_soc(self):
        d=LiteAnnualDriver(LiteStore());o=d.advance(stop='lower',upper=RULES['B0'][0])
        for h in range(24):
            d.submit_lower([0,0,0]);o=d.advance(stop='lower',upper=RULES['B0'][0])
        self.assertEqual(d.c.t,24);self.assertEqual(d.c.devices.hour,24)
        self.assertAlmostEqual(o['devices'][1],100/3,places=5)
        self.assertEqual(o['devices'][0],0);self.assertTrue(np.array_equal(o['actual_state_mask'],[0,0]))
        self.assertEqual(len(d.daily_receipts),1)

    def test_invalid_and_duplicate_hour_action_do_not_charge(self):
        d=LiteAnnualDriver(LiteStore());d.advance(stop='lower',upper=RULES['B0'][0])
        before=(d.c.ledger.digest(),d.c.total_purchase,d.c.devices.hour)
        with self.assertRaises(ValueError):d.submit_lower([0,2,0])
        self.assertEqual(before,(d.c.ledger.digest(),d.c.total_purchase,d.c.devices.hour))
        d.submit_lower([0,0,0]);before=d.c.total_purchase
        with self.assertRaises(RuntimeError):d.submit_lower([0,0,0])
        self.assertEqual(before,d.c.total_purchase)

    def test_source_failure_retains_pending_and_stops(self):
        d=LiteAnnualDriver(LiteStore());d.advance(stop='lower',upper=RULES['B0'][0]);d.submit_lower([0,0,0])
        with patch.object(d.c.world,'reveal',side_effect=ValueError('controlled source fault')):
            with self.assertRaises(ValueError):d.advance(stop='lower',upper=RULES['B0'][0])
        self.assertTrue(d.aborted);self.assertIsNotNone(d.c.pending);self.assertEqual(d.c.total_purchase,0)
        with self.assertRaises(RuntimeError):d.advance(stop='lower',upper=RULES['B0'][0])

    def test_cache_requires_end_audit_and_source_hash(self):
        s=LiteStore()
        s.context.sealed={i:'unit-only-sentinel' for i in range(s.n//24)}
        with self.assertRaises(RuntimeError):s.replay_copy()
        # Unit-test metadata only, no delivery/clock is fabricated or executed.
        s.completed_year=True;s.arrays=dict(s.arrays);s.arrays['balancing_price']=s.arrays['balancing_price']+1
        with self.assertRaises(RuntimeError):s.replay_copy()

    def test_encoding_keeps_mixed_category_mask_and_shapes(self):
        d=LiteAnnualDriver(LiteStore());raw=d.advance(stop='upper',lower=RULES['B2'][1]);obs=encode(raw,'upper')
        self.assertTrue(np.array_equal(raw['calendar'],obs['calendar']))
        self.assertTrue(np.array_equal(raw['stage'],obs['stage']))
        self.assertEqual(obs['resource_outlook'].shape,(4,));self.assertTrue(all(v.dtype==np.float32 for v in obs.values()))

    def test_gym_upper_real_macro_and_reset_restriction(self):
        spec=dict(year=2033,family='unit',group='G00',split='train',fixture=True)
        e=AnnualDecisionEnv(WorldCache([spec]),'upper');o,_=e.reset(seed=0)
        self.assertTrue(e.observation_space.contains(o));o2,_=e.reset(seed=0)
        self.assertTrue(all(np.array_equal(v,o2[k]) for k,v in o.items()))
        _,r,done,trunc,info=e.step(np.r_[2.,np.full(12,.3)].astype(np.float32))
        self.assertEqual(len(e.driver.c.ledger.lots),1);self.assertFalse(done);self.assertFalse(trunc)
        self.assertEqual(r,info['raw_reward']/1e6)
        with self.assertRaises(RuntimeError):e.reset()

    def test_gym_lower_reward_and_actual_execution(self):
        spec=dict(year=2033,family='unit',group='G00',split='train',fixture=True)
        e=AnnualDecisionEnv(WorldCache([spec]),'lower',partner=RULES['B0'][0]);o,_=e.reset(seed=0)
        self.assertTrue(e.observation_space.contains(o));_,r,done,trunc,info=e.step(np.zeros(3,np.float32))
        row=e.driver.c.hours[0];self.assertAlmostEqual(r,-row['objective_cost']/1e6)
        self.assertFalse(done);self.assertFalse(trunc)
        with self.assertRaises(RuntimeError):e.reset()

    def test_stage_a_actual_ordinary_leap_and_repeat_evidence(self):
        root=Path(__file__).resolve().parents[1]/'outputs/v31/p5_lite/stage_a_20261017'
        p=json.loads((root/'gate_passed.json').read_text())
        self.assertTrue(p['passed']);self.assertEqual(p['actual_full_annual_runs'],12)
        for year,n in ((2033,8760),(2036,8784)):
            for strategy in RULES:
                rr=[r for r in p['results'] if r['year']==year and r['strategy']==strategy]
                self.assertEqual(len(rr),2);self.assertEqual(rr[0]['hours'],n)
                self.assertEqual(rr[0]['row_sha256'],rr[1]['row_sha256'])


if __name__=='__main__':unittest.main()
