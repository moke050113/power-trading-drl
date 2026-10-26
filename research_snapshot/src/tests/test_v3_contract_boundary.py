"""D7a独立数值/原子性/同轨迹验收，无PPO训练或validation评价。"""
import ast,inspect,json,copy,unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from v3.core import PositionLedger,opportunities,year_axis,contract_coverage,vector,number,settlement
from v3.coordinator import YearCoordinator
from v3.environments import decode_upper
from experiments.verify_contract_boundary_v3 import before_modules,constant_fixture,commit,signature,ROOT,DOC,OUT,D7,rows

def first(scope,year=2011):return next(x for x in opportunities(year) if x.scope==scope and x.index==0)

def stopped_ledger():
 old,_=before_modules();records=rows(D7/'seed_0/upper_steps.csv')
 start=max(i for i,x in enumerate(records) if x['stage']=='Y' and x['opportunity']=='0')
 rr=records[start:];world=rr[0]['world'];year=int(world.split('_')[1])
 with np.load(ROOT/f'data/v3/synthetic_multiscale_v3/{world}.npz',allow_pickle=False) as z:r=z['reference'].copy();quotes=z['quotes'].copy()
 ledger=old.PositionLedger(year,r)
 for row in rr:
  op=next(x for x in opportunities(year) if x.scope==row['scope'] and x.index==int(row['opportunity']))
  action=np.array(json.loads(row['action']),dtype=np.float32);verb,u=decode_upper({'calendar':np.array([0,0,op.index])},action)
  if op.stage=='Y':
   target=np.zeros(len(r))
   for m in range(1,13):
    scope=first(f'M{m}',year);target[scope.start:scope.end]=u[m-1]
  else:
   oldratio=ledger.quantity[op.start:op.end]/r[op.start:op.end];target=oldratio+(1-oldratio)*u[0]
  price=float(quotes[opportunities(year).index(op)])
  ledger.decide(op,verb,target,price,now=op.decision_at,published_at=op.published_at,expires_at=op.decision_at,expected_version=ledger.version,trade_id=f'{op.scope}:{op.index}')
 return ledger,next(x for x in opportunities(year) if x.scope=='I2-3' and x.index==1)

class BoundaryTests(unittest.TestCase):
 def test_original_fixture_fails_and_fixed_succeeds(self):
  old,_=before_modules();legacy,op=constant_fixture(old.PositionLedger);q=legacy.quantity[op.start:op.end];r=legacy.reference[op.start:op.end]
  self.assertEqual(float((q/r).max()),np.nextafter(1.,np.inf))
  for u in (0.,.1,.5,.9,1.):
   oldcase=copy.deepcopy(legacy);target=q/r+(1-q/r)*u
   if u<.5:
    state=signature(oldcase)
    with self.assertRaises(ValueError):commit(oldcase,op,target)
    self.assertEqual(state,signature(oldcase))
   upgraded=copy.deepcopy(legacy);upgraded.__class__=PositionLedger
   prior=(upgraded.quantity,upgraded.fee,upgraded.lots)
   ratio=contract_coverage(q,r);commit(upgraded,op,ratio+(1-ratio)*u)
   np.testing.assert_array_equal(prior[0],upgraded.quantity);np.testing.assert_array_equal(prior[1],upgraded.fee)
   self.assertEqual(prior[2],upgraded.lots[:-1]);self.assertFalse(any(upgraded.lots[-1].quantities))
 def test_new_fixture_has_no_excess_and_lot_fee_agree(self):
  ledger,op=constant_fixture();self.assertTrue((ledger.quantity<=ledger.reference).all())
  old,_=before_modules();legacy,_=constant_fixture(old.PositionLedger)
  self.assertLessEqual(float(abs(ledger.quantity-legacy.quantity).max()),2*float(np.spacing(89.178170972871)))
  expected=np.zeros(8760)
  for lot in ledger.lots:expected[lot.start:lot.end]+=np.array(lot.quantities)*lot.unit_price
  np.testing.assert_array_equal(expected,ledger.fee)
  for u in (0.,.1,.5,.9,1.):
   case=copy.deepcopy(ledger);q=case.quantity[op.start:op.end];r=case.reference[op.start:op.end];ratio=contract_coverage(q,r)
   commit(case,op,ratio+(1-ratio)*u);self.assertTrue((case.quantity<=case.reference).all())
 def test_actual_D7_fourteen_hours_and_legacy_compatibility(self):
  legacy,op=stopped_ledger();q=legacy.quantity[op.start:op.end];r=legacy.reference[op.start:op.end]
  bad=np.flatnonzero(q>r)
  self.assertEqual(len(bad),14)
  original=rows(D7/'stopped_diagnostic/position_boundary.csv')
  self.assertEqual({op.start+int(x) for x in bad},{int(x['hour']) for x in original})
  for u in (0.,.1,.5,.9,1.):
   case=copy.deepcopy(legacy);case.__class__=PositionLedger;before=(case.quantity,case.fee,case.lots)
   ratio=contract_coverage(q,r);commit(case,op,ratio+(1-ratio)*u)
   mask=before[0]>case.reference
   np.testing.assert_array_equal(before[0][mask],case.quantity[mask]);np.testing.assert_array_equal(before[1][mask],case.fee[mask])
   self.assertEqual(before[2],case.lots[:-1]);self.assertTrue((case.quantity>=before[0]).all())
   self.assertTrue(all(case.lots[-1].quantities[int(j)]==0. for j in bad))
 def test_local_capacity_ulps_read_only(self):
  r=np.array([89.178170972871,1.,1e-100,1e100]);q=r.copy()
  for _ in range(2):
   q=np.nextafter(q,np.inf);old=q.copy();np.testing.assert_array_equal(contract_coverage(q,r),np.ones(4));np.testing.assert_array_equal(q,old)
  with self.assertRaises(ValueError):contract_coverage(np.nextafter(q,np.inf),r)
  with self.assertRaises(ValueError):contract_coverage(r*(1+1e-6),r)
 def test_zero_reference_does_not_accept_nonzero(self):
  np.testing.assert_array_equal(contract_coverage(np.zeros(2),np.zeros(2)),np.zeros(2))
  with self.assertRaises(ValueError):contract_coverage(np.array([np.nextafter(0.,1.)]),np.zeros(1))
 def test_external_targets_remain_strict_and_atomic(self):
  ledger=PositionLedger(2011,np.full(8760,10.));op=first('Y')
  for value in (1+1e-6,np.nextafter(1.,np.inf),-.1,np.nan,np.inf,-np.inf):
   state=signature(ledger)
   with self.assertRaises(ValueError):commit(ledger,op,np.full(8760,value))
   self.assertEqual(state,signature(ledger))
  for array in (np.zeros(3),np.zeros((8760,1)),np.ones(8760,dtype=bool)):
   state=signature(ledger)
   with self.assertRaises(ValueError):commit(ledger,op,array)
   self.assertEqual(state,signature(ledger))
 def test_clock_quote_version_repeat_fail_atomically(self):
  ledger=PositionLedger(2011,np.full(8760,10.));op=first('Y');target=np.full(8760,.3)
  for bad in ({'now':op.decision_at+timedelta(seconds=1)},{'published_at':op.decision_at+timedelta(seconds=1)},
     {'expires_at':op.decision_at-timedelta(seconds=1)},{'expected_version':999}):
   state=signature(ledger)
   with self.assertRaises((ValueError,RuntimeError)):commit(ledger,op,target,**bad)
   self.assertEqual(state,signature(ledger))
  commit(ledger,op,target);state=signature(ledger)
  with self.assertRaises(RuntimeError):commit(ledger,op,target)
  self.assertEqual(state,signature(ledger))
 def test_true_revocation_still_rejected(self):
  ledger=PositionLedger(2011,np.full(8760,10.));commit(ledger,first('Y'),np.full(8760,.7));op=first('M1')
  before=signature(ledger)
  with self.assertRaises(ValueError):commit(ledger,op,np.full(op.end-op.start,.7-1e-6))
  self.assertEqual(before,signature(ledger))
  q=ledger.quantity;commit(ledger,op,np.full(op.end-op.start,.7-5e-12));np.testing.assert_array_equal(q,ledger.quantity)
 def test_started_hour_and_overflow_are_atomic(self):
  ledger=PositionLedger(2011,np.full(8760,10.));op=first('Y');ledger.start_hour(0);before=signature(ledger)
  with self.assertRaises(RuntimeError):commit(ledger,op,np.ones(8760))
  self.assertEqual(before,signature(ledger))
  ledger=PositionLedger(2011,np.full(8760,10.));before=signature(ledger)
  with np.errstate(over='ignore'):
   with self.assertRaises(ValueError):commit(ledger,op,np.ones(8760),np.finfo(np.float64).max)
  self.assertEqual(before,signature(ledger))
 def test_three_two_two_and_price_handcalculation(self):
  ledger=PositionLedger(2011,np.full(8760,10.))
  for scope,target,price in [('Y',.3,400),('M1',.5,420),('I1-1',.7,450)]:
   op=first(scope);commit(ledger,op,np.full(op.end-op.start,target),price)
  self.assertEqual([x.quantities[0] for x in ledger.lots],[3.,2.,2.]);self.assertEqual(ledger.fee[0],2940.)
  self.assertEqual(settlement(9,0,0,7,2940,1,500,600)['purchase_cost'],4040.)
 def test_repeated_zero_increment_scopes_and_leap_years(self):
  for year in (2010,2011,2020):
   n=year_axis(year)[1];r=np.full(n,89.178170972871);ledger=PositionLedger(year,r)
   for op in opportunities(year):
    if op.index:continue
    q=ledger.quantity;f=ledger.fee;lots=ledger.lots;commit(ledger,op,np.ones(op.end-op.start),400.)
    if op.stage!='Y':
     np.testing.assert_array_equal(q,ledger.quantity);np.testing.assert_array_equal(f,ledger.fee);self.assertFalse(any(ledger.lots[-1].quantities));self.assertEqual(lots,ledger.lots[:-1])
   self.assertEqual(len(ledger.lots),49);np.testing.assert_array_equal(ledger.quantity,r)
 def test_fixed_seed_reference_scale_stress(self):
  rng=np.random.default_rng(20261015)
  for scale in (1e-100,1e-6,1.,100.,1e6,1e100):
   for _ in range(5):
    r=np.full(8760,scale*float(rng.uniform(.5,1.5)));ledger=PositionLedger(2011,r)
    alpha=float(rng.random());commit(ledger,first('Y'),np.full(8760,alpha),1.)
    q=ledger.quantity;op=first('M1');commit(ledger,op,np.ones(op.end-op.start),1.)
    self.assertTrue((ledger.quantity<=r).all());self.assertTrue((ledger.quantity>=q).all());self.assertTrue(np.isfinite(ledger.fee).all())
 def test_decode_and_reward_helpers_unchanged(self):
  old,coord=before_modules()
  for fn in ('vector','number','settlement'):
   self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(old,fn)))),ast.dump(ast.parse(inspect.getsource(globals()[fn]))))
  self.assertEqual((ROOT/'v3/environments.py').read_bytes(),(DOC/'before/v3/environments.py').read_bytes())
  for i,t in [(0,1/3),(0,2/3),(1,.5)]:
   for x in (np.nextafter(t,0),t,np.nextafter(t,1)):
    a=np.zeros(13);a[0]=x;verb,_=decode_upper({'calendar':np.array([0,0,i])},a)
    expected=('SKIP' if x<.5 else 'COMMIT') if i else ('WAIT' if x<1/3 else 'SKIP' if x<2/3 else 'COMMIT')
    self.assertEqual(verb,expected)
 def test_candidate_quantity_lot_and_fee_use_identical_delta(self):
  ledger=PositionLedger(2011,np.full(8760,89.178170972871));commit(ledger,first('Y'),np.full(8760,.2719821631908417),400.)
  before_q=ledger.quantity;before_f=ledger.fee;op=first('M2');commit(ledger,op,np.ones(op.end-op.start),420.)
  delta=np.array(ledger.lots[-1].quantities);a,z=op.start,op.end
  np.testing.assert_array_equal(ledger.quantity[a:z],before_q[a:z]+delta)
  np.testing.assert_array_equal(ledger.fee[a:z],before_f[a:z]+delta*420.)
  self.assertTrue((ledger.quantity<=ledger.reference).all())
 def test_corrupt_legacy_three_ulps_rejects_without_commit(self):
  # 仅独立测试故障注入；不改历史文件或真实停止状态。
  for n in (2,3):
   ledger=PositionLedger(2011,np.full(8760,89.178170972871));commit(ledger,first('Y'),np.ones(8760))
   op=first('M2');q=ledger.reference[op.start:op.end]
   for _ in range(n):q=np.nextafter(q,np.inf)
   ledger._quantity[op.start:op.end]=q
   before=signature(ledger)
   if n==3:
    with self.assertRaises(ValueError):commit(ledger,op,np.ones(op.end-op.start))
    self.assertEqual(before,signature(ledger))
   else:
    oldq=ledger.quantity;oldf=ledger.fee;commit(ledger,op,np.ones(op.end-op.start))
    np.testing.assert_array_equal(oldq,ledger.quantity);np.testing.assert_array_equal(oldf,ledger.fee)
 def test_coordinator_legacy_five_requests_no_repricing(self):
  old,_=before_modules();legacy,op=constant_fixture(old.PositionLedger)
  for u in (0.,.1,.5,.9,1.):
   c=object.__new__(YearCoordinator);c.ledger=copy.deepcopy(legacy);c.ledger.__class__=PositionLedger
   allops=opportunities(2011);c.op_cursor=allops.index(op);c.now=op.decision_at;c.macros=[]
   c.world=SimpleNamespace(reference=legacy.reference,n=8760,ops=allops,
     macro=lambda o,at:dict(price=450.,published_at=o.published_at,expires_at=o.decision_at))
   q=c.ledger.quantity;f=c.ledger.fee;lots=c.ledger.lots;c.submit_macro('COMMIT',np.full(12,u))
   np.testing.assert_array_equal(q,c.ledger.quantity);np.testing.assert_array_equal(f,c.ledger.fee)
   self.assertEqual(lots,c.ledger.lots[:-1]);self.assertEqual(c.macros[-1]['verb'],'COMMIT')
 def test_old_source_lock_verified_against_saved_version(self):
  from v3.data import sha
  pre=json.loads((ROOT/'outputs/v3/continuous_20261015/pretrain_lock.json').read_text())
  for name,h in pre['source_sha256'].items():
   if name in ('v3/core.py','v3/coordinator.py'):
    self.assertEqual(sha(DOC/'before'/name),h);self.assertNotEqual(sha(ROOT/name),h)
   else:self.assertEqual(sha(ROOT/name),h)

@unittest.skipUnless((OUT/'completion.json').is_file(),'同轨迹验证尚未完成')
class ReplayArtifactTests(unittest.TestCase):
 def test_before_matches_original_and_after_same_actions(self):
  s=json.loads((OUT/'completion.json').read_text());self.assertEqual(s['status'],'COMPLETE_NO_LEARNING')
  self.assertEqual(s['before']['hours'],106339);self.assertEqual(s['before']['changed_hours'],0)
  self.assertEqual(s['after']['hours'],106339);self.assertEqual(s['after']['full_years'],12)
  self.assertLess(s['max_macro_economic_delta'],1e-5);self.assertFalse(s['learn_called']);self.assertFalse(s['validation']);self.assertFalse(s['test'])
  before=rows(OUT/'replay_before_macros.csv');after=rows(OUT/'replay_after_macros.csv');self.assertEqual(len(before),966)
  for a,b in zip(before,after):self.assertEqual((a['index'],a['verb'],a['scope']),(b['index'],b['verb'],b['scope']))
 def test_original_14_and_five_successful_probes(self):
  self.assertEqual(len(rows(OUT/'original_14_hours.csv')),14)
  rr=rows(OUT/'legacy_market_quote_accounting.csv');self.assertEqual(len(rr),5)
  with np.load(ROOT/'data/v3/synthetic_multiscale_v3/train_2011_G00.npz',allow_pickle=False) as z:
   expected=float(z['quotes'][opportunities(2011).index(next(x for x in opportunities(2011) if x.scope=='I2-3' and x.index==1))])
  self.assertTrue(all(float(x['unit_price'])==expected for x in rr))
  self.assertTrue(all(x['old_overcap_quantity_unchanged']=='True' and x['old_overcap_fee_unchanged']=='True' and x['after_outcome']=='ACCEPT' for x in rr))
  rr=rows(OUT/'new_state_probes.csv');self.assertEqual(len(rr),5);self.assertTrue(all(x['outcome']=='ACCEPT' for x in rr))

if __name__=='__main__':unittest.main()
