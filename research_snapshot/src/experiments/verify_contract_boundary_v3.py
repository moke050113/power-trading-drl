"""D7a：冻结前源码与修复版的无学习train重放及只读旧仓位兼容探针。"""
from pathlib import Path
import importlib.util,sys,json,csv,copy,hashlib,time,contextlib
from unittest.mock import patch
import numpy as np
from functools import lru_cache
import v3.core as current_core
import v3.coordinator as current_coord
from v3.environments import UpperTradingEnv,decode_upper
from v3.data import json_new,sha

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/v3_d7a'
OUT=ROOT/'outputs/v3/d7a_contract_boundary_20261015'
D7=ROOT/'outputs/v3/d7_upper_budget_20261015'

def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

@lru_cache(maxsize=1)
def before_modules():
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 core=load('_d7a_before_core',DOC/'before/v3/core.py')
 # 未改时钟/dataclass与现有data共享类型，避免仅模块名不同导致dataclass不等。
 core.Opportunity=current_core.Opportunity
 coordinator=load('_d7a_before_coordinator',DOC/'before/v3/coordinator.py')
 coordinator.PositionLedger=core.PositionLedger
 return core,coordinator

def commit(ledger,op,target,price=400.,**kwargs):
 kw=dict(now=op.decision_at,published_at=op.published_at,expires_at=op.decision_at,
  expected_version=ledger.version,trade_id=f'{op.scope}:{op.index}')
 kw.update(kwargs);ledger.decide(op,'COMMIT',target,price,**kw)

def constant_fixture(klass=current_core.PositionLedger,year=2011):
 _,n=current_core.year_axis(year);ledger=klass(year,np.full(n,89.178170972871))
 ops=current_core.opportunities(year)
 for scope,target,p in [('Y',.2719821631908417,400.),('M2',1.,420.)]:
  op=next(x for x in ops if x.scope==scope and x.index==0);commit(ledger,op,np.full(op.end-op.start,target),p)
 return ledger,next(x for x in ops if x.scope=='I2-3' and x.index==0)

def signature(ledger):
 return (ledger.digest(),ledger.version,ledger.lots,ledger.fee.tobytes(),
   tuple(ledger._stage_fee[k].tobytes() for k in 'YMI'),ledger._consumed,ledger._last_time,ledger._trade_ids,ledger._decided)

class Stream:
 def __init__(self,path):self.file=Path(path).open('x',encoding='utf-8-sig',newline='');self.writer=None;self.n=0
 def emit(self,x):
  if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(x));self.writer.writeheader()
  self.writer.writerow(x);self.n+=1
 def close(self):self.file.close()

def legacy_probes(ledger,op,out,price=400.):
 rr=[]
 q=ledger.quantity[op.start:op.end];r=ledger.reference[op.start:op.end]
 for u in (0.,.1,.5,.9,1.):
  old=copy.deepcopy(ledger);old_target=q/r+(1-q/r)*u;old_before=signature(old)
  try:commit(old,op,old_target,price);old_outcome='ACCEPT'
  except Exception as e:old_outcome=repr(e);assert signature(old)==old_before
  updated=copy.deepcopy(ledger);updated.__class__=current_core.PositionLedger
  before_q=updated.quantity;before_fee=updated.fee;before_lots=updated.lots
  over_mask=before_q>updated.reference
  ratio=current_core.contract_coverage(q,r);target=ratio+(1-ratio)*u
  commit(updated,op,target,price)
  if not np.array_equal(before_q[over_mask],updated.quantity[over_mask]) or not np.array_equal(before_fee[over_mask],updated.fee[over_mask]) or updated.lots[:-1]!=before_lots or (updated.quantity<before_q).any():raise RuntimeError('只读兼容修改旧责任')
  rr.append(dict(u=u,unit_price=price,before_outcome=old_outcome,after_outcome='ACCEPT',old_q_max_ratio=float((q/r).max()),
   canonical_max_ratio=float(ratio.max()),old_overcap_quantity_unchanged=True,old_overcap_fee_unchanged=True,old_lots_unchanged=True,
   new_lot_mwh=sum(updated.lots[-1].quantities),new_lot_cost=float(np.sum(np.array(updated.lots[-1].quantities)*updated.lots[-1].unit_price)),
   fee_array_increase=float((updated.fee-before_fee).sum()),aggregate_fee_sum_difference=float(updated.fee.sum()-before_fee.sum())))
 stream=Stream(out)
 for x in rr:stream.emit(x)
 stream.close();return rr

def replay(which,before_data=None):
 if which not in ('before','after'):raise PermissionError('只允许train固定重放')
 core,coord=before_modules();context=patch('v3.environments.YearCoordinator',coord.YearCoordinator) if which=='before' else contextlib.nullcontext()
 archive=(D7/'seed_0/upper_hours.csv').open(encoding='utf-8-sig',newline='');original_hours=iter(csv.DictReader(archive))
 audit=Stream(OUT/f'replay_{which}_macros.csv');hour_diffs=Stream(OUT/f'hour_differences_{which}.csv')
 observed=[];obs_diffs=[];count=0;changed_hours=0;max_hour_delta=0.;failure_before=False
 action_rows=rows(D7/'seed_0/upper_steps.csv');started=time.perf_counter()
 with context:
  env=UpperTradingEnv(seed=0);obs,_=env.reset(seed=0)
  def emit(row):
   nonlocal count,changed_hours,max_hour_delta
   previous=next(original_hours);world=env.c.world.metadata['id'];assert previous['world']==world
   diffs={k:float(row[k])-float(previous[k]) for k in row if float(row[k])!=float(previous[k])}
   if which=='before' and diffs:raise RuntimeError('修复前重放未复现原D7小时记录')
   if diffs:
    changed_hours+=1
    for key,diff in diffs.items():
     hour_diffs.emit(dict(world=world,t=row['t'],field=key,before=float(previous[key]),after=float(row[key]),difference=diff))
     if 'cost' in key or key=='reward':max_hour_delta=max(max_hour_delta,abs(diff))
   count+=1
   if not np.isclose(row['purchase_cost'],row['contract_cost']+row['spot_cost']+row['balance_cost'],rtol=1e-12,atol=1e-8):raise RuntimeError('购电组成')
   if not np.isclose(row['reward'],-row['purchase_cost']-row['flex_cost'],rtol=1e-12,atol=1e-8):raise RuntimeError('Reward恒等式')
   if time.perf_counter()-started>600:raise TimeoutError('只读重放超过600秒停止')
  env.c.on_hour=emit
  try:
   for i,row in enumerate(action_rows):
    c=env.c;op=c.next_op();before_obs={k:v.copy() for k,v in obs.items()};action=np.array(json.loads(row['action']),dtype=np.float32)
    verb,_=decode_upper(obs,action)
    if verb!=row['verb']:raise RuntimeError('动作解码改变')
    obs,raw,done,trunc,_=env.step(action)
    q=c.ledger.quantity;r=c.world.reference
    if which=='after' and (q>r).any():raise RuntimeError('新轨迹仍超R')
    if which=='before' and abs(raw-float(row['raw_reward']))>1e-5:raise RuntimeError('原宏Reward不匹配')
    record=dict(index=i,world=c.world.metadata['id'],stage=op.stage,scope=op.scope,opportunity=op.index,verb=verb,
     raw_reward=raw,purchase_cost=c.total_purchase,flex_cost=c.total_flex,contract_fee_committed=float(c.ledger.fee.sum()),
     contract_cost_settled=sum(h['contract_cost'] for h in c.hours),unserved_liability=float(c.ledger.fee[c.t:].sum()),
     quantity_sum=float(q.sum()),max_quantity_minus_reference=float((q-r).max()),terminated=done)
    if abs(record['contract_fee_committed']-record['contract_cost_settled']-record['unserved_liability'])>1e-5:raise RuntimeError('合同责任对账')
    if which=='after':
     original=before_data['records'][i]
     for k in ('raw_reward','purchase_cost','flex_cost','contract_fee_committed','contract_cost_settled','unserved_liability','quantity_sum'):
      record[k+'_difference']=record[k]-original[k]
     for k,v in before_obs.items():
      old=before_data['observations'][i][k]
      if v.shape!=old.shape:raise RuntimeError('观察形状改变')
      if not np.array_equal(v,old):obs_diffs.append(dict(index=i,key=k,max_abs_difference=float(abs(v-old).max())))
    observed.append(before_obs);audit.emit(record)
    if done:
     for day in range(c.world.n//24):
      hh=c.hours[day*24:(day+1)*24]
      if abs(sum(x['ev_grid_mwh'] for x in hh)-100/3)>1e-8 or abs(sum(x['industrial_mwh'] for x in hh)-12)>1e-8 or abs(hh[-1]['ev_soc']-.7)>1e-10:raise RuntimeError('每日设备任务')
     obs,_=env.reset();env.c.on_hour=emit
   if next(original_hours,None) is not None:raise RuntimeError('漏重放原小时')
   c=env.c;op=c.next_op();q=c.ledger.quantity;r=c.world.reference;ratio=q[op.start:op.end]/r[op.start:op.end]
   if which=='before':
    stream=Stream(OUT/'original_14_hours.csv')
    for j in np.flatnonzero(ratio>1):stream.emit(dict(hour=int(op.start+j),quantity=float(q[op.start+j]),reference=float(r[op.start+j]),ratio=float(ratio[j])))
    stream.close()
    legacy_probes(c.ledger,op,OUT/'legacy_14_hours_probes.csv')
   else:
    # 原失败13维action缺失，仅提交五组明确标注的合法探针；不推进任何交割。
    stream=Stream(OUT/'new_state_probes.csv')
    for u in (0.,.1,.5,.9,1.):
     clone=copy.copy(c);clone.ledger=copy.deepcopy(c.ledger);clone.macros=list(c.macros)
     oldq=clone.ledger.quantity;oldfee=clone.ledger.fee;clone.submit_macro('COMMIT',np.full(12,u))
     if (clone.ledger.quantity>r).any():raise RuntimeError('探针超仓')
     stream.emit(dict(u=u,outcome='ACCEPT',added_mwh=float((clone.ledger.quantity-oldq).sum()),new_fee=float((clone.ledger.fee-oldfee).sum()),probe_not_original_failed_action=True))
    stream.close()
   result=dict(version=which,records=[],observations=observed,hours=count,changed_hours=changed_hours,max_hour_economic_delta=max_hour_delta,
    observation_differences=obs_diffs,full_years=len(env.episodes),stop_hour=c.t,world=c.world.metadata['id'],pending=c.pending is not None,
    max_stop_ratio=float(ratio.max()),seconds=time.perf_counter()-started)
  finally:audit.close();hour_diffs.close();archive.close()
 result['records']=[{k:(float(v) if k in ('raw_reward','purchase_cost','flex_cost','contract_fee_committed','contract_cost_settled','unserved_liability','quantity_sum') else v) for k,v in x.items()} for x in rows(OUT/f'replay_{which}_macros.csv')]
 return result

def main():
 if len(sys.argv)>1:raise PermissionError('没有split/model/train参数；不允许test')
 if OUT.exists():raise FileExistsError('修复验证拒绝覆盖')
 OUT.mkdir(parents=True)
 json_new(OUT/'protocol.json',dict(stage='V3-D7a',learn=False,test=False,validation=False,actions=966,source=D7.relative_to(ROOT).as_posix(),
  max_seconds_per_replay=600,legacy_capacity_ulps=2,source_sha256={p:sha(ROOT/p) for p in ('v3/core.py','v3/coordinator.py','v3/environments.py')}))
 started=time.perf_counter()
 try:
  # 即使误接算法入口也立即拒绝；核心验证不需要加载或初始化任何模型。
  try:
   from stable_baselines3 import PPO
   import torch
   guards=contextlib.ExitStack();guards.enter_context(patch.object(PPO,'learn',side_effect=AssertionError('D7a禁止learn')));guards.enter_context(patch.object(torch.optim.Adam,'step',side_effect=AssertionError('D7a禁止优化器更新')))
  except ImportError:guards=contextlib.nullcontext()
  with guards:
   before=replay('before');after=replay('after',before)
  differences=[]
  for a,b in zip(before['records'],after['records']):
   for key in ('raw_reward','purchase_cost','flex_cost','contract_fee_committed','contract_cost_settled','unserved_liability','quantity_sum'):
    diff=b[key]-a[key]
    if diff:differences.append(dict(index=a['index'],world=a['world'],field=key,before=a[key],after=b[key],difference=diff))
  stream=Stream(OUT/'macro_economic_differences.csv')
  for x in differences:stream.emit(x)
  stream.close()
  for x in differences:
   if 'cost' in x['field'] or x['field'] in ('raw_reward','unserved_liability','contract_fee_committed'):
    if abs(x['difference'])>1e-5:raise RuntimeError('费用差异超过机器精度解释范围')
  summary=dict(status='COMPLETE_NO_LEARNING',before={k:v for k,v in before.items() if k not in ('records','observations')},after={k:v for k,v in after.items() if k not in ('records','observations')},
   macro_difference_rows=len(differences),max_macro_economic_delta=max([abs(x['difference']) for x in differences if x['field']!='quantity_sum'] or [0.]),
   failed_action_not_archived=True,probes_are_not_original_failed_action=True,seed1_started=False,validation=False,test=False,learn_called=False,seconds=time.perf_counter()-started)
  json_new(OUT/'completion.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
 except Exception as e:json_new(OUT/'failure.json',dict(exception=repr(e)));raise

if __name__=='__main__':main()
