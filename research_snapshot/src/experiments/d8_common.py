"""D7b修复版独立协议、源码锁与只读统计；不变更任何历史配置。"""
from pathlib import Path
import json,platform,importlib.metadata,hashlib
from v3.training import CONFIG,policy_hash,write_csv
from v3.data import DATA_ROOT,sha,json_new
from experiments.upper_budget_v3_common import train_probes,actor_probe,head_stats,optimizer_steps,CsvStream,rows

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/v3_d8'
OLD=ROOT/'outputs/v3/continuous_20261015'
OUT=ROOT/'outputs/v3/d8_hybrid_upper_20261016'
APPROVED={
 'v3/core.py':'00ea5cd96a87e2c1beb4e5b780caa2b6c47f4b49e5da363b23ecffa70cc961b4',
 'v3/coordinator.py':'3e846e6c555337f840fbf752efca1caa1a36935733d9b0534914cfa83cf1634c',
 'v3/environments.py':'6dcc365b81d6cb142b56791c3f78cdc9eef8a4fa4835641550822f3040853b7d'}
PARAMS=dict(CONFIG,upper_timesteps=1536)
CANDIDATES=('d7b_long_ppo_0','d7b_long_ppo_1','d8_long_ppo_0','d8_long_ppo_1','rolling_deadline','rolling_forecast')
NEW_SOURCES=('experiments/hybrid_upper_v3.py','experiments/d8_common.py','experiments/train_upper_hybrid_d8.py','experiments/evaluate_upper_hybrid_d8.py')
PROTOCOL=dict(stage='V3-D8 masked Categorical x Beta action experiment',parameters=PARAMS,seeds=[0,1],
 trained_levels=['upper'],lower='forecast_dispatch',split_train='train',train_worlds=8,validation_worlds=8,
 candidates=list(CANDIDATES),approved_source=APPROVED,per_seed_seconds=900,total_train_seconds=1800,
 evaluation_per_split_seconds=900,output=OUT.relative_to(ROOT).as_posix(),test_authorized=False,
 probes='256 archived D6 train states only; no sampling, no RNG changes',
 selection='both final models, no checkpoint selection, no validation adaptation',
 optimizer_reporting='96 actual Adam step diagnostics; 48 SB3 aggregated update logs, no invented per-step KL',
 classification='positive COMMIT means actual Lot sum>0; also report material >1e-9 MWh separately',
 hybrid=dict(category_codes={'WAIT':0,'SKIP':1,'COMMIT':2},beta_concentration='1.0001+softplus',deterministic='argmax category + interior Beta mode',auxiliary_ratios='joint density includes all 12 ratios even WAIT/SKIP and M/I',smoke_steps=32,smoke_seed=0),
 comparison='D8 and D7b PPO repaired D7a kernel; rolling rules pre-D7a, compatibility evidence cited',
 failure='stop whole round, preserve attempts/partial model, never retry or resume')

def check_params(p):
 old=json.loads((OLD/'experiment_config.json').read_text())
 if p!=PARAMS or set(p)!=set(old) or {k for k in old if old[k]!=p[k]}!={'upper_timesteps'}:raise ValueError('额外配置变化')

def check_sources(expected=None):
 old=json.loads((OLD/'pretrain_lock.json').read_text())['source_sha256'];wanted={**old,**APPROVED}
 for name,h in wanted.items():
  if sha(ROOT/name)!=h:raise ValueError('已批准V3源码不符:'+name)
 if expected:
  for name,h in expected.items():
   if sha(ROOT/name)!=h:raise ValueError('冻结新源码/依赖改变:'+name)

def preflight():
 check_params(PARAMS);check_sources()
 old=json.loads((OLD/'pretrain_lock.json').read_text());model_lock=json.loads((OLD/'prevalidation_model_lock.json').read_text())
 deps=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','stable-baselines3','torch')})
 if deps!=old['dependencies']:raise ValueError('依赖版本改变')
 for path,h in [(OLD/'experiment_config.json',model_lock['config_sha256']),(OLD/'frozen_rules.json',model_lock['rules_sha256']),(DATA_ROOT/'manifest.json',model_lock['manifest_sha256'])]:
  if sha(path)!=h:raise ValueError('旧配置/市场规则/manifest哈希改变')
 for m in model_lock['models']:
  if sha(m['path'])!=m['sha256'] or sha(OLD/f'seed_{m["seed"]}/config.json')!=m['config_sha256']:raise ValueError('旧模型/种子配置改变')
 replay=ROOT/'outputs/v3/d7a_contract_boundary_20261015/completion.json'
 if json.loads(replay.read_text())['status']!='COMPLETE_NO_LEARNING':raise ValueError('D7a兼容验收缺失')
 manifest=json.loads((DATA_ROOT/'manifest.json').read_text())
 for world in manifest['worlds']:
  if world['split'] not in ('train','validation') or sha(DATA_ROOT/world['path'])!=world['sha256']:raise ValueError('冻结情景文件哈希错误')
 dependencies={n:sha(ROOT/n) for n in [*NEW_SOURCES,'experiments/upper_budget_v3_common.py','diagnostics_v3/upper_no_contract.py',
   'load_models/ev_fleet.py','load_models/ev_load.py','load_models/industrial_load.py']}
 inputs=[*[f'outputs/v3/d7b_upper_budget_20261015/validation/{n}' for n in ('combined_annual.csv','annual.csv','daily.csv','macros.csv','hourly.csv')],'outputs/v3/d6_no_contract_20261015/observations.jsonl','outputs/v3/d7a_contract_boundary_20261015/completion.json',
   'docs/v3_d7a/source_versions.json',*[f'outputs/v3/continuous_20261015/validation/{n}' for n in ('annual.csv','daily.csv','macros.csv','hourly.csv')]]
 return dict(dependencies=deps,source_sha256={**{p:sha(ROOT/p) for p in old['source_sha256']},**dependencies},
  readonly_input_sha256={p:sha(ROOT/p) for p in inputs},manifest_sha256=sha(DATA_ROOT/'manifest.json'),
  kernel_version='approved D7a float64 boundary repair',historical_kernel='pre-D7a frozen original',old_model_lock=model_lock,test_opened=False)

def check_lock():
 lock=json.loads((OUT/'pretrain_lock.json').read_text());check_sources(lock['source_sha256'])
 for p,h in lock['readonly_input_sha256'].items():
  if sha(ROOT/p)!=h:raise ValueError('只读历史输入改变')
 if sha(DATA_ROOT/'manifest.json')!=lock['manifest_sha256'] or sha(OUT/'protocol.json')!=lock['protocol_sha256'] or sha(OUT/'experiment_config.json')!=lock['config_sha256']:raise ValueError('D7b锁不符')
 return lock

def verify_protection():
 before=json.loads((DOC/'protection_before.json').read_text(encoding='utf-8'))
 for p,h in before['files'].items():
  if not (ROOT/p).is_file() or sha(ROOT/p)!=h:raise ValueError('历史保护失败:'+p)
 for p,h in before['git_core'].items():
  if sha(ROOT/'.git'/p)!=h:raise ValueError('Git核心改变')

def attempt_context(c,action):
 import numpy as np
 from experiments.hybrid_upper_v3 import decode_hybrid as decode_upper
 op=c.next_op();obs=c.upper_view();verb,_=decode_upper(obs,action);quote=c.world.macro(op,c.now)
 return dict(world=c.world.metadata['id'],scope=op.scope,stage=op.stage,opportunity=op.index,
  decision_at=op.decision_at.isoformat(),quote_id=f'{op.scope}:{op.index}',price=quote['price'],
  quote_published_at=quote['published_at'].isoformat(),action=np.asarray(action).tolist(),decoded_verb=verb,
  observation={k:v.tolist() for k,v in obs.items()},ledger_digest=c.ledger.digest(),hour=c.t,
  contract_quantity_total=float(c.ledger.quantity.sum()),unserved_liability=float(c.ledger.fee[c.t:].sum()),
  trusted_log_only=True,no_future_realizations=True)

def attach_optimizer_audit(model,stream,check):
 """纯观测包装真实Adam.step；不改变梯度、参数或随机流。返回恢复函数。"""
 import numpy as np
 original=model.policy.optimizer.step
 def step(*args,**kwargs):
  check();before=optimizer_steps(model)
  full_gradient=model.policy.action_net.weight.grad.detach().cpu().numpy().astype(np.float64)
  g=full_gradient[0]
  gb=float(model.policy.action_net.bias.grad[0].detach().cpu())
  if not np.isfinite(full_gradient).all() or not np.isfinite(model.policy.action_net.bias.grad.detach().cpu().numpy()).all():raise RuntimeError('梯度非有限')
  result=original(*args,**kwargs)
  after=optimizer_steps(model);head=head_stats(model)
  if any(after[k]!=before[k]+1 for k in before):raise RuntimeError('Adam步骤不连续')
  stream.emit(dict(update=after['action_net.weight'],rollout_timesteps=model.num_timesteps,
    learning_rate=model.policy.optimizer.param_groups[0]['lr'],mode_weight_grad_l2=float(np.linalg.norm(g)),
    mode_bias_grad=gb,category_grad_l2=float(np.linalg.norm(full_gradient[:3])),beta_grad_l2=float(np.linalg.norm(full_gradient[3:])),mode_absolute_bound=head['all_input_absolute_bound'],policy_sha256=policy_hash(model)))
  stream.flush();check();return result
 model.policy.optimizer.step=step
 return lambda:setattr(model.policy.optimizer,'step',original)


def head_stats(model):
 import numpy as np
 w=model.policy.action_net.weight.detach().cpu().numpy().astype(np.float64)
 b=model.policy.action_net.bias.detach().cpu().numpy().astype(np.float64)
 return dict(all_input_absolute_bound=float(np.max(np.abs(w[:3]).sum(1)+np.abs(b[:3]))),
   head='3 category logits + 12 alpha + 12 beta',max_abs_weight=float(np.abs(w).max()),max_abs_bias=float(np.abs(b).max()))

def actor_probe(model,states,step):
 import numpy as np, torch
 from experiments.hybrid_upper_v3 import decode_hybrid
 rng=torch.get_rng_state().clone();before=policy_hash(model);details=[]
 for ix,s in enumerate(states):
  obs={k:np.array(v,dtype=np.float32) for k,v in s['obs'].items()}
  with torch.no_grad():
   ot,_=model.policy.obs_to_tensor(obs);d=model.policy.get_distribution(ot)
   a=d.mode()[0].cpu().numpy();p=d.category.probs[0].cpu().numpy()
   aa=d.a[0].cpu().numpy();bb=d.b[0].cpu().numpy()
   entropy=float(d.entropy()[0]);lp=float(d.log_prob(d.mode())[0])
  verb,u=decode_hybrid(obs,a)
  details.append(dict(timesteps=step,index=ix,stage=s.get('stage',int(obs['calendar'][0])),opportunity=int(obs['calendar'][2]),
    p_WAIT=float(p[0]),p_SKIP=float(p[1]),p_COMMIT=float(p[2]),verb=verb,
    fractions=json.dumps(u.tolist()),beta_alpha=json.dumps(aa.tolist()),beta_beta=json.dumps(bb.tolist()),
    entropy=entropy,deterministic_log_prob=lp,action=json.dumps(a.tolist())))
 if policy_hash(model)!=before or not torch.equal(rng,torch.get_rng_state()):raise RuntimeError('probe changed weights/RNG')
 return dict(timesteps=step,policy_sha256=before,n_states=len(states),
   mean_p_COMMIT=float(np.mean([x['p_COMMIT'] for x in details])),
   deterministic_COMMIT=sum(x['verb']=='COMMIT' for x in details),
   mean_entropy=float(np.mean([x['entropy'] for x in details])),**head_stats(model)),details
