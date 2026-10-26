"""D7实验的只读冻结检查与统计；不修改任何V3市场实现。"""
from pathlib import Path
import csv,json,hashlib,platform,importlib.metadata
import numpy as np
from v3.training import CONFIG,policy_hash,write_csv
from v3.data import DATA_ROOT,sha,json_new
from diagnostics_v3.upper_no_contract import verify_frozen,observation_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v3/d7_upper_budget_20261015'
DOC=ROOT/'docs/v3_d7'
OLD=ROOT/'outputs/v3/continuous_20261015'
CANDIDATES=('long_ppo_0','long_ppo_1','d7_long_ppo_0','d7_long_ppo_1','rolling_deadline','rolling_forecast')
PARAMS=dict(CONFIG,upper_timesteps=1536)
PROTOCOL=dict(stage='V3-D7 upper budget only',parameters=PARAMS,seeds=[0,1],trained_levels=['upper'],
 lower='forecast_dispatch',split_train='train',validation_worlds=8,candidates=list(CANDIDATES),
 per_seed_limit_seconds=900,training_total_limit_seconds=1800,output=OUT.relative_to(ROOT).as_posix(),
 probes='all 256 archived legal D6 train states, no validation probes',
 rollout_statistics='initial and after every completed update; no RNG draws',
 selection='no selection, both models frozen before one validation round',
 evaluation_per_split_limit_seconds=900,
 test_authorized=False,observation_normalization=False,
 comparisons='same world purchase cost; objective separately includes flex cost',
 failure='stop whole experiment, preserve all partial logs; no retry')

def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def check_params(params):
 if params!=PARAMS:raise ValueError('不允许额外配置变化')
 old=json.loads((OLD/'experiment_config.json').read_text(encoding='utf-8'))
 changes={k for k in old if old[k]!=params[k]}
 if set(params)!=set(old) or changes!={'upper_timesteps'} or old['upper_timesteps']!=128:raise ValueError('唯一变量失败')

def preflight():
 check_params(PARAMS);lock=verify_frozen()
 deps=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','stable-baselines3','torch')})
 old=json.loads((OLD/'pretrain_lock.json').read_text())
 if deps!=old['dependencies']:raise ValueError('冻结依赖版本改变')
 return dict(old_lock=lock,dependencies=deps,manifest_sha256=sha(DATA_ROOT/'manifest.json'),
   readonly_input_sha256={p:sha(ROOT/p) for p in ['outputs/v3/d6_no_contract_20261015/observations.jsonl',
      *['outputs/v3/continuous_20261015/validation/'+n for n in ('annual.csv','daily.csv','macros.csv','hourly.csv')]]},
   source_sha256={k:sha(ROOT/k) for k in ('experiments/train_upper_budget_v3.py','experiments/evaluate_upper_budget_v3.py','experiments/upper_budget_v3_common.py')},
   changed_parameter={'upper_timesteps':{'before':128,'after':1536}},test_opened=False)

def train_probes():
 path=ROOT/'outputs/v3/d6_no_contract_20261015/observations.jsonl'
 states=[]
 with path.open(encoding='utf-8') as f:
  for line in f:
   s=json.loads(line)
   if s['split']=='train':states.append(s)
 if len(states)!=256 or any(not x['world'].startswith('train_') for x in states):raise ValueError('probe必须冻结train')
 return states

def optimizer_steps(model):
 opt=model.policy.optimizer
 return {name:int(opt.state[p].get('step',0)) for name,p in model.policy.named_parameters() if name in ('action_net.weight','action_net.bias')}

def head_stats(model):
 w=model.policy.action_net.weight[0].detach().cpu().numpy().astype(np.float64)
 b=float(model.policy.action_net.bias[0].detach().cpu())
 return dict(mode_bias=b,mode_weight_l1=float(abs(w).sum()),mode_weight_l2=float(np.linalg.norm(w)),
  all_input_absolute_bound=float(abs(w).sum()+abs(b)),mode_weight=w.tolist(),optimizer_steps=optimizer_steps(model))

def actor_probe(model,states,step):
 import torch,pickle,random
 def rng():return hashlib.sha256(pickle.dumps((np.random.get_state(),random.getstate(),torch.get_rng_state().numpy()))).hexdigest()
 before=rng();digest=policy_hash(model)
 batch={k:np.stack([np.array(s['obs'][k],dtype=np.float32) for s in states]) for k in states[0]['obs']}
 with torch.inference_mode():
  tensor,_=model.policy.obs_to_tensor(batch);dist=model.policy.get_distribution(tensor).distribution
  mean=dist.mean.cpu().numpy();std=dist.stddev.cpu().numpy()
 if mean.shape!=(len(states),13) or not np.isfinite(mean).all() or not np.isfinite(std).all():raise ValueError('Actor异常')
 if before!=rng() or digest!=policy_hash(model):raise RuntimeError('诊断改变随机流或权重')
 from v3.environments import decode_upper
 result=[]
 for i,s in enumerate(states):
  clipped=np.clip(mean[i],0,1).astype(np.float32);obs={k:np.array(v,dtype=np.float32) for k,v in s['obs'].items()}
  verb,u=decode_upper(obs,clipped)
  delta=0. if verb!='COMMIT' else float(np.dot(u,s['month_remaining_mwh'])) if s['stage']=='Y' else float(u[0]*s['scope_remaining_mwh'])
  result.append(dict(timesteps=step,case=s['case'],world=s['world'],split='train',stage=s['stage'],opportunity=s['opportunity'],
   observation_sha256=observation_hash(obs),mode_mean=float(mean[i,0]),mode_std=float(std[i,0]),
   actor_mean=json.dumps(mean[i].tolist()),actor_std=json.dumps(std[i].tolist()),raw_action=json.dumps(mean[i].tolist()),
   clipped_action=json.dumps(clipped.tolist()),verb=verb,increment_mwh=delta,
   commit_threshold_distance=(.5 if s['opportunity'] else 2/3)-float(mean[i,0])))
 modes=mean[:,0];h=head_stats(model)
 summary=dict(timesteps=step,policy_sha256=digest,mode_mean_min=float(modes.min()),mode_mean_max=float(modes.max()),
  mode_mean_average=float(modes.mean()),mode_std_average=float(std[:,0].mean()),all_input_absolute_bound=h['all_input_absolute_bound'],
  mode_bias=h['mode_bias'],head_optimizer_steps=h['optimizer_steps']['action_net.weight'],
  WAIT=sum(x['verb']=='WAIT' for x in result),SKIP=sum(x['verb']=='SKIP' for x in result),COMMIT=sum(x['verb']=='COMMIT' for x in result),
  positive_COMMIT=sum(x['verb']=='COMMIT' and x['increment_mwh']>1e-9 for x in result),rng_unchanged=True)
 return summary,result

class CsvStream:
 def __init__(self,path):self.file=Path(path).open('x',encoding='utf-8-sig',newline='');self.writer=None;self.n=0
 def emit(self,row):
  if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
  self.writer.writerow(row);self.n+=1
 def flush(self):self.file.flush()
 def close(self):self.file.close()
