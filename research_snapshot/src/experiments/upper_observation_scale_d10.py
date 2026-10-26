"""D10 fixed physical-unit observation transform and experiment contract; no fitting."""
from pathlib import Path
import json,hashlib,platform,importlib.metadata
import numpy as np
import gymnasium as gym
from experiments import d8_common as prior
from v3.data import DATA_ROOT,sha,json_new
from v3.training import policy_hash,write_csv

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/v3_d10'
OUT=ROOT/'outputs/v3/d10_scaled_upper_20261016'
D8=prior.OUT
PARAMS=dict(prior.PARAMS)
CANDIDATES=('d8_long_ppo_0','d8_long_ppo_1','d10_long_ppo_0','d10_long_ppo_1','rolling_deadline','rolling_forecast')
DIVISORS=dict(stage=[1]*3,calendar=[1]*3,coverage=[1]*12,reference=[100]*12,
    remaining_capacity=[100]*12,forecast=[100,500],quote=[500],forecast_gap=[100],
    flex_capacity=[42,32,100/3,12],settled_cost=[500,.01],device_state=[1,12])
SCALE_CONFIG=dict(version='D10 fixed physical units v1',formula='scaled=raw/divisor; no centering/clipping/fitting',
    divisors=DIVISORS,effective_float32_divisors={k:np.array(v,np.float32).tolist() for k,v in DIVISORS.items()},
    untouched_controls=['stage','calendar'],dimension=54,train_fitted=False,validation_fitted=False,
    physical_cost_units=dict(raw_settled_cost='million CNY',scaled_purchase=500_000_000,scaled_flex=10_000))
PROTOCOL=dict(stage='V3-D10 economic observation scale only',parameters=PARAMS,seeds=[0,1],
    candidates=list(CANDIDATES),training_steps_each=1536,per_seed_seconds=900,total_train_seconds=1800,
    trained_levels=['upper'],lower='forecast_dispatch',initialization='same seed and policy weights as D8 before learning',
    scale=SCALE_CONFIG,extra_smoke_training=False,train_gate='8 full train scenario years after both models frozen',
    validation='one native pass of eight existing scenario years; all six results; exploratory',
    test_authorized=False,failure='preserve and stop, no retry or scale search')
NEW_SOURCES=('experiments/upper_observation_scale_d10.py','experiments/train_upper_scaled_d10.py','experiments/evaluate_upper_scaled_d10.py')
CsvStream=prior.CsvStream
rows=prior.rows
optimizer_steps=prior.optimizer_steps
head_stats=prior.head_stats
attach_optimizer_audit=prior.attach_optimizer_audit
attempt_context=prior.attempt_context
train_probes=prior.train_probes

def transform_observation(obs,*,inverse=False):
    if not isinstance(obs,dict) or set(obs)!=set(DIVISORS):raise ValueError('observation keys must match 11 legal upper fields')
    result={}
    for key,divisor in DIVISORS.items():
        x=np.asarray(obs[key],dtype=np.float32)
        d=np.asarray(divisor,dtype=np.float32)
        if x.shape!=d.shape or not np.isfinite(x).all():raise ValueError('shape/nonfinite: '+key)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            y=x*d if inverse else x/d
        if not np.isfinite(y).all():raise ValueError('transformed nonfinite: '+key)
        result[key]=np.array(y,dtype=np.float32,copy=True)
    return result

class FixedUpperObservationScale(gym.ObservationWrapper):
    def observation(self,observation):return transform_observation(observation)

def predict_on_raw_observation(model,raw_observation):
    scaled=transform_observation(raw_observation)
    action,_=model.predict(scaled,deterministic=True)
    return action,scaled

def verify_protection():
    before=json.loads((DOC/'protection_before.json').read_text(encoding='utf-8'))
    for p,h in before['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('historical protection failure:'+p)
    for p,h in before['git_core'].items():
        if sha(ROOT/'.git'/p)!=h:raise ValueError('Git protection failure')

def preflight():
    # Exact D8 sources/config/dependencies; no validation statistics are loaded for scales.
    old=json.loads((D8/'pretrain_lock.json').read_text())
    for p,h in old['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('frozen D8 dependency changed:'+p)
    if json.loads((D8/'experiment_config.json').read_text())!=PARAMS:raise ValueError('PPO parameter changes')
    deps=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','stable-baselines3','torch')})
    if deps!=old['dependencies']:raise ValueError('dependency versions changed')
    models=json.loads((ROOT/'docs/v3_d8/model_manifest.json').read_text())['models']
    for m in models:
        if sha(m['model_path'])!=m['model_sha256']:raise ValueError('old D8 model changed')
    if sha(DATA_ROOT/'manifest.json')!=old['manifest_sha256']:raise ValueError('manifest changed')
    sources={**old['source_sha256'],**{p:sha(ROOT/p) for p in NEW_SOURCES},
        'experiments/train_upper_hybrid_d8.py':sha(ROOT/'experiments/train_upper_hybrid_d8.py')}
    inputs=['outputs/v3/d6_no_contract_20261015/observations.jsonl',
        'outputs/v3/d8_hybrid_upper_20261016/validation/combined_annual.csv',
        'outputs/v3/d8_hybrid_upper_20261016/validation/stage_coverage.csv',
        *[f'outputs/v3/d8_hybrid_upper_20261016/seed_{s}/initial_policy.json' for s in (0,1)]]
    return dict(dependencies=deps,source_sha256=sources,readonly_input_sha256={p:sha(ROOT/p) for p in inputs},
        manifest_sha256=old['manifest_sha256'],old_models=models,test_opened=False)

def check_lock():
    lock=json.loads((OUT/'pretrain_lock.json').read_text())
    for p,h in {**lock['source_sha256'],**lock['readonly_input_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('source/input lock changed:'+p)
    for name,key in (('protocol.json','protocol_sha256'),('experiment_config.json','config_sha256'),('scale_parameters.json','scale_sha256')):
        if sha(OUT/name)!=lock[key]:raise ValueError('frozen '+name+' changed')
    if sha(DATA_ROOT/'manifest.json')!=lock['manifest_sha256']:raise ValueError('manifest changed')
    for model in lock['old_models']:
        if sha(model['model_path'])!=model['model_sha256']:raise ValueError('old model changed')
    return lock

def actor_probe(model,states,step):
    """Identical 256 archived legal train states, actor/critic input scaling consistent."""
    import torch
    rng=torch.get_rng_state().clone();before=policy_hash(model)
    scaled=[transform_observation({k:np.array(v,np.float32) for k,v in s['obs'].items()}) for s in states]
    batch={k:np.stack([s[k] for s in scaled]) for k in DIVISORS}
    with torch.no_grad():
        ot,_=model.policy.obs_to_tensor(batch);d=model.policy.get_distribution(ot)
        actions=d.mode().cpu().numpy();probs=d.category.probs.cpu().numpy();aa=d.a.cpu().numpy();bb=d.b.cpu().numpy()
        entropy=d.entropy().cpu().numpy();lp=d.log_prob(d.mode()).cpu().numpy();values=model.policy.predict_values(ot).flatten().cpu().numpy()
        features=model.policy.extract_features(ot);saturation={}
        for name,network in (('actor',model.policy.mlp_extractor.policy_net),('critic',model.policy.mlp_extractor.value_net)):
            first=network[1](network[0](features));saturation[name]=float((first.abs()>.99).float().mean())
    details=[]
    for i,s in enumerate(states):
        details.append(dict(timesteps=step,index=i,stage=s['stage'],opportunity=int(s['obs']['calendar'][2]),
            p_WAIT=float(probs[i,0]),p_SKIP=float(probs[i,1]),p_COMMIT=float(probs[i,2]),
            verb=('WAIT','SKIP','COMMIT')[int(actions[i,0])],fractions=json.dumps(actions[i,1:].tolist()),
            beta_alpha=json.dumps(aa[i].tolist()),beta_beta=json.dumps(bb[i].tolist()),entropy=float(entropy[i]),
            deterministic_log_prob=float(lp[i]),final_value_scaled=float(values[i]),action=json.dumps(actions[i].tolist())))
    if not torch.equal(rng,torch.get_rng_state()) or policy_hash(model)!=before:raise RuntimeError('probe changed RNG/weights')
    return dict(timesteps=step,policy_sha256=before,n_states=len(states),mean_p_COMMIT=float(probs[:,2].mean()),
        deterministic_COMMIT=sum(x['verb']=='COMMIT' for x in details),mean_entropy=float(entropy.mean()),
        actor_first_saturation=saturation['actor'],critic_first_saturation=saturation['critic'],
        value_mean=float(values.mean()),value_std=float(values.std()),value_min=float(values.min()),value_max=float(values.max()),**head_stats(model)),details

def initial_raw_scaled_saturation(model,states):
    import torch
    rng=torch.get_rng_state().clone();before=policy_hash(model);result=[]
    for scaled in (False,True):
        obs=[{k:np.array(v,np.float32) for k,v in s['obs'].items()} for s in states]
        if scaled:obs=[transform_observation(s) for s in obs]
        batch={k:np.stack([s[k] for s in obs]) for k in DIVISORS}
        with torch.no_grad():
            ot,_=model.policy.obs_to_tensor(batch);features=model.policy.extract_features(ot)
            for name,net in (('actor',model.policy.mlp_extractor.policy_net),('critic',model.policy.mlp_extractor.value_net)):
                pre=net[0](features);activation=net[1](pre)
                result.append(dict(input_units='D10_scaled' if scaled else 'D8_raw',branch=name,states=len(states),policy_sha256=policy_hash(model),
                    first_layer_saturation=float((activation.abs()>.99).float().mean()),preactivation_abs_mean=float(pre.abs().mean())))
    if not torch.equal(rng,torch.get_rng_state()) or policy_hash(model)!=before:raise RuntimeError('saturation probe changed RNG/weights')
    return result
