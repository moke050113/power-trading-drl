"""Bounded staged PPO pilot. No validation/test access and no new PPO mathematics."""
import argparse,hashlib,json,time,platform,importlib.metadata
from pathlib import Path
import numpy as np
from .lite_gym import AnnualDecisionEnv,WorldCache,upper_policy,encode,DIVISORS
from .lite_stage_a import write,csv_new

ROOT=Path(__file__).resolve().parents[1]
TRAIN_SPECS=[dict(year=y,family=f'train_{f}',group=g,split='train',fixture=False)
    for f in (0,1) for y in (2033,2036) for g in ('G00','G01','G10','G11')]
VAL_SPECS=[dict(year=2038,family='validation_0',group=g,split='validation',fixture=False)
    for g in ('G00','G01','G10','G11')]
PARAMS=dict(policy_upper='original D8 HybridUpperPolicy',policy_lower='MultiInputPolicy',network=[32,32],
    learning_rate=.0003,gamma=1.,gae_lambda=.95,n_steps_upper=32,n_steps_lower=48,
    batch_upper=32,batch_lower=48,n_epochs=2,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
    reward_divisor=1e6,observation_normalization='fixed physical divisors, no fitted/running statistics',
    device='cpu',threads=1,seed_seconds=900,test_execution=False)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def policy_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.policy.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def peak_memory():
    import ctypes,os
    if os.name!='nt':
        import resource
        return dict(peak_wset_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,measurement='ru_maxrss KiB converted to bytes')
    from ctypes import wintypes
    class Counts(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    psapi=ctypes.WinDLL('psapi');kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counts),wintypes.DWORD]
    c=Counts();c.cb=ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise OSError('memory counter unavailable')
    return dict(rss_bytes=c.WorkingSetSize,peak_wset_bytes=c.PeakWorkingSetSize,measurement='Windows native GetProcessMemoryInfo, process peak incl libraries and archives')

def sources():
    files=[*sorted((ROOT/'v31').glob('lite_*.py')),ROOT/'experiments/hybrid_upper_v3.py',ROOT/'experiments/upper_observation_scale_d10.py',
        ROOT/'v3/core.py',ROOT/'v3/coordinator.py',ROOT/'v3/devices.py',ROOT/'v3/data.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def freeze(path):
    """Called only AFTER smoke, BEFORE formal train and every validation cost."""
    p=Path(path);p.mkdir(parents=True,exist_ok=False)
    protocol=dict(version='V31 P5 Lite controlled pilot v1',parameters=PARAMS,seeds=[0,1],
        upper_steps=128,lower_steps=384,smoke_discarded_steps=dict(upper=32,lower=48,seed=0),
        authorization_caps=dict(upper_per_seed=1536,lower_per_seed=8192),
        training='upper with fixed B2 lower; freeze upper; lower with that frozen upper; no alternating updates',
        tail='complete any open annual obligation with frozen final models, no learn; separate from gradient steps',
        train_specs=TRAIN_SPECS,validation_specs=VAL_SPECS,
        candidates=['B0','B1','B2','R1_seed0','R1_seed1','R2_seed0','R2_seed1'],
        validation='one complete native annual pass per frozen candidate and four matched conditions; one independent validation family',
        risk='N=1 independent validation family per condition; no reliable tail inference or 95% CVaR improvement claim',
        test_definition=dict(year=2040,families=40,groups=4,status='definition/seed commitments ONLY, no realizations generated',
            seed_commitments=[hashlib.sha256(f'V31 future independent sealed test {i}'.encode()).hexdigest() for i in range(40)]),
        scales=dict(upper=DIVISORS,resource_outlook=[100/3,12,100/3,12],
            lower=dict(hour=24,forecast_load=100,forecast_price=500,contract_plan=100,contract_fee=100000,
                offer=[500,160],devices=[1,100/3,7,24,24,12,4,24,24],history=[100,500,7,4]*24,
                history_mask=1,actual_state_mask=1,declared_initial_soc=1,forecast_offer_price=500)),
        no_model_selection=True,no_hyperparameter_search=True)
    write(p/'protocol.json',protocol)
    write(p/'pretrain_lock.json',dict(protocol_sha256=sha(p/'protocol.json'),source_sha256=sources(),
        dependencies=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','torch','stable-baselines3')})))
    return protocol

def save_table(path,rows):
    if rows:csv_new(path,[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows])

def dump_environment(out,env,tail):
    save_table(out/'steps.csv',env.steps);save_table(out/'resets.csv',env.resets)
    save_table(out/'training_complete_years.csv',env.episodes)
    write(out/'no_learning_tail.json',tail)
    # Every annual driver is owned by this environment, and archived before reset.
    save_table(out/'hourly.csv',env.recorded_hours)
    save_table(out/'macro.csv',env.recorded_macros)
    save_table(out/'daily_tasks.csv',env.recorded_daily)
    if env.driver:
        save_table(out/'last_year_hourly.csv',env.driver.c.hours)
        save_table(out/'last_year_macro.csv',env.driver.macro_receipts)
        write(out/'last_year_data_manifest.json',env.driver.store.manifest())
        write(out/'last_year_forecast_origins.json',env.driver.context.origins)

def train_layer(out,cache,level,seed,steps,*,partner=None,deadline=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from experiments.hybrid_upper_v3 import HybridUpperPolicy
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    env=AnnualDecisionEnv(cache,level,seed=seed,partner=partner)
    model=PPO(HybridUpperPolicy if level=='upper' else 'MultiInputPolicy',env,seed=seed,device='cpu',
        learning_rate=PARAMS['learning_rate'],n_steps=PARAMS['n_steps_'+level],batch_size=PARAMS['batch_'+level],
        n_epochs=2,gamma=1.,gae_lambda=.95,clip_range=.2,policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32])),verbose=0)
    model.set_logger(configure(str(out),['csv']))
    initial=policy_sha(model);adam=[];rollouts=[];tick=time.perf_counter()
    write(out/'initial_and_config.json',dict(initial_policy_sha256=initial,seed=seed,level=level,steps=steps,parameters=PARAMS))
    original_step=model.policy.optimizer.step
    def optimizer_step(*args,**kwargs):
        value=original_step(*args,**kwargs)
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):raise FloatingPointError('nonfinite learned parameter')
        adam.append(dict(update=len(adam)+1,timesteps=model.num_timesteps))
        return value
    model.policy.optimizer.step=optimizer_step
    class Guard(BaseCallback):
        def _on_step(self):
            if time.perf_counter()>deadline:raise TimeoutError('frozen per-seed time budget exceeded')
            if not np.isfinite(self.locals['rewards']).all() or not np.isfinite(self.locals['actions']).all():raise FloatingPointError('nonfinite rollout')
            return True
        def _on_rollout_end(self):
            ob=env.raw
            if ob is None:return
            with torch.no_grad():
                tensor,_=model.policy.obs_to_tensor(encode(ob,level));dist=model.policy.get_distribution(tensor)
                if level=='upper':
                    probs=dist.category.probs.cpu().numpy()[0].tolist();a=dist.a.cpu().numpy()[0].tolist();b=dist.b.cpu().numpy()[0].tolist()
                    rollouts.append(dict(timesteps=model.num_timesteps,category=probs,beta_a=a,beta_b=b))
                else:rollouts.append(dict(timesteps=model.num_timesteps,mean=dist.distribution.mean.cpu().numpy()[0].tolist(),std=dist.distribution.stddev.cpu().numpy()[0].tolist()))
    try:
        model.learn(total_timesteps=steps,callback=Guard())
        model.logger.dump(step=model.num_timesteps) # persist actual LAST update, never infer it
        seconds=time.perf_counter()-tick
        if model.num_timesteps!=steps or len(env.steps)!=steps:raise AssertionError('step budget mismatch')
        if len(adam)!=steps//PARAMS['n_steps_'+level]*PARAMS['n_epochs']:raise AssertionError('actual optimizer update mismatch')
        if policy_sha(model)==initial:raise AssertionError('no real parameter update')
        model.save(out/'model.zip');file_sha=sha(out/'model.zip');before=policy_sha(model)
        reloaded=PPO.load(out/'model.zip',device='cpu')
        if policy_sha(reloaded)!=before:raise AssertionError('reloaded parameters differ')
        if reloaded.num_timesteps!=steps or reloaded.observation_space!=model.observation_space or reloaded.action_space!=model.action_space:raise AssertionError('reload budget/spaces mismatch')
        if env.raw is not None:
            obs=encode(env.raw,level)
            if not np.array_equal(model.predict(obs,deterministic=True)[0],reloaded.predict(obs,deterministic=True)[0]):raise AssertionError('reload prediction mismatch')
        tail_tick=time.perf_counter();tail=env.finish_without_learning(reloaded)
        if policy_sha(reloaded)!=before or sha(out/'model.zip')!=file_sha:raise AssertionError('inference updated model')
        dump_environment(out,env,tail);save_table(out/'optimizer_updates.csv',adam);save_table(out/'rollout_distributions.csv',rollouts)
        receipt=dict(level=level,seed=seed,gradient_steps=model.num_timesteps,actual_adam_steps=len(adam),
            initial_policy_sha256=initial,policy_sha256=before,model_sha256=file_sha,model_path=str((out/'model.zip').resolve()),
            training_seconds=seconds,no_learning_tail_seconds=time.perf_counter()-tail_tick,
            resets=env.resets,complete_years_during_learning=len(env.episodes),tail=tail,memory=peak_memory())
        write(out/'integrity.json',receipt)
        for store in cache.compiled.values():
            data=out/store.key
            if not data.exists():
                data.mkdir();np.savez_compressed(data/'owner_market_arrays.npz',**store.arrays)
                write(data/'manifest.json',store.manifest())
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('tail','resets')},ensure_ascii=False),flush=True)
        return model,receipt
    except Exception as exc:
        model.save(out/'partial_model.zip');write(out/'failure.json',dict(error=repr(exc),steps=model.num_timesteps,t=env.driver.c.t if env.driver else None))
        failure_dir=out/'failure_snapshot';failure_dir.mkdir()
        dump_environment(failure_dir,env,dict(failed=True));raise

def run(out,*,smoke=False,protocol_dir=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not smoke:
        p=Path(protocol_dir);lock=json.loads((p/'pretrain_lock.json').read_text())
        if sha(p/'protocol.json')!=lock['protocol_sha256']:raise RuntimeError('protocol changed')
        for name,h in lock['source_sha256'].items():
            if sha(ROOT/name)!=h:raise RuntimeError('frozen source changed: '+name)
    receipts=[]
    for seed in ([0] if smoke else [0,1]):
        cache=WorldCache(TRAIN_SPECS);deadline=time.perf_counter()+900
        upper,r=train_layer(out/f'seed_{seed}'/'upper',cache,'upper',seed,32 if smoke else 128,deadline=deadline);receipts.append(r)
        lower,r=train_layer(out/f'seed_{seed}'/'lower',cache,'lower',seed,48 if smoke else 384,partner=upper_policy(upper),deadline=deadline);receipts.append(r)
    write(out/'all_models.json',receipts);write(out/'complete.json',dict(smoke=smoke,test_access=False,models=len(receipts)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','freeze','train'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--protocol-dir',type=Path)
    a=p.parse_args()
    if a.mode=='freeze':freeze(a.output_dir)
    else:run(a.output_dir,smoke=a.mode=='smoke',protocol_dir=a.protocol_dir)
