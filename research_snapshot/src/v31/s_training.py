"""S: one upper training-unit ablation, otherwise identical R continuation."""
import argparse,gc,json,time,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .lite_training import sha,policy_sha,peak_memory,save_table,dump_environment,PARAMS,ROOT
from .lite_gym import WorldCache,upper_policy,lower_policy
from .lite_stage_a import write
from .s_env import ScaledAnnualEnv as BalancedAnnualEnv,specs
from .r_diagnostics import model_probe,OLD

DOC=ROOT/'docs/v31_p5_lite_s'
OUT=ROOT/'outputs/v31/p5_lite_s'
R_OUT=ROOT/'outputs/v31/p5_lite_r'
DEVELOPMENT=[dict(year=2041,family='development_s_0',group=g,split='validation',fixture=False) for g in ('G00','G01','G10','G11')]
CANDIDATES=['B0','B1','B2','R_R1_seed0','R_R1_seed1','R_R2_seed0','R_R2_seed1','S_R1_seed0','S_R1_seed1','S_R2_seed0','S_R2_seed1']

def protect():
    before=json.loads((DOC/'protection_before.json').read_text(encoding='utf8'))
    for p,h in before['sha256'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('historical SHA changed: '+p)

def freeze(out):
    protect();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if not (OUT/'s1_signal/complete.json').exists() or not (OUT/'s1_economic/economic_summary.csv').exists():raise RuntimeError('S1 incomplete')
    previous=json.loads((R_OUT/'protocol_20261018/protocol.json').read_text())
    config=dict(previous)
    config.update(version='V31-P5-Lite-S upper fixed training-reward scale ablation',
        unique_change=dict(upper_training_divisor_before=1000000,upper_training_divisor_after=100000000,
            lower_training_divisor=1000000,environment_and_objective_unchanged=True,
            proof='For every fully fulfilled annual trajectory G_S=.01*G_R; positive constant preserves expected original objective ranking. PPO/GAE/Adam trajectories need not be invariant.',
            old_value_units='Old value weights and Adam inherited unchanged; no value-head rescaling. Initial estimate is not fitted to the new unit.',
            not_implemented='bounded lower distribution, reward shaping, changed network, new market parameters'),
        development=DEVELOPMENT,candidates=CANDIDATES,
        controls='R six phase training models frozen; same original128/384 initialization+Adam, seeds, sampler, exact phase budgets and hyperparameters. Only upper algorithm reward units differ; partner changes are endogenous.',
        budget=dict(cpu_threads=1,max_seconds_per_seed=900,max_train_total_seconds=1400,max_development_seconds=1500,
            max_rss_bytes=3*1024**3,estimated_two_seed_training_seconds=1113.59,
            estimate_basis='one current legal train cold year98.60s plus R training receipts; three seeds estimated1670.39s exceeds predeclared1400s session budget',
            development_estimate_seconds=1210.20,stop='preserve partial; no retry, extended budget, extra seed or parameter search'),
        model_selection='all final seed0/1 models; no checkpoint selection; old2038/2039 effects are not used to tune',
        test_enabled=False)
    write(out/'protocol.json',config)
    files=[*sorted((ROOT/'v31').glob('*.py')),ROOT/'experiments/hybrid_upper_v3.py',ROOT/'v3/core.py',ROOT/'v3/coordinator.py',ROOT/'v3/devices.py',ROOT/'v3/data.py']
    source={p.relative_to(ROOT).as_posix():sha(p) for p in files if p.name not in ('s_results.py','s_bundle.py')}
    write(out/'lock.json',dict(protocol_sha256=sha(out/'protocol.json'),sources=source,
        old_models=json.loads((OLD/'all_models.json').read_text()),
        R_models=json.loads((R_OUT/'training_20261018/all_models.json').read_text()),
        s1_evidence={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'s1_signal/complete.json',OUT/'s1_economic/economic_summary.csv',DOC/'resource_precheck.json']},
        dependencies=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','torch','stable-baselines3')})))
    print(json.dumps(dict(protocol_sha256=sha(out/'protocol.json'),budget=config['budget'])),flush=True)

class AuditedPPO(PPO):
    """Only observes AFTER original PPO.train; no loss/gradient modifications."""
    def train(self):
        super().train()
        hook=getattr(self,'_r_hook',None)
        if hook:hook(self)
    def _excluded_save_params(self):return super()._excluded_save_params()+['_r_hook']

def probe_rows(model,level):
    p=json.loads((R_OUT/'r1_consistent_probes/probe_states.json').read_text())
    values=p['upper'] if level=='upper' else [r for r in p['lower'] if r['hour'] in (8,12,18,20,23)]
    return [dict(origin=v.get('stage',v.get('hour')),**model_probe(model,{k:np.array(x,np.float32) for k,x in v['obs'].items()},level)) for v in values]

def train_phase(out,cache,seed,level,additional,input_model,partner,checkpoints,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def check_budget():
        if time.perf_counter()>deadline:raise TimeoutError('frozen seed wall-clock budget including fulfillment exceeded')
    check_budget()
    if peak_memory()['rss_bytes']>3*1024**3:raise MemoryError('frozen memory budget exceeded')
    env=BalancedAnnualEnv(cache,level,seed=seed,partner=partner,budget_guard=check_budget)
    model=AuditedPPO.load(input_model,env=env,device='cpu');start_steps=model.num_timesteps;initial=policy_sha(model)
    model.set_logger(configure(str(out),['csv']));updates=[];rollouts=[];checkpoints_done=[];tick=time.perf_counter()
    optimizer_count=lambda:max((int(v['step'].item()) for v in model.policy.optimizer.state.values() if 'step' in v),default=0)
    initial_optimizer_steps=optimizer_count()
    write(out/'initial.json',dict(input_model=str(Path(input_model).resolve()),input_sha256=sha(input_model),policy_sha256=initial,
        start_steps=start_steps,additional=additional,seed=seed,level=level,initial_optimizer_steps=initial_optimizer_steps,
        parameters=PARAMS,algorithm_reward_divisor=(1e8 if level=='upper' else 1e6),gradient_semantics='post SB3 global gradient clipping; advantages before minibatch normalization',probes=probe_rows(model,level)))
    original_step=model.policy.optimizer.step
    def optimizer_step(*a,**kw):
        groups=dict(actor=0.,critic=0.)
        for name,p in model.policy.named_parameters():
            if p.grad is None:continue
            if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite actual gradient')
            key='critic' if 'value_net' in name else 'actor'
            groups[key]+=float(torch.sum(p.grad.detach()**2))
        value=original_step(*a,**kw)
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):raise FloatingPointError('nonfinite actual parameter')
        updates.append(dict(update=len(updates)+1,cumulative_steps=model.num_timesteps,actor_grad_l2=groups['actor']**.5,critic_grad_l2=groups['critic']**.5))
        return value
    model.policy.optimizer.step=optimizer_step
    def after_update(m):
        check_budget()
        if peak_memory()['rss_bytes']>3*1024**3:raise MemoryError('frozen memory budget exceeded')
        elapsed=time.perf_counter()-tick;done=m.num_timesteps-start_steps
        metrics={k:float(v) for k,v in m.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        b=m.rollout_buffer
        rr=dict(additional_steps=done,cumulative_steps=m.num_timesteps,seconds=elapsed,
            advantage_mean=float(b.advantages.mean()),advantage_std=float(b.advantages.std()),
            return_min=float(b.returns.min()),return_max=float(b.returns.max()),value_mean=float(b.values.mean()),
            rollout_reward_sum=float(b.rewards.sum()),actual_updates=len(updates),**metrics)
        rollouts.append(rr);m.logger.dump(step=m.num_timesteps)
        if done in checkpoints:
            p=out/f'checkpoint_{m.num_timesteps}.zip';m.save(p)
            write(out/f'checkpoint_{m.num_timesteps}.json',dict(steps=m.num_timesteps,additional=done,model_sha256=sha(p),policy_sha256=policy_sha(m),probes=probe_rows(m,level)))
            checkpoints_done.append(done)
    model._r_hook=after_update
    class Guard(BaseCallback):
        def _on_step(self):
            if time.perf_counter()>deadline:raise TimeoutError('frozen seed CPU budget exceeded')
            if self.n_calls%1000==0 and peak_memory()['rss_bytes']>3*1024**3:raise MemoryError('frozen memory budget exceeded')
            if not np.isfinite(self.locals['rewards']).all() or not np.isfinite(self.locals['actions']).all():raise FloatingPointError('nonfinite rollout action/reward')
            return True
    try:
        model.learn(total_timesteps=additional,reset_num_timesteps=False,callback=Guard())
        seconds=time.perf_counter()-tick
        if model.num_timesteps!=start_steps+additional or len(env.steps)!=additional:raise AssertionError('exact training budget failed')
        n_steps=32 if level=='upper' else 48
        if len(updates)!=additional//n_steps*2 or checkpoints_done!=checkpoints:raise AssertionError('update/checkpoint completeness failed')
        if initial==policy_sha(model):raise AssertionError('no real learning update')
        if level=='lower' and len(env.episodes)!=4:raise AssertionError('four full natural training years required')
        model.save(out/'model.zip');fsha=sha(out/'model.zip');psha=policy_sha(model)
        loaded=PPO.load(out/'model.zip',device='cpu')
        if policy_sha(loaded)!=psha or probe_rows(model,level)!=probe_rows(loaded,level):raise AssertionError('save/reload/predict mismatch')
        tail_start=time.perf_counter();tail=env.finish_without_learning(loaded)
        check_budget()
        if optimizer_count()!=initial_optimizer_steps+len(updates):raise AssertionError('resumed Adam state count mismatch')
        if policy_sha(loaded)!=psha or sha(out/'model.zip')!=fsha:raise AssertionError('tail inference changed model')
        dump_environment(out,env,tail);save_table(out/'optimizer_updates.csv',updates);save_table(out/'learning_curve.csv',rollouts)
        receipt=dict(seed=seed,level=level,start_steps=start_steps,additional=additional,cumulative_steps=model.num_timesteps,
            actual_updates=len(updates),model_path=str((out/'model.zip').resolve()),model_sha256=fsha,policy_sha256=psha,
            initial_optimizer_steps=initial_optimizer_steps,final_optimizer_steps=optimizer_count(),
            initial_policy_sha256=initial,training_seconds=seconds,tail_seconds=time.perf_counter()-tail_start,tail=tail,
            completed_learning_years=len(env.episodes),resets=env.resets,memory=peak_memory())
        write(out/'integrity.json',receipt);print(json.dumps(receipt),flush=True)
        return loaded,receipt
    except Exception as exc:
        model.save(out/'partial_model.zip');write(out/'failure.json',dict(error=repr(exc),steps=model.num_timesteps,t=env.driver.c.t if env.driver else None))
        fail=out/'failure_snapshot';fail.mkdir();dump_environment(fail,env,dict(failed=True));save_table(fail/'updates.csv',updates);save_table(fail/'rollouts.csv',rollouts);raise

def verify_lock(protocol):
    protocol=Path(protocol);lock=json.loads((protocol/'lock.json').read_text())
    if sha(protocol/'protocol.json')!=lock['protocol_sha256']:raise RuntimeError('protocol changed')
    for p,h in lock['sources'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('frozen source changed '+p)
    protect();return json.loads((protocol/'protocol.json').read_text()),lock

def run(out,protocol):
    torch.set_num_threads(1);config,lock=verify_lock(protocol);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    receipts=[];tick=time.perf_counter()
    for seed in (0,1):
        deadline=min(time.perf_counter()+900,tick+1400);cache=WorldCache(specs(seed));old={m['level']:m for m in lock['old_models'] if m['seed']==seed}
        for m in old.values():
            if sha(m['model_path'])!=m['model_sha256']:raise RuntimeError('old model changed')
        upper,a=train_phase(out/f'seed_{seed}/upper_main',cache,seed,'upper',384,old['upper']['model_path'],None,[128,256,384],deadline);receipts.append(a)
        lower,b=train_phase(out/f'seed_{seed}/lower',cache,seed,'lower',35088,old['lower']['model_path'],upper_policy(upper),[8784,17568,26304,35088],deadline);receipts.append(b)
        final_upper,c=train_phase(out/f'seed_{seed}/upper_coordination',cache,seed,'upper',128,a['model_path'],lower_policy(lower),[128],deadline);receipts.append(c)
        # Current source/predictions remain locked. No development observation yet.
        protect();del cache,upper,lower,final_upper;gc.collect()
    write(out/'all_models.json',receipts);write(out/'complete.json',dict(phases=6,seeds=[0,1],seconds=time.perf_counter()-tick,development_started=False,test=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','train'],required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--protocol-dir',type=Path)
    a=p.parse_args();freeze(a.output_dir) if a.mode=='freeze' else run(a.output_dir,a.protocol_dir)
