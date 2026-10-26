"""唯一冻结有限PPO实验；先上层+规则，再冻结上层train仓位训练下层。"""
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import gymnasium as gym
from v3.data import YearStore,DATA_ROOT,sha,json_new
from v3.environments import UpperTradingEnv,LowerDispatchEnv,decode_upper
from v3.coordinator import YearCoordinator
from v3.baselines import forecast_dispatch

CONFIG=dict(stage='V3 limited controlled pilot',seeds=[0,1],policy='MultiInputPolicy',device='cpu',
            network=[32,32],learning_rate=.0003,gamma=1.,gae_lambda=.95,batch_upper=32,batch_lower=48,
            n_steps_upper=32,n_steps_lower=48,n_epochs=2,upper_timesteps=128,lower_timesteps=384,
            reward_divisor=1_000_000.,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
            normalize_advantage=True,observation_normalization=False,torch_threads=1,
            max_seed_train_seconds=900,max_total_seconds=1800,hierarchical_rounds=1,
            cvar_optimization=False,model_selection='all frozen candidates, no validation selection',
            flex_kappa=2.,contract_reference='C1 annual frozen base forecast',test_execution=False)

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def policy_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.policy.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def upper_predict(model):
    def act(obs):
        action,_=model.predict(obs,deterministic=True)
        return decode_upper(obs,action)
    return act

def lower_predict(model):
    def act(obs):return model.predict(obs,deterministic=True)[0]
    return act

class TrainRewardScale(gym.Wrapper):
    def __init__(self,env):
        if env.store.split!='train':raise PermissionError('训练只train')
        super().__init__(env);self.logs=[];self.hour_logs=[]
    def reset(self,**kw):return self.env.reset(**kw)
    def step(self,action):
        before=len(self.env.c.hours) if isinstance(self.env,UpperTradingEnv) else 0
        obs,raw,done,truncated,info=self.env.step(action)
        scaled=raw/CONFIG['reward_divisor']
        self.logs.append(dict(index=len(self.logs),action=json.dumps(np.asarray(action).tolist()),raw_reward=raw,scaled_reward=scaled,terminated=done,truncated=truncated))
        if isinstance(self.env,UpperTradingEnv):
            self.hour_logs.extend(dict(r,world=self.env.c.world.metadata['id']) for r in self.env.c.hours[before:])
        else:self.hour_logs.append(dict(info['hour_record'],world=self.env.audit_key))
        return obs,scaled,done,truncated,info

def freeze_config(path):
    json_new(path,CONFIG)
    return sha(path)

def train_seed(out,seed,data_root=DATA_ROOT,*,deadline=None):
    if type(seed) is not int or seed not in CONFIG['seeds']:raise ValueError('只允许冻结种子')
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    json_new(out/'config.json',dict(CONFIG,seed=seed))
    start=perf_counter();torch.set_num_threads(1)
    limit=min(start+CONFIG['max_seed_train_seconds'],deadline if deadline is not None else start+CONFIG['max_total_seconds'])
    def check_budget(*_):
        if perf_counter()>limit:raise TimeoutError('超过冻结预算，停止不扩实验')
    class Budget(BaseCallback):
        def _on_step(self):
            check_budget()
            return True
    integrity=[];models={};plans=None
    try:
        for level in ('upper','lower'):
            wrapped=None
            if level=='upper':env=UpperTradingEnv(data_root,seed=seed)
            else:env=LowerDispatchEnv(YearStore(data_root),plans,seed=seed)
            wrapped=TrainRewardScale(env)
            steps=CONFIG[level+'_timesteps'];rollout=CONFIG['n_steps_'+level];batch=CONFIG['batch_'+level]
            check_budget()
            model=PPO(CONFIG['policy'],wrapped,device=CONFIG['device'],seed=seed,learning_rate=CONFIG['learning_rate'],
                      gamma=CONFIG['gamma'],gae_lambda=CONFIG['gae_lambda'],n_steps=rollout,batch_size=batch,n_epochs=CONFIG['n_epochs'],
                      clip_range=CONFIG['clip_range'],ent_coef=CONFIG['ent_coef'],vf_coef=CONFIG['vf_coef'],max_grad_norm=CONFIG['max_grad_norm'],
                      normalize_advantage=CONFIG['normalize_advantage'],policy_kwargs={'net_arch':CONFIG['network']},verbose=0)
            model.set_logger(configure(str(out/(level+'_logger')),['csv']))
            tick=perf_counter();model.learn(total_timesteps=steps,callback=Budget());elapsed=perf_counter()-tick
            model.logger.close()
            if model.num_timesteps!=steps:raise RuntimeError('训练步数不一致')
            path=out/(level+'.zip');model.save(path)
            digest=policy_hash(model);file_hash=sha(path)
            loaded=PPO.load(path,device='cpu')
            if policy_hash(loaded)!=digest or loaded.num_timesteps!=steps:raise RuntimeError('保存重载不一致')
            if loaded.action_space!=env.action_space or loaded.observation_space!=env.observation_space:raise RuntimeError('模型空间不兼容')
            actual=dict(learning_rate=loaded.learning_rate,gamma=loaded.gamma,gae_lambda=loaded.gae_lambda,
                        n_steps=loaded.n_steps,batch_size=loaded.batch_size,n_epochs=loaded.n_epochs,
                        ent_coef=loaded.ent_coef,vf_coef=loaded.vf_coef,max_grad_norm=loaded.max_grad_norm,
                        normalize_advantage=loaded.normalize_advantage,clip_range=loaded.clip_range(1.),
                        net_arch=loaded.policy.net_arch,use_sde=loaded.use_sde,target_kl=loaded.target_kl,
                        activation_fn=loaded.policy.activation_fn.__name__,ortho_init=loaded.policy.ortho_init)
            for k in ('learning_rate','gamma','gae_lambda','n_epochs','ent_coef','vf_coef','max_grad_norm','normalize_advantage','clip_range'):
                if actual[k]!=CONFIG[k]:raise RuntimeError('执行参数与冻结配置不一致')
            json_new(out/(level+'_effective_config.json'),actual)
            # 新建train实例，同一seed的固定观察、确定性动作重复核对，不learn。
            fresh=UpperTradingEnv(data_root,seed=seed) if level=='upper' else LowerDispatchEnv(YearStore(data_root),plans,seed=seed)
            obs,_=fresh.reset(seed=seed)
            a=model.predict(obs,deterministic=True)[0];b=loaded.predict(obs,deterministic=True)[0]
            if not np.array_equal(a,b) or not fresh.action_space.contains(a):raise RuntimeError('重载确定性动作失配')
            if policy_hash(loaded)!=digest or sha(path)!=file_hash:raise RuntimeError('推理更新权重')
            write_csv(out/(level+'_steps.csv'),wrapped.logs)
            write_csv(out/(level+'_hours.csv'),wrapped.hour_logs)
            write_csv(out/(level+'_episodes.csv'),env.episodes)
            write_csv(out/(level+'_resets.csv'),env.resets)
            if level=='upper':
                json_new(out/'upper_stop_state.json',dict(hours=env.c.t,year_hours=env.c.world.n,world=env.c.world.metadata['id'],
                  pending_hour=env.c.pending is not None,unsettled_contract_commitment=float(env.c.ledger.fee[env.c.t:].sum()),
                  note='训练rollout停止可能位于未完年度，费用责任保留，不声称每个年度训练episode都已完成'))
            integrity.append(dict(level=level,seed=seed,path=str(path.resolve()),sha256=file_hash,policy_sha256=digest,
                                  timesteps=steps,seconds=elapsed,config_sha256=sha(out/'config.json'),save_reload_predict_verified=True))
            models[level]=loaded
            if level=='upper':
                # 唯一冻结上层，train全年度+规则下层产生已签轨迹；日训练不得自行重签。
                store=YearStore(data_root);plans={};snap=[]
                for w in store.worlds:
                    check_budget()
                    c=YearCoordinator(store,w['id'],on_hour=check_budget);c.run(upper_predict(loaded),forecast_dispatch)
                    check_budget()
                    q,f=c.ledger.plan(0,c.world.n);q.flags.writeable=False;f.flags.writeable=False
                    plans[w['id']]=(q,f)
                    snap.append(dict(world=w['id'],q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),fee_sha256=hashlib.sha256(f.tobytes()).hexdigest(),upper_policy_sha256=digest))
                json_new(out/'frozen_train_positions.json',snap)
        json_new(out/'model_integrity.json',integrity)
        json_new(out/'timing.json',dict(total_seconds=perf_counter()-start,status='COMPLETE'))
        return models,integrity
    except Exception as exc:
        if 'model' in locals():model.logger.close()
        json_new(out/'failure.json',dict(status='STOPPED',exception=repr(exc),seconds=perf_counter()-start))
        if 'wrapped' in locals() and wrapped is not None:
            write_csv(out/(level+'_partial_steps.csv'),wrapped.logs)
            write_csv(out/(level+'_partial_hours.csv'),wrapped.hour_logs)
            write_csv(out/(level+'_partial_episodes.csv'),env.episodes)
            write_csv(out/(level+'_partial_resets.csv'),env.resets)
            if isinstance(env,UpperTradingEnv) and env.c is not None:
                json_new(out/'partial_liabilities.json',dict(world=env.c.world.metadata['id'],hour=env.c.t,
                    pending_hour=env.c.pending is not None,unsettled_contract_commitment=float(env.c.ledger.fee[env.c.t:].sum())))
        raise
