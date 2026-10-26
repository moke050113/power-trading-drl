"""两个独立Gym投影；正式年度评价同一Coordinator，不共享两个训练时钟。"""
from datetime import timedelta
from copy import copy
from pathlib import Path
from time import perf_counter
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from v3.core import vector,settlement
from v3.data import YearStore, PublishedWorld
from v3.devices import DailyDevices
from v3.coordinator import YearCoordinator,lower_view
from v3.baselines import forecast_dispatch


def dict_space(sample):
    return spaces.Dict({k:spaces.Box(-np.inf,np.inf,shape=v.shape,dtype=np.float32) for k,v in sample.items()})

def decode_upper(obs,action):
    a=vector(action,13,0,1)
    last=int(obs['calendar'][2])==1
    # 明确的可行动作解码，第二机会屏蔽WAIT；不是随机修复非法市场请求。
    if last:verb='SKIP' if a[0]<.5 else 'COMMIT'
    else:verb=('WAIT','SKIP','COMMIT')[min(2,int(a[0]*3))]
    return verb,a[1:]


class UpperTradingEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,root=None,*,split='train',seed=0,lower=forecast_dispatch):
        self.store=YearStore(**({'root':root} if root is not None else {}),split=split)
        self.split=split;self.rng=np.random.default_rng(seed);self.seed_value=seed;self.cursor=0
        self.lower=lower;self.c=None;self.episodes=[];self.actions=[];self.resets=[]
        sample=YearCoordinator(self.store,self.store.worlds[0]['id']).upper_view()
        self.observation_space=dict_space(sample)
        self.action_space=spaces.Box(0.,1.,shape=(13,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('不接受日期/仓位覆盖options')
        if self.c is not None and self.c.t<self.c.world.n:
            if self.c.macros or self.c.pending is not None:raise RuntimeError('已交易的年度不能reset逃债')
            if seed is not None and seed!=self.seed_value:raise ValueError('幂等reset参数冲突')
            return self.c.upper_view(),{}
        if self.split=='validation' and self.cursor>=len(self.store.worlds):raise StopIteration('validation年度穷尽')
        if seed is not None:
            if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('非法seed')
            self.rng=np.random.default_rng(seed);self.seed_value=int(seed)
        super().reset(seed=seed)
        ix=int(self.rng.integers(len(self.store.worlds))) if self.split=='train' else self.cursor
        w=self.store.worlds[ix];start=perf_counter()
        c=YearCoordinator(self.store,w['id']);obs=c.upper_view()
        if not self.observation_space.contains(obs):raise ValueError('观察空间错误')
        self.c=c
        if self.split=='validation':self.cursor+=1
        self.resets.append(dict(world=w['id'],split=self.split,seed=self.seed_value,seconds=perf_counter()-start))
        return obs,{}
    def step(self,action):
        if self.c is None or self.c.next_op() is None:raise RuntimeError('先reset；年终不能再次step')
        obs=self.c.upper_view();verb,u=decode_upper(obs,action)
        before=self.c.total_purchase+self.c.total_flex
        self.c.submit_macro(verb,u)
        op=self.c.next_op()
        at=op.decision_at if op else self.c.world.start+timedelta(hours=self.c.world.n)
        self.c.run_until(at,self.lower)
        raw=-(self.c.total_purchase+self.c.total_flex-before)
        done=op is None
        observation={k:np.zeros_like(v) for k,v in obs.items()} if done else self.c.upper_view()
        self.actions.append(dict(world=self.c.world.metadata['id'],verb=verb,raw_reward=raw,ended=done))
        if done:self.episodes.append(dict(world=self.c.world.metadata['id'],hours=self.c.t,purchase_cost=self.c.total_purchase,flex_cost=self.c.total_flex,macro_steps=len(self.c.macros)))
        return observation,float(raw),done,False,{'purchase_cost':self.c.total_purchase,'flex_cost':self.c.total_flex}


class LowerDispatchEnv(gym.Env):
    """只train：从已冻结上层年账本抽独立日会话，不在reset重签合同。"""
    metadata={'render_modes':[]}
    def __init__(self,store,plans,*,seed=0):
        if store.split!='train':raise PermissionError('训练下层不接收validation/test')
        if set(plans)!={w['id'] for w in store.worlds}:raise ValueError('缓存仓位不是完整train集合')
        self.store=store;self.plans=plans;self.rng=np.random.default_rng(seed);self.seed_value=seed
        self.world=None;self.h=24;self.rows=[];self.episodes=[];self.resets=[];self.audit_key=None
        self.action_space=spaces.Box(0.,1.,shape=(3,),dtype=np.float32)
        sample={k:np.zeros(n,dtype=np.float32) for k,n in {'hour':1,'forecast_load':24,'forecast_price':24,'contract_plan':24,'contract_fee':24,'offer':2,'devices':9,'history':96,'history_mask':24}.items()}
        self.observation_space=dict_space(sample)
    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('不接受日期/仓位覆盖options')
        if self.world is not None and self.h<24:
            if self.h>0:raise RuntimeError('不允许中途reset')
            if seed is not None and seed!=self.seed_value:raise ValueError('幂等seed冲突')
            return self._obs(),{}
        if seed is not None:
            if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('非法seed')
            self.rng=np.random.default_rng(seed);self.seed_value=int(seed)
        super().reset(seed=seed);start=perf_counter()
        w=self.store.worlds[int(self.rng.integers(len(self.store.worlds)))];day=int(self.rng.integers(w['hours']//24))
        world=PublishedWorld(self.store,w['id'])
        # 明确的可信日子通道，起点由train选日器选定；并非修改旧适配器split。
        world.open_day(day)
        devices=DailyDevices(w['group'].endswith('1'))
        q,f=self.plans[w['id']]
        self.q=vector(q[day*24:day*24+24],24);self.f=vector(f[day*24:day*24+24],24)
        self.world,self.devices,self.day,self.h,self.rows=world,devices,day,0,[]
        self.audit_key=w['id'];self.episode_purchase=self.episode_flex=0.
        self.resets.append(dict(world=w['id'],day=day,seed=self.seed_value,seconds=perf_counter()-start))
        return self._obs(),{}
    def _obs(self):
        if self.h==24:return {k:np.zeros(s.shape,dtype=np.float32) for k,s in self.observation_space.spaces.items()}
        return lower_view(self.world,self.day*24+self.h,self.devices,self.q,self.f,self.rows)
    def step(self,action):
        if self.world is None or self.h>=24:raise RuntimeError('先reset；日终不能再次step')
        x=vector(action,3,0,1);t=self.day*24+self.h
        p,k=self.world.offer(t,self.world.start+timedelta(hours=t))
        candidate,d=self.devices.prepare(x[1]*7,x[2]*4)
        b,pb=self.world.reveal(t,self.world.start+timedelta(hours=t+1))
        row=settlement(b,d['industrial_actual_mw'],d['ev_actual_mw'],self.q[self.h],self.f[self.h],x[0]*k,p,pb)
        row.update(d,t=t,hour=self.h,u_spot=float(x[0]),u_ev=float(x[1]),u_industrial=float(x[2]))
        row['objective_cost']=row['purchase_cost']+row['flex_cost'];row['reward']=-row['objective_cost']
        if not np.isfinite(list(row.values())).all():raise ValueError('结算非有限')
        world=copy(self.world);world.commit_reveal(t)
        done=self.h+1==24
        obs={k:np.zeros(s.shape,dtype=np.float32) for k,s in self.observation_space.spaces.items()} if done else lower_view(world,t+1,candidate,self.q,self.f,self.rows+[row])
        if not self.observation_space.contains(obs) or not all(np.isfinite(a).all() for a in obs.values()):raise ValueError('候选观察非法')
        self.world=world;self.devices=candidate;self.h+=1;self.rows.append(row)
        self.episode_purchase+=row['purchase_cost'];self.episode_flex+=row['flex_cost']
        if done:self.episodes.append(dict(world=self.audit_key,day=self.day,purchase_cost=self.episode_purchase,flex_cost=self.episode_flex,hours=24))
        return obs,row['reward'],done,False,{'hour_record':row.copy()}
