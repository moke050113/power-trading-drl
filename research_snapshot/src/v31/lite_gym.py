"""Two decision views of one continuous trusted annual driver, no new PPO math."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .lite_data import LiteStore
from .lite_engine import LiteAnnualDriver
from .lite_rules import RULES
from .lite_audit import audit_year

# Identical fixed physical divisors to D10, kept lightweight: Gym/basic tests
# must not import PyTorch or old experiment preflight machinery.
DIVISORS=dict(stage=[1]*3,calendar=[1]*3,coverage=[1]*12,reference=[100]*12,
    remaining_capacity=[100]*12,forecast=[100,500],quote=[500],forecast_gap=[100],
    flex_capacity=[42,32,100/3,12],settled_cost=[500,.01],device_state=[1,12])

def decode_hybrid(obs,action):
    a=np.asarray(action,dtype=np.float64)
    if a.shape!=(13,) or not np.isfinite(a).all():raise ValueError('finite hybrid (13,) required')
    if a[0] not in (0.,1.,2.) or (a[1:]<0).any() or (a[1:]>1).any():raise ValueError('invalid hybrid action')
    if int(obs['calendar'][2])==1 and a[0]==0:raise ValueError('WAIT forbidden at final opportunity')
    return ('WAIT','SKIP','COMMIT')[int(a[0])],a[1:].copy()


UPPER_SHAPES=dict(stage=3,calendar=3,coverage=12,reference=12,remaining_capacity=12,forecast=2,
    quote=1,forecast_gap=1,flex_capacity=4,settled_cost=2,device_state=2,resource_outlook=4)
LOWER_SHAPES=dict(hour=1,forecast_load=24,forecast_price=24,contract_plan=24,contract_fee=24,
    offer=2,devices=9,history=96,history_mask=24,actual_state_mask=2,declared_initial_soc=1,forecast_offer_price=24)


def encode(obs,level):
    if level=='upper':
        result={k:np.array(obs[k]/np.array(d,np.float32),np.float32) for k,d in DIVISORS.items()}
        result['resource_outlook']=np.array(obs['resource_outlook']/np.array([100/3,12,100/3,12],np.float32),np.float32)
    else:
        div=dict(hour=24,forecast_load=100,forecast_price=500,contract_plan=100,contract_fee=100000,
            offer=np.array([500,160],np.float32),devices=np.array([1,100/3,7,24,24,12,4,24,24],np.float32),
            history=np.tile([100,500,7,4],24).astype(np.float32),history_mask=1,actual_state_mask=1,
            declared_initial_soc=1,forecast_offer_price=500)
        if set(obs)!=set(div):raise ValueError('lower shape schema mismatch')
        result={k:np.array(v/div[k],np.float32) for k,v in obs.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite encoded observation')
    return result


def upper_policy(model):
    def act(raw):
        action,_=model.predict(encode(raw,'upper'),deterministic=True)
        return decode_hybrid(raw,action)
    return act


def lower_policy(model):
    return lambda raw:model.predict(encode(raw,'lower'),deterministic=True)[0]


class WorldCache:
    """Owner-only cache. Each driver always receives independent ledger/devices."""
    def __init__(self,specs):
        if not specs or any(s['split']!='train' for s in specs):raise PermissionError('training cache accepts train specs only')
        self.specs=list(specs);self.compiled={}
    def make(self,index):
        spec=self.specs[index];key=tuple(sorted(spec.items()))
        if key in self.compiled:return self.compiled[key].replay_copy()
        return LiteStore(**spec)
    def complete(self,store):self.compiled[tuple(sorted(dict(year=store.year,family=store.family,group=store.group,split=store.split,fixture=store.fixture).items()))]=store


class AnnualDecisionEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,cache,level,*,seed=0,partner=None):
        if level not in ('upper','lower'):raise ValueError('unknown layer')
        self.cache=cache;self.level=level;self.partner=partner;self.rng=np.random.default_rng(seed);self.seed_value=seed
        shape=UPPER_SHAPES if level=='upper' else LOWER_SHAPES
        self.observation_space=spaces.Dict({k:spaces.Box(-np.inf,np.inf,(n,),np.float32) for k,n in shape.items()})
        self.action_space=spaces.Box(np.zeros(13,np.float32),np.array([2]+[1]*12,np.float32),dtype=np.float32) if level=='upper' else spaces.Box(0,1,(3,),np.float32)
        self.driver=None;self.raw=None;self.ended=True;self.actions=0;self.resets=[];self.steps=[];self.episodes=[]
        self.recorded_hours=[];self.recorded_macros=[];self.recorded_daily=[];self._archived=False

    def _archive(self):
        if self.driver is None or self._archived:return
        key=self.driver.store.key
        instance=len(self.resets)-1
        learning_hours=max((s['completed_hour'] for s in self.steps if s['episode_instance']==instance),default=0)
        self.recorded_hours.extend(dict(r,world=key,episode_instance=instance,during_gradient_collection=r['t']<learning_hours) for r in self.driver.c.hours)
        self.recorded_macros.extend(dict(r,world=key,episode_instance=instance) for r in self.driver.macro_receipts)
        self.recorded_daily.extend(dict(r,world=key,episode_instance=instance) for r in self.driver.daily_receipts)
        self._archived=True

    def _advance(self):
        if self.level=='upper':return self.driver.advance(stop='upper',lower=self.partner or RULES['B2'][1])
        if self.partner is None:raise ValueError('fixed frozen upper partner required')
        return self.driver.advance(stop='lower',upper=self.partner)

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('no date/position override')
        if self.driver is not None and not self.ended:
            if self.actions:raise RuntimeError('cannot reset an unfinished annual commitment')
            if seed is not None and seed!=self.seed_value:raise ValueError('conflicting idempotent seed')
            return encode(self.raw,self.level),{}
        if seed is not None:
            if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('bad seed')
            self.rng=np.random.default_rng(seed);self.seed_value=int(seed)
        super().reset(seed=seed)
        from time import perf_counter
        tick=perf_counter();index=int(self.rng.integers(len(self.cache.specs)))
        self._archive()
        self.driver=LiteAnnualDriver(self.cache.make(index));self._archived=False;self.ended=False;self.actions=0
        self.raw=self._advance()
        self.resets.append(dict(index=index,key=self.driver.store.key,seed=self.seed_value,seconds=perf_counter()-tick))
        obs=encode(self.raw,self.level)
        if not self.observation_space.contains(obs):raise ValueError('space mismatch')
        return obs,{}

    def step(self,action):
        if self.driver is None or self.ended:raise RuntimeError('reset required / annual episode finished')
        c=self.driver.c;before=c.total_purchase+c.total_flex
        if self.level=='upper':
            verb,u=decode_hybrid(self.raw,action);self.driver.submit_upper(verb,u)
        else:self.driver.submit_lower(action)
        self.actions+=1;self.raw=self._advance();self.ended=self.raw is None
        raw=-(c.total_purchase+c.total_flex-before)
        self.steps.append(dict(index=len(self.steps),world=self.driver.store.key,episode_instance=len(self.resets)-1,action=np.asarray(action).tolist(),
            raw_reward=raw,training_reward=raw/1e6,ended=self.ended,completed_hour=c.t,
            cumulative_purchase=c.total_purchase,cumulative_flex=c.total_flex,
            signed_contract_fee=float(c.ledger.fee.sum()),unsettled_contract_fee=float(c.ledger.fee[c.t:].sum())))
        if self.ended:
            record=audit_year(self.driver);record['world']=self.driver.store.key;self.episodes.append(record);self.cache.complete(self.driver.store)
            obs={k:np.zeros(s.shape,np.float32) for k,s in self.observation_space.spaces.items()}
        else:obs=encode(self.raw,self.level)
        if not self.observation_space.contains(obs):raise ValueError('encoded observation invalid')
        return obs,float(raw/1e6),self.ended,False,{'purchase_cost':c.total_purchase,'flex_cost':c.total_flex,'raw_reward':raw}

    def finish_without_learning(self,model):
        """Fixed final policies discharge the active year; never calls learn."""
        if self.ended:return None
        upper=upper_policy(model) if self.level=='upper' else self.partner
        lower=self.partner or RULES['B2'][1] if self.level=='upper' else lower_policy(model)
        # A VecEnv may have reset to a fresh empty upper instance after final step.
        # That empty instance has no liability. Lower reset already signed contracts.
        if self.actions==0 and self.level=='upper' and not self.driver.c.macros:
            return {'empty_auto_reset':True}
        tail_start=self.driver.c.t
        while self.driver.c.t<self.driver.store.n:
            if self.driver.phase=='upper':
                raw=self.driver.upper_observation();verb,u=upper(raw);self.driver.submit_upper(verb,u)
            elif self.driver.phase=='lower':
                self.driver.submit_lower(lower(self.driver.lower_observation()))
            obs=self.driver.advance(stop='upper',lower=lower)
            if obs is None:break
        record=audit_year(self.driver);record['world']=self.driver.store.key;record['no_learning_tail']=True
        record['tail_start_hour']=tail_start;record['no_learning_tail_hours']=self.driver.c.t-tail_start
        self.cache.complete(self.driver.store);self.ended=True;self._archive()
        return record
