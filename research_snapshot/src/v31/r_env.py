"""Balanced natural-year sampling only; original trusted env is unchanged."""
import numpy as np
from .lite_gym import AnnualDecisionEnv

def specs(seed):
    if seed not in (0,1):raise ValueError('frozen seeds 0 and 1 only')
    return [dict(year=(2033 if (j+seed)%2==0 else 2036),family=f'train_{j//2}',
        group=g,split='train',fixture=False) for j,g in enumerate(('G00','G01','G10','G11'))]

class BalancedSampler:
    def __init__(self,seed,size):
        self.order=np.random.default_rng(seed).permutation(size).tolist();self.cursor=0
    def integers(self,size):
        if size!=len(self.order):raise ValueError('curriculum size changed')
        index=self.order[self.cursor%size];self.cursor+=1;return index

class BalancedAnnualEnv(AnnualDecisionEnv):
    def __init__(self,*args,seed=0,budget_guard=None,**kwargs):
        super().__init__(*args,seed=seed,**kwargs)
        self.rng=BalancedSampler(seed,len(self.cache.specs))
        self.budget_guard=budget_guard
    def reset(self,*,seed=None,options=None):
        if self.driver is not None and not self.ended:
            if seed is not None and seed!=self.seed_value:raise ValueError('cannot reseed active annual responsibility')
            return super().reset(seed=None,options=options)
        if seed is not None:
            if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('bad seed')
            self.rng=BalancedSampler(int(seed),len(self.cache.specs));self.seed_value=int(seed)
        result=super().reset(seed=None,options=options)
        if self.budget_guard:
            self.budget_guard()
            self.driver.c.on_hour=lambda row:self.budget_guard()
        return result
