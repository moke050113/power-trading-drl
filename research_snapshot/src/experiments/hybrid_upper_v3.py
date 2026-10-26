"""D8: genuine masked Categorical x independent Beta PPO, Box only transports samples."""
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Beta
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from gymnasium import spaces
from v3.environments import UpperTradingEnv, decode_upper

VERBS = ('WAIT', 'SKIP', 'COMMIT')

def decode_hybrid(obs, action):
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (13,) or not np.isfinite(a).all():
        raise ValueError('hybrid action must be finite (13,)')
    if a[0] not in (0., 1., 2.) or (a[1:] < 0).any() or (a[1:] > 1).any():
        raise ValueError('invalid category or fraction')
    if int(obs['calendar'][2]) == 1 and a[0] == 0:
        raise ValueError('WAIT forbidden at final opportunity')
    return VERBS[int(a[0])], a[1:].copy()

def canonical_action(obs, action):
    verb, u = decode_hybrid(obs, action)
    last = int(obs['calendar'][2]) == 1
    code = 1. if verb == 'COMMIT' else (0. if last or verb == 'WAIT' else .5)
    a = np.r_[code, u]
    old_verb, old_u = decode_upper(obs, a)
    if old_verb != verb or not np.array_equal(old_u, u):
        raise RuntimeError('semantic mapping mismatch')
    return a

class HybridUpperEnv(UpperTradingEnv):
    def __init__(self, *args, **kwargs):
        if kwargs.get('split', 'train') not in ('train', 'validation'):
            raise PermissionError('no test before storage construction')
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(np.zeros(13, np.float32),
            np.array([2.] + [1.] * 12, np.float32), dtype=np.float32)

    def step(self, action):
        if self.c is None or self.c.next_op() is None:
            raise RuntimeError('reset required or year finished')
        return super().step(canonical_action(self.c.upper_view(), action))

class HybridDistribution(Distribution):
    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, 27)

    def proba_distribution(self, parameters, last):
        if parameters.ndim != 2 or parameters.shape[1] != 27 or not th.isfinite(parameters).all():
            raise ValueError('invalid distribution parameters')
        logits = parameters[:, :3]
        mask = th.zeros_like(logits, dtype=th.bool)
        mask[:, 0] = last.reshape(-1).bool()
        self.category = Categorical(logits=logits.masked_fill(mask, -th.inf))
        # >1 ensures finite density/entropy and unique interior mode; no clipping samples.
        self.a = 1.0001 + th.nn.functional.softplus(parameters[:, 3:15])
        self.b = 1.0001 + th.nn.functional.softplus(parameters[:, 15:27])
        self.ratios = Beta(self.a, self.b)
        return self

    def log_prob(self, actions):
        if actions.ndim != 2 or actions.shape[1] != 13 or not th.isfinite(actions).all():
            raise ValueError('invalid action tensor')
        c, u = actions[:, 0], actions[:, 1:]
        if ((c != c.round()) | (c < 0) | (c > 2)).any() or ((u <= 0) | (u >= 1)).any():
            raise ValueError('category integer and open Beta support required')
        return self.category.log_prob(c.long()) + self.ratios.log_prob(u).sum(-1)

    def entropy(self):
        return self.category.entropy() + self.ratios.entropy().sum(-1)

    def sample(self):
        return th.cat((self.category.sample().float().unsqueeze(-1), self.ratios.sample()), -1)

    def mode(self):
        u = (self.a - 1) / (self.a + self.b - 2)
        return th.cat((self.category.probs.argmax(-1).float().unsqueeze(-1), u), -1)

    def actions_from_params(self, parameters, last, deterministic=False):
        return self.proba_distribution(parameters, last).get_actions(deterministic)

    def log_prob_from_params(self, parameters, last):
        actions = self.actions_from_params(parameters, last)
        return actions, self.log_prob(actions)

class HybridUpperPolicy(MultiInputActorCriticPolicy):
    def _build(self, lr_schedule):
        # Reuse SB3's MLP/value/init/optimizer builder, creating exactly one 27-output head.
        self.action_dist = CategoricalDistribution(27)
        super()._build(lr_schedule)
        self.action_dist = HybridDistribution()

    def _joint(self, obs):
        features = self.extract_features(obs)
        if not self.share_features_extractor:
            raise ValueError('D8 freezes shared extractor')
        pi, vf = self.mlp_extractor(features)
        last = obs['calendar'][:, 2] == 1
        return self.action_dist.proba_distribution(self.action_net(pi), last), vf

    def get_distribution(self, obs):
        return self._joint(obs)[0]

    def forward(self, obs, deterministic=False):
        dist, vf = self._joint(obs)
        actions = dist.get_actions(deterministic)
        return actions, self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist, vf = self._joint(obs)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def _get_action_dist_from_latent(self, latent_pi):
        raise RuntimeError('masked distribution requires original legal observation')
