"""Upper-only fixed training-unit change. No trusted execution changes."""
from .r_env import BalancedAnnualEnv, specs

UPPER_REWARD_DIVISOR = 100_000_000.0
LOWER_REWARD_DIVISOR = 1_000_000.0


class ScaledAnnualEnv(BalancedAnnualEnv):
    def step(self, action):
        obs, environment_reward, ended, truncated, info = super().step(action)
        divisor = UPPER_REWARD_DIVISOR if self.level == 'upper' else LOWER_REWARD_DIVISOR
        training_reward = info['raw_reward'] / divisor
        # Own newly produced audit row only; historical files and the original
        # AnnualDecisionEnv implementation remain untouched.
        self.steps[-1].update(environment_reward=environment_reward,
                              training_reward=training_reward,
                              reward_divisor=divisor)
        return obs, training_reward, ended, truncated, info
