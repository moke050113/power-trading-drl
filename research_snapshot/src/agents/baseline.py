"""策略输入现有环境观察值，输出合同覆盖电量数组（MWh）。"""
import numpy as np


class BaselinePolicy:
    def __init__(self, strategy='spot', seed=0):
        if strategy not in ('spot', 'fixed70', 'random'):
            raise ValueError('strategy必须是spot、fixed70或random')
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):
        """兼容SB3的predict返回格式；random策略始终从自身随机流采样。"""
        state = np.asarray(observation, dtype=float)
        if state.shape not in ((3,), (4,)) or not np.all(np.isfinite(state)) or state[0] < 0:
            raise ValueError('观察值须为[负荷, 合同价, 现货价, 可选小时]的有限数值')
        ratio = {'spot': 0.0, 'fixed70': 0.7}.get(self.strategy)
        if ratio is None:
            ratio = float(self.rng.uniform(0, 1))
        return np.array([state[0] * ratio], dtype=np.float32), None
