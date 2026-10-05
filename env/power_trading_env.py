import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PowerTradingEnv(gym.Env):
    """
    简化的电力交易强化学习环境

    Agent的任务：
    决定中长期市场购买多少电量。
    剩余电量由现货市场补充。
    """

    def __init__(self):
        super().__init__()

        # ==========================
        # 1. 基本市场参数
        # ==========================

        # 用户总负荷：100 MWh
        self.load = 100.0

        # 中长期电价：400 元/MWh
        self.long_term_price = 400.0

        # 现货电价：600 元/MWh
        self.spot_price = 600.0

        # ==========================
        # 2. 定义Action空间
        # ==========================

        # Agent决定中长期购买量
        # 最少0 MWh，最多100 MWh
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==========================
        # 3. 定义State空间
        # ==========================

        # 当前状态包含：
        # [负荷, 中长期价格, 现货价格]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 2000.0, 2000.0], dtype=np.float32),
            dtype=np.float32
        )

        # 当前状态
        self.state = None

    def reset(self, seed=None, options=None):
        """
        重置环境
        """

        super().reset(seed=seed)

        self.state = np.array(
            [
                self.load,
                self.long_term_price,
                self.spot_price
            ],
            dtype=np.float32
        )

        return self.state, {}

    def step(self, action):
        """
        Agent执行一次交易决策
        """

        # 获取Agent决定的中长期购买量
        long_term_quantity = float(action[0])

        # 防止超过实际负荷
        long_term_quantity = np.clip(
            long_term_quantity,
            0.0,
            self.load
        )

        # 剩余电量进入现货市场
        spot_quantity = self.load - long_term_quantity

        # 计算中长期成本
        long_term_cost = (
            long_term_quantity * self.long_term_price
        )

        # 计算现货成本
        spot_cost = (
            spot_quantity * self.spot_price
        )

        # 计算总成本
        total_cost = (
            long_term_cost + spot_cost
        )

        # 强化学习Reward
        # 成本越低，Reward越高
        reward = -total_cost

        # 当前版本每次交易后结束
        terminated = True

        # 没有特殊的截断情况
        truncated = False

        # 下一状态
        next_state = self.state

        # 一些额外信息
        info = {
            "long_term_quantity": long_term_quantity,
            "spot_quantity": spot_quantity,
            "long_term_cost": long_term_cost,
            "spot_cost": spot_cost,
            "total_cost": total_cost
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )