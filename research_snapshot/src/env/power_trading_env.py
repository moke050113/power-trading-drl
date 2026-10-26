import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PowerTradingEnv(gym.Env):
    """逐小时交易教学环境：默认24步，每步结算一个小时。

    状态：[小时负荷(MWh), 合同价格(元/MWh), 现货价格(元/MWh), 小时]。
    动作：[本小时合同覆盖电量(MWh)]，剩余负荷由现货补足。
    合同动作是简化教学模型，不代表真实中长期合同可逐小时重签。
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon=24, load_profile=None, spot_price_profile=None,
                 long_term_price=400.0):
        super().__init__()
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon必须是正整数")
        self.horizon = int(horizon)
        price = np.asarray(long_term_price, dtype=np.float64)
        if price.shape != () or not np.isfinite(price) or not 0 <= price <= 2000:
            raise ValueError("合同价格必须是0到2000之间的有限标量")
        self.long_term_price = float(price)
        # 固定合成曲线便于复现，首小时与原版一致。
        default_load = [100, 95, 90, 85, 80, 85, 100, 120, 140, 150, 145, 135,
                        130, 125, 120, 125, 140, 160, 170, 165, 150, 135, 120, 110]
        default_spot = [600, 350, 300, 280, 260, 300, 380, 450, 550, 600, 580, 500,
                        420, 380, 360, 400, 550, 700, 800, 750, 650, 550, 450, 400]
        self.load_profile = self._profile(load_profile, default_load, "负荷", 1000)
        self.spot_price_profile = self._profile(spot_price_profile, default_spot, "现货价格", 2000)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([max(100.0, float(self.load_profile.max()))], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([1000, 2000, 2000, self.horizon], dtype=np.float32),
            dtype=np.float32,
        )
        self.hour = 0
        self.episode_cost = 0.0
        self.state = None

    def _profile(self, values, default, name, maximum):
        if values is None:
            values = [default[i % len(default)] for i in range(self.horizon)]
        result = np.array(values, dtype=np.float64, copy=True)
        if result.shape != (self.horizon,):
            raise ValueError(f"{name}曲线必须是一维且长度等于horizon")
        if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > maximum):
            raise ValueError(f"{name}必须为0到{maximum}之间的有限数值")
        result.flags.writeable = False
        return result

    def _observation(self):
        # 终止状态保留最后一小时信息，小时标记为horizon，避免越界。
        index = min(self.hour, self.horizon - 1)
        self.load = float(self.load_profile[index])
        self.spot_price = float(self.spot_price_profile[index])
        self.state = np.array([self.load, self.long_term_price, self.spot_price, self.hour],
                              dtype=np.float32)
        return self.state.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.episode_cost = 0.0
        return self._observation(), {}

    def step(self, action):
        if self.state is None or self.hour >= self.horizon:
            raise RuntimeError("请先reset；一天结束后也必须reset再交易")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.all(np.isfinite(action)):
            raise ValueError("动作必须是仅含一个有限数值的数组，例如[70.0]")
        hour = self.hour
        load = float(self.load_profile[hour])
        spot_price = float(self.spot_price_profile[hour])
        # 沿用原版裁剪语义，购买量限制在[0, 当前负荷]。
        long_term_quantity = float(np.clip(action[0], 0.0, load))
        spot_quantity = load - long_term_quantity
        long_term_cost = long_term_quantity * self.long_term_price
        spot_cost = spot_quantity * spot_price
        total_cost = long_term_cost + spot_cost
        self.episode_cost += total_cost
        self.hour += 1
        info = {
            "hour": hour,
            "load": load,
            "long_term_price": self.long_term_price,
            "spot_price": spot_price,
            "long_term_quantity": long_term_quantity,
            "spot_quantity": spot_quantity,
            "long_term_cost": long_term_cost,
            "spot_cost": spot_cost,
            "total_cost": total_cost,
            "episode_cost": self.episode_cost,
        }
        return self._observation(), -total_cost, self.hour == self.horizon, False, info
