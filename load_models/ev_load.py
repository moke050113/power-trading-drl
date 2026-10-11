"""简化EV聚合充电：电池容量MWh，功率MW，SOC为0～1，每步1小时。"""
from load_models._validation import check_hour, finite, window


class EVLoad:
    def __init__(self, capacity_mwh=0.06, initial_soc=0.2, max_power_mw=0.007,
                 start_hour=18, end_hour=24, efficiency=0.9, target_soc=1.0):
        window(start_hour, end_hour)
        self.capacity_mwh = finite(capacity_mwh, '容量')
        if self.capacity_mwh == 0:
            raise ValueError('容量必须大于0')
        self.initial_soc = finite(initial_soc, '初始SOC', maximum=1)
        self.target_soc = finite(target_soc, '目标SOC', minimum=self.initial_soc, maximum=1)
        self.max_power_mw = finite(max_power_mw, '最大功率')
        self.efficiency = finite(efficiency, '效率', maximum=1)
        if self.efficiency == 0:
            raise ValueError('效率必须大于0')
        self.start_hour, self.end_hour = start_hour, end_hour
        self.reset()

    def reset(self):
        self.hour = 0
        self.soc = self.initial_soc
        return self.soc

    def step(self, hour, requested_power_mw):
        check_hour(hour, self.hour)
        requested = finite(requested_power_mw, '请求功率')
        energy = 0.0
        if self.start_hour <= hour < self.end_hour:
            grid_needed = max(0.0, (self.target_soc - self.soc) * self.capacity_mwh / self.efficiency)
            energy = min(requested, self.max_power_mw, grid_needed)  # 1小时：MW × h = MWh
            self.soc = min(self.target_soc, self.soc + energy * self.efficiency / self.capacity_mwh)
        self.hour += 1
        return {'hour': hour, 'load_mwh': energy, 'power_mw': energy,
                'battery_energy_mwh': energy * self.efficiency, 'soc': self.soc,
                'target_reached': self.soc >= self.target_soc - 1e-12}
