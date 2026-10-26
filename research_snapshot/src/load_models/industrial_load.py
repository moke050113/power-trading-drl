"""单个可分割、可暂停工业任务；每步1小时，截止前保证任务电量守恒。"""
from load_models._validation import check_hour, finite, window


class IndustrialLoad:
    def __init__(self, task_energy_mwh=12.0, max_power_mw=4.0, start_hour=8, end_hour=16):
        window(start_hour, end_hour)
        self.task_energy_mwh = finite(task_energy_mwh, '任务电量')
        self.max_power_mw = finite(max_power_mw, '最大功率')
        if self.task_energy_mwh > self.max_power_mw * (end_hour - start_hour):
            raise ValueError('任务电量超过时间窗口内可供电量')
        self.start_hour, self.end_hour = start_hour, end_hour
        self.reset()

    def reset(self):
        self.hour = 0
        self.remaining_energy_mwh = self.task_energy_mwh
        return self.remaining_energy_mwh

    def step(self, hour, requested_power_mw):
        check_hour(hour, self.hour)
        requested = finite(requested_power_mw, '请求功率')
        energy = minimum = 0.0
        if self.start_hour <= hour < self.end_hour:
            future_capacity = (self.end_hour - hour - 1) * self.max_power_mw
            minimum = max(0.0, self.remaining_energy_mwh - future_capacity)
            maximum = min(self.max_power_mw, self.remaining_energy_mwh)
            energy = min(maximum, max(requested, minimum))
            self.remaining_energy_mwh = max(0.0, self.remaining_energy_mwh - energy)
        self.hour += 1
        return {'hour': hour, 'load_mwh': energy, 'power_mw': energy,
                'minimum_required_mwh': minimum, 'remaining_energy_mwh': self.remaining_energy_mwh,
                'completed': self.remaining_energy_mwh <= 1e-12}
