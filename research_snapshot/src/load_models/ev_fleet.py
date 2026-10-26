"""S2a同质EV车队约束执行；仅设备，不接交易/Gym/价格数据。

复用原EVLoad一次step；单车kWh/kW转换为聚合MWh/MW，固定1小时。
冻结配置、截止可达性检查、候选状态提交；无V2G/异质车队/跨日SOC。
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real

from load_models._validation import check_hour, window
from load_models.ev_load import EVLoad


def _number(value, name, *, minimum=0.0, maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name}必须是有限实数标量')
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f'{name}超出数值范围') from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f'{name}必须在[{minimum},{maximum}]内')
    return result


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


@dataclass(frozen=True)
class EVFleetConfig:
    """默认是已批准1000辆教学参数；其他配置仅用于独立可行性/缩放验证。"""
    vehicle_count: int = 1000
    battery_capacity_kwh: float = 60.0
    max_charging_power_kw: float = 7.0
    initial_soc: float = 0.2
    target_soc: float = 0.7
    efficiency: float = 0.9
    start_hour: int = 18
    end_hour: int = 24

    def __post_init__(self):
        if isinstance(self.vehicle_count, bool) or not isinstance(self.vehicle_count, Integral) or self.vehicle_count < 1:
            raise ValueError('车数必须是正整数')
        object.__setattr__(self, 'vehicle_count', int(self.vehicle_count))
        for field in ('battery_capacity_kwh', 'max_charging_power_kw', 'initial_soc', 'target_soc', 'efficiency'):
            maximum = 1.0 if field in ('initial_soc', 'target_soc', 'efficiency') else math.inf
            object.__setattr__(self, field, _number(getattr(self, field), field, maximum=maximum))
        if self.battery_capacity_kwh == 0 or self.efficiency == 0 or self.target_soc < self.initial_soc:
            raise ValueError('容量/效率必须大于0，目标SOC不得低于初始SOC')
        window(self.start_hour, self.end_hour)
        object.__setattr__(self, 'start_hour', int(self.start_hour))
        object.__setattr__(self, 'end_hour', int(self.end_hour))
        for value in (self.capacity_mwh, self.max_power_mw):
            _number(value, '聚合量')
        if self.capacity_mwh == 0:
            raise ValueError('聚合容量过小导致数值下溢')

    @property
    def capacity_mwh(self):
        try:
            return self.vehicle_count * self.battery_capacity_kwh / 1000.0
        except OverflowError:
            raise ValueError('聚合容量数值溢出') from None

    @property
    def max_power_mw(self):
        try:
            return self.vehicle_count * self.max_charging_power_kw / 1000.0
        except OverflowError:
            raise ValueError('聚合功率数值溢出') from None

    @property
    def required_grid_mwh(self):
        return self.capacity_mwh * (self.target_soc - self.initial_soc) / self.efficiency


@dataclass(frozen=True)
class _FleetState:
    config: EVFleetConfig
    model: EVLoad
    total_grid_mwh: float = 0.0
    total_battery_mwh: float = 0.0
    total_forced_mwh: float = 0.0
    records: tuple = ()


class EVFleetExecutor:
    """reset后step(hour, scalar_MW)顺序0～23；一次提交一个候选状态。

    尚未开始可reset；开始后须完成24小时，不允许中途清SOC/重开一天。
    本模块单线程；交交易内核的未来跨设备事务须另行设计。
    """
    def __init__(self, config=None):
        self._initial_config = self._config_type(EVFleetConfig() if config is None else config)
        self._state = None

    @staticmethod
    def _config_type(config):
        if type(config) is not EVFleetConfig:
            raise TypeError('需要冻结EVFleetConfig')
        return config

    @property
    def config(self):
        return self._initial_config if self._state is None else self._state.config

    def _active(self):
        if self._state is None:
            raise RuntimeError('请先reset并通过截止可达性检查')
        return self._state

    @property
    def hour(self):
        return self._active().model.hour

    @property
    def soc(self):
        return self._active().model.soc

    @property
    def status(self):
        if self._state is None:
            return 'UNINITIALIZED'
        return 'FINISHED' if self.hour == 24 else 'ACTIVE'

    @property
    def hourly_records(self):
        return [record.copy() for record in self._active().records]

    def reset(self, *, config=None):
        """全部校验及候选EV初始化成功才替换状态；不可达明确报错。"""
        if self._state is not None and 0 < self.hour < 24:
            raise RuntimeError('设备执行中不能reset，须完成24小时')
        cfg = self._config_type(self.config if config is None else config)
        need = _number(cfg.required_grid_mwh, '目标电网需求')
        capacity = _number(cfg.max_power_mw * (cfg.end_hour - cfg.start_hour), '窗口供电容量')
        if need > capacity:
            raise ValueError(f'截止目标不可达：需{need}MWh，窗口最多{capacity}MWh；不降低目标SOC')
        model = EVLoad(capacity_mwh=cfg.capacity_mwh, initial_soc=cfg.initial_soc,
                       max_power_mw=cfg.max_power_mw, start_hour=cfg.start_hour,
                       end_hour=cfg.end_hour, efficiency=cfg.efficiency, target_soc=cfg.target_soc)
        self._validate_model_config(model, cfg)
        if type(model.hour) is not int or model.hour != 0 or model.soc != cfg.initial_soc:
            raise RuntimeError('候选EV初始化状态异常')
        self._state = _FleetState(cfg, model)
        return self.state_view()

    @staticmethod
    def _validate_model_config(model, cfg):
        for name, expected in {'capacity_mwh': cfg.capacity_mwh, 'max_power_mw': cfg.max_power_mw,
                               'initial_soc': cfg.initial_soc, 'target_soc': cfg.target_soc,
                               'efficiency': cfg.efficiency, 'start_hour': cfg.start_hour,
                               'end_hour': cfg.end_hour}.items():
            if getattr(model, name) != expected:
                raise RuntimeError('候选EV配置意外改变')

    @staticmethod
    def _remaining(cfg, soc):
        return max(0.0, (cfg.target_soc - soc) * cfg.capacity_mwh / cfg.efficiency)

    def _minimum(self, state):
        cfg, model = state.config, state.model
        if not cfg.start_hour <= model.hour < cfg.end_hour:
            return 0.0
        remaining = self._remaining(cfg, model.soc)
        future_capacity = (cfg.end_hour - model.hour - 1) * cfg.max_power_mw
        minimum = max(0.0, remaining - future_capacity)  # MWh / 1h = MW。
        if minimum > cfg.max_power_mw:
            if not _close(minimum, cfg.max_power_mw):
                raise RuntimeError('当前EV状态已不可达，禁止执行')
            minimum = cfg.max_power_mw  # 仅浮点舍入误差，不修改目标或真实不可行配置。
        return minimum

    def state_view(self):
        state = self._active();cfg = state.config
        return {'hour': state.model.hour, 'soc': state.model.soc,
                'capacity_mwh': cfg.capacity_mwh, 'max_power_mw': cfg.max_power_mw,
                'target_soc': cfg.target_soc, 'efficiency': cfg.efficiency,
                'start_hour': cfg.start_hour, 'end_hour': cfg.end_hour,
                'remaining_grid_mwh': self._remaining(cfg, state.model.soc),
                'minimum_power_mw': self._minimum(state),
                'total_grid_mwh': state.total_grid_mwh, 'total_battery_mwh': state.total_battery_mwh,
                'total_forced_mwh': state.total_forced_mwh,
                'target_reached': state.model.soc >= cfg.target_soc - 1e-12}

    def step(self, hour, requested_power_mw):
        state = self._active();cfg, model = state.config, state.model
        if self.status == 'FINISHED':
            raise RuntimeError('24小时已完成，不能重复执行')
        check_hour(hour, model.hour)
        request = _number(requested_power_mw, '请求功率MW', maximum=cfg.max_power_mw)
        minimum = self._minimum(state)
        remaining_before = self._remaining(cfg, model.soc)
        in_window = cfg.start_hour <= hour < cfg.end_hour
        protected = max(request, minimum) if in_window and remaining_before > 0 else 0.0
        candidate = deepcopy(model)
        output = candidate.step(hour, protected)  # 唯一一次EVLoad.step；不先充一份保底电。
        check_hour(output['hour'], hour)
        grid = _number(output['load_mwh'], '实际电网MWh', maximum=cfg.max_power_mw)
        power = _number(output['power_mw'], '实际功率MW', maximum=cfg.max_power_mw)
        battery = _number(output['battery_energy_mwh'], '电池MWh', maximum=cfg.capacity_mwh)
        soc = _number(output['soc'], 'SOC', minimum=cfg.initial_soc, maximum=cfg.target_soc)
        expected_grid = min(protected, cfg.max_power_mw, remaining_before) if in_window else 0.0
        expected_soc = min(cfg.target_soc, model.soc + grid * cfg.efficiency / cfg.capacity_mwh)
        reached = soc >= cfg.target_soc - 1e-12
        if (type(candidate.hour) is not int or candidate.hour != hour + 1 or not _close(grid, expected_grid)
                or not _close(power, grid) or not _close(battery, grid * cfg.efficiency)
                or not _close(soc, expected_soc) or not _close(candidate.soc, soc)
                or type(output['target_reached']) is not bool or output['target_reached'] != reached
                or (not in_window and (grid != 0 or battery != 0 or power != 0))
                or (hour >= cfg.end_hour - 1 and not reached)):
            raise RuntimeError('候选EV时钟/电量/SOC/截止目标校验失败')
        self._validate_model_config(candidate, cfg)
        forced = max(0.0, grid - request)  # 固定1h；实际多执行量，不是另一份电量。
        total_grid = _number(state.total_grid_mwh + grid, '累计电网MWh')
        total_battery = _number(state.total_battery_mwh + battery, '累计电池MWh')
        total_forced = _number(state.total_forced_mwh + forced, '累计强制MWh')
        if not _close(total_battery, (soc - cfg.initial_soc) * cfg.capacity_mwh):
            raise RuntimeError('累计电池电量与SOC不守恒')
        row = {'hour': int(hour), 'requested_power_mw': request, 'minimum_power_mw': minimum,
               'protected_power_mw': protected, 'actual_power_mw': power,
               'grid_energy_mwh': grid, 'battery_energy_mwh': battery, 'soc': soc,
               'remaining_grid_mwh': self._remaining(cfg, soc),
               'forced_charging_mwh': forced, 'target_reached': reached,
               'total_grid_mwh': total_grid, 'total_battery_mwh': total_battery,
               'total_forced_mwh': total_forced}
        new_state = _FleetState(cfg, candidate, total_grid, total_battery, total_forced,
                                state.records + (row.copy(),))
        self._state = new_state  # 全部验证通过后的单引用提交。
        return row.copy()
