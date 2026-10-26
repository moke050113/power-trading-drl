"""V2.0方案A的24小时结算内核；独立于V1、CSV、Gym与训练。

教学时序为D-1 18:00发布预测/已知合同价，18:05一次锁定。
这里的reset装入受信任调用方已校验时间来源的数组，不做墙钟等待。
时间戳、日期划分及来源验证由未来独立场景适配器承担。
"""
import numpy as np


def _numeric(values, shape, name, maximum):
    """只接受实数数值、固定形状；拷贝为float64，不裁剪。"""
    try:
        raw = np.asarray(values)
        if raw.shape != shape or raw.dtype.kind not in 'iuf':
            raise ValueError
        with np.errstate(over='raise', invalid='raise'):
            result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        raise ValueError(f'{name}必须是形状{shape}的实数数值') from None
    if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > maximum):
        raise ValueError(f'{name}必须为0到{maximum}之间的有限数值；负价不兼容，禁止裁剪')
    result.flags.writeable = False
    return result


class MultiscaleTradingDay:
    """一次锁定、24次交割；调用方是受信任内核驱动器，不是Agent。

    reset接收forecast_load、forecast_spot_price、realized_base_load、
    realized_spot_price四条(24,)曲线及标量contract_price。
    电量MWh、价格元/MWh、成本元；只实现非负教学价格方案A。
    内部实现数组不通过公开日前投影返回，不提供修改计划的接口。
    """

    HORIZON = 24
    SETTLEMENT_MODEL = 'A'

    def __init__(self):
        self._status = 'UNINITIALIZED'
        self._hour = 0
        self._episode_cost = np.float64(0)
        self._contract_plan = None
        self._contract_ratio = None
        self._records = []

    @property
    def status(self):
        return self._status

    @property
    def hour(self):
        return self._hour

    @property
    def episode_cost(self):
        return self._episode_cost

    @property
    def contract_ratio(self):
        if self._contract_ratio is None:
            raise RuntimeError('合同尚未锁定')
        return self._contract_ratio

    @property
    def contract_plan(self):
        if self._contract_plan is None:
            raise RuntimeError('合同尚未锁定')
        return self._contract_plan.copy()

    @property
    def hourly_records(self):
        if self._status != 'FINISHED':
            raise RuntimeError('全天明细只能在24小时结算结束后读取')
        return [record.copy() for record in self._records]

    def reset(self, *, forecast_load, forecast_spot_price, realized_base_load,
              realized_spot_price, contract_price=400.0):
        """先验证所有输入再替换单日状态；锁定/结算期间禁止reset。"""
        if self._status == 'LOCKED':
            raise RuntimeError('合同已锁定，请先完成24小时结算；不能reset绕过重签限制')
        # 所有验证完成前不修改旧场景、计划、时钟或成本。
        forecasts = _numeric(forecast_load, (24,), '预测负荷', 1000)
        forecast_prices = _numeric(forecast_spot_price, (24,), '预测现货价', 2000)
        loads = _numeric(realized_base_load, (24,), '实现基础负荷', 1000)
        prices = _numeric(realized_spot_price, (24,), '实现现货价', 2000)
        price = _numeric(contract_price, (), '合同价格', 2000)[()]
        self._forecast_load = forecasts
        self._forecast_spot_price = forecast_prices
        self._realized_base_load = loads
        self._realized_spot_price = prices
        self._contract_price = price
        self._contract_plan = None
        self._contract_ratio = None
        self._records = []
        self._hour = 0
        self._episode_cost = np.float64(0)
        self._status = 'UNLOCKED'
        return self.decision_view()

    def decision_view(self):
        """独立副本，仅日前已知值；没有实现曲线、场景ID或日志。"""
        if self._status == 'UNINITIALIZED':
            raise RuntimeError('请先reset装入一日场景')
        return {'forecast_load': self._forecast_load.copy(),
                'forecast_spot_price': self._forecast_spot_price.copy(),
                'contract_price': np.array([self._contract_price], dtype=np.float64)}

    def lock_contract(self, action):
        """action必须为[alpha]，alpha∈[0,1]；锁定时不收费、不推进小时。"""
        if self._status != 'UNLOCKED':
            raise RuntimeError('只能在reset后、未锁定时签约一次')
        alpha = _numeric(action, (1,), '合同覆盖比例alpha', 1)[0]
        plan = alpha * self._forecast_load
        plan.flags.writeable = False
        self._contract_ratio = alpha
        self._contract_plan = plan
        self._status = 'LOCKED'
        return self.contract_plan

    def settle_next_hour(self):
        """受信任调用方推进一次交割，返回该小时明细；不接收新合同。"""
        if self._status != 'LOCKED':
            raise RuntimeError('请先reset并锁定合同；结束后需reset，不能重复结算')
        h = self._hour
        quantity = self._contract_plan[h]
        load = self._realized_base_load[h]
        price = self._realized_spot_price[h]
        used = np.minimum(quantity, load)
        spot_buy = np.maximum(load - quantity, np.float64(0))
        surplus = np.maximum(quantity - load, np.float64(0))
        contract_cost = quantity * self._contract_price
        spot_cost = spot_buy * price
        total_cost = contract_cost + spot_cost
        cumulative = self._episode_cost + total_cost
        record = {'hour': h, 'settlement_model': self.SETTLEMENT_MODEL,
                  'forecast_load': self._forecast_load[h],
                  'forecast_spot_price': self._forecast_spot_price[h],
                  'realized_base_load': load, 'realized_spot_price': price,
                  'contract_price': self._contract_price, 'contract_quantity': quantity,
                  'contract_used': used, 'spot_buy': spot_buy, 'contract_surplus': surplus,
                  'contract_cost': contract_cost, 'spot_cost': spot_cost,
                  'total_cost': total_cost, 'episode_cost': cumulative,
                  'hour_reward': -total_cost}
        self._records.append(record)
        self._episode_cost = cumulative
        self._hour += 1
        if self._hour == self.HORIZON:
            self._status = 'FINISHED'
        return record.copy()
