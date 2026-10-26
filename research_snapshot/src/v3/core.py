"""绝对时钟、W1窗口、不可撤销批次账本；数量/费用float64。"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import calendar
import hashlib
import numpy as np

TZ = timezone(timedelta(hours=8))


def number(x, lo=0., hi=float('inf')):
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
        raise ValueError('需要有限实数')
    try: v = float(x)
    except (ValueError, TypeError, OverflowError): raise ValueError('需要有限实数') from None
    if not np.isfinite(v) or not lo <= v <= hi: raise ValueError('数值越界')
    return v


def vector(x, n, lo=0., hi=float('inf')):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu' or a.shape != (n,): raise ValueError('数组形状/类型错误')
    a = np.array(a, dtype=np.float64, copy=True)
    if not np.isfinite(a).all() or (a < lo).any() or (a > hi).any(): raise ValueError('数组越界/非有限')
    return a


def _two_ulp_ceiling(x):
    """合同内部float64边界；绝不用于放宽外部比例或全局vector。"""
    with np.errstate(over='ignore'):
        return np.nextafter(np.nextafter(x, np.inf), np.inf)


def contract_coverage(quantity, reference):
    """只读累计比例。兼容旧正参考量上方至多2 ULP，不改仓位/Lot/费用。

    新仓位仍不得超过R；此兼容仅防止已有舍入尾差在合法后续动作中放大。
    R=0时只允许q=0，真正超仓和非有限值必须拒绝。
    """
    r = np.asarray(reference)
    if r.ndim != 1: raise ValueError('合同参考维度错误')
    r = vector(r, r.size)
    q = vector(quantity, r.size)
    if ((r == 0) & (q != 0)).any() or (q > _two_ulp_ceiling(r)).any():
        raise ValueError('合同仓位超出机器精度边界')
    return np.minimum(1., np.divide(q, r, out=np.zeros_like(r), where=r > 0))


def year_axis(year):
    if type(year) is not int or not 1900 <= year <= 2100: raise ValueError('年份错误')
    start = datetime(year, 1, 1, tzinfo=TZ)
    n = (366 if calendar.isleap(year) else 365) * 24
    return start, n


@dataclass(frozen=True)
class Opportunity:
    scope: str
    stage: str
    month: int
    block: int
    start: int
    end: int
    index: int
    decision_at: datetime

    @property
    def published_at(self): return self.decision_at - timedelta(minutes=5)


def opportunities(year):
    start, n = year_axis(year)
    result = []
    def put(scope, stage, m, b, a, z, times):
        for i, when in enumerate(times):
            result.append(Opportunity(scope, stage, m, b, a, z, i, when))
    put('Y', 'Y', 0, 0, 0, n, [start-timedelta(days=d)+timedelta(hours=17,minutes=5) for d in (20,10)])
    for m in range(1,13):
        d = datetime(year,m,1,tzinfo=TZ)
        a = int((d-start).total_seconds()/3600)
        days = calendar.monthrange(year,m)[1]
        z = a+days*24
        put(f'M{m}', 'M', m,0,a,z,[d-timedelta(days=k)+timedelta(hours=17,minutes=5) for k in (3,1)])
        for b,day in enumerate((1,11,21),1):
            bd = datetime(year,m,day,tzinfo=TZ)
            ba = a+(day-1)*24
            bz = a+(10 if b==1 else 20 if b==2 else days)*24
            put(f'I{m}-{b}', 'I',m,b,ba,bz,[bd-timedelta(days=1)+timedelta(hours=h,minutes=5) for h in (18,19)])
    return sorted(result,key=lambda o:o.decision_at)


@dataclass(frozen=True)
class ContractLot:
    trade_id: str
    scope: str
    stage: str
    signed_at: datetime
    start: int
    end: int
    quantities: tuple
    unit_price: float
    quote_id: str
    published_at: datetime
    expires_at: datetime


class PositionLedger:
    """不提供reset/撤销。成交责任独立于费用确认，开始交割小时不再可买。"""
    def __init__(self, year, reference):
        self.year = year
        self.start_at, self.n = year_axis(year)
        r = vector(reference,self.n)
        r.flags.writeable = False
        self._reference = r
        self._quantity = np.zeros(self.n,dtype=np.float64)
        self._fee = np.zeros(self.n,dtype=np.float64)
        self._stage_fee = {s:np.zeros(self.n,dtype=np.float64) for s in 'YMI'}
        self._lots = ()
        self._decided = frozenset()
        self._started_through = -1
        self._settled = -1
        self._last_time = None
        self._consumed = frozenset()
        self._trade_ids = frozenset()
        self.version = 0

    @property
    def lots(self): return self._lots
    @property
    def reference(self): return self._reference.copy()
    @property
    def quantity(self): return self._quantity.copy()
    @property
    def fee(self): return self._fee.copy()
    def decided(self,scope): return scope in self._decided
    def plan(self,a,z): return self._quantity[a:z].copy(),self._fee[a:z].copy()
    def stage_fee(self,t): return {s:float(a[t]) for s,a in self._stage_fee.items()}
    def digest(self):
        return hashlib.sha256(self._quantity.tobytes()+self._fee.tobytes()+str(self.version).encode()).hexdigest()

    def decide(self, op, verb, target, price, *, now, published_at, expires_at, expected_version, trade_id):
        if op not in opportunities(self.year): raise ValueError('未知scope/日期')
        if now != op.decision_at or published_at > now or expires_at < now or published_at.tzinfo is None:
            raise ValueError('报价/决策时刻非法')
        if self._last_time is not None and now < self._last_time: raise RuntimeError('时钟不得倒退')
        parent = 'Y' if op.stage=='M' else f'M{op.month}' if op.stage=='I' else None
        if parent and not self.decided(parent): raise RuntimeError('父stage尚未完成')
        if expected_version != self.version or self.decided(op.scope): raise RuntimeError('过期仓位或重复签约')
        if (op.scope,op.index) in self._consumed: raise RuntimeError('机会已消费')
        if verb not in ('WAIT','COMMIT','SKIP') or (op.index==1 and verb=='WAIT'): raise ValueError('窗口动作非法')
        if verb=='WAIT':
            self._last_time = now
            self._consumed = self._consumed | {(op.scope,op.index)}
            return
        if op.start <= self._started_through: raise RuntimeError('不得购买已开始交割小时')
        if not isinstance(trade_id,str) or not trade_id or trade_id in self._trade_ids: raise ValueError('重复/空trade_id')
        if verb=='SKIP':
            self._decided = self._decided | {op.scope}
            self.version += 1
            self._last_time = now
            self._consumed = self._consumed | {(op.scope,op.index)}
            self._trade_ids = self._trade_ids | {trade_id}
            return
        p = number(price)
        target = vector(target,op.end-op.start,0,1)
        reference = self._reference[op.start:op.end]
        wanted = target * reference
        old = self._quantity[op.start:op.end]
        contract_coverage(old, reference)  # 校验旧仓位；不修剪历史合同。
        # 保留旧版“极小目标回退只作零增量”的兼容性；它绝不减仓，非超仓许可。
        if ((wanted < old-1e-10) & (old > _two_ulp_ceiling(wanted))).any():
            raise ValueError('不得撤销已成交仓位')
        delta = np.maximum(wanted-old,0.)
        candidate_quantity = old+delta
        # old+(wanted-old)可能比R多1 ULP。只下修本次delta，Lot及费用使用同一delta。
        # 两次nextafter足够覆盖这次减法/加法的边界舍入；仍失败则原子拒绝。
        for _ in range(2):
            over = (old <= reference) & (candidate_quantity > reference)
            if not over.any(): break
            delta[over] = np.nextafter(delta[over], 0.)
            candidate_quantity = old+delta
        if ((old <= reference) & (candidate_quantity > reference)).any():
            raise ValueError('新增合同仍超参考容量')
        if ((old > reference) & (delta != 0)).any() or not np.isfinite(candidate_quantity).all():
            raise ValueError('旧边界仓位不得新增电量')
        fees = delta*p
        if not np.isfinite(fees).all(): raise ValueError('合同费用溢出')
        q = self._quantity.copy(); f = self._fee.copy(); sf = self._stage_fee[op.stage].copy()
        q[op.start:op.end] = candidate_quantity  # 不撤销旧量；新增量/费用/Lot一致。
        f[op.start:op.end] += fees
        sf[op.start:op.end] += fees
        if not np.isfinite(f).all(): raise ValueError('累计费用溢出')
        lot = ContractLot(trade_id,op.scope,op.stage,now,op.start,op.end,tuple(delta),p,
                          f'{op.scope}:{op.index}',published_at,expires_at)
        # 候选全部验证成功后提交；无签约Reward第二次收费。
        self._quantity,self._fee = q,f
        self._stage_fee = dict(self._stage_fee,**{op.stage:sf})
        self._lots += (lot,)
        self._decided = self._decided | {op.scope}
        self.version += 1
        self._last_time = now
        self._consumed = self._consumed | {(op.scope,op.index)}
        self._trade_ids = self._trade_ids | {trade_id}

    def start_hour(self,t):
        if type(t) is not int or t!=self._started_through+1 or t!=self._settled+1 or t>=self.n: raise RuntimeError('小时开始顺序错误')
        self._started_through = t

    def commit_hour(self,t):
        if type(t) is not int or t!=self._settled+1 or t!=self._started_through: raise RuntimeError('重复/乱序确认')
        self._settled = t


def settlement(base, industrial, ev_grid, contract_q, contract_fee, spot_q, offer_price, balancing_price):
    b,i,e,q,f,s,ps,pb = [number(x) for x in (base,industrial,ev_grid,contract_q,contract_fee,spot_q,offer_price,balancing_price)]
    d = b+i+e
    balance = max(0.,d-q-s)
    used_l = min(q,d); used_s = min(s,max(0.,d-q))
    cost = f+s*ps+balance*pb
    if not np.isfinite(cost): raise ValueError('结算溢出')
    return dict(base_load_mwh=b,industrial_mwh=i,ev_grid_mwh=e,total_load_mwh=d,
                contract_mwh=q,contract_cost=f,spot_mwh=s,spot_cost=s*ps,
                balance_mwh=balance,balance_cost=balance*pb,unused_contract_mwh=q-used_l,
                unused_spot_mwh=s-used_s,purchase_cost=cost,
                offer_price=ps,balancing_price=pb)
