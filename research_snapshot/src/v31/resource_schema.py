"""Structured exogenous resource demands, independent of frozen P1 scalar schema."""
from dataclasses import dataclass
from datetime import timedelta
import math
from .schema import ContentRecord, HourGrid, instant, text, sha, stamp


def number(v, minimum=0., maximum=math.inf):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError('finite scalar required')
    v=float(v)
    if not math.isfinite(v) or not minimum <= v <= maximum:
        raise ValueError('resource value outside physical domain')
    return v


@dataclass(frozen=True)
class TaskDemand(ContentRecord):
    task_id: str
    resource: str
    pool_id: str
    origin: str
    start: str
    deadline: str
    max_power_mw: float
    energy_mwh: float = 0.
    capacity_mwh: float = 0.
    initial_soc: float = 0.
    target_soc: float = 0.
    efficiency: float = 1.

    def __post_init__(self):
        for v in (self.task_id,self.pool_id):text(v)
        if self.resource not in ('EV','industrial') or self.origin not in ('booked','unannounced'):
            raise ValueError('unknown resource/origin')
        grid=HourGrid.between(self.start,self.deadline)
        if grid.end > grid.start.replace(hour=0)+timedelta(days=1):
            raise ValueError('P3 same-day tasks only; no cross-day SOC')
        object.__setattr__(self,'start',grid.start.isoformat());object.__setattr__(self,'deadline',grid.end.isoformat())
        for k in ('max_power_mw','energy_mwh','capacity_mwh','initial_soc','target_soc','efficiency'):
            limit=1. if k in ('initial_soc','target_soc','efficiency') else math.inf
            object.__setattr__(self,k,number(getattr(self,k),maximum=limit))
        if self.resource=='EV':
            if self.energy_mwh!=0 or self.capacity_mwh<=0 or self.efficiency<=0 or self.target_soc<self.initial_soc:
                raise ValueError('EV requires capacity/SOC/efficiency, not duplicate energy')
        elif (self.capacity_mwh or self.initial_soc or self.target_soc or self.efficiency!=1):
            raise ValueError('industrial energy has no SOC/capacity/efficiency conversion')
        number(self.required_grid_mwh)

    @property
    def required_grid_mwh(self):
        return self.capacity_mwh*(self.target_soc-self.initial_soc)/self.efficiency if self.resource=='EV' else self.energy_mwh


def ev_fleet_demand(*,task_id,pool_id,origin,start,deadline,vehicle_count=1000,
                    capacity_kwh=60.,power_kw=7.,initial_soc=.2,target_soc=.7,efficiency=.9):
    """Unit conversion only, not 1000 independent agents or a device executor."""
    if type(vehicle_count) is not int or vehicle_count<1:raise ValueError('positive integer fleet size required')
    capacity=number(capacity_kwh)*vehicle_count/1000
    power=number(power_kw)*vehicle_count/1000
    return TaskDemand(task_id,'EV',pool_id,origin,start,deadline,power,
                      capacity_mwh=capacity,initial_soc=initial_soc,target_soc=target_soc,efficiency=efficiency)


@dataclass(frozen=True)
class ResourceState(ContentRecord):
    observed_at: str
    soc: float = 0.
    remaining_grid_mwh: float = 0.

    def __post_init__(self):
        object.__setattr__(self,'observed_at',stamp(self.observed_at))
        object.__setattr__(self,'soc',number(self.soc,maximum=1))
        object.__setattr__(self,'remaining_grid_mwh',number(self.remaining_grid_mwh))

    def validate_task(self,task):
        t=instant(self.observed_at)
        if not instant(task.start)<=t<=instant(task.deadline):raise ValueError('state outside task window')
        if task.resource=='EV':
            expected=task.capacity_mwh*(task.target_soc-self.soc)/task.efficiency
            if not task.initial_soc<=self.soc<=task.target_soc or not math.isclose(expected,self.remaining_grid_mwh,rel_tol=1e-12,abs_tol=1e-10):
                raise ValueError('SOC/remaining grid energy mismatch')
        elif self.soc!=0 or self.remaining_grid_mwh>task.energy_mwh:
            raise ValueError('industrial remaining exceeds declared task')


@dataclass(frozen=True)
class ResourceRecord(ContentRecord):
    record_id: str
    series_id: str
    revision: int
    record_type: str
    day: str
    tasks: tuple
    published_at: str
    available_at: str
    source: str
    source_sha256: str
    state: ResourceState | None = None
    supersedes: str | None = None
    demand_basis: str = 'declared_requirement_not_dispatch'
    schema_version: int = 1

    def __post_init__(self):
        for x in (self.record_id,self.series_id,self.source):text(x)
        sha(self.source_sha256)
        if type(self.schema_version) is not int or self.schema_version!=1 or type(self.revision) is not int or self.revision<0:
            raise ValueError('resource schema/revision invalid')
        if self.record_type not in ('history_day','booking','arrived','state') or self.demand_basis!='declared_requirement_not_dispatch':
            raise ValueError('dispatch/unknown records cannot be demand labels')
        d=instant(self.day)
        if d.hour or d.minute or d.second or d.microsecond:raise ValueError('explicit midnight day required')
        object.__setattr__(self,'day',d.isoformat())
        tasks=tuple(self.tasks)
        if any(type(t) is not TaskDemand for t in tasks) or len({t.task_id for t in tasks})!=len(tasks):
            raise ValueError('invalid/duplicate tasks')
        if any(instant(t.start).date()!=d.date() for t in tasks):raise ValueError('task belongs to another day')
        object.__setattr__(self,'tasks',tasks)
        p,a=instant(self.published_at),instant(self.available_at)
        if p>a:raise ValueError('availability before publication')
        if self.record_type=='history_day':
            if p<d+timedelta(days=1) or self.state is not None:raise ValueError('history labels require complete released day, no dispatch state')
        else:
            if len(tasks)!=1:raise ValueError('one lifecycle task per known series')
            if self.record_type=='booking':
                if self.state is not None or tasks[0].origin!='booked' or p>instant(tasks[0].start):
                    raise ValueError('future public bookings only')
            else:
                if type(self.state) is not ResourceState:raise ValueError('confirmed state required')
                self.state.validate_task(tasks[0])
                if p<instant(self.state.observed_at):raise ValueError('state not observed yet')
                if self.record_type=='arrived' and instant(self.state.observed_at)!=instant(tasks[0].start):
                    raise ValueError('arrival state must be at actual window start')
        if self.supersedes is not None:text(self.supersedes)
        object.__setattr__(self,'published_at',p.isoformat());object.__setattr__(self,'available_at',a.isoformat())

    @classmethod
    def from_dict(cls,data):
        d=dict(data);d['tasks']=tuple(TaskDemand(**v) for v in d['tasks'])
        if d['state'] is not None:d['state']=ResourceState(**d['state'])
        return cls(**d)
