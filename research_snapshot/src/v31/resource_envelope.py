"""Task-coupled energy feasibility, not a schedule or a device executor."""
from dataclasses import dataclass
from datetime import timedelta
import math
from .schema import ContentRecord,instant
from .resource_schema import TaskDemand,ResourceState,number


@dataclass(frozen=True)
class TaskEnvelope(ContentRecord):
    task: TaskDemand
    certainty: str
    as_of: str
    remaining_grid_mwh: float
    first_slot: str
    boundaries: tuple
    cumulative_min_mwh: tuple
    cumulative_max_mwh: tuple
    minimum_first_slot_mw: float
    feasible: bool
    feasibility_scope: str = 'independent_task_only'

    def accepts(self,grid_energy_mwh):
        """Read-only validation of a caller's hypothetical plan, no state commit."""
        v=tuple(number(x) for x in grid_energy_mwh)
        if not self.feasible or len(v)!=len(self.boundaries)-1:return False
        if any(x>self.task.max_power_mw for x in v):return False
        total=0.
        for k,x in enumerate(v,1):
            total+=x
            if total<self.cumulative_min_mwh[k]-1e-10 or total>self.cumulative_max_mwh[k]+1e-10:return False
        return math.isclose(total,self.remaining_grid_mwh,rel_tol=1e-12,abs_tol=1e-10)


def task_envelope(task,*,as_of,certainty='known',state=None):
    if type(task) is not TaskDemand or certainty not in ('known','predicted_unknown'):
        raise ValueError('task and knowledge type required')
    at=instant(as_of);start=instant(task.start);end=instant(task.deadline)
    energy=task.required_grid_mwh
    if state is not None:
        if type(state) is not ResourceState or certainty!='known':raise ValueError('only confirmed actual state can restrict a known task')
        state.validate_task(task)
        if instant(state.observed_at)!=at:raise ValueError('current state must be observed at the requested as_of')
        energy=state.remaining_grid_mwh
    elif at>start:
        raise ValueError('started task requires fresh confirmed state; never assume zero execution')
    ceil=at.replace(minute=0,second=0,microsecond=0)
    if ceil<at:ceil+=timedelta(hours=1)
    first=min(max(start,ceil),end)
    n=int((end-first).total_seconds()/3600)
    power=task.max_power_mw
    feasible=energy<=n*power
    lower=tuple(max(0.,energy-(n-k)*power) for k in range(n+1))
    upper=tuple(min(energy,k*power) for k in range(n+1))
    boundaries=tuple((first+timedelta(hours=k)).isoformat() for k in range(n+1))
    return TaskEnvelope(task,certainty,at.isoformat(),energy,first.isoformat(),boundaries,lower,upper,
                        max(0.,energy-(n-1)*power) if n else 0.,feasible)


def independent_envelopes(tasks,*,as_of,certainty):
    """Conservatively refuse shared overlapping pools, rather than invent capacity."""
    tasks=tuple(tasks)
    pools={}
    for task in tasks:pools.setdefault(task.pool_id,[]).append(task)
    for group in pools.values():
        group=sorted(group,key=lambda t:instant(t.start))
        for a,b in zip(group,group[1:]):
            if instant(b.start)<instant(a.deadline):
                raise ValueError('overlapping shared pool needs a separately approved joint capacity model')
    return tuple(task_envelope(t,as_of=as_of,certainty=certainty) for t in tasks)


def current_resource_envelopes(archive,*,as_of,source):
    """Known arrived/state projection only; do not infer unobserved execution."""
    rows=archive.snapshot(as_of,source=source)
    return tuple(task_envelope(r.tasks[0],as_of=as_of,certainty='known',state=r.state)
                 for r in rows if r.record_type in ('arrived','state') and instant(r.tasks[0].deadline)>=instant(as_of))
