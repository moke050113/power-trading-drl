"""Read-only fixed-time P4a protocol. No step/settlement/device/Agent methods."""
from dataclasses import dataclass, replace
from datetime import timedelta
import math
from v3.core import PositionLedger, opportunities, contract_coverage
from .schema import HourGrid, UNITS, canonical, digest, instant
from .forecasts import ForecastRepository
from .resource_archive import ResourceArchive
from .resource_forecast import ResourceForecastProvider, ResourceOutlook
from .resource_envelope import task_envelope, independent_envelopes
from .resource_schema import number
from .coordination_schema import (LotView, PositionView, ForecastView, ConstraintView,
                                  TaskView, Observation, MarketToLoad, LoadToMarket)


TARGETS=('base_load','offer_price','balancing_price')


@dataclass(frozen=True)
class ContractQuote:
    scope: str
    opportunity: int
    price: float
    published_at: str
    expires_at: str


@dataclass(frozen=True)
class SpotQuote:
    hour_start: str
    price: float
    capacity_mwh: float
    published_at: str


def _constraint(envelope):
    return ConstraintView(envelope.boundaries,envelope.cumulative_min_mwh,
                          envelope.cumulative_max_mwh,envelope.minimum_first_slot_mw,envelope.feasible)


def _task(task, at, slot, certainty, state=None, completed=False):
    unknown=certainty.startswith('predicted')
    constraint=None
    # A begun UNKNOWN template is a prior, not a known remaining-energy state.
    if not completed and (not unknown or instant(task.start)>=instant(at)):
        constraint=_constraint(task_envelope(task,as_of=at,certainty='predicted_unknown' if unknown else 'known',state=state))
    return TaskView(slot,task.resource,certainty,task.start,task.deadline,task.max_power_mw,
                    state.remaining_grid_mwh if state else task.required_grid_mwh,
                    task.capacity_mwh,task.efficiency,
                    task.initial_soc if task.resource=='EV' and not state else None,
                    state.soc if state and task.resource=='EV' else None,
                    state.observed_at if state else None,constraint)


def _single_session(tasks):
    keys=[(t.pool_id,t.resource,instant(t.start).date()) for t in tasks]
    if len(set(keys))!=len(keys):
        raise ValueError('P4a supports one session per resource pool/day only')


class LedgerReadPort:
    """Capture public V3 state once. Clock is attested by the trusted caller, not Agent."""
    def __init__(self,ledger,*,observed_at):
        if type(ledger) is not PositionLedger:raise ValueError('V3 PositionLedger required')
        self._ledger=ledger;self.observed_at=instant(observed_at)
        self._version=ledger.version;self._digest=ledger.digest()
        self._reference=tuple(float(v) for v in ledger.reference)
        self._quantity=tuple(float(v) for v in ledger.quantity)
        self._fee=tuple(float(v) for v in ledger.fee)
        self._lots=tuple(ledger.lots)
        self._reference_sha=digest(self._reference)
        contract_coverage(self._quantity,self._reference)
        if any(l.signed_at>self.observed_at for l in self._lots):raise ValueError('ledger has later contracts; cannot replay past using live state')
        # Independent Lot-to-hour identity, no signing/settlement and no repricing.
        q=[0.]*ledger.n;fee=[0.]*ledger.n
        for lot in self._lots:
            for i,v in enumerate(lot.quantities,lot.start):
                q[i]+=float(v);fee[i]+=float(v)*lot.unit_price
        if any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10) for a,b in zip(q,self._quantity)) or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8) for a,b in zip(fee,self._fee)):
            raise ValueError('Lot/position accounting mismatch')
        self.assert_unchanged()

    def assert_unchanged(self):
        if (self._ledger.version!=self._version or self._ledger.digest()!=self._digest or
            tuple(self._ledger.lots)!=self._lots or digest(tuple(float(v) for v in self._ledger.reference))!=self._reference_sha):
            raise RuntimeError('source ledger changed after fixed-time capture')

    def view(self,grid):
        self.assert_unchanged()
        if type(grid) is not HourGrid:raise ValueError('HourGrid required')
        a=(grid.start-self._ledger.start_at).total_seconds()/3600
        z=a+len(grid.hours)
        if not a.is_integer() or not 0<=a<z<=self._ledger.n:raise ValueError('position target outside ledger year')
        a,z=int(a),int(z)
        lots=[]
        for l in self._lots:
            if max(a,l.start)>=min(z,l.end):continue
            values=tuple(float(l.quantities[i-l.start]) if l.start<=i<l.end else 0. for i in range(a,z))
            lots.append(LotView(l.stage,l.signed_at.isoformat(),float(l.unit_price),values))
        return PositionView(grid.hours,self._version,self._reference_sha,self._reference[a:z],self._quantity[a:z],self._fee[a:z],
                            tuple(max(0.,r-q) for r,q in zip(self._reference[a:z],self._quantity[a:z])),tuple(lots))


def _forecasts(repository,at,scope,grid):
    rows=tuple(repository.get_published(decision_at=at,scope=scope,target_variable=t,target_interval=grid) for t in TARGETS)
    for t,r in zip(TARGETS,rows):
        if r.target_variable!=t or r.unit!=UNITS[t] or r.grid!=grid:raise ValueError('forecast target/unit/hour mismatch')
    return rows


def _forecast_views(rows,usage):
    return tuple(ForecastView(r.target_variable,r.unit,r.as_of,r.published_at,usage,tuple(r.values)) for r in rows)


class DayAheadMemory:
    """Trusted append-only archive of LEGALLY captured priors, never given to Agent."""
    def __init__(self):self._days={}

    def _capture(self,at,grid,forecasts,outlook):
        rows=self._days.get(grid.hours[0],())
        if rows and instant(at)<instant(rows[-1][0]):raise ValueError('day-ahead capture cannot backdate')
        packet=(at,grid,forecasts,outlook)
        if packet not in rows:self._days[grid.hours[0]]=rows+(packet,)

    def _read(self,at):
        dt=instant(at);day=dt.replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
        rows=[r for r in self._days.get(day,()) if instant(r[0])<=dt]
        if not rows:raise LookupError('no prior legitimately sealed before this delivery day')
        return rows[-1]


class TrustedObservationCoordinator:
    def __init__(self,*,decision_at,ledger,forecasts,resources,resource_provider,memory=None):
        self._at=instant(decision_at).isoformat()
        if type(forecasts) is not ForecastRepository or type(resources) is not ResourceArchive or type(resource_provider) is not ResourceForecastProvider:
            raise ValueError('trusted P1/P3 ports required')
        if resource_provider.archive is not resources:raise ValueError('resource provider/archive mismatch')
        if memory is not None and type(memory) is not DayAheadMemory:raise ValueError('trusted DayAheadMemory required')
        self._positions=LedgerReadPort(ledger,observed_at=self._at)
        self._ledger=ledger;self._forecasts=forecasts;self._resources=resources;self._provider=resource_provider
        self._memory=memory;self._audit=()

    @property
    def audit(self):
        import json
        return tuple(json.loads(r) for r in self._audit)

    def _future(self,scope,grid,outlook):
        rows=_forecasts(self._forecasts,self._at,scope,grid)
        visible=self._provider.get_visible(outlook,decision_at=self._at,scope=scope,target_interval=grid)
        _single_session(visible.known_tasks+visible.predicted_unknown_tasks)
        visible.envelopes()
        return rows,visible

    def _finish(self,obs):
        self._positions.assert_unchanged()
        self._audit+=(canonical({'decision_at':self._at,'level':obs.level,'projection_sha256':obs.content_sha256,
                                 'position_digest':self._ledger.digest()}),)
        return obs

    def upper(self,*,opportunity,quote,outlook):
        op=opportunity
        if op not in opportunities(self._ledger.year) or op.stage not in 'YMI' or op.decision_at!=instant(self._at):
            raise ValueError('wrong macro opportunity/clock')
        parent='Y' if op.stage=='M' else f'M{op.month}' if op.stage=='I' else None
        if self._ledger.decided(op.scope) or (parent and not self._ledger.decided(parent)):
            raise ValueError('scope closed or parent phase not finalized')
        if type(quote) is not ContractQuote or (quote.scope,quote.opportunity)!=(op.scope,op.index):raise ValueError('wrong quote scope/opportunity')
        if instant(quote.published_at)!=op.published_at or instant(quote.expires_at)!=op.decision_at:raise ValueError('quote release/expiry mismatch')
        price=number(quote.price)
        grid=HourGrid.between((self._ledger.start_at+timedelta(hours=op.start)).isoformat(),(self._ledger.start_at+timedelta(hours=op.end)).isoformat())
        rows,r=self._future(op.stage,grid,outlook)
        known=tuple(_task(t,self._at,i,'known_planned') for i,t in enumerate(r.known_tasks))
        unknown=tuple(_task(t,self._at,i,'predicted_unknown') for i,t in enumerate(r.predicted_unknown_tasks))
        return self._finish(Observation(self._at,op.stage,op.index,price,None,None,self._positions.view(grid),
                                         _forecast_views(rows,'currently_published'),known,unknown,'currently_published_future_outlook'))

    def day_ahead(self,*,grid,outlook):
        if self._memory is None:raise ValueError('DayAheadMemory required to seal a delivery day')
        grid.validate_scope('DA')
        rows,r=self._future('DA',grid,outlook)
        known=tuple(_task(t,self._at,i,'known_planned') for i,t in enumerate(r.known_tasks))
        unknown=tuple(_task(t,self._at,i,'predicted_unknown') for i,t in enumerate(r.predicted_unknown_tasks))
        obs=self._finish(Observation(self._at,'DA',None,None,None,None,self._positions.view(grid),
                                    _forecast_views(rows,'currently_published'),known,unknown,'currently_published_future_outlook'))
        self._memory._capture(self._at,grid,rows,r)
        return obs

    def hour(self,*,spot_quote):
        at=instant(self._at)
        if at.minute or at.second or at.microsecond or self._memory is None:raise ValueError('hour start and sealed prior required')
        if type(spot_quote) is not SpotQuote or instant(spot_quote.hour_start)!=at or instant(spot_quote.published_at)!=at-timedelta(minutes=5):raise ValueError('spot quote not legally released for current hour')
        price,capacity=number(spot_quote.price),number(spot_quote.capacity_mwh)
        _,grid,rows,prior=self._memory._read(self._at)
        current=[r for r in self._resources.snapshot(self._at,source=self._provider.config.source)
                 if r.record_type!='history_day' and instant(r.day)==grid.start]
        all_current=tuple(r.tasks[0] for r in current)
        _single_session(all_current)
        bykey={(t.pool_id,t.resource,instant(t.start).date()):r for t,r in zip(all_current,current)}
        known=[];active=[]
        for i,r in enumerate(current):
            t=r.tasks[0]
            if r.state is not None:
                elapsed=(instant(r.state.observed_at)-instant(t.start)).total_seconds()/3600
                delivered=t.required_grid_mwh-r.state.remaining_grid_mwh
                if delivered>elapsed*t.max_power_mw+1e-10:
                    raise ValueError('confirmed state implies impossible energy before observation time')
            if r.record_type=='booking':
                if instant(t.start)<=at:raise ValueError('scheduled arrival requires confirmation, not a booking-as-state')
                known.append(_task(t,self._at,i,'known_planned'));active.append(t)
            elif r.state.remaining_grid_mwh==0:
                known.append(_task(t,self._at,i,'confirmed_completed',state=r.state,completed=True))
            else:
                known.append(_task(t,self._at,i,'confirmed_active',state=r.state));active.append(t)
        unknown=[]
        remaining_templates=[]
        for t in prior.predicted_unknown_tasks:
            key=(t.pool_id,t.resource,instant(t.start).date());actual=bykey.get(key)
            if actual is not None:
                if actual.tasks[0].origin!='unannounced':raise ValueError('booked and unknown demand cannot share a pool/day in P4a')
                continue  # Actual unannounced cohort REPLACES prior; never double-count.
            label='predicted_unknown' if instant(t.start)>=at else 'predicted_unknown_prior_started_no_state'
            unknown.append(_task(t,self._at,len(unknown),label));remaining_templates.append(t)
        # P3's conflict rule, without assuming any execution since earlier hours.
        all_plans=active+remaining_templates
        pools={}
        for t in all_plans:
            pools.setdefault(t.pool_id,[]).append(t)
        for group in pools.values():
            group=sorted(group,key=lambda t:t.start)
            if any(instant(b.start)<instant(a.deadline) for a,b in zip(group,group[1:])):raise ValueError('shared pool overlap')
        return self._finish(Observation(self._at,'H',None,None,price,capacity,self._positions.view(grid),
                                        _forecast_views(rows,'archived_day_ahead_forecast_not_current_state'),
                                        tuple(known),tuple(unknown),'archived_prior_reconciled_with_fresh_actual_state'))


def market_to_load(observation):
    if type(observation) is not Observation:raise ValueError('value-only observation required')
    base=next(f.values for f in observation.forecasts if f.target=='base_load')
    q=observation.position.contract_mwh
    forward=[i for i,h in enumerate(observation.position.hours) if instant(h)>=instant(observation.decision_at)]
    known=math.fsum(t.grid_demand_mwh for t in observation.known_tasks)
    # Begun, unobserved templates are NOT a current remaining-energy estimate.
    unknown=math.fsum(t.grid_demand_mwh for t in observation.unknown_demand_forecast if t.certainty=='predicted_unknown')
    return MarketToLoad(observation.decision_at,observation.position,
                        tuple(observation.position.hours[i] for i in forward),tuple(base[i]-q[i] for i in forward),known,unknown,
                        math.fsum(base[i] for i in forward)+known+unknown-math.fsum(q[i] for i in forward),
                        tuple(instant(h)>instant(observation.decision_at) and c>0 for h,c in zip(observation.position.hours,observation.position.remaining_c1_capacity_mwh)),
                        sum(t.certainty=='predicted_unknown_prior_started_no_state' for t in observation.unknown_demand_forecast))


def load_to_market(observation):
    if type(observation) is not Observation:raise ValueError('value-only observation required')
    at=instant(observation.decision_at)
    return LoadToMarket(observation.decision_at,observation.position.version,
                         observation.position.reference_sha256,observation.position.content_sha256,
                         digest({'known':tuple(t.to_dict() for t in observation.known_tasks),'unknown':tuple(t.to_dict() for t in observation.unknown_demand_forecast)}),
                         tuple(h for h in observation.position.hours if instant(h)>at),
                         tuple(t for t in observation.known_tasks if t.certainty!='confirmed_completed' and instant(t.deadline)>at),
                         tuple(t for t in observation.unknown_demand_forecast if t.certainty=='predicted_unknown' and instant(t.window_start)>at))
