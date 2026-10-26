"""Continuous annual public-event driver around the unchanged V3 trusted core."""
from copy import copy
from datetime import timedelta
import math
import numpy as np
from v3.core import vector
from v3.coordinator import YearCoordinator
from v3.data import PublishedWorld
from v3.devices import DailyDevices
from v31.schema import HourGrid, digest
from v31.coordination import DayAheadMemory, TrustedObservationCoordinator
from v31.resource_forecast import ResourceForecastConfig,ResourceForecastProvider
from .lite_data import SOURCE


class HistoricalWorld(PublishedWorld):
    def __init__(self,store):
        super().__init__(store,store.key);self.context=store.context

    def macro(self,op,at):
        quote=super().macro(op,at);tag='op:'+str(self.ops.index(op))
        if tag not in self.context.records:raise RuntimeError('forecast publication event not executed')
        rows=self.context.records[tag]
        if any(r.published_at>at.isoformat() for r in rows):raise RuntimeError('unpublished prediction')
        quote['forecast_load']=float(np.mean(rows[0].values));quote['forecast_price']=float(np.mean(rows[2].values))
        return quote

    def day_forecast(self,t,at):
        day=t//24
        if day not in self.context.sealed:raise RuntimeError('no legally sealed DA prior')
        rows=self.context.records['day:'+str(day)]
        if at<self.start+timedelta(hours=day*24):raise RuntimeError('hour projection before delivery day')
        return np.array(rows[0].values,dtype=np.float64),np.array(rows[2].values,dtype=np.float64)


class LiteAnnualDriver:
    """Agent receives arrays only. Owner holds this driver/store/archive, never Agent.

    All actual contracts, device transitions and costs go through YearCoordinator.
    Public now is driven forward; t/realization/ledger private clocks are NOT set.
    """
    def __init__(self,store,*,on_hour=None):
        self.store=store;self.context=store.context;self.c=YearCoordinator(store,store.key,on_hour=on_hour)
        self.c.world=HistoricalWorld(store)  # before any action; same arrays/R/year
        self.memory=DayAheadMemory();self.events=[];self.phase=None;self.aborted=False;self.last_token=None
        self._queue=[]
        for j,op in enumerate(self.c.world.ops):self._queue.append((op.published_at,1,'macro_publish',j))
        for day in range(store.n//24):
            pub=store.start+timedelta(days=day-1,hours=18)
            self._queue.extend([(pub,1,'DA_publish',day),(pub+timedelta(minutes=5),3,'DA_seal',day)])
        self._queue.sort();self._cursor=0
        self.c.now=self._queue[0][0] # initial public event time, before all actions
        self.macro_receipts=[];self.daily_receipts=[]

    def _guard(self):
        if self.aborted:raise RuntimeError('aborted; preserve pending and obligations, no reset/retry')

    def _token(self):
        return (self.c.t,self.c.now,self.c.ledger.version,self.c.ledger.digest(),tuple(sorted(self._current_devices().state().items())))

    def _current_devices(self):
        # Original coordinator uses exactly this read-only next-session preview.
        return DailyDevices(self.c.devices.limited) if self.c.devices.hour==24 else self.c.devices

    def upper_observation(self):
        self._guard()
        if self.phase!='upper':raise RuntimeError('not at an upper action')
        obs=self.c.upper_view();obs['device_state'][:]=0 # no future/session-initial actual SOC in upper view
        op=self.c.next_op();outlook=self.context.outlooks['op:'+str(self.c.world.ops.index(op))]
        days=len(outlook.grid.hours)/24
        obs['resource_outlook']=np.array([sum(t.required_grid_mwh for t in outlook.known_tasks if t.resource==r)/days
            for r in ('EV','industrial')]+[sum(t.required_grid_mwh for t in outlook.predicted_unknown_tasks if t.resource==r)/days
            for r in ('EV','industrial')],np.float32)
        self.last_token=self._token()
        return {k:v.copy() for k,v in obs.items()}

    def lower_observation(self):
        self._guard()
        if self.phase!='lower':raise RuntimeError('not at an hourly action')
        obs=self.c.observation();h=self.c.t%24;d=obs['devices']
        obs['actual_state_mask']=np.array([h>=d[3],h>=d[7]],np.float32)
        if h<d[3]:d[0]=0 # planned SOC .2 is a declared plan, not actual pre-arrival SOC
        obs['declared_initial_soc']=np.array([.2],np.float32)
        obs['forecast_offer_price']=np.array(self.context.records['day:'+str(self.c.t//24)][1].values,np.float32)
        self.last_token=self._token()
        return {k:v.copy() for k,v in obs.items()}

    def submit_upper(self,verb,u):
        self._guard()
        if self.phase!='upper' or self.last_token!=self._token():raise RuntimeError('stale upper state')
        vector(u,12,0,1);op=self.c.next_op();before=self.c.ledger.quantity
        self.c.submit_macro(verb,u)
        delta=self.c.ledger.quantity-before
        self.macro_receipts.append(dict(self.c.macros[-1],positive_mwh=float(delta.sum()),
            liability_cny=float(self.c.ledger.fee.sum()),completed_hour=self.c.t,pending=self.c.pending is not None))
        self.phase=None;self.last_token=None

    def submit_lower(self,action):
        self._guard()
        if self.phase!='lower' or self.last_token!=self._token():raise RuntimeError('stale hourly state')
        x=vector(action,3,0,1)
        try:self.c.prepare_hour(x)
        except Exception as exc:
            self.aborted=True;self.events.append(dict(event='H_START_FAILED',t=self.c.t,error=repr(exc)));raise
        self.phase=None;self.last_token=None

    def _publication(self,event,index):
        at=self.c.now
        if event=='macro_publish':
            op=self.c.world.ops[index]
            grid=HourGrid.between((self.store.start+timedelta(hours=op.start)).isoformat(),(self.store.start+timedelta(hours=op.end)).isoformat())
            self.context.publish(op.stage,grid,at,'op:'+str(index),self.c.hours)
        elif event=='DA_publish':
            day=self.store.start+timedelta(days=index);grid=HourGrid.between(day.isoformat(),(day+timedelta(days=1)).isoformat())
            self.context.publish('DA',grid,at,'day:'+str(index),self.c.hours)
        else:
            tag='day:'+str(index);outlook=self.context.outlooks[tag]
            rows=self.context.records[tag]
            stamp=digest({'forecasts':tuple(r.content_sha256 for r in rows),'outlook':outlook.content_sha256})
            if self.context.replay:
                # Same immutable source values validated by P1/P2/P3/P4a on first
                # chronological pass. Never cache a previous strategy's position.
                if self.context.validation_stamps.get(tag)!=stamp:raise RuntimeError('forecast cache hash changed')
                if not all(r.published_at<=at.isoformat()<r.valid_until for r in rows):raise RuntimeError('cache time boundary')
                self.context.sealed[index]=stamp
                self.events.append(dict(event=event,at=at.isoformat(),index=index,source_stamp=stamp,replay=True))
                return
            resources,rp=self.context.resource_ports[tag]
            projection=TrustedObservationCoordinator(decision_at=at.isoformat(),ledger=self.c.ledger,
                forecasts=self.context.repo,resources=resources,resource_provider=rp,memory=self.memory)
            obs=projection.day_ahead(grid=outlook.grid,outlook=outlook)
            self.context.sealed[index]=obs.content_sha256
            self.context.validation_stamps[tag]=stamp
        self.events.append(dict(event=event,at=at.isoformat(),index=index))

    def advance(self,*,stop='upper',upper=None,lower=None):
        """Pause at chosen layer; fixed partner handles the other. No time skipping."""
        self._guard()
        if self.phase is not None:raise RuntimeError('submit pending policy decision before advancing')
        while self.c.t<self.store.n:
            op=self.c.next_op()
            candidates=[]
            if self.c.pending is not None:
                candidates.append((self.store.start+timedelta(hours=self.c.t+1),0,'H_END',0))
            else:candidates.append((self.store.start+timedelta(hours=self.c.t),4,'H_START',0))
            if op is not None:candidates.append((op.decision_at,2,'MACRO',0))
            if self._cursor<len(self._queue):candidates.append(self._queue[self._cursor])
            at,priority,event,index=min(candidates)
            if at<self.c.now:raise RuntimeError('event clock cannot go backwards')
            self.c.now=at
            if event=='MACRO':
                self.phase='upper';obs=self.upper_observation()
                if stop=='upper':return obs
                if upper is None:raise RuntimeError('fixed upper partner required')
                verb,u=upper(obs);self.submit_upper(verb,u)
            elif event=='H_START':
                self.phase='lower';obs=self.lower_observation()
                if stop=='lower':return obs
                if lower is None:raise RuntimeError('fixed lower partner required')
                self.submit_lower(lower(obs))
            elif event=='H_END':
                t=self.c.t
                try:row=self.c.settle_pending()
                except Exception as exc:
                    self.aborted=True;self.events.append(dict(event='H_END_FAILED',t=t,
                        committed=self.c.t!=t,pending=self.c.pending is not None,error=repr(exc)));raise
                if row['hour']==23:
                    self.daily_receipts.append(dict(day=t//24,ev_mwh=self.c.devices.ev_total,industrial_mwh=self.c.devices.ind_total,
                        ev_soc=self.c.devices.ev.soc,ev_forced=self.c.devices.ev_forced,industrial_forced=self.c.devices.ind_forced))
            else:
                self._publication(event,index);self._cursor+=1
        return None

    def run(self,upper,lower):
        while (obs:=self.advance(stop='upper',lower=lower)) is not None:
            verb,u=upper(obs);self.submit_upper(verb,u)
        return self.c.total_purchase,self.c.total_flex
