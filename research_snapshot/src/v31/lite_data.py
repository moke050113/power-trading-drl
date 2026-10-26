"""Independent Lite teaching worlds. This module NEVER materializes a test world."""
from datetime import timedelta
from copy import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from v3.core import opportunities, year_axis
from .schema import HourGrid, HistoryRecord, UNITS, digest, instant
from .history import HistoryArchive
from .forecasts import ForecastRepository
from .prediction import ForecastProvider, BaselineConfig
from .resource_archive import ResourceArchive
from .resource_forecast import ResourceForecastProvider, ResourceForecastConfig
from .resource_schema import ResourceRecord, TaskDemand, ev_fleet_demand

VERSION='synthetic_history_multiscale_v31_lite_v1'
SOURCE=VERSION+':declared_history'
GROUPS=('G00','G01','G10','G11')
MASTER=20261017


def seed_for(family,day,channel):
    return int.from_bytes(hashlib.sha256(f'{MASTER}:{family}:{day}:{channel}'.encode()).digest()[:8],'little')


def day_values(family,group,day,fixture=False):
    h=np.arange(24,dtype=np.float64)
    if fixture:
        return {'base_load':97+h%4,'offer_price':450+2*h,'balancing_price':550-5*h}
    season=np.cos(2*np.pi*(day.timetuple().tm_yday-1)/365.25)
    daily=np.sin(2*np.pi*(h-8)/24)
    rng=np.random.default_rng(seed_for(family,day.date(),'realized'))
    # Bounded synthetic innovations are a generative assumption, not clipping data.
    return {'base_load':100+10*season+12*daily+rng.uniform(-3,3,24),
            'offer_price':500+50*season+55*daily+np.random.default_rng(seed_for(family,day.date(),'public_offer')).uniform(-12,12,24),
            'balancing_price':550+60*season+50*daily+rng.uniform(-120 if group[1]=='1' else -25,120 if group[1]=='1' else 25,24)}


def tasks(day,limited):
    return (ev_fleet_demand(task_id=f'EV:{day.date()}',pool_id='fixed_EV',origin='booked',
        start=(day+timedelta(hours=19 if limited else 18)).isoformat(),deadline=(day+timedelta(days=1)).isoformat()),
        TaskDemand(f'IND:{day.date()}','industrial','fixed_IND','booked',
          (day+timedelta(hours=12 if limited else 8)).isoformat(),(day+timedelta(hours=16)).isoformat(),4.,12.))


class HistoryContext:
    """Causal publication manager. Completed simulation days come from H_END rows."""
    def __init__(self,store):
        self.store=store;self.history=HistoryArchive();self.repo=ForecastRepository(self.history)
        self.resources=ResourceArchive();self.records={};self.outlooks={};self.sealed={};self.origins=[]
        self.hist_index={};self.resource_history_index={};self.booking_index={};self.resource_ports={}
        self.validation_stamps={};self.replay=False
        first=opportunities(store.year)[0].published_at
        self.next_day=first.replace(hour=0,minute=0)-timedelta(days=28)
        self.fixed_train_end='2037-01-01T00:00:00+08:00'
        self.advance_history(first,())
        # A fixed daily public plan, not a prediction of unknown arrivals.
        bookings=[]
        for k in range(store.n//24):
            day=store.start+timedelta(days=k)
            self.booking_index[day.date()]=[]
            for task in tasks(day,store.limited):
                rid='booking:'+task.task_id
                record=ResourceRecord(rid,rid,0,'booking',day.isoformat(),(task,),
                    first.isoformat(),first.isoformat(),SOURCE,digest(task.to_dict()))
                bookings.append(record);self.booking_index[day.date()].append(record)
        self.resources.append_many(bookings)

    def advance_history(self,at,rows):
        end=at.replace(hour=0,minute=0,second=0,microsecond=0)
        while self.next_day<end:
            day=self.next_day;grid=HourGrid.between(day.isoformat(),(day+timedelta(days=1)).isoformat())
            if day<self.store.start:
                values=day_values(self.store.family,self.store.group,day,self.store.fixture)
            else:
                a=int((day-self.store.start).total_seconds()/3600)
                completed=rows[a:a+24]
                if len(completed)!=24 or [r['t'] for r in completed]!=list(range(a,a+24)):
                    raise RuntimeError('historical labels must come from 24 committed H_END rows')
                values={'base_load':[r['base_load_mwh'] for r in completed],
                        'offer_price':[r['offer_price'] for r in completed],
                        'balancing_price':[r['balancing_price'] for r in completed]}
            pub=grid.end.isoformat()
            for target,vals in values.items():
                rid=f'{target}:{day.date()}';vals=tuple(float(v) for v in vals)
                record=HistoryRecord(rid,rid,0,'observation',target,grid,pub,pub,SOURCE,digest(vals),vals,UNITS[target])
                self.history.append(record);self.hist_index[(day.date(),target)]=record
            rid='resource_history:'+str(day.date())
            record=ResourceRecord(rid,rid,0,'history_day',day.isoformat(),(),pub,pub,SOURCE,
                digest({'unknown_demand':0,'basis':'fixed daily declared appointments; no dispatch labels'}))
            self.resources.append(record);self.resource_history_index[day.date()]=record
            self.next_day+=timedelta(days=1)

    def publish(self,scope,grid,at,tag,rows):
        if tag in self.records:return
        self.advance_history(at,rows)
        cutoff=at.replace(hour=0,minute=0,second=0,microsecond=0).isoformat()
        config=BaselineConfig('seasonal_last',SOURCE,cutoff)
        # Indexed immutable views avoid repeatedly reconstructing unrelated years.
        # All P1/P3 validators still run; the global repository resolves exact refs.
        end=instant(cutoff);days=[(end-timedelta(days=k)).date() for k in range(28,0,-1)]
        thin_history=HistoryArchive()
        thin_history.append_many(tuple(self.hist_index[(day,target)] for day in days
                                      for target in ('base_load','offer_price','balancing_price')))
        provider=ForecastProvider(thin_history,config)
        result=[]
        for target in ('base_load','offer_price','balancing_price'):
            rec=provider.issue(as_of=at.isoformat(),published_at=at.isoformat(),scope=scope,
                target_variable=target,target_interval=grid,forecast_id=f'{tag}:{target}',vintage=int(tag.rsplit(':',1)[-1]) if tag.startswith('op:') else 0)
            self.repo.publish(rec);result.append(rec)
        thin_resources=ResourceArchive()
        selected=[self.resource_history_index[day] for day in days]
        for k in range(len(grid.hours)//24):
            selected.extend(self.booking_index[(grid.start+timedelta(days=k)).date()])
        thin_resources.append_many(selected)
        rp=ResourceForecastProvider(thin_resources,ResourceForecastConfig(SOURCE,cutoff))
        outlook=rp.issue(as_of=at.isoformat(),published_at=at.isoformat(),scope=scope,target_interval=grid,forecast_id=tag+':resource')
        self.records[tag]=tuple(result);self.outlooks[tag]=outlook
        self.resource_ports[tag]=(thin_resources,rp)
        self.origins.append(dict(tag=tag,scope=scope,as_of=at.isoformat(),published_at=at.isoformat(),
            fixed_train_end=self.fixed_train_end,rolling_context_cutoff=cutoff,origin_fit_end=cutoff,
            forecasts=[r.content_sha256 for r in result],resource_sha256=outlook.content_sha256,config_sha256=config.content_sha256))


class LiteStore:
    """Owner-only in-memory new data port. Not original YearStore or test access."""
    def __init__(self,year=2033,family='fixture',group='G00',*,split='fixture',fixture=True):
        if split not in ('fixture','train','validation'):raise PermissionError('test denied before data construction')
        if group not in GROUPS:raise ValueError('unknown condition')
        self.year,self.family,self.group,self.fixture,self.split=year,family,group,bool(fixture),split
        self.start,self.n=year_axis(year);self.limited=group.endswith('1')
        self.key=f'{split}_{family}_{year}_{group}'
        self.context=HistoryContext(self)
        op=opportunities(year)[0];grid=HourGrid.between(self.start.isoformat(),(self.start+timedelta(hours=self.n)).isoformat())
        self.context.publish('Y',grid,op.published_at,'op:0',())
        ref=np.array(self.context.records['op:0'][0].values,dtype=np.float64)
        values=[day_values(family,group,self.start+timedelta(days=d),fixture) for d in range(self.n//24)]
        arrays={k:np.concatenate([v[k] for v in values]) for k in ('base_load','offer_price','balancing_price')}
        # Future realizations remain owner-only and are revealed by original PublishedWorld.
        self.arrays=dict(reference=ref,realized_load=arrays['base_load'],balancing_price=arrays['balancing_price'],
            offer_price=arrays['offer_price'],capacity=np.full(self.n,160.),
            forecast_load=np.zeros(self.n),forecast_price=np.zeros(self.n),
            quotes=np.array([{'Y':400.,'M':420.,'I':450.}[o.stage] if fixture else
                500+{'Y':-20,'M':-10,'I':-5}[o.stage]+np.random.default_rng(seed_for(family,o.scope,str(o.index))).uniform(-15,15)
                for o in opportunities(year)]),macro_load_factor=np.ones(98),macro_price_offset=np.zeros(98))
        for a in self.arrays.values():
            if not np.isfinite(a).all() or (a<0).any():raise ValueError('nonnegative V3 teaching domain required; no clipping')
            a.flags.writeable=False
        self.completed_year=False
        self._source_hashes=tuple((k,hashlib.sha256(v.tobytes()).hexdigest()) for k,v in sorted(self.arrays.items()))

    def load(self,key):
        if key!=self.key:raise PermissionError('unknown/cross-world key')
        return dict(id=self.key,year=self.year,group=self.group,split=self.split,hours=self.n),dict(self.arrays)

    def manifest(self):
        return dict(version=VERSION,key=self.key,split=self.split,year=self.year,family=self.family,group=self.group,
            fixture=self.fixture,master_seed=MASTER,reference_sha256=hashlib.sha256(self.arrays['reference'].tobytes()).hexdigest(),
            arrays_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in self.arrays.items()},
            task_semantics='fixed booked daily EV/industrial, zero unknown demand; no V2/V3 old data')

    def replay_copy(self):
        """Reuse fully verified immutable exogenous vintages; no ledger/device shared.

        This is allowed only after a complete year. All inputs are action-independent
        base/price observations and declared tasks, not historical device dispatch.
        Publication gates are replayed; the Agent never receives the tape/archive.
        """
        if not self.completed_year or len(self.context.sealed)!=self.n//24:raise RuntimeError('cannot cache an unfinished year')
        current=tuple((k,hashlib.sha256(v.tobytes()).hexdigest()) for k,v in sorted(self.arrays.items()))
        if current!=self._source_hashes:raise RuntimeError('source arrays changed after prediction validation')
        other=copy(self);ctx=copy(self.context)
        ctx.records=dict(ctx.records);ctx.outlooks=dict(ctx.outlooks);ctx.resource_ports=dict(ctx.resource_ports)
        ctx.sealed={};ctx.replay=True;other.context=ctx;other.completed_year=False
        return other
