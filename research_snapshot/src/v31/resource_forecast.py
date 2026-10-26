"""Published bookings plus historical UNANNOUNCED demand; no dispatch labels."""
from dataclasses import dataclass,asdict,replace
from datetime import timedelta
import json
import math
from pathlib import Path
from .schema import ContentRecord,HourGrid,InputRef,instant,canonical,digest,text
from .resource_schema import TaskDemand,number
from .resource_archive import ResourceArchive
from .resource_envelope import independent_envelopes


@dataclass(frozen=True)
class ResourceForecastConfig(ContentRecord):
    source: str
    training_split_end: str
    window_days: int = 28
    method: str = 'latest_weekday_unannounced'

    def __post_init__(self):
        text(self.source)
        end=instant(self.training_split_end)
        if end.hour or end.minute or end.second or end.microsecond:raise ValueError('midnight training cutoff required')
        if type(self.window_days) is not int or self.window_days<7 or self.window_days%7 or self.method!='latest_weekday_unannounced':
            raise ValueError('fixed complete-week historical method required')
        object.__setattr__(self,'training_split_end',end.isoformat())


@dataclass(frozen=True)
class DailyDemandStats(ContentRecord):
    day: str
    resource: str
    point_unknown_grid_mwh: float
    historical_min_grid_mwh: float
    historical_max_grid_mwh: float
    historical_mean_grid_mwh: float
    historical_std_grid_mwh: float
    sample_days: int
    historical_earliest_start_hour: float
    historical_latest_deadline_hour: float
    historical_min_task_count: int
    historical_max_task_count: int
    interpretation: str = 'descriptive_history_not_calibrated_interval_or_guarantee'

    def __post_init__(self):
        instant(self.day)
        if self.resource not in ('EV','industrial') or type(self.sample_days) is not int or self.sample_days<1:raise ValueError('invalid demand statistics')
        for field in ('point_unknown_grid_mwh','historical_min_grid_mwh','historical_max_grid_mwh','historical_mean_grid_mwh','historical_std_grid_mwh'):
            object.__setattr__(self,field,number(getattr(self,field)))
        if not self.historical_min_grid_mwh<=self.historical_mean_grid_mwh<=self.historical_max_grid_mwh:raise ValueError('inconsistent sample range')
        if not 0<=self.historical_earliest_start_hour<=self.historical_latest_deadline_hour<=24:raise ValueError('invalid historical window range')
        if (type(self.historical_min_task_count) is not int or type(self.historical_max_task_count) is not int or
            not 0<=self.historical_min_task_count<=self.historical_max_task_count):raise ValueError('invalid count range')


@dataclass(frozen=True)
class ResourceOutlook(ContentRecord):
    forecast_id: str
    scope: str
    grid: HourGrid
    as_of: str
    published_at: str
    valid_until: str
    known_tasks: tuple
    predicted_unknown_tasks: tuple
    daily_stats: tuple
    input_vintages: tuple
    config_json: str
    config_sha256: str
    schema_version: int = 1

    def __post_init__(self):
        text(self.forecast_id)
        if type(self.schema_version) is not int or self.schema_version!=1 or type(self.grid) is not HourGrid:raise ValueError('resource outlook schema')
        self.grid.validate_scope(self.scope)
        a,p,z=map(instant,(self.as_of,self.published_at,self.valid_until))
        if not a<=p<z<=self.grid.start:raise ValueError('outlook time order')
        for key,origin in (('known_tasks','booked'),('predicted_unknown_tasks','unannounced')):
            values=tuple(getattr(self,key))
            if any(type(t) is not TaskDemand or t.origin!=origin or instant(t.start)<self.grid.start or instant(t.deadline)>self.grid.end for t in values):raise ValueError('outlook task origin/range')
            object.__setattr__(self,key,values)
        all_tasks=self.known_tasks+self.predicted_unknown_tasks
        if len({t.task_id for t in all_tasks})!=len(all_tasks):raise ValueError('duplicate outlook task')
        stats=tuple(self.daily_stats)
        if any(type(s) is not DailyDemandStats for s in stats):raise ValueError('invalid stats payload')
        object.__setattr__(self,'daily_stats',stats)
        refs=tuple(self.input_vintages)
        if any(type(r) is not InputRef for r in refs) or len({r.record_id for r in refs})!=len(refs):raise ValueError('invalid resource lineage')
        object.__setattr__(self,'input_vintages',refs)
        if digest(json.loads(self.config_json))!=self.config_sha256:raise ValueError('outlook config SHA mismatch')

    @property
    def expected_total_grid_mwh(self):
        return math.fsum(t.required_grid_mwh for t in self.known_tasks+self.predicted_unknown_tasks)

    def envelopes(self):
        # Check conflicts across BOTH categories before separately labelling certainty.
        independent_envelopes(self.known_tasks+self.predicted_unknown_tasks,as_of=self.as_of,certainty='known')
        return (independent_envelopes(self.known_tasks,as_of=self.as_of,certainty='known')+
                independent_envelopes(self.predicted_unknown_tasks,as_of=self.as_of,certainty='predicted_unknown'))


class ResourceForecastProvider:
    def __init__(self,archive,config):
        if type(archive) is not ResourceArchive or type(config) is not ResourceForecastConfig:raise ValueError('resource archive/config required')
        self.archive=archive;self.config=config;self._audit=()

    @property
    def audit(self):return tuple(json.loads(v) for v in self._audit)

    def issue(self,*,as_of,published_at,scope,target_interval,forecast_id):
        text(forecast_id)
        if type(target_interval) is not HourGrid:raise ValueError('HourGrid required')
        target_interval.validate_scope(scope);a,p=instant(as_of),instant(published_at)
        if not a<=p<target_interval.start:raise ValueError('resource forecast must precede target day')
        end=min(a.replace(hour=0,minute=0,second=0,microsecond=0),instant(self.config.training_split_end))
        start=end-timedelta(days=self.config.window_days)
        rows=self.archive.snapshot(as_of,source=self.config.source)
        history={}
        for r in rows:
            if r.record_type=='history_day' and start<=instant(r.day)<end:
                if instant(r.day) in history:raise ValueError('duplicate historical day')
                history[instant(r.day)]=r
        required=tuple(start+timedelta(days=i) for i in range(self.config.window_days))
        if set(history)!=set(required):raise LookupError('complete released resource history required, including zero-demand days')
        known=[];used={r.record_id:r for r in history.values()}
        for r in rows:
            if r.record_type=='booking' and target_interval.start<=instant(r.day)<target_interval.end:
                known.extend(r.tasks);used[r.record_id]=r
        groups={}
        for day in required:groups.setdefault(day.weekday(),[]).append(history[day])
        unknown=[];stats=[]
        for i in range(len(target_interval.hours)//24):
            day=target_interval.start+timedelta(days=i);samples=groups[day.weekday()];template=samples[-1]
            for task in template.tasks:
                if task.origin!='unannounced':continue
                delta=day-instant(template.day)
                unknown.append(replace(task,task_id=f'predicted:{day.date()}:{task.task_id}',
                                       start=(instant(task.start)+delta).isoformat(),deadline=(instant(task.deadline)+delta).isoformat()))
            for resource in ('EV','industrial'):
                energies=[math.fsum(t.required_grid_mwh for t in r.tasks if t.origin=='unannounced' and t.resource==resource) for r in samples]
                mean=math.fsum(e/len(energies) for e in energies)
                std=math.sqrt(math.fsum((e-mean)**2/len(energies) for e in energies))
                sets=[tuple(t for t in r.tasks if t.origin=='unannounced' and t.resource==resource) for r in samples]
                windows=[((instant(t.start)-instant(r.day)).total_seconds()/3600,
                          (instant(t.deadline)-instant(r.day)).total_seconds()/3600) for r,ts in zip(samples,sets) for t in ts]
                stats.append(DailyDemandStats(day.isoformat(),resource,energies[-1],min(energies),max(energies),mean,std,len(energies),
                                              min((w[0] for w in windows),default=0.),max((w[1] for w in windows),default=0.),
                                              min(len(ts) for ts in sets),max(len(ts) for ts in sets)))
        refs=tuple(InputRef(k,r.content_sha256) for k,r in sorted(used.items()))
        cfg={**asdict(self.config),'window_start':start.isoformat(),'window_end':end.isoformat(),
             'unknown_process':'unannounced_only_disjoint_from_published_bookings','units':{'energy':'MWh_grid','power':'MW','soc':'fraction'}}
        outlook=ResourceOutlook(forecast_id,scope,target_interval,a.isoformat(),p.isoformat(),target_interval.start.isoformat(),
                                tuple(known),tuple(unknown),tuple(stats),refs,canonical(cfg),digest(cfg))
        outlook.envelopes()  # shared-pool ambiguity rejects; infeasible individual tasks remain explicitly flagged.
        self._audit+=(canonical({'operation':'resource_forecast','as_of':a.isoformat(),'published_at':p.isoformat(),
                                 'forecast_sha256':outlook.content_sha256,'input_vintages':[asdict(r) for r in refs]}),)
        return outlook

    def save(self,path,outlook):
        self.verify(outlook)
        with Path(path).open('x',encoding='utf-8') as f:f.write(canonical({'record':outlook.to_dict(),'sha256':outlook.content_sha256})+'\n')

    def get_visible(self,outlook,*,decision_at,scope,target_interval):
        at=instant(decision_at)
        if scope!=outlook.scope or target_interval!=outlook.grid:raise ValueError('wrong resource scope/target')
        if not instant(outlook.published_at)<=at<instant(outlook.valid_until):raise PermissionError('resource outlook unpublished/expired')
        self.verify(outlook)
        return replace(outlook)

    def load(self,path):
        e=json.loads(Path(path).read_text(encoding='utf-8'));d=dict(e['record'])
        d['grid']=HourGrid(**d['grid']);d['known_tasks']=tuple(TaskDemand(**t) for t in d['known_tasks'])
        d['predicted_unknown_tasks']=tuple(TaskDemand(**t) for t in d['predicted_unknown_tasks'])
        d['daily_stats']=tuple(DailyDemandStats(**t) for t in d['daily_stats']);d['input_vintages']=tuple(InputRef(**v) for v in d['input_vintages'])
        result=ResourceOutlook(**d)
        if result.content_sha256!=e['sha256']:raise ValueError('resource outlook SHA mismatch')
        self.verify(result);return result

    def verify(self,outlook):
        if type(outlook) is not ResourceOutlook:raise ValueError('ResourceOutlook required')
        for ref in outlook.input_vintages:self.archive.resolve(ref,outlook.as_of)
        expected=ResourceForecastProvider(self.archive,self.config).issue(as_of=outlook.as_of,published_at=outlook.published_at,scope=outlook.scope,
                            target_interval=outlook.grid,forecast_id=outlook.forecast_id)
        if expected!=outlook:raise ValueError('outlook inconsistent with actual historical inputs/configuration')
