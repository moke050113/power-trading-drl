"""新年度合成数据；test仅封存计划，门禁先于任何实现文件读取。"""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import numpy as np
from v3.core import year_axis, opportunities, number

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT/'data/v3/synthetic_multiscale_v3'
GROUPS = ('G00','G01','G10','G11')
CONFIG = dict(version='synthetic_multiscale_v3', master_seed=20261015,
              train_years=[2010,2011],validation_years=[2018,2019],sealed_test_years=[2020],
              groups=list(GROUPS),replicates=1,units={'energy':'MWh','power':'MW','price':'CNY/MWh','cost':'CNY'},
              generator='independent_public_and_realization_streams_calendar_prior',test_materialized=False)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_new(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def _world(year,group):
    start,n=year_axis(year); k=np.arange(n,dtype=np.float64)
    day=k//24;hour=k%24
    season=np.cos(2*np.pi*day/(n/24)); daily=np.sin(2*np.pi*(hour-8)/24)
    load_prior=100+12*season+18*daily
    price_prior=480+50*season+70*daily
    g=GROUPS.index(group)
    public,actual=np.random.SeedSequence([CONFIG['master_seed'],year,g]).spawn(2)
    p=np.random.default_rng(public);a=np.random.default_rng(actual)
    high=group[1]=='1';scale=75 if high else 25
    # 所有预测只取已知日历及独立公开信号；不使用实际序列的未来值。
    fl=load_prior+p.normal(0,2,n)
    fp=price_prior+p.normal(0,scale*.3,n)
    offers=fp+p.normal(0,15,n)
    base=load_prior+a.normal(0,4,n)
    latent=np.zeros(n)
    shocks=a.normal(0,scale,n)
    for t in range(n):latent[t]=(.65*latent[t-1] if t else 0)+shocks[t]
    # bounded真实生成函数，不是裁剪外部真实价格。
    bp=100+1000/(1+np.exp(-(price_prior-600+latent)/220))
    quote=np.array([float(price_prior[o.start:o.end].mean())+p.normal(0,35) for o in opportunities(year)])
    mf=p.normal(0,12,(98,2))
    return dict(reference=load_prior,forecast_load=fl,forecast_price=fp,offer_price=offers,
                capacity=np.full(n,160.),realized_load=base,balancing_price=bp,quotes=quote,
                macro_load_factor=1+mf[:,0]/500,macro_price_offset=mf[:,1])

def generate(root=DATA_ROOT):
    root=Path(root)
    if root.exists():raise FileExistsError('不覆盖年度数据')
    root.mkdir(parents=True)
    json_new(root/'generator_config.json',CONFIG)
    worlds=[]
    for split in ('train','validation'):
        for year in CONFIG[split+'_years']:
            for group in GROUPS:
                key=f'{split}_{year}_{group}'
                arrays=_world(year,group)
                # 单文件每世界；加载时校验SHA及全部数值/单位/时间维度。
                path=root/(key+'.npz')
                np.savez_compressed(path,**arrays)
                worlds.append(dict(id=key,split=split,year=year,group=group,path=path.name,sha256=sha(path),hours=year_axis(year)[1]))
    sealed=[dict(id=f'test_{y}_{g}',split='test',year=y,group=g,hours=year_axis(y)[1],materialized=False,
                 seed_commitment=hashlib.sha256(f'{CONFIG["master_seed"]}:{y}:{g}:test'.encode()).hexdigest())
            for y in CONFIG['sealed_test_years'] for g in GROUPS]
    json_new(root/'manifest.json',dict(config_sha256=sha(root/'generator_config.json'),version=CONFIG['version'],
                                      units=CONFIG['units'],worlds=worlds,sealed_test=sealed))
    return root

class YearStore:
    """只读可信存储；策略只能收到投影的numpy副本，不能收到本实例。"""
    def __init__(self,root=DATA_ROOT,*,split='train'):
        if split not in ('train','validation'):raise PermissionError('V3正式test未授权，构造前拒绝')
        self.root=Path(root);self.split=split
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['version']!=CONFIG['version'] or self.manifest['units']!=CONFIG['units']:raise ValueError('来源/单位错误')
        if sha(self.root/'generator_config.json')!=self.manifest['config_sha256']:raise ValueError('生成配置hash错误')
        if json.loads((self.root/'generator_config.json').read_text())!=CONFIG:raise ValueError('不是冻结数据配置')
        self.worlds=tuple(w for w in self.manifest['worlds'] if w['split']==split)
        if {(w['year'],w['group']) for w in self.worlds}!={(y,g) for y in CONFIG[split+'_years'] for g in GROUPS} or len(self.worlds)!=8:
            raise ValueError('冻结日期/分组错误')
        self._loaded={}

    def load(self,key):
        rows=[w for w in self.worlds if w['id']==key]
        if len(rows)!=1:raise PermissionError('未知或跨split世界')
        w=rows[0]
        if key not in self._loaded:
            path=(self.root/w['path']).resolve()
            if path.parent!=self.root.resolve() or sha(path)!=w['sha256']:raise ValueError('文件hash/路径错误')
            with np.load(path,allow_pickle=False) as z:arr={k:np.array(z[k],dtype=np.float64,copy=True) for k in z.files}
            lengths={'quotes':98,'macro_load_factor':98,'macro_price_offset':98}
            if set(arr)!={'reference','forecast_load','forecast_price','offer_price','capacity','realized_load','balancing_price',*lengths}:raise ValueError('字段错误')
            for k,a in arr.items():
                if a.shape!=(lengths.get(k,w['hours']),) or not np.isfinite(a).all():raise ValueError('维度/有限值错误')
                if k!='macro_price_offset' and (a<0).any():raise ValueError('负值不兼容，不裁剪')
                a.flags.writeable=False
            if w['hours']!=year_axis(w['year'])[1]:raise ValueError('年度长度错误')
            self._loaded[key]=(dict(w),arr)
        return self._loaded[key]


class PublishedWorld:
    def __init__(self,store,key):
        self.metadata,self._arrays=store.load(key)
        self.start,self.n=year_axis(self.metadata['year'])
        self.ops=opportunities(self.metadata['year'])
        self._next_realization=0
        self._channel_end=self.n
    def open_day(self,day):
        if type(day) is not int or not 0<=day<self.n//24 or self._next_realization!=0:raise RuntimeError('非法/已用日通道')
        self._next_realization=day*24;self._channel_end=(day+1)*24
    @property
    def reference(self):return self._arrays['reference'].copy()
    def macro(self,op,at):
        if op not in self.ops or at<op.published_at or at>op.decision_at:raise ValueError('非法报价时间')
        ix=self.ops.index(op);a,z=op.start,op.end
        price=float(self._arrays['quotes'][ix])
        return dict(price=price,forecast_load=float(np.mean(self._arrays['reference'][a:z]))*self._arrays['macro_load_factor'][ix],
                    forecast_price=float(np.mean(self._prior_price()[a:z]))+self._arrays['macro_price_offset'][ix],
                    published_at=op.published_at,expires_at=op.decision_at)
    def _prior_price(self):
        k=np.arange(self.n);return 480+50*np.cos(2*np.pi*(k//24)/(self.n/24))+70*np.sin(2*np.pi*(k%24-8)/24)
    def day_forecast(self,t,at):
        d=t//24;issued=self.start+timedelta(hours=d*24-6)
        if at<issued:raise ValueError('预测未发布')
        a=d*24;z=a+24
        return self._arrays['forecast_load'][a:z].copy(),self._arrays['forecast_price'][a:z].copy()
    def offer(self,t,at):
        published=self.start+timedelta(hours=t,minutes=-5)
        if not 0<=t<self.n or not published<=at<=self.start+timedelta(hours=t):raise ValueError('现货报价未发布/过期')
        return float(self._arrays['offer_price'][t]),float(self._arrays['capacity'][t])
    def reveal(self,t,at):
        if t!=self._next_realization or t>=self._channel_end or at!=self.start+timedelta(hours=t+1):raise RuntimeError('只能小时结束依次读取')
        # read不推进；和全部设备/费用一起由commit_reveal提交。
        return float(self._arrays['realized_load'][t]),float(self._arrays['balancing_price'][t])
    def commit_reveal(self,t):
        if t!=self._next_realization:raise RuntimeError('实现读取时钟错误')
        self._next_realization+=1
