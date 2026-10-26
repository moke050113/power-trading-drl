"""年度唯一时钟。H_START只建立候选；H_END才读取实现与提交。"""
from datetime import timedelta
from copy import copy
import math
import numpy as np
from v3.core import PositionLedger, vector, settlement, contract_coverage
from v3.data import PublishedWorld
from v3.devices import DailyDevices


def lower_view(world,t,devices,quantity,fee,history):
    when=world.start+timedelta(hours=t)
    fl,fp=world.day_forecast(t,when)
    p,k=world.offer(t,when)
    state=devices.state();h=t%24
    past=np.zeros((24,4));mask=np.zeros(24)
    for row in history:
        if row['t']//24==t//24 and row['t']<t:
            j=row['t']%24;past[j]=[row['base_load_mwh'],row['balancing_price'],row['ev_grid_mwh'],row['industrial_mwh']];mask[j]=1
    return {name:np.array(value,dtype=np.float32,copy=True) for name,value in dict(
        hour=[h],forecast_load=fl,forecast_price=fp,contract_plan=quantity,contract_fee=fee,
        offer=[p,k],devices=[state[x] for x in ['ev_soc','ev_remaining','ev_min','ev_start','ev_end',
                                              'industrial_remaining','industrial_min','industrial_start','industrial_end']],
        history=past.flatten(),history_mask=mask).items()}


class YearCoordinator:
    def __init__(self,store,key,*,on_hour=None):
        self.world=PublishedWorld(store,key)
        self.ledger=PositionLedger(self.world.metadata['year'],self.world.reference)
        self.devices=DailyDevices(self.world.metadata['group'].endswith('1'))
        self.t=0;self.op_cursor=0;self.pending=None
        self.now=self.world.ops[0].decision_at
        self.total_purchase=self.total_flex=0.
        self.hours=[];self.macros=[];self.on_hour=on_hour
        self.current_day_history=[]

    def next_op(self):
        while self.op_cursor<len(self.world.ops) and self.ledger.decided(self.world.ops[self.op_cursor].scope):self.op_cursor+=1
        return self.world.ops[self.op_cursor] if self.op_cursor<len(self.world.ops) else None

    def upper_view(self):
        op=self.next_op()
        if op is None:raise RuntimeError('年度宏决策已完成')
        if self.now!=op.decision_at:raise RuntimeError('尚未到宏决策时刻')
        quote=self.world.macro(op,self.now)
        r=self.world.reference;q=self.ledger.quantity
        contract_coverage(q,r)  # 只读检查旧机器精度边界，不改历史仓位。
        ratios=[];refs=[]
        for m in range(1,13):
            scope=next(x for x in self.world.ops if x.scope==f'M{m}')
            a,z=scope.start,scope.end
            denominator=float(r[a:z].sum())
            ratios.append(min(1.,float(np.minimum(q[a:z],r[a:z]).sum()/denominator)) if denominator>0 else 0.);refs.append(float(r[a:z].mean()))
        return {k:np.array(v,dtype=np.float32) for k,v in dict(stage=[int(op.stage==s) for s in 'YMI'],
           calendar=[op.month,op.block,op.index],coverage=ratios,reference=refs,
           remaining_capacity=[max(0.,r*(1-c)) for r,c in zip(refs,ratios)],
           forecast=[quote['forecast_load'],quote['forecast_price']],quote=[quote['price']],
           forecast_gap=[quote['forecast_load']-float(q[op.start:op.end].mean())],
           flex_capacity=[(24-self.devices.ev.config.start_hour)*7,(16-self.devices.industrial.start_hour)*4,100/3,12],
           settled_cost=[self.total_purchase/1e6,self.total_flex/1e6],
           # 已确认任务状态，pending候选不能投影给上层。
           device_state=[self.devices.ev.soc,self.devices.industrial.remaining_energy_mwh]).items()}

    def submit_macro(self,verb,ratios):
        op=self.next_op()
        if op is None or self.now!=op.decision_at:raise RuntimeError('先推进到正确宏机会')
        quote=self.world.macro(op,self.now)
        u=vector(ratios,12,0,1)
        r=self.world.reference;q=self.ledger.quantity
        if op.stage=='Y':
            target=np.zeros(self.world.n)
            for m in range(1,13):
                s=next(o for o in self.world.ops if o.scope==f'M{m}')
                target[s.start:s.end]=u[m-1]
        else:
            old=contract_coverage(q[op.start:op.end],r[op.start:op.end])
            target=old+(1-old)*u[0]
        self.ledger.decide(op,verb,target,quote['price'],now=self.now,published_at=quote['published_at'],
                           expires_at=quote['expires_at'],expected_version=self.ledger.version,
                           trade_id=f'{op.scope}:{op.index}')
        self.macros.append(dict(stage=op.stage,scope=op.scope,opportunity=op.index,at=self.now.isoformat(),verb=verb,
                                u=list(u),price=quote['price'],published_at=quote['published_at'].isoformat(),
                                expires_at=quote['expires_at'].isoformat(),ledger_version=self.ledger.version))
        self.op_cursor+=1

    def observation(self):
        if self.t>=self.world.n:raise RuntimeError('年终无下一小时')
        start=self.world.start+timedelta(hours=self.t)
        op=self.next_op()
        if self.now!=start or (op is not None and op.decision_at<=start):raise RuntimeError('未到小时动作时点/宏事件未处理')
        if self.devices.hour==24:
            # 只提供下一会话初始化投影；正式会话在成功H_START提交。
            devices=DailyDevices(self.devices.limited)
        else:devices=self.devices
        a=self.t//24*24
        q,f=self.ledger.plan(a,a+24)
        return lower_view(self.world,self.t,devices,q,f,self.current_day_history)

    def prepare_hour(self,action):
        if self.pending is not None or self.t>=self.world.n:raise RuntimeError('已准备/已结束')
        now=self.world.start+timedelta(hours=self.t)
        op=self.next_op()
        if self.now!=now or (op is not None and op.decision_at<=now):raise RuntimeError('不能跳过宏时钟开始交割')
        x=vector(action,3,0,1)
        d=DailyDevices(self.devices.limited) if self.devices.hour==24 else self.devices
        if d.hour!=self.t%24:raise RuntimeError('设备与年度时钟不同步')
        p,k=self.world.offer(self.t,now)
        candidate,device_row=d.prepare(x[1]*7,x[2]*4)
        q,f=self.ledger.plan(self.t,self.t+1)
        staged=dict(t=self.t,devices=candidate,device_row=device_row,action=x,
                    contract=float(q[0]),fee=float(f[0]),stage_fee=self.ledger.stage_fee(self.t),
                    spot=float(x[0]*k),offer=p)
        self.ledger.start_hour(self.t)
        self.pending=staged;self.now=now

    def settle_pending(self):
        s=self.pending
        if s is None or s['t']!=self.t:raise RuntimeError('没有本小时候选')
        if self.world._next_realization!=self.t or self.ledger._settled!=self.t-1:raise RuntimeError('可信时钟错位')
        at=self.world.start+timedelta(hours=self.t+1)
        op=self.next_op()
        if op is not None and op.decision_at<at:raise RuntimeError('小时内宏机会必须先处理')
        b,pb=self.world.reveal(self.t,at)
        q,f=self.ledger.plan(self.t,self.t+1)
        if q[0]!=s['contract'] or f[0]!=s['fee']:raise RuntimeError('小时开始后仓位被改变')
        d=s['device_row']
        row=settlement(b,d['industrial_actual_mw'],d['ev_actual_mw'],s['contract'],s['fee'],s['spot'],s['offer'],pb)
        row.update(d,t=self.t,hour=self.t%24,u_spot=float(s['action'][0]),u_ev=float(s['action'][1]),u_industrial=float(s['action'][2]))
        row.update({f'{k}_contract_cost':v for k,v in s['stage_fee'].items()})
        row['objective_cost']=row['purchase_cost']+row['flex_cost'];row['reward']=-row['objective_cost']
        if not math.isclose(row['total_load_mwh'],min(row['contract_mwh'],row['total_load_mwh'])+
                             min(row['spot_mwh'],max(0.,row['total_load_mwh']-row['contract_mwh']))+row['balance_mwh'],abs_tol=1e-9):
            raise RuntimeError('电量守恒失败')
        if not np.isfinite(list(row.values())).all():raise ValueError('小时结果非有限')
        # 候选设备/读取/结算都验证后，无可失败的状态赋值一起提交。
        ledger=copy(self.ledger);world=copy(self.world)
        ledger.commit_hour(self.t);world.commit_reveal(self.t)
        self.ledger,self.world=ledger,world
        self.devices=s['devices'];self.total_purchase+=row['purchase_cost'];self.total_flex+=row['flex_cost']
        self.t+=1;self.now=at;self.pending=None
        self.hours.append(row.copy())
        if row['hour']==0:self.current_day_history=[]
        self.current_day_history.append(row.copy())
        if self.on_hour:self.on_hour(row.copy())
        return row.copy()

    def run_until(self,at,lower):
        if at<self.now:raise RuntimeError('全局时钟倒退')
        op=self.next_op()
        if op is not None and at>op.decision_at:raise RuntimeError('不得跳过宏交易机会')
        while self.t<self.world.n:
            start=self.world.start+timedelta(hours=self.t)
            end=start+timedelta(hours=1)
            if start>at:break
            if self.pending is None:
                self.now=start
                action=lower(self.observation())
                self.prepare_hour(action)
            if end>at:break
            self.settle_pending()
        self.now=at

    def run(self,upper,lower):
        while (op:=self.next_op()) is not None:
            self.run_until(op.decision_at,lower)
            verb,u=upper(self.upper_view())
            self.submit_macro(verb,u)
        self.run_until(self.world.start+timedelta(hours=self.world.n),lower)
        if self.pending is not None or len(self.hours)!=self.world.n:raise RuntimeError('年交付不完整')
        return self.total_purchase,self.total_flex
