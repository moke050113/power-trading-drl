"""冻结一套合法教学规则；输入仅投影观察，禁止事后优化。"""
import numpy as np

def fixed_contract(obs):
    stage=int(np.argmax(obs['stage']));coverage=obs['coverage']
    desired=(.3,.5,.7)[stage]
    if stage==0:u=np.full(12,desired)
    else:
        old=float(coverage[int(obs['calendar'][0])-1])
        # I下coverage月均可能混合已交付不同段；submit内部基于本段仓位。
        # 此规则固定增量比例选择：M (.5-.3)/(.7)，I (.7-.5)/(.5)。
        u=np.full(12,(.5-.3)/(1-.3) if stage==1 else (.7-.5)/(1-.5))
    return 'COMMIT',u

def rolling_contract(obs):
    stage=int(np.argmax(obs['stage']));last=int(obs['calendar'][2])==1
    quote=float(obs['quote'][0]);forecast=float(obs['forecast'][1])
    if not last and quote>=forecast:return 'WAIT',np.zeros(12)
    if quote>=forecast:return 'SKIP',np.zeros(12)
    return 'COMMIT',np.full(12,.3 if stage==0 else .25)

def zero_dispatch(obs):return np.zeros(3,dtype=np.float32)
def max_dispatch(obs):return np.ones(3,dtype=np.float32)

def _allocate(need,maximum,start,end,h,load,price,contract):
    plan=np.zeros(24)
    # 合同余量零边际，超出按合法预测边际；两个窗口不重叠。
    segments=[]
    for t in range(max(start,h),end):
        free=min(maximum,max(0.,float(contract[t]-load[t])))
        if free>0:segments.append((0.,t,0,free))
        if maximum-free>0:segments.append((float(price[t]),t,1,maximum-free))
    for cost,t,kind,cap in sorted(segments):
        e=min(need,cap);plan[t]+=e;need-=e
    if need>1e-6:raise ValueError('未来可行性失败，不减少任务')
    return plan[h]

def forecast_dispatch(obs):
    h=int(obs['hour'][0]);d=obs['devices'];fl=obs['forecast_load'];fp=obs['forecast_price'];q=obs['contract_plan']
    ev=_allocate(float(d[1]),7,int(d[3]),int(d[4]),h,fl,fp,q)
    ind=_allocate(float(d[5]),4,int(d[7]),int(d[8]),h,fl,fp,q)
    # 本小时申报数量根据预测总用电，非按未知实际缺口。
    price,cap=obs['offer'];gap=max(0.,float(fl[h]+ev+ind-q[h]))
    spot=min(float(cap),gap) if price<fp[h] else 0.
    return np.array([spot/cap,ev/7,ind/4],dtype=np.float32)
