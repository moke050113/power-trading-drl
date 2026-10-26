"""D6冻结模型只读Actor/归档状态/合法train反事实；无learn、无test通道。"""
import argparse
import ast
import csv
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from v3.core import opportunities,year_axis,PositionLedger,vector
from v3.data import DATA_ROOT,YearStore,sha,json_new
from v3.environments import UpperTradingEnv,decode_upper
from v3.training import policy_hash,write_csv
from v3.coordinator import YearCoordinator
from v3.baselines import rolling_contract,forecast_dispatch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/v3/continuous_20261015'
PROTOCOL=ROOT/'docs/v3_d6/protocol.json'
OUTPUT=ROOT/'outputs/v3/d6_no_contract_20261015'
UPPER_KEYS={'stage','calendar','coverage','reference','remaining_capacity','forecast','quote','forecast_gap','flex_capacity','settled_cost','device_state'}

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify_frozen():
    pre=json.loads((OLD/'pretrain_lock.json').read_text());lock=json.loads((OLD/'prevalidation_model_lock.json').read_text())
    for p,digest in pre['source_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('冻结源码改变')
    checks=[(DATA_ROOT/'manifest.json',lock['manifest_sha256']),
            (OLD/'experiment_config.json',lock['config_sha256']),(OLD/'frozen_rules.json',lock['rules_sha256'])]
    for item in lock['models']:
        checks.extend([(Path(item['path']),item['sha256']),(OLD/f'seed_{item["seed"]}/config.json',item['config_sha256'])])
    for p,digest in checks:
        if sha(p)!=digest:raise ValueError('冻结模型/配置/规则/manifest SHA不匹配')
    return lock

def observation_hash(obs):
    h=hashlib.sha256()
    for k,v in sorted(obs.items()):h.update(k.encode());h.update(np.asarray(v,dtype=np.float32).tobytes())
    return h.hexdigest()

def clipped_verb(raw,opportunity):
    a=vector(raw,13,-float('inf'),float('inf'))
    obs={'calendar':np.array([0,0,opportunity],dtype=np.float32)}
    return decode_upper(obs,np.clip(a,0,1))[0]

def threshold_verb(raw_mode,opportunity):
    """仅数学扩展用于比较裁剪；不是允许向市场提交Box外动作。"""
    return ('SKIP' if raw_mode<.5 else 'COMMIT') if opportunity else ('WAIT' if raw_mode<1/3 else 'SKIP' if raw_mode<2/3 else 'COMMIT')

def caps(c):
    op=c.next_op();r=c.world.reference;q=c.ledger.quantity
    monthly=[]
    for m in range(1,13):
        s=next(o for o in c.world.ops if o.scope==f'M{m}')
        monthly.append(float((r[s.start:s.end]-q[s.start:s.end]).sum()))
    return dict(stage=op.stage,scope=op.scope,opportunity=op.index,at=op.decision_at.isoformat(),
                scope_remaining_mwh=float((r[op.start:op.end]-q[op.start:op.end]).sum()),month_remaining_mwh=monthly)

def increment(state,action):
    verb,u=decode_upper(state['obs'],action)
    if verb!='COMMIT':return 0.
    if state['stage']=='Y':return float(np.dot(u,state['month_remaining_mwh']))
    return float(u[0]*state['scope_remaining_mwh'])

def replay_train():
    """执行已保存合法train动作以恢复真实观察，逐步核验已归档Reward。"""
    result=[];counts=[]
    for seed in (0,1):
        e=UpperTradingEnv(seed=seed);obs,_=e.reset(seed=seed)
        old=rows(OLD/f'seed_{seed}/upper_steps.csv');reset=rows(OLD/f'seed_{seed}/upper_resets.csv')
        verbs={v:0 for v in ('WAIT','SKIP','COMMIT')};positive=zero=0;max_diff=0.
        for row in old:
            c=e.c;meta=caps(c);action=np.array(json.loads(row['action']),dtype=np.float32)
            verb,_=decode_upper(obs,action)
            st=dict(meta,obs={k:v.copy() for k,v in obs.items()},split='train',origin=f'archived_train_seed_{seed}',
                    case=f'train-{seed}-{row["index"]}',world=c.world.metadata['id'],historical_verb=verb,
                    historical_action=action.tolist())
            delta=increment(st,action);st['historical_increment_mwh']=delta
            verbs[verb]+=1
            if verb=='COMMIT':
                positive+=int(delta>1e-9);zero+=int(delta<=1e-9)
            result.append(st)
            obs,reward,done,_,_=e.step(action)
            diff=abs(reward-float(row['raw_reward']));max_diff=max(max_diff,diff)
            if diff>1e-5 or done!=(row['terminated']=='True'):raise ValueError('归档训练轨迹不匹配')
            if done:obs,_=e.reset()
        if [r['world'] for r in e.resets]!=[r['world'] for r in reset]:raise ValueError('归档reset轨迹失配')
        counts.append(dict(seed=seed,steps=len(old),**verbs,positive_increment_commits=positive,zero_increment_commits=zero,
             full_year_episodes=len(e.episodes),reset_count=len(e.resets),unique_train_worlds=len({r['world'] for r in reset}),
             total_train_worlds=8,max_raw_reward_replay_error=max_diff,partial_year_hour=e.c.t,
             partial_year_remaining_commitment=float(e.c.ledger.fee[e.c.t:].sum())))
    return result,counts

def parse_archived_u(text):
    # 不eval CSV字符串，只允许普通数字列表或明确的np.float64数字包装。
    tree=ast.parse(text,mode='eval')
    def parse(n):
        if isinstance(n,ast.List):return [parse(x) for x in n.elts]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            x=parse(n.operand);return -x if isinstance(n.op,ast.USub) else x
        if isinstance(n,ast.Call) and not n.keywords and len(n.args)==1 and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr=='float64':return parse(n.args[0])
        raise ValueError('非法归档表达式')
    return vector(parse(tree.body),12,0,1)

def public_archive_world(key):
    """只提取归档validation公开字段，绝不加载实现NPZ键。"""
    if not key.startswith('validation_'):raise PermissionError('这里只重建归档validation状态')
    manifest=json.loads((DATA_ROOT/'manifest.json').read_text())
    ws=[w for w in manifest['worlds'] if w['id']==key and w['split']=='validation']
    if len(ws)!=1:raise PermissionError('未知/跨split世界')
    w=ws[0];p=(DATA_ROOT/w['path']).resolve()
    if p.parent!=DATA_ROOT.resolve() or sha(p)!=w['sha256']:raise ValueError('公开档案hash/路径错误')
    with np.load(p,allow_pickle=False) as z:
        arr={k:np.array(z[k],dtype=np.float64) for k in ('reference','quotes','macro_load_factor','macro_price_offset')}
    return w,arr

def archived_upper_view(w,arr,ledger,op,history):
    """源码同义重建：只汇总截止前H_END已完成信息，不能倒灌后续费用。"""
    start,n=year_axis(w['year']);ops=opportunities(w['year']);r=arr['reference'];q=ledger.quantity
    idx=ops.index(op);t=max(0,min(n,int((op.decision_at-start).total_seconds()//3600)))
    past=history[:t]
    cost=float(np.cumsum(past[:,0])[-1]) if len(past) else 0.
    flex=float(np.cumsum(past[:,1])[-1]) if len(past) else 0.
    soc,remaining=tuple(past[-1,2:]) if len(past) else (.2,12.)
    refs=[];cover=[];months=[]
    for m in range(1,13):
        s=next(o for o in ops if o.scope==f'M{m}');a,z=s.start,s.end
        refs.append(float(r[a:z].mean()));cover.append(float(q[a:z].sum()/r[a:z].sum()));months.append(float((r[a:z]-q[a:z]).sum()))
    k=np.arange(n);prior=480+50*np.cos(2*np.pi*(k//24)/(n/24))+70*np.sin(2*np.pi*(k%24-8)/24)
    fl=float(r[op.start:op.end].mean())*arr['macro_load_factor'][idx]
    price=float(prior[op.start:op.end].mean())+arr['macro_price_offset'][idx]
    limited=w['group'].endswith('1')
    obs={key:np.array(value,dtype=np.float32) for key,value in dict(stage=[int(op.stage==s) for s in 'YMI'],
      calendar=[op.month,op.block,op.index],coverage=cover,reference=refs,
      remaining_capacity=[max(0.,x*(1-y)) for x,y in zip(refs,cover)],forecast=[fl,price],quote=[arr['quotes'][idx]],
      forecast_gap=[fl-float(q[op.start:op.end].mean())],flex_capacity=[35 if limited else 42,16 if limited else 32,100/3,12],
      settled_cost=[cost/1e6,flex/1e6],device_state=[soc,remaining]).items()}
    return obs,dict(stage=op.stage,scope=op.scope,opportunity=op.index,at=op.decision_at.isoformat(),
                    scope_remaining_mwh=float((r[op.start:op.end]-q[op.start:op.end]).sum()),month_remaining_mwh=months,
                    last_visible_hour=t-1)

def validation_archive():
    """只读历史CSV；不创建validation环境、不结算、不调用策略推进日期。"""
    names={f'{family}_{seed}' for family in ('long_ppo','hierarchical_ppo') for seed in (0,1)}
    macros=[r for r in rows(OLD/'validation/macros.csv') if r['strategy'] in names]
    histories={};counts={}
    with (OLD/'validation/hourly.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['strategy'] not in names:continue
            key=(r['strategy'],r['world']);h=histories.setdefault(key,np.zeros((8760,4),dtype=np.float64))
            t=int(r['t']);h[t]=[float(r[x]) for x in ('purchase_cost','flex_cost','ev_soc','industrial_remaining')]
            counts[key]=counts.get(key,0)+1
    if any(v!=8760 for v in counts.values()):raise ValueError('归档年小时不足')
    result=[]
    for name,key in sorted(histories):
        w,arr=public_archive_world(key);ledger=PositionLedger(w['year'],arr['reference']);ops=opportunities(w['year'])
        recs=[r for r in macros if r['strategy']==name and r['world']==key]
        for j,r in enumerate(recs):
            op=next(o for o in ops if o.scope==r['scope'] and o.index==int(r['opportunity']))
            if op.decision_at.isoformat()!=r['at'] or op.published_at.isoformat()!=r['published_at']:raise ValueError('归档时刻错误')
            obs,meta=archived_upper_view(w,arr,ledger,op,histories[(name,key)])
            u=parse_archived_u(r['u'])
            result.append(dict(meta,obs=obs,split='validation',origin=name,case=f'{name}-{key}-{j}',world=key,
                               historical_verb=r['verb'],historical_u=u.tolist()))
            if op.stage=='Y':
                target=np.zeros(w['hours'])
                for m in range(1,13):
                    s=next(x for x in ops if x.scope==f'M{m}');target[s.start:s.end]=u[m-1]
            else:
                old=ledger.quantity[op.start:op.end]/arr['reference'][op.start:op.end];target=old+(1-old)*u[0]
            ledger.decide(op,r['verb'],target,float(r['price']),now=op.decision_at,published_at=op.published_at,
                          expires_at=op.decision_at,expected_version=ledger.version,trade_id=f'{op.scope}:{op.index}')
    if len(result)!=3136:raise ValueError('冻结归档候选/机会数量不匹配')
    return result

def actor(model,obs):
    import torch
    if set(obs)!=UPPER_KEYS or not model.observation_space.contains(obs) or not all(np.isfinite(x).all() for x in obs.values()):raise ValueError('非法策略投影')
    model.policy.set_training_mode(False)
    with torch.inference_mode():
        tensor,_=model.policy.obs_to_tensor(obs);dist=model.policy.get_distribution(tensor)
        mean=dist.distribution.mean.cpu().numpy()[0].copy();std=dist.distribution.stddev.cpu().numpy()[0].copy()
        raw=dist.get_actions(deterministic=True).cpu().numpy()[0].copy()
    predicted=model.predict(obs,deterministic=True)[0]
    clipped=np.clip(raw,model.action_space.low,model.action_space.high)
    if not np.array_equal(raw,mean) or not np.array_equal(predicted,clipped) or not np.isfinite(std).all() or (std<=0).any():raise ValueError('原始/裁剪推理不一致')
    return mean,std,raw,clipped

def sample_train(state,mean,std,rng,count):
    if state['split']!='train':raise PermissionError('随机诊断仅train，validation不采样')
    samples=rng.normal(mean,std,size=(count,13));clipped=np.clip(samples,0,1)
    verbs=[decode_upper(state['obs'],a)[0] for a in clipped]
    positives=sum(increment(state,a)>1e-9 for a in clipped)
    threshold=.5 if state['opportunity'] else 2/3
    analytic=.5*math.erfc((threshold-mean[0])/(std[0]*math.sqrt(2)))
    return dict(case=state['case'],samples=count,WAIT=verbs.count('WAIT'),SKIP=verbs.count('SKIP'),COMMIT=verbs.count('COMMIT'),
                positive_increment_samples=positives,analytic_commit_probability=analytic,
                mode_lower_clipped_fraction=float(np.mean(samples[:,0]<0)),mode_upper_clipped_fraction=float(np.mean(samples[:,0]>1)))

def counterfactual(out,deadline):
    store=YearStore(split='train');results=[];rewards=[]
    for w in store.worlds:
        for target_scope in ('Y','M1','I1-1'):
            prefix_hash=None
            for branch in ('WAIT','SKIP','COMMIT'):
                def budget(_):
                    if perf_counter()>deadline:raise TimeoutError('固定诊断预算到期')
                c=YearCoordinator(store,w['id'],on_hour=budget);signed=0.;signed_fee=0.;hash_at=None
                while (op:=c.next_op()) is not None:
                    c.run_until(op.decision_at,forecast_dispatch);obs=c.upper_view()
                    target=op.scope==target_scope and op.index==0
                    if target:
                        hash_at=observation_hash(obs)
                        if prefix_hash is None:prefix_hash=hash_at
                        if hash_at!=prefix_hash:raise ValueError('反事实前缀状态不一致')
                        verb,u=branch,np.full(12,.3)
                    else:verb,u=rolling_contract(obs)
                    before=c.total_purchase+c.total_flex;t0=c.t;q0=float(c.ledger.quantity.sum());f0=float(c.ledger.fee.sum())
                    c.submit_macro(verb,u)
                    if target:
                        signed=float(c.ledger.quantity.sum())-q0;signed_fee=float(c.ledger.fee.sum())-f0
                    nxt=c.next_op();at=nxt.decision_at if nxt else c.world.start+timedelta(hours=c.world.n)
                    c.run_until(at,forecast_dispatch);segment=c.hours[t0:c.t]
                    objective=sum(r['objective_cost'] for r in segment)
                    if not math.isclose(c.total_purchase+c.total_flex-before,objective,abs_tol=1e-5):raise ValueError('宏Reward费用不一致')
                    rewards.append(dict(world=w['id'],target_scope=target_scope,branch=branch,scope=op.scope,opportunity=op.index,
                       verb=verb,at=op.decision_at.isoformat(),end_at=at.isoformat(),duration_hours=(at-op.decision_at).total_seconds()/3600,
                       first_settled_hour=t0,last_settled_hour=c.t-1,contract_cost=sum(r['contract_cost'] for r in segment),
                       spot_cost=sum(r['spot_cost'] for r in segment),balance_cost=sum(r['balance_cost'] for r in segment),
                       flex_cost=sum(r['flex_cost'] for r in segment),raw_reward=-objective,scaled_reward=-objective/1e6,target_decision=target))
                fields=['contract_cost','spot_cost','balance_cost','ev_grid_mwh','industrial_mwh','unused_contract_mwh']
                if c.t!=8760 or abs(sum(r['ev_grid_mwh'] for r in c.hours)-365*100/3)>1e-6 or abs(sum(r['industrial_mwh'] for r in c.hours)-365*12)>1e-6:raise ValueError('反事实没有完成同等任务')
                for day in range(365):
                    hrs=c.hours[day*24:day*24+24]
                    if abs(sum(r['ev_grid_mwh'] for r in hrs)-100/3)>1e-8 or abs(sum(r['industrial_mwh'] for r in hrs)-12)>1e-8 or abs(hrs[-1]['ev_soc']-.7)>1e-10:
                        raise ValueError('每日设备任务/SOC不满足')
                results.append(dict(world=w['id'],split='train',target_scope=target_scope,branch=branch,scenario_sha256=w['sha256'],prefix_observation_sha256=hash_at,
                   target_increment_mwh=signed,target_new_contract_commitment=signed_fee,purchase_cost=c.total_purchase,flex_cost=c.total_flex,objective_cost=c.total_purchase+c.total_flex,
                   **{k:sum(r[k] for r in c.hours) for k in fields},final_soc=c.devices.ev.soc,hours=c.t))
                write_csv(out/f'counterfactual_{w["id"]}_{target_scope}_{branch}.csv',[results[-1]])
    write_csv(out/'counterfactual.csv',results);write_csv(out/'counterfactual_macro_rewards.csv',rewards)
    return results

def run(output=OUTPUT):
    from stable_baselines3 import PPO
    import torch
    out=Path(output).resolve()
    if out!=OUTPUT.resolve():raise ValueError('诊断只写预定新目录')
    if out.exists():raise FileExistsError('拒绝覆盖诊断结果')
    out.mkdir(parents=True);cfg=json.loads(PROTOCOL.read_text());json_new(out/'protocol.json',cfg)
    start=perf_counter();deadline=start+cfg['max_seconds'];torch.set_num_threads(1)
    lock=verify_frozen()
    models={};integrity=[]
    for item in lock['models']:
        if item['level']!='upper':continue
        if sha(item['path'])!=item['sha256']:raise ValueError('模型SHA不匹配，停止不修复')
        m=PPO.load(item['path'],device='cpu')
        if policy_hash(m)!=item['policy_sha256'] or m.num_timesteps!=128 or m.action_space.shape!=(13,):raise ValueError('policy/训练步数/空间不匹配')
        models[item['seed']]=m;integrity.append(dict(item,final_n_updates=m._n_updates))
    try:
        with patch.object(PPO,'learn',side_effect=AssertionError('D6禁止learn')):
            train,coverage=replay_train();validation=validation_archive();states=train+validation
            write_csv(out/'training_coverage.csv',coverage)
            with (out/'observations.jsonl').open('x',encoding='utf-8') as f:
                for state in states:f.write(json.dumps(dict(state,obs={k:v.tolist() for k,v in state['obs'].items()}),ensure_ascii=False)+'\n')
            actions=[];samples=[];rng={s:np.random.default_rng(np.random.SeedSequence([cfg['diagnostic_seed'],s])) for s in models}
            for seed,m in models.items():
                for state in states:
                    mean,std,raw,clipped=actor(m,state['obs']);verb,_=decode_upper(state['obs'],clipped)
                    threshold=.5 if state['opportunity'] else 2/3
                    matching=state['split']=='validation' and state['origin'].endswith(f'_{seed}')
                    if matching and (verb!=state['historical_verb'] or not np.allclose(clipped[1:],state['historical_u'],rtol=0,atol=1e-7)):raise ValueError('重建归档观察与已部署动作不一致')
                    actions.append(dict(model_seed=seed,case=state['case'],split=state['split'],origin=state['origin'],world=state['world'],
                        stage=state['stage'],scope=state['scope'],opportunity=state['opportunity'],at=state['at'],observation_sha256=observation_hash(state['obs']),
                        actor_mean=json.dumps(mean.tolist()),actor_std=json.dumps(std.tolist()),deterministic_raw_action=json.dumps(raw.tolist()),
                        clipped_action=json.dumps(clipped.tolist()),decoded_verb=verb,increment_mwh=increment(state,clipped),
                        raw_mode=float(raw[0]),clipped_mode=float(clipped[0]),commit_threshold=threshold,commit_threshold_distance=float(threshold-raw[0]),
                        first_wait_boundary_distance=float(1/3-raw[0]) if not state['opportunity'] else '',
                        clip_changed_threshold_category=threshold_verb(raw[0],state['opportunity'])!=verb,
                        historical_match_checked=matching,model_sha256=next(i['sha256'] for i in integrity if i['seed']==seed),policy_sha256=policy_hash(m)))
                    if state['split']=='train':samples.append(dict(model_seed=seed,**sample_train(state,mean,std,rng[seed],cfg['samples_per_train_state'])))
                    if perf_counter()>deadline:raise TimeoutError('固定诊断预算到期')
            write_csv(out/'actor_actions.csv',actions);write_csv(out/'frozen_train_sampling.csv',samples)
            signals=[]
            for seed in models:
                for r in rows(OLD/f'seed_{seed}/upper_logger/progress.csv'):
                    signals.extend(dict(seed=seed,rollout_timesteps=r['time/total_timesteps'],metric=k,value=v,available=v!='') for k,v in r.items())
            write_csv(out/'training_signals.csv',signals)
            result=counterfactual(out,deadline)
        for item in integrity:
            if sha(item['path'])!=item['sha256'] or policy_hash(models[item['seed']])!=item['policy_sha256']:raise ValueError('只读推理改变模型')
            item['file_sha256_after']=sha(item['path']);item['policy_sha256_after']=policy_hash(models[item['seed']])
        json_new(out/'model_integrity.json',dict(models=integrity,learn_called=False,protocol_sha256=sha(PROTOCOL)))
        json_new(out/'completion.json',dict(status='COMPLETE',seconds=perf_counter()-start,actor_rows=len(actions),observations=len(states),
           train_sampling_rows=len(samples),counterfactual_year_runs=len(result),validation_re_evaluated=False,test_loaded=False,
           training_raw_actions_available=False,initial_weights_available=False,logger_last_update=6,model_last_update=8))
        return out
    except Exception as exc:
        json_new(out/'failure.json',dict(status='STOPPED',exception=repr(exc),seconds=perf_counter()-start,retry_automatically=False));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',default='train',choices=['train','validation','test']);a=p.parse_args()
    if a.split!='train':raise PermissionError('入口固定train诊断，test门禁在任何存储/模型构造前')
    print(run())
if __name__=='__main__':main()
