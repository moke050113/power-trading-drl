"""Frozen-model input sensitivity on legal train origins, no learn/evaluation set."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .lite_data import LiteStore
from .lite_engine import LiteAnnualDriver
from .lite_gym import encode,decode_hybrid
from .lite_rules import RULES
from .lite_training import sha,policy_sha,save_table,peak_memory,ROOT
from .lite_stage_a import write

OLD=ROOT/'outputs/v31/p5_lite/training_20261017'

def load_old():
    from stable_baselines3 import PPO
    models={};records=json.loads((OLD/'all_models.json').read_text())
    for r in records:
        p=Path(r['model_path'])
        if sha(p)!=r['model_sha256']:raise RuntimeError('frozen old model hash mismatch')
        m=PPO.load(p,device='cpu')
        if policy_sha(m)!=r['policy_sha256']:raise RuntimeError('frozen old policy mismatch')
        models[(r['seed'],r['level'])]=m
    return models,records

def train_probes():
    """Real driver origin states; do not teleport devices, ledger or hour."""
    d=LiteAnnualDriver(LiteStore(year=2033,family='train_0',group='G00',split='train',fixture=False))
    upper=[]
    while (o:=d.advance(stop='upper',lower=RULES['B2'][1])) is not None:
        stage='YMI'[int(np.argmax(o['stage']))]
        if not any(s==stage for s,_ in upper):upper.append((stage,o))
        verb,u=RULES['B2'][0](o);d.submit_upper(verb,u)
        if len(upper)==3:break
    lower=[]
    while len(lower)<24:
        o=d.advance(stop='lower',upper=RULES['B2'][0])
        if o is None:raise RuntimeError('missing train day')
        lower.append((int(o['hour'][0]),o));d.submit_lower(RULES['B2'][1](o))
    return upper,lower

def perturb_upper(raw):
    result=[('unchanged',raw)]
    for key,factors in [('quote',(.8,1.2)),('forecast_price',(.8,1.2)),('coverage',(.2,.8)),('flexible_demand',(.9,1.1))]:
        for value in factors:
            o={k:v.copy() for k,v in raw.items()}
            if key=='quote':o['quote']*=value
            elif key=='forecast_price':o['forecast'][1]*=value
            elif key=='coverage':
                o['coverage'][:]=value;o['remaining_capacity']=o['reference']*(1-value)
                o['forecast_gap'][0]=o['forecast'][0]-np.mean(o['reference']*value)
            else:
                o['resource_outlook'][:2]*=value;o['flex_capacity'][2:]*=value
                if (o['flex_capacity'][2:]>o['flex_capacity'][:2]+1e-6).any():raise ValueError('probe demand not feasible')
            result.append((f'{key}_{value}',o))
    return result

def perturb_lower(raw):
    result=[('unchanged',raw)]
    for key in ('offer','contract','forecast_price','remaining'):
        for factor in (.8,1.2):
            o={k:v.copy() for k,v in raw.items()}
            if key=='offer':o['offer'][0]*=factor
            elif key=='forecast_price':o['forecast_price']*=factor;o['forecast_offer_price']*=factor
            elif key=='contract':o['contract_plan']*=factor;o['contract_fee']*=factor
            else:
                # Reachable reduced remaining demand from earlier legitimate charging.
                # Never claim these synthetic policy probes are committed ledger states.
                if factor>1:continue
                h=int(o['hour'][0]);history=o['history'].reshape(24,4)
                changed=False
                for kind,need_index,start_index,end_index,column,cap in [('EV',1,3,4,2,7.),('IND',5,7,8,3,4.)]:
                    start=int(o['devices'][start_index]);end=int(o['devices'][end_index]);need=float(o['devices'][need_index])
                    if not start<h<=end or need<=1e-8:continue
                    room=sum(max(0.,cap-float(history[j,column])) for j in range(start,h))
                    used=min(need*(1-factor),room)
                    if used<=1e-8:continue
                    remaining=used
                    for j in range(start,h):
                        delta=min(remaining,max(0.,cap-float(history[j,column])));history[j,column]+=delta;remaining-=delta
                    o['devices'][need_index]-=used
                    if kind=='EV':o['devices'][0]+=.9*used/60
                    changed=True
                if not changed:continue
            result.append((f'{key}_{factor}',o))
    h=int(raw['hour'][0]);need=float(raw['devices'][5]);old_end=int(raw['devices'][8])
    if h<old_end-1 and need<=4*(old_end-1-max(h,int(raw['devices'][7]))):
        o={k:v.copy() for k,v in raw.items()};o['devices'][8]-=1
        result.append(('hypothetical_known_industrial_deadline_one_hour_earlier',o))
    return result

def model_probe(model,raw,level):
    import torch
    with torch.no_grad():
        tensor,_=model.policy.obs_to_tensor(encode(raw,level));dist=model.policy.get_distribution(tensor)
        action=dist.mode().cpu().numpy()[0] if level=='upper' else dist.distribution.mean.cpu().numpy()[0]
        result=dict(raw_action=action.tolist(),executed_box_action=np.clip(action,model.action_space.low,model.action_space.high).tolist(),
            entropy=dist.entropy().cpu().numpy().reshape(-1).tolist(),value=float(model.policy.predict_values(tensor).reshape(-1)[0]))
        if level=='upper':
            verb,u=decode_hybrid(raw,action)
            result.update(category_probabilities=dist.category.probs.cpu().numpy()[0].tolist(),verb=verb,
                quantity_mode=u.tolist(),quantity_mean=(dist.a/(dist.a+dist.b)).cpu().numpy()[0].tolist())
        else:result['std']=dist.distribution.stddev.cpu().numpy()[0].tolist()
    return result

def run(out):
    import torch
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    models,records=load_old();upper,lower=train_probes();results=[]
    # Input probes change legally known, physically plausible hypothetical inputs.
    # They are not claims of executable transactions with an altered official quote.
    write(out/'probe_states.json',dict(upper=[dict(stage=s,obs={k:v.tolist() for k,v in o.items()}) for s,o in upper],
        lower=[dict(hour=h,obs={k:v.tolist() for k,v in o.items()}) for h,o in lower]))
    for (seed,level),model in models.items():
        origins=upper if level=='upper' else [(h,o) for h,o in lower if h in (8,12,15,18,20,23)]
        for origin,raw in origins:
            for change,obs in (perturb_upper(raw) if level=='upper' else perturb_lower(raw)):
                results.append(dict(seed=seed,level=level,origin=origin,change=change,**model_probe(model,obs,level)))
    keys=tuple(dict.fromkeys(k for r in results for k in r))
    save_table(out/'actor_sensitivity.csv',[{k:r.get(k) for k in keys} for r in results])
    # Archive actual diagnostics with a fixed declared summary, never tune on response.
    summary=[]
    for seed in (0,1):
        for level in ('upper','lower'):
            rs=[r for r in results if r['seed']==seed and r['level']==level]
            a=np.array([r['raw_action'] for r in rs]);summary.append(dict(seed=seed,level=level,probes=len(rs),raw_action_range=np.ptp(a,axis=0).tolist(),
                mean_entropy=float(np.mean([np.mean(r['entropy']) for r in rs]))))
    for r in records:
        if sha(r['model_path'])!=r['model_sha256'] or policy_sha(models[(r['seed'],r['level'])])!=r['policy_sha256']:raise RuntimeError('read-only inference changed model')
    write(out/'summary.json',dict(results=summary,old_models=records,seconds=time.perf_counter()-tick,memory=peak_memory(),
        input_semantics='synthetic known-input sensitivity around real train origins; no transactions on edited quote/ledger projections',
        test_or_validation_loaded=False,learn_called=False))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);run(p.parse_args().output_dir)
