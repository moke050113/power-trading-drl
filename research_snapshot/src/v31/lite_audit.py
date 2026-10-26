"""Independent accounting/physics checks on actual V3 records; no settlement call."""
import math
import hashlib
import json
import numpy as np
from .schema import canonical


def close(a,b,atol=1e-7):
    if not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=atol):
        raise AssertionError(f'accounting mismatch: {a} != {b}')


def audit_year(driver):
    c=driver.c;rows=c.hours;n=driver.store.n
    if c.t!=n or len(rows)!=n or c.pending is not None:raise AssertionError('incomplete annual delivery')
    q=np.zeros(n);fee=np.zeros(n)
    for lot in c.ledger.lots:
        if lot.signed_at>c.world.start+__import__('datetime').timedelta(hours=lot.start):raise AssertionError('late contract')
        q[lot.start:lot.end]+=np.array(lot.quantities)
        fee[lot.start:lot.end]+=np.array(lot.quantities)*lot.unit_price
    if (q>c.ledger.reference+1e-10).any():raise AssertionError('capacity exceeded')
    esum=isum=0.;purchase=flex=0.
    for t,r in enumerate(rows):
        if r['t']!=t or r['hour']!=t%24:raise AssertionError('hour order')
        h=t%24
        if h==0:esum=isum=0.
        e=r['ev_grid_mwh'];i=r['industrial_mwh'];esum+=e;isum+=i
        if not 0<=e<=7+1e-10 or not 0<=i<=4+1e-10:raise AssertionError('power bound')
        ev_start=19 if driver.store.limited else 18;ind_start=12 if driver.store.limited else 8
        if h<ev_start and e!=0:raise AssertionError('EV window')
        if not ind_start<=h<16 and i!=0:raise AssertionError('industrial window')
        close(r['ev_battery_mwh'],.9*e);close(r['ev_soc'],.2+.9*esum/60)
        close(r['ev_remaining'],100/3-esum);close(r['industrial_remaining'],12-isum)
        close(r['contract_mwh'],q[t]);close(r['contract_cost'],fee[t])
        close(sum(r[s+'_contract_cost'] for s in 'YMI'),fee[t])
        load=r['base_load_mwh']+e+i;spot=r['spot_mwh'];balance=max(0.,load-q[t]-spot)
        close(r['total_load_mwh'],load);close(r['balance_mwh'],balance)
        close(load,min(q[t],load)+min(spot,max(0.,load-q[t]))+balance)
        cost=fee[t]+spot*r['offer_price']+balance*r['balancing_price']
        nominal_e=(100/3)/(24-ev_start) if h>=ev_start else 0.
        nominal_i=12/(16-ind_start) if ind_start<=h<16 else 0.
        fc=abs(e-nominal_e)+abs(i-nominal_i)
        close(r['purchase_cost'],cost);close(r['flex_cost'],fc);close(r['reward'],-cost-fc)
        purchase+=cost;flex+=fc
        if h==23:close(esum,100/3);close(isum,12);close(r['ev_soc'],.7)
    close(purchase,c.total_purchase);close(flex,c.total_flex)
    close(sum(r['contract_cost'] for r in rows),sum(fee))
    close(sum(fee),sum(sum(l.quantities)*l.unit_price for l in c.ledger.lots),atol=1e-5)
    driver.store.completed_year=True  # Only AFTER all end-of-year and accounting checks.
    return dict(hours=n,days=n//24,contract_lots=len(c.ledger.lots),contract_fee=float(sum(fee)),
        purchase_cost=purchase,flex_cost=flex,objective_cost=purchase+flex,
        active_spot_cost=sum(r['spot_cost'] for r in rows),balancing_cost=sum(r['balance_cost'] for r in rows),
        ev_grid_mwh=sum(r['ev_grid_mwh'] for r in rows),industrial_mwh=sum(r['industrial_mwh'] for r in rows),
        ev_forced_mwh=sum(r['ev_forced_mwh'] for r in rows),industrial_forced_mwh=sum(r['industrial_forced_mwh'] for r in rows),
        positive_commits=sum(m['positive_mwh']>1e-10 for m in driver.macro_receipts),
        row_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        market_sha256=hashlib.sha256(np.array([[r['base_load_mwh'],r['offer_price'],r['balancing_price']] for r in rows],np.float64).tobytes()).hexdigest())
