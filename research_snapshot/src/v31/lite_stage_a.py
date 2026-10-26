"""A gate runner; no training, validation or test. New output only."""
import argparse,csv,json,os,time,threading
from pathlib import Path
from .lite_data import LiteStore
from .lite_engine import LiteAnnualDriver
from .lite_rules import RULES,PROTOCOL
from .lite_audit import audit_year


def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def csv_new(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);write(out/'rules_frozen.json',PROTOCOL)
    results=[]
    for year in (2033,2036):
        compiled=None
        for name,(upper,lower) in RULES.items():
            first=None
            for repeat in (0,1):
                tick=time.perf_counter();store=LiteStore(year=year) if compiled is None else compiled.replay_copy()
                driver=LiteAnnualDriver(store)
                try:
                    driver.run(upper,lower);result=audit_year(driver)
                    if compiled is None:compiled=store
                except Exception as exc:
                    write(out/'failure.json',dict(year=year,strategy=name,repeat=repeat,error=repr(exc),
                        t=driver.c.t,pending=driver.c.pending is not None,ledger_digest=driver.c.ledger.digest(),events=driver.events[-20:]))
                    raise
                result.update(year=year,strategy=name,repeat=repeat,seconds=time.perf_counter()-tick)
                if repeat==0:
                    first=result['row_sha256'];path=out/f'{year}_{name}';path.mkdir()
                    csv_new(path/'hourly.csv',driver.c.hours);csv_new(path/'macro.csv',driver.macro_receipts)
                    csv_new(path/'daily_tasks.csv',driver.daily_receipts)
                    write(path/'market_manifest.json',store.manifest());write(path/'forecast_origins.json',store.context.origins)
                    write(path/'summary.json',result)
                elif result['row_sha256']!=first:raise AssertionError('repeat nonreproducible')
                results.append(result);csv_path=out/f'progress_{year}_{name}_{repeat}.json';write(csv_path,result)
                print(json.dumps(result),flush=True)
    csv_new(out/'gate_summary.csv',results);write(out/'gate_passed.json',{'passed':True,'actual_full_annual_runs':12,'results':results})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);run(p.parse_args().output_dir)
