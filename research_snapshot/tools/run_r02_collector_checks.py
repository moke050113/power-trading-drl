"""R02 proposed real-collector prefix check; explicit scope approval required."""
import sys
sys.dont_write_bytecode = True
import os, pathlib, json, hashlib, time, subprocess, importlib.util
import importlib.metadata, importlib.abc, tempfile, traceback
from contextlib import ExitStack
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[3]
CANDIDATE = pathlib.Path(__file__).resolve().parents[1]
SRC = CANDIDATE / "src"
RECORD = ROOT / "TASK_RECORD.json"
PREFIX = ROOT / "logs/r02_collector_1"
OUTPUT_LIMITS = dict(raw=7*1024*1024, log=256*1024,
                     worker_receipt=512*1024, supervisor_receipt=128*1024)
assert sum(OUTPUT_LIMITS.values()) <= 8*1024*1024
class CappedLog:
    def __init__(self, path):
        self.file=path.open("x",encoding="utf-8",newline="\n"); self.bytes=0
    def write(self, text):
        size=len(text.encode("utf-8"))
        if self.bytes+size > OUTPUT_LIMITS["log"]:
            raise RuntimeError("log byte cap")
        value=self.file.write(text); self.file.flush(); self.bytes+=size
        return value
    def flush(self): self.file.flush()
    def close(self): self.file.close()

def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def store_record(record):
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def source_id():
    files = {p.relative_to(CANDIDATE).as_posix(): sha(p) for p in
             sorted(list(SRC.rglob("*.py"))+list((CANDIDATE/"tools").glob("*.py")))}
    return dict(files=files, sha256=hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest())

def supervisor():
    started = time.monotonic()
    record = read(RECORD); scope = record["collector_scope"]
    if scope.get("approval_state") != "APPROVED" or scope["state"] != "AUTHORIZED":
        raise PermissionError("annual scope not authorized and ready")
    if scope.get("actual", {}).get("verification_invocations", 0):
        raise RuntimeError("single invocation already consumed")
    manifest = read(CANDIDATE/"MAIN_VERSION_MANIFEST.json")
    if scope.get("fixed_manifest_sha256") != sha(CANDIDATE/"MAIN_VERSION_MANIFEST.json"):
        raise RuntimeError("candidate not fixed to approved review")
    if any(sha(CANDIDATE/k) != v for k,v in manifest["candidate_files"].items()):
        raise RuntimeError("fixed candidate mismatch")
    paths = [pathlib.Path(str(PREFIX)+suffix) for suffix in
             (".log", ".json", "_events.jsonl", "_views.jsonl", "_supervisor.json")]
    if any(p.exists() for p in paths): raise FileExistsError("preserve all prior evidence")
    scope["state"] = "RUNNING"
    scope["actual"] = dict(verification_invocations=1, counts={}, output_bytes=0)
    scope["worker_nonce"] = os.urandom(16).hex()
    record["actual"]["test_invocations"] += 1
    store_record(record)
    command = [sys.executable, "-I", "-B", str(pathlib.Path(__file__).resolve()),
               "--worker", scope["worker_nonce"]]
    remaining = 120 - (time.monotonic()-started)
    status = "UNKNOWN"; code = None; error = None
    try:
        if remaining <= 0:
            status = "PRE_LAUNCH_BUDGET_EXHAUSTED"
        else:
            completed = subprocess.run(command, timeout=remaining,
                                       capture_output=True, text=True)
            code = completed.returncode
            status = "WORKER_EXITED"
            error = completed.stderr[-4000:] if code else None
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps this one worker on timeout.
        status = "HARD_TIMEOUT_STOPPED"
    except BaseException as exc:
        status = "SUPERVISOR_FAILED"; error = repr(exc)
    worker_seconds = time.monotonic()-started
    record = read(RECORD); scope = record["collector_scope"]
    scope["observed_supervised_wall_seconds"] = worker_seconds
    scope["observed_process_exit_code"] = code
    if status != "WORKER_EXITED" or code != 0:
        scope["state"] = "STOPPED_FAILURE_NO_AUTOMATIC_REPAIR"
    # A killed worker cannot finalize RAM counters. Recover durable attempts.
    counts = {}; last_boundaries = {}; journal = pathlib.Path(str(PREFIX)+"_events.jsonl")
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try: row = json.loads(line)
            except ValueError: continue
            if row.get("event") == "attempt": counts[row["kind"]] = row["count"]
            if row.get("event") == "liability_boundary":last_boundaries[row["layer"]]=row
    scope["actual"]["durable_attempt_counts"] = counts
    scope["last_durable_liability_boundaries"]=last_boundaries
    scope["liability_state_note"]="Annual obligations remain OPEN. RAM state beyond last durable boundary is UNKNOWN if hard-stopped, never assumed zero."
    scope["actual"]["output_bytes"] = sum(p.stat().st_size for p in paths if p.exists())
    store_record(record)
    out = dict(command=command, status=status, worker_exit_code=code,
               supervised_wall_seconds=worker_seconds, error=error,
               counts=counts, limits=scope["budgets"],
               last_durable_liability_boundaries=last_boundaries,
               liability_note=scope["liability_state_note"],
               note="worker compute hard capped120s; post-stop receipt finalization separately measured")
    out["finalization_seconds_before_receipt"] = time.monotonic()-started-worker_seconds
    encoded=json.dumps(out,indent=2)+"\n"
    if len(encoded.encode("utf-8")) > OUTPUT_LIMITS["supervisor_receipt"]:
        raise RuntimeError("supervisor receipt cap before write")
    with paths[-1].open("x", encoding="utf-8", newline="\n") as f: f.write(encoded)
    record=read(RECORD)
    record["collector_scope"]["actual"]["output_bytes"]=sum(
        p.stat().st_size for p in paths if p.exists())
    store_record(record)
    print(json.dumps(dict(status=status, exit_code=code, seconds=worker_seconds,
                          counts=counts, receipt=str(PREFIX)+".json")))
    return 0 if code == 0 and status == "WORKER_EXITED" else 1

def worker(nonce):
    start = time.monotonic()
    record = read(RECORD); scope = record["collector_scope"]
    if scope.get("approval_state") != "APPROVED" or scope["state"] != "RUNNING" or scope["worker_nonce"] != nonce:
        raise PermissionError("unapproved worker")
    before = source_id()
    runtime = ROOT/"logs/collector_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(runtime)
    tempfile.tempdir = str(runtime)
    os.environ["MPLCONFIGDIR"]=str(runtime/"matplotlib")
    os.environ["TORCHINDUCTOR_CACHE_DIR"]=str(runtime/"torch_cache")
    import platform
    native_version=sys.getwindowsversion().platform_version
    platform._syscmd_ver=lambda *a,**k:("Windows","", ".".join(map(str,native_version)))
    # The already-checked absent stdlib zip search placeholder isn't an archive input.
    z = pathlib.Path(sys.base_prefix)/("python"+str(sys.version_info.major)+str(sys.version_info.minor)+".zip")
    if z.exists(): raise RuntimeError("unexpected real standard library archive")
    sys.path[:] = [p for p in sys.path if pathlib.Path(p).resolve() != z.resolve()]
    sys.path.insert(0, str(SRC))
    paths = {s:pathlib.Path(str(PREFIX)+s) for s in
             (".log", ".json", "_events.jsonl", "_views.jsonl")}
    write_allow = {p.resolve() for p in paths.values()} | {RECORD.resolve()}
    original = pathlib.Path(record["source_repository"]).resolve()
    envroot = pathlib.Path(sys.prefix).resolve()
    guard = dict(forbidden_reads=0, forbidden_writes=0, processes=0, forbidden_entry_imports=0, forbidden_operations=0)
    denied = []
    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
            guard["processes"] += 1; raise RuntimeError("worker child process denied")
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
            for p in (args[:2] if event == "os.rename" else args[:1]):
                if isinstance(p, (str,bytes)) and not pathlib.Path(p).resolve().is_relative_to(runtime):
                    guard["forbidden_writes"] += 1; raise RuntimeError("mutation outside runtime denied")
        if event != "open" or not args or not isinstance(args[0], (str,bytes)): return
        p = pathlib.Path(args[0]).resolve()
        if p == pathlib.Path(os.devnull).resolve(): return
        mode, flags = args[1], args[2] if len(args)>2 else 0
        writing = (isinstance(mode,str) and any(s in mode for s in "wax+")) or (
            isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if p not in write_allow and not p.is_relative_to(runtime):
                guard["forbidden_writes"] += 1; denied.append(str(p)); raise PermissionError("write denied")
            return
        if p.is_relative_to(envroot): return  # existing dependency runtime only
        if p.is_relative_to(original) or p.suffix.lower() in (
            ".csv",".npz",".npy",".pt",".pth",".pkl",".pickle",".zip",".ckpt"):
            guard["forbidden_reads"] += 1; denied.append(str(p)); raise PermissionError("dataset/model/original denied")
        if any(x.lower() in ("data","outputs","runs","sealed","review_bundles") for x in p.parts):
            guard["forbidden_reads"] += 1; denied.append(str(p)); raise PermissionError("historical path denied")
    class NoLegacy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in ("v31.lite_training","v31.r_training","v31.r_env","v31.lite_stage_a"):
                guard["forbidden_entry_imports"]+=1
                raise PermissionError("legacy training/evaluation entry denied")
    sys.addaudithook(audit);sys.meta_path.insert(0,NoLegacy())
    class Context:
        def __init__(self):
            self.counts={};self.bytes=0;self.active=None;self.models={};self.envs={}
            self.last_bootstrap={};self.last_boundaries={};self.saved_lots=set();self.boundary_errors=[]
            self.journal=paths["_events.jsonl"].open("x",encoding="utf-8",newline="\n")
            self.views=paths["_views.jsonl"].open("x",encoding="utf-8",newline="\n")
        def convert(self,x):
            if torch.is_tensor(x):return x.detach().cpu().numpy().tolist()
            if hasattr(x,"tolist"):return x.tolist()
            if hasattr(x,"isoformat"):return x.isoformat()
            raise TypeError(type(x).__name__)
        def emit(self,stream,obj):
            line=json.dumps(obj,default=self.convert,ensure_ascii=False,allow_nan=False)+"\n"
            n=len(line.encode("utf-8"))
            if self.bytes+n>OUTPUT_LIMITS["raw"]:raise RuntimeError("evidence byte cap")
            stream.write(line);stream.flush();self.bytes+=n
        def check_deadline(self):
            if time.monotonic()-start>=110:raise TimeoutError("cooperative pre-stop110; hard worker120")
        def charge(self,kind,cap):
            self.check_deadline()
            n=self.counts.get(kind,0)
            if n>=cap:raise RuntimeError("attempt cap: "+kind)
            self.counts[kind]=n+1
            self.emit(self.journal,dict(event="attempt",kind=kind,count=n+1,layer=self.active))
        def fingerprint(self,model):
            def freeze(v):
                if torch.is_tensor(v):
                    a=v.detach().cpu().contiguous()
                    return dict(dtype=str(a.dtype),shape=list(a.shape),
                                sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())
                if isinstance(v,dict):return {str(k):freeze(x) for k,x in sorted(v.items(),key=lambda z:str(z[0]))}
                if isinstance(v,(list,tuple)):return [freeze(x) for x in v]
                return v
            def digest(v):return hashlib.sha256(json.dumps(freeze(v),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            return dict(policy_sha256=digest(model.policy.state_dict()),
                        optimizer_sha256=digest(model.policy.optimizer.state_dict()))
        def new_model(self,layer):
            if layer!=("upper","lower")[len(self.models)]:raise PermissionError("ordered two controllers only")
            self.active=layer
            spec=scope["spec"].copy()
            policy=input_boundary.ResearchInputPolicy([spec])
            partner=lite_rules.RULES["B2"][1 if layer=="upper" else 0]
            env=input_boundary.build_environment([spec],policy=policy,level=layer,seed=20261025,partner=partner)
            self.envs[layer]=env
            model=PPO(HybridUpperPolicy if layer=="upper" else "MultiInputPolicy",env,
                seed=20261025,device="cpu",n_steps=2,batch_size=2,n_epochs=2,
                learning_rate=.0003,gamma=1.,gae_lambda=.95,clip_range=.2,
                ent_coef=0.,vf_coef=.5,max_grad_norm=.5,use_sde=False,
                policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),ortho_init=True,normalize_images=True),
                verbose=0)
            model.set_logger(Logger(None,[]))
            self.models[layer]=model
            for name in ("forward","evaluate_actions","get_distribution","predict_values"):
                original_method=getattr(model.policy,name)
                def invoke(*args,_method=original_method,_name=name,_layer=layer,**kw):
                    if not args or type(args[0]) is not dict:raise ValueError("Dict tensor batch required")
                    rows={int(v.shape[0]) for v in args[0].values()}
                    if len(rows)!=1 or not 1<=next(iter(rows))<=2:raise ValueError("batch rows cap2")
                    if any(v.device.type!="cpu" or v.dtype!=torch.float32 for v in args[0].values()):
                        raise ValueError("CPU float32 required")
                    self.charge("policy_value_api",16)
                    self.emit(self.journal,dict(event="api",layer=_layer,method=_name,rows=next(iter(rows))))
                    out=_method(*args,**kw)
                    if _name=="predict_values":self.last_bootstrap[_layer]=out.detach().cpu().clone()
                    return out
                setattr(model.policy,name,invoke)
            return model,env
        def record(self,layer,data):
            self.emit(self.views,dict(layer=layer,details=data,
                scope="real PPO collector prefix, annual economic liability remains open"))
        def boundary(self,layer,reason):
            # Pure owner-side state inspection; no model calls, simulation or tail.
            from dataclasses import asdict
            env=self.envs.get(layer)
            if env is None or env.driver is None:
                row=dict(event="liability_boundary",layer=layer,reason=reason,
                         state="UNKNOWN",note="no assigned driver; do not assume zero liability")
            else:
                driver=env.driver;c=driver.c;lot_refs=[]
                for lot in c.ledger.lots:
                    content=asdict(lot)
                    encoded=json.dumps(content,default=self.convert,sort_keys=True,
                                       separators=(",",":"),allow_nan=False)
                    identity=hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                    if identity not in self.saved_lots:
                        self.emit(self.journal,dict(event="contract_lot",identity=identity,lot=content))
                        self.saved_lots.add(identity)
                    lot_refs.append(identity)
                row=dict(event="liability_boundary",layer=layer,reason=reason,state="OPEN",
                    now=c.now.isoformat(),completed_hours=c.t,remaining_hours=driver.store.n-c.t,
                    annual_complete=driver.store.completed_year,env_ended=env.ended,
                    phase=driver.phase,pending=c.pending is not None,
                    signed_contract_fee=float(c.ledger.fee.sum()),
                    delivered_contract_fee=sum(x["contract_cost"] for x in c.hours),
                    remaining_signed_contract_fee=float(c.ledger.fee[c.t:].sum()),
                    remaining_signed_contract_mwh=float(c.ledger.quantity[c.t:].sum()),
                    cumulative_purchase_cost=c.total_purchase,cumulative_flex_cost=c.total_flex,
                    ledger_version=c.ledger.version,contract_lots=lot_refs,
                    pending_note="prepared but uncommitted hour" if c.pending is not None else None)
            self.emit(self.journal,row)
            self.last_boundaries[layer]=row
        def finalize_boundaries(self):
            for layer in tuple(self.envs):
                try:self.boundary(layer,"worker_finalization")
                except BaseException as exc:
                    self.boundary_errors.append(dict(layer=layer,error=repr(exc)[:1024],
                        current_state="UNKNOWN; use last persisted boundary"))
        def check_final_counts(self):
            expected=dict(ppo_models=2,policy_objects=2,envs=2,stores=2,drivers=2,
                          coordinators=2,published_worlds=4,resets=2,setup=2,
                          collect=2,env_steps=4,H_START=2,H_END=2)
            for k,n in expected.items():
                if self.counts.get(k)!=n:raise AssertionError((k,self.counts.get(k),n))
        def close(self):self.journal.close();self.views.close()
    ctx=Context();result=None;failure=None;selected=[]
    log=CappedLog(paths[".log"])
    try:
        import unittest,numpy as np,torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        from stable_baselines3.common.logger import Logger
        from experiments.hybrid_upper_v3 import HybridUpperPolicy
        from research import input_boundary
        from v31 import lite_gym,lite_engine,lite_data,lite_rules
        from v3 import coordinator,data
        with ExitStack() as stack:
            def blocked(*a,**kw):
                guard["forbidden_operations"]+=1
                raise PermissionError("no updates,checkpoint,legacy data,tail or replay")
            for target in ("stable_baselines3.PPO.learn","stable_baselines3.PPO.train",
                "stable_baselines3.common.base_class.BaseAlgorithm.save",
                "stable_baselines3.common.base_class.BaseAlgorithm.load",
                "stable_baselines3.common.policies.BaseModel.save",
                "stable_baselines3.common.policies.BaseModel.load",
                "torch.save","torch.load","torch.optim.Optimizer.step",
                "torch.optim.Adam.step","torch.optim.SGD.step",
                "v3.data.YearStore.__init__","v3.data.generate","v3.data._world",
                "v31.history.save_new","v31.history.load_records",
                "v31.lite_gym.AnnualDecisionEnv.finish_without_learning",
                "v31.lite_data.LiteStore.replay_copy"):
                stack.enter_context(patch(target,new=blocked))
            def install(cls,name,kind,cap,check=None,after=None):
                original_method=getattr(cls,name)
                def invoke(obj,*args,**kw):
                    ctx.charge(kind,cap)
                    if check:check(obj,args,kw)
                    value=original_method(obj,*args,**kw)
                    if after:after(obj,value)
                    return value
                stack.enter_context(patch.object(cls,name,new=invoke))
            def store_check(obj,args,kw):
                spec=dict(zip(("year","family","group"),args));spec.update(kw)
                if spec!=scope["spec"]:raise PermissionError("explicit fixture SPEC only")
            def reset_check(obj,args,kw):
                if getattr(obj,"_r02_reset_used",False):raise PermissionError("no automatic/reset retry")
                obj._r02_reset_used=True
            def controller_check(obj,args,kw):
                if type(obj) is not PPO or ctx.active not in ("upper","lower"):
                    raise PermissionError("selected PPO only")
                if ctx.active in ctx.models or kw["n_steps"]!=2 or kw["batch_size"]!=2:
                    raise PermissionError("one two-transition controller per layer")
            def policy_check(obj,args,kw):
                if type(obj) not in (HybridUpperPolicy,MultiInputActorCriticPolicy):
                    raise PermissionError("selected policy only")
            def setup_check(obj,args,kw):
                if not args or args[0]!=2 or not obj._custom_logger or obj.logger.dir is not None or obj.logger.output_formats:
                    raise PermissionError("explicit no-output setup only")
            def collect_check(obj,args,kw):
                n=kw.get("n_rollout_steps",args[3] if len(args)>3 else None)
                if n!=2 or args[0] is not obj.get_env() or args[2] is not obj.rollout_buffer:
                    raise PermissionError("actual two-step collector only")
            install(PPO,"__init__","ppo_models",2,controller_check)
            install(MultiInputActorCriticPolicy,"__init__","policy_objects",2,policy_check)
            install(BaseAlgorithm,"_setup_learn","setup",2,setup_check)
            install(OnPolicyAlgorithm,"collect_rollouts","collect",2,collect_check)
            install(lite_gym.AnnualDecisionEnv,"__init__","envs",2)
            install(lite_gym.AnnualDecisionEnv,"reset","resets",2,reset_check,
                    after=lambda obj,value:ctx.boundary(ctx.active,"after_initial_reset"))
            install(lite_gym.AnnualDecisionEnv,"step","env_steps",4,
                    after=lambda obj,value:ctx.boundary(ctx.active,"after_environment_step"))
            install(lite_data.LiteStore,"__init__","stores",2,store_check)
            install(lite_engine.LiteAnnualDriver,"__init__","drivers",2)
            install(coordinator.YearCoordinator,"__init__","coordinators",2)
            install(data.PublishedWorld,"__init__","published_worlds",4)
            install(coordinator.YearCoordinator,"prepare_hour","H_START",2)
            install(coordinator.YearCoordinator,"settle_pending","H_END",2)
            install(coordinator.YearCoordinator,"submit_macro","macros",16)
            install(lite_engine.LiteAnnualDriver,"_publication","publications",32)
            spec=importlib.util.spec_from_file_location("r02_collection_unit",SRC/"tests/test_research_collector_integration.py")
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            module.CONTEXT=ctx
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.CollectorIntegrationTests)
            selected=[x.id() for x in suite]
            result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
            ctx.check_deadline()
    except BaseException as exc:
        failure=repr(exc)[:4096]
        try:traceback.print_exc(file=log)
        except BaseException:failure+="; exception log capped/unavailable"
    finally:
        ctx.finalize_boundaries()
        ctx.close();log.close()
    after=source_id()
    success=result is not None and result.wasSuccessful() and failure is None and before==after and not any(guard.values()) and not ctx.boundary_errors
    receipt=dict(source_before=before,source_after=after,source_unchanged=before==after,
        selected_tests=selected,tests=result.testsRun if result else 0,
        failures=len(result.failures) if result else 0,errors=len(result.errors) if result else 0,
        skipped=len(result.skipped) if result else 0,runner_error=failure,guard=guard,denied=denied,
        counts=ctx.counts,last_durable_liability_boundaries=ctx.last_boundaries,
        liability_snapshot_errors=ctx.boundary_errors,worker_seconds=time.monotonic()-start,exit_code=0 if success else 1,
        artifacts={str(p):sha(p) for suffix,p in paths.items() if suffix!=".json"},
        interpreter=dict(path=sys.executable,version=sys.version,sha256=sha(pathlib.Path(sys.executable))),
        dependencies={n:importlib.metadata.version(n) for n in ("numpy","gymnasium","torch","stable-baselines3")},
        dependency_sources={str(p):sha(p) for p in (
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/on_policy_algorithm.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/base_class.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/buffers.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/policies.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/logger.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/callbacks.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/ppo/ppo.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/monitor.py")},
        limits=scope["budgets"],
        claim_scope="2 real untrained PPO collectors,2 transitions each; upper0/lower2 delivered hours; open annual obligation; no updates/economic claim")
    encoded=json.dumps(receipt,ensure_ascii=False,indent=2)+"\n"
    if len(encoded.encode("utf-8"))>OUTPUT_LIMITS["worker_receipt"]:raise RuntimeError("receipt bytes cap")
    with paths[".json"].open("x",encoding="utf-8",newline="\n") as f:f.write(encoded)
    scope["state"]="EXECUTED_PASS_PENDING_REVIEW" if success else "STOPPED_FAILURE_NO_AUTOMATIC_REPAIR"
    scope["actual"].update(counts=ctx.counts,worker_seconds=receipt["worker_seconds"],tests=receipt["tests"])
    record["collector_scope"]=scope;store_record(record)
    return receipt["exit_code"]

if __name__=="__main__":
    if len(sys.argv)==1:sys.exit(supervisor())
    if len(sys.argv)==3 and sys.argv[1]=="--worker":sys.exit(worker(sys.argv[2]))
    raise PermissionError("unsupported invocation")
