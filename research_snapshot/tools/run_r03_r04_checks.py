"""Explicit combined R03/R04 owner: no training, worlds, checkpoints or datasets.

The only child is a foreground, hard-limited worker. No automatic next task.
"""
import sys
sys.dont_write_bytecode=True
import os,json,pathlib,hashlib,time,subprocess,importlib.util,importlib.metadata,importlib.abc,tempfile,traceback
from contextlib import ExitStack
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[3]
CANDIDATE=pathlib.Path(__file__).resolve().parents[1]
SRC=CANDIDATE/"src"
RECORD=ROOT/"TASK_RECORD.json"
DEPS=ROOT/"dependencies/scipy_1_16_2"

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(d):RECORD.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def source_id():
    files={p.relative_to(CANDIDATE).as_posix():sha(p) for p in sorted(list(SRC.rglob("*.py"))+list((CANDIDATE/"tools").glob("*.py")))}
    return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest())
def verify_r03_reuse(scope):
    reuse=scope.get("r03_reuse")
    if not reuse:raise PermissionError("R04-only repair needs accepted R03 evidence")
    receipt_path=pathlib.Path(reuse["receipt"]);log_path=pathlib.Path(reuse["log"])
    if sha(receipt_path)!=reuse["receipt_sha256"] or sha(log_path)!=reuse["log_sha256"]:raise RuntimeError("reused R03 evidence changed")
    rec=read(receipt_path)
    if rec["source_before"]!=rec["source_after"] or any(rec["guard"].values()) or rec["runner_error"] is not None:raise RuntimeError("R03 source/guard provenance invalid")
    for k in ("src/research/prediction_optimization_baseline.py","src/tests/test_research_baselines.py"):
        if sha(CANDIDATE/k)!=rec["source_before"]["files"][k]:raise RuntimeError("R03 reused implementation/test changed")
    names=[name for name in rec["selected_tests"] if name.startswith("r03_baseline_unit.")]
    text=log_path.read_text(encoding="utf-8")
    if len(names)!=10 or any(name.rsplit(".",1)[1]+" ("+name+") ... ok" not in text for name in names):raise RuntimeError("R03 ten executed passes not established")
    for item in reuse["acceptance_reviews"]:
        if sha(pathlib.Path(item["path"]))!=item["sha256"] or read(pathlib.Path(item["path"])).get("verdict")!="PASS":raise RuntimeError("R03 accepted-reuse review missing")
    return names

def fixed(scope):
    p=CANDIDATE/"MAIN_VERSION_MANIFEST.json"
    if sha(p)!=scope.get("fixed_manifest_sha256"):raise RuntimeError("reviewed manifest mismatch")
    m=read(p)
    if any(sha(CANDIDATE/k)!=v for k,v in m["candidate_files"].items()):raise RuntimeError("candidate changed")
    entries=read(CANDIDATE/"EVIDENCE_INDEX.json")["entries"]
    expected=next(x["proposed_budgets"] for x in entries if x["id"]=="E45")
    for actual in (scope.get("budgets"),scope.get("approved_budgets")):
        if type(actual) is not dict or set(actual)!=set(expected) or any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in expected.items()):
            raise PermissionError("approved budgets differ from fixed candidate, including zero permissions")

class CappedLog:
    def __init__(self,p,limit):self.file=p.open("x",encoding="utf-8",newline="\n");self.n=0;self.limit=limit
    def write(self,s):
        size=len(s.encode("utf-8"))
        if self.n+size>self.limit:raise RuntimeError("log byte limit")
        self.file.write(s);self.file.flush();self.n+=size;return len(s)
    def flush(self):self.file.flush()
    def close(self):self.file.close()

def supervisor(dependency=False):
    d=read(RECORD);scope=d["r03_r04_package"]
    if scope.get("approval_state")!="APPROVED":raise PermissionError("combined explicit approval required")
    fixed(scope);budget=scope["budgets"];actual=scope.setdefault("actual",{})
    if not dependency and scope.get("verification_selection")=="R04_ONLY":verify_r03_reuse(scope)
    if dependency:
        if actual.get("dependency_installation_attempts",0):raise RuntimeError("dependency attempt consumed")
        if not budget.get("solver_installations_max"):raise PermissionError("solver installation not approved")
        nonce=os.urandom(16).hex();scope["dependency_worker_nonce"]=nonce;save(d)
        cmd=[sys.executable,"-I","-B",str(CANDIDATE/"tools/prepare_r03_solver.py"),"--worker",nonce]
        started=time.monotonic();code=None;error=None
        try:code=subprocess.run(cmd,timeout=budget["dependency_hard_seconds_max"],capture_output=True,text=True).returncode
        except BaseException as exc:error=repr(exc)
        receipt=dict(kind="dependency_supervisor",exit_code=code,error=error,seconds=time.monotonic()-started,note="hard limit covers network/extraction; partial installation is retained, never retried automatically")
        p=ROOT/"logs/r03_dependency_supervisor_1.json"
        encoded=(json.dumps(receipt,indent=2)+"\n").encode("utf-8")
        if len(encoded)>128*1024:raise RuntimeError("dependency supervisor receipt cap")
        with p.open("xb") as f:f.write(encoded)
        d=read(RECORD);actual=d["r03_r04_package"].setdefault("actual",{})
        actual["output_bytes"]=sum(q.stat().st_size for q in (ROOT/"logs/r03_dependency_1").glob("*.json"))+p.stat().st_size
        if actual["output_bytes"]>d["r03_r04_package"]["budgets"]["evidence_output_bytes_max"]:raise RuntimeError("dependency evidence total cap")
        save(d)
        print(json.dumps(receipt));return 0 if code==0 else 1
    if scope.get("dependency_status")!="INSTALLED_NOT_IMPORTED":raise PermissionError("approved pinned dependency not ready")
    installed=read(ROOT/"logs/r03_dependency_1/receipt.json")
    if installed.get("wheel_sha256")!="91e9e8a37befa5a69e9cacbe0bcb79ae5afb4a0b130fd6db6ee6cc0d491695fa":raise RuntimeError("wrong solver wheel")
    if any(sha(DEPS/k)!=v for k,v in installed["files"].items()):raise RuntimeError("isolated dependency changed")
    used=actual.get("verification_invocations",0)
    if used>=budget["verification_invocations_max"]:raise RuntimeError("verification quota exhausted")
    # Only the initial call and bounded, changed/reviewed failed-candidate repairs.
    if used:
        if scope.get("state")!="STOPPED_FAILURE":raise RuntimeError("successful or pending scope cannot be rerun")
        if actual.get("repair_rounds",0)>=budget["repair_rounds_max"]:raise RuntimeError("repair cap exhausted")
        proof=scope.get("repair_review_records",[])
        if len(proof)<2:raise RuntimeError("independent and algorithm repair records required")
        for item in proof:
            rp=pathlib.Path(item["path"])
            if sha(rp)!=item["sha256"] or read(rp).get("verdict")!="PASS" or read(rp).get("manifest_sha256")!=scope["fixed_manifest_sha256"]:
                raise RuntimeError("repair evidence does not bind current candidate")
        prior=actual["attempts"][-1].get("manifest_sha256")
        if prior==scope["fixed_manifest_sha256"]:raise RuntimeError("no unchanged failing command retry")
        actual["repair_rounds"]=actual.get("repair_rounds",0)+1
    n=used+1;runtime=ROOT/("logs/r03_r04_runtime_"+str(n));prefix=ROOT/("logs/r03_r04_"+str(n))
    if runtime.exists() or any(pathlib.Path(str(prefix)+x).exists() for x in (".json",".log","_events.jsonl","_supervisor.json")):raise FileExistsError("preserve prior attempts")
    invocation_limit=min(budget["per_invocation_output_bytes_max"],budget["evidence_output_bytes_max"]-actual.get("output_bytes",0))
    if invocation_limit < 3*1024*1024:raise RuntimeError("insufficient reserved evidence partitions")
    installed=read(ROOT/"logs/r03_dependency_1/receipt.json")
    cache_used=installed["extracted_bytes"]+installed["wheel_bytes"]
    for folder in (ROOT/"logs").glob("r03_r04_runtime_*"):
        cache_used+=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and "r04_contract" not in p.parts)
    cache_remaining=budget["dependency_storage_bytes_max"]-cache_used
    if cache_remaining<=0:raise RuntimeError("installation/cache storage budget exhausted")
    nonce=os.urandom(16).hex();scope.update(state="RUNNING",worker_nonce=nonce,invocation=n,current_invocation_output_limit=invocation_limit,current_cache_write_limit=cache_remaining)
    actual["verification_invocations"]=n;d["actual"]["test_invocations"]+=1;save(d)
    command=[sys.executable,"-I","-B",str(pathlib.Path(__file__).resolve()),"--worker",nonce]
    started=time.monotonic();code=None;error=None
    try:
        completed=subprocess.run(command,timeout=budget["per_worker_hard_seconds_max"],capture_output=True,text=True)
        code=completed.returncode;error=completed.stderr[-4096:] if code else None
    except BaseException as exc:error=repr(exc)
    seconds=time.monotonic()-started;events=pathlib.Path(str(prefix)+"_events.jsonl");counts={};last_boundary=None
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if row.get("event")=="attempt":counts[row["kind"]]=row["count"]
            if row.get("event")=="runner_receipt":last_boundary=row
    d=read(RECORD);scope=d["r03_r04_package"];actual=scope["actual"]
    scope["state"]="EXECUTED_PASS_PENDING_REVIEW" if code==0 else "STOPPED_FAILURE"
    actual.setdefault("attempts",[]).append(dict(invocation=n,seconds=seconds,exit_code=code,counts=counts,manifest_sha256=scope["fixed_manifest_sha256"]))
    for key,value in counts.items():actual[key]=actual.get(key,0)+value
    for globalkey,localkey in (("new_PPO_controllers","ppo_models"),("new_untrained_policies","policy_objects"),("model_inference_steps","policy_value_api"),("policy_value_api_calls","policy_value_api")):
        d["actual"][globalkey]=d["actual"].get(globalkey,0)+counts.get(localkey,0)
    d["actual"].setdefault("test_process_wall_seconds_observed",[]).append(seconds)
    d["actual"]["test_process_wall_seconds_total"]=sum(d["actual"]["test_process_wall_seconds_observed"])
    worker_path=pathlib.Path(str(prefix)+".json")
    if worker_path.exists():d["actual"]["test_seconds"]+=read(worker_path).get("worker_seconds",0)
    artifact_paths=[p for p in (runtime/"r04_contract").rglob("*") if p.is_file()]+[pathlib.Path(str(prefix)+x) for x in (".json",".log","_events.jsonl") if pathlib.Path(str(prefix)+x).exists()]
    output_bytes=sum(p.stat().st_size for p in artifact_paths)
    actual["output_bytes"]=actual.get("output_bytes",0)+output_bytes
    if output_bytes>scope["current_invocation_output_limit"] or actual["output_bytes"]>budget["evidence_output_bytes_max"]:
        code=1;scope["state"]="STOPPED_FAILURE";error="evidence byte cap exceeded"
    durable={}
    folder=runtime/"r04_contract"
    if folder.exists():
        for p in folder.glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                try:boundary=json.loads(line)
                except ValueError:continue
                if boundary.get("status") in ("READY_FOR_TAIL","COMPLETE","STOPPED"):durable[p.name]=boundary
    scope["last_durable_runner_boundaries"]=durable
    scope["last_durable_runner_receipt"]=last_boundary
    scope["hard_stop_note"]="state beyond last durable journal boundary UNKNOWN after hard termination; never infer zero liability or completed year"
    save(d)
    receipt=dict(command=command,exit_code=code,error=error,seconds=seconds,counts=counts,output_bytes_before_supervisor_receipt=output_bytes,last_durable_runner_receipt=last_boundary)
    supervisor_path=pathlib.Path(str(prefix)+"_supervisor.json")
    for _ in range(8):
        encoded=(json.dumps(receipt,indent=2)+"\n").encode("utf-8")
        total=output_bytes+len(encoded)
        if receipt.get("output_bytes") == total:break
        receipt["output_bytes"]=total
    if len(encoded)>128*1024 or total>scope["current_invocation_output_limit"] or actual["output_bytes"]+len(encoded)>budget["evidence_output_bytes_max"]:
        raise RuntimeError("supervisor reserved evidence cap before write")
    with supervisor_path.open("xb") as f:f.write(encoded)
    actual["output_bytes"]+=len(encoded);save(d)
    if total!=sum(p.stat().st_size for p in artifact_paths+[supervisor_path]):raise RuntimeError("complete evidence byte accounting mismatch")
    print(json.dumps(receipt));return 0 if code==0 else 1

def worker(nonce):
    started=time.monotonic();d=read(RECORD);scope=d["r03_r04_package"];budget=scope["budgets"]
    if scope.get("approval_state")!="APPROVED" or scope.get("state")!="RUNNING" or scope.get("worker_nonce")!=nonce:raise PermissionError("unapproved worker")
    fixed(scope);before=source_id();n=scope["invocation"];prefix=ROOT/("logs/r03_r04_"+str(n));runtime=ROOT/("logs/r03_r04_runtime_"+str(n));runtime.mkdir()
    os.environ.update(TMP=str(runtime),TEMP=str(runtime),MPLCONFIGDIR=str(runtime/"mpl"),TORCHINDUCTOR_CACHE_DIR=str(runtime/"torch"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    tempfile.tempdir=str(runtime)
    import platform
    pv=sys.getwindowsversion().platform_version;platform._syscmd_ver=lambda *a,**k:("Windows","",".".join(map(str,pv)))
    z=pathlib.Path(sys.base_prefix)/("python"+str(sys.version_info.major)+str(sys.version_info.minor)+".zip")
    if z.exists():raise RuntimeError("unexpected runtime archive")
    sys.path[:]=[p for p in sys.path if pathlib.Path(p).resolve()!=z.resolve()]
    sys.path[:0]=[str(SRC),str(DEPS)]
    envroot=pathlib.Path(sys.prefix).resolve();original=pathlib.Path(d["source_repository"]).resolve()
    paths={s:pathlib.Path(str(prefix)+s) for s in (".json",".log","_events.jsonl")}
    allow_write={p.resolve() for p in paths.values()};guard=dict(reads=0,writes=0,processes=0,forbidden_operations=0);denied=[]
    def audit(event,args):
        if event in ("subprocess.Popen","os.system","os.posix_spawn"):
            guard["processes"]+=1;raise PermissionError("no worker child processes")
        if event in ("os.mkdir","os.remove","os.rmdir","os.rename"):
            for p in (args[:2] if event=="os.rename" else args[:1]):
                if isinstance(p,(str,bytes)) and not pathlib.Path(p).resolve().is_relative_to(runtime):
                    guard["writes"]+=1;raise PermissionError("mutation outside isolated runtime")
        if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
        p=pathlib.Path(args[0]).resolve()
        if p==pathlib.Path(os.devnull).resolve():return
        mode,flags=args[1],args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(x in mode for x in "wax+")) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if p.suffix.lower() in (".pt",".pth",".pkl",".pickle",".ckpt",".npz",".npy",".zip"):
                guard["writes"]+=1;denied.append(str(p));raise PermissionError("model/array archive writing denied")
            if p not in allow_write and not p.is_relative_to(runtime):guard["writes"]+=1;denied.append(str(p));raise PermissionError("write denied")
            return
        if p.is_relative_to(envroot) or p.is_relative_to(DEPS):return
        if p.is_relative_to(original) or p.suffix.lower() in (".csv",".npz",".npy",".pt",".pth",".pkl",".pickle",".ckpt",".zip") or any(x.lower() in ("data","runs","outputs","sealed","review_bundles") for x in p.parts):
            guard["reads"]+=1;denied.append(str(p));raise PermissionError("data/model/formal repository read denied")
    class NoLegacy(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname in ("v31.lite_training","v31.r_training","v31.s_training","v31.r_env","v31.lite_stage_a","agents.train_ppo") or fullname.startswith("experiments.train_"):
                guard["forbidden_operations"]+=1;raise PermissionError("legacy training entry denied")
    sys.addaudithook(audit);sys.meta_path.insert(0,NoLegacy())
    import builtins,io,threading
    raw_io_open=io.open;raw_builtin_open=builtins.open;raw_os_open=os.open;raw_os_write=os.write
    fd_paths={};pool={"evidence":0,"cache":0};lock=threading.RLock()
    evidence_limit=scope["current_invocation_output_limit"]-(2*1024*1024+256*1024+512*1024+128*1024)
    def reserve_write(path,size):
        category="evidence" if path.is_relative_to(runtime/"r04_contract") else "cache"
        limit=evidence_limit if category=="evidence" else scope["current_cache_write_limit"]
        with lock:
            if pool[category]+size>limit:raise RuntimeError("runtime "+category+" bytes cap before write")
            pool[category]+=size
    class RuntimeFile:
        def __init__(self,file,path,newline):self.file=file;self.path=path;self.newline=newline;self.end=max(file.tell(),os.fstat(file.fileno()).st_size)
        def __getattr__(self,key):return getattr(self.file,key)
        def __enter__(self):self.file.__enter__();return self
        def __exit__(self,*args):return self.file.__exit__(*args)
        def write(self,value):
            if isinstance(value,str):
                text=value.replace("\n",os.linesep) if self.newline is None else value
                size=len(text.encode(self.file.encoding or "utf-8"))
            else:size=len(value)
            if self.file.tell()>max(self.end,os.fstat(self.file.fileno()).st_size):raise PermissionError("no sparse runtime evidence writes")
            reserve_write(self.path,size);written=self.file.write(value);self.end=max(self.end,self.file.tell());return written
        def writelines(self,lines):
            for line in lines:self.write(line)
        def truncate(self,size=None):
            size=self.file.tell() if size is None else size
            if size>max(self.end,os.fstat(self.file.fileno()).st_size):raise PermissionError("no growing runtime truncation")
            self.end=size;return self.file.truncate(size)
    def wrapped_open(original,file,*args,**kw):
        handle=original(file,*args,**kw)
        mode=args[0] if args else kw.get("mode","r")
        path=fd_paths.get(file) if type(file) is int else pathlib.Path(file).resolve()
        if path is not None and path.is_relative_to(runtime) and any(x in mode for x in "wax+"):
            newline=kw.get("newline",args[4] if len(args)>4 else None)
            return RuntimeFile(handle,path,newline)
        return handle
    def runtime_os_open(file,flags,*args,**kw):
        fd=raw_os_open(file,flags,*args,**kw);path=pathlib.Path(file).resolve()
        if path.is_relative_to(runtime):fd_paths[fd]=path
        else:fd_paths.pop(fd,None)
        return fd
    def runtime_os_write(fd,data):
        if fd in fd_paths:reserve_write(fd_paths[fd],len(data))
        return raw_os_write(fd,data)
    io.open=lambda file,*a,**k:wrapped_open(raw_io_open,file,*a,**k)
    builtins.open=lambda file,*a,**k:wrapped_open(raw_builtin_open,file,*a,**k)
    os.open=runtime_os_open;os.write=runtime_os_write
    log=CappedLog(paths[".log"],256*1024);journal=CappedLog(paths["_events.jsonl"],2*1024*1024)
    class Context:
        def __init__(self):self.counts={};self.runtime=runtime;self.owner=object();self.raw_results=[]
        def check_deadline(self):
            if time.monotonic()-started>=110:raise TimeoutError("cooperative pre-stop110s")
        def emit(self,row):
            if row.get("kind")=="r04_real_tail":row=dict(row,event="runner_receipt")
            journal.write(json.dumps(row,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x),allow_nan=False)+"\n")
        def charge(self,kind,limit):
            self.check_deadline();value=self.counts.get(kind,0)
            prior=scope["actual"].get(kind,0)
            if value>=limit or value+prior>=budget.get("total_"+kind+"_max",limit*budget["verification_invocations_max"]):raise RuntimeError("attempt budget "+kind)
            self.counts[kind]=value+1;self.emit(dict(event="attempt",kind=kind,count=value+1))
        def make_model(self,level,env):
            self.active=level
            model=PPO(HybridUpperPolicy if level=="upper" else "MultiInputPolicy",env,device="cpu",seed=20261025,n_steps=2,batch_size=2,n_epochs=2,learning_rate=.0003,gamma=1.,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,use_sde=False,policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32]),ortho_init=True,normalize_images=True),verbose=0)
            model.set_logger(Logger(None,[]));model.policy.set_training_mode(False)
            self.emit(dict(event="fresh_model_created",level=level,model_class=type(model).__name__,policy_class=type(model.policy).__name__,config=dict(seed=20261025,n_steps=2,batch_size=2,n_epochs=2,gamma=1.,gae_lambda=.95,learning_rate=.0003,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,device="cpu",net_arch=[32,32]),checkpoint=None,normalization="fixed_physical_divisors"))
            return model
    ctx=Context();result=None;error=None;selected=[]
    try:
        import unittest,numpy as np,torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy,BaseModel
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        from stable_baselines3.common.logger import Logger
        from experiments.hybrid_upper_v3 import HybridUpperPolicy
        from v3 import data,coordinator,devices
        from v31 import lite_data,lite_engine
        from research.prediction_optimization_baseline import SolveBudget
        import scipy.optimize
        with ExitStack() as stack:
            def blocked(*a,**k):guard["forbidden_operations"]+=1;raise PermissionError("no updates/checkpoints/worlds/legacy storage")
            for target in ("stable_baselines3.PPO.learn","stable_baselines3.PPO.train","stable_baselines3.common.base_class.BaseAlgorithm.save","stable_baselines3.common.base_class.BaseAlgorithm.load","stable_baselines3.common.policies.BaseModel.save","stable_baselines3.common.policies.BaseModel.load","torch.save","torch.load","torch.optim.Adam.step","torch.optim.SGD.step","torch.optim.Optimizer.step","v3.data.YearStore.__init__","v3.data.generate","v3.data._world","v31.lite_data.LiteStore.__init__","v31.lite_engine.LiteAnnualDriver.__init__","v3.coordinator.YearCoordinator.__init__","v31.history.save_new","v31.history.load_records","stable_baselines3.common.base_class.BaseAlgorithm._setup_learn","stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm.collect_rollouts","v31.lite_gym.AnnualDecisionEnv.__init__","v31.lite_gym.AnnualDecisionEnv.reset","v31.lite_gym.AnnualDecisionEnv.step"):
                stack.enter_context(patch(target,new=blocked))
            def wrap(cls,name,kind,cap):
                fn=getattr(cls,name)
                def checked(*a,**k):ctx.charge(kind,cap);return fn(*a,**k)
                stack.enter_context(patch.object(cls,name,new=checked))
            wrap(PPO,"__init__","ppo_models",2);wrap(MultiInputActorCriticPolicy,"__init__","policy_objects",2)
            for name in ("forward","evaluate_actions","get_distribution","predict_values"):wrap(MultiInputActorCriticPolicy,name,"policy_value_api",16)
            # Hybrid overrides must also be counted; inherited methods aren't wrapped twice.
            for name in ("forward","evaluate_actions","get_distribution"):
                if name in HybridUpperPolicy.__dict__:wrap(HybridUpperPolicy,name,"policy_value_api",16)
            wrap(devices.DailyDevices,"prepare","device_hours",96)
            actual_linprog=scipy.optimize.linprog
            def guarded_linprog(*a,**k):
                ctx.charge("lp_solver",64)
                options=k.get("options",{})
                if not 0<float(options.get("time_limit",0))<=5:raise PermissionError("solver time limit required")
                options=dict(options,threads=1);k["options"]=options
                return actual_linprog(*a,**k)
            stack.enter_context(patch.object(scipy.optimize,"linprog",new=guarded_linprog))
            original_reserve=SolveBudget.reserve
            def reserve_attempt(obj,*args,**kw):
                reservation=original_reserve(obj,*args,**kw)
                ctx.charge("lp_attempts",64)
                return reservation
            stack.enter_context(patch.object(SolveBudget,"reserve",new=reserve_attempt))
            suite=unittest.TestSuite();lp_context=dict(solver_budget=SolveBudget(64,5.),results=[])
            for name,rel in (("r03_baseline_unit","tests/test_research_baselines.py"),("r04_runner_unit","tests/test_research_runner_contract.py")):
                if name.startswith("r03") and scope.get("verification_selection")=="R04_ONLY":
                    verify_r03_reuse(scope);continue
                spec=importlib.util.spec_from_file_location(name,SRC/rel);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
                if name.startswith("r03"):module.RUN_CONTEXT=lp_context
                else:
                    module.CONTEXT=ctx
                    wrap(module.ArtificialDriver,"submit_lower","artificial_settlements",64)
                group=unittest.defaultTestLoader.loadTestsFromModule(module);suite.addTests(group)
            def ids(s):
                for item in s:
                    if isinstance(item,unittest.TestSuite):yield from ids(item)
                    else:yield item.id()
            selected=list(ids(suite));result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
            ctx.raw_results=lp_context["results"];ctx.emit(dict(event="lp_budget",attempts=lp_context["solver_budget"].attempts,results=ctx.raw_results))
            if scope.get("verification_selection")=="R04_ONLY":ctx.emit(dict(event="reused_R03",tests=verify_r03_reuse(scope),receipt=scope["r03_reuse"]["receipt"],not_current_execution=True))
            ctx.check_deadline()
    except BaseException as exc:
        error=repr(exc)
        try:traceback.print_exc(file=log)
        except BaseException:pass
    finally:journal.close();log.close()
    after=source_id();success=result is not None and result.wasSuccessful() and not result.skipped and error is None and not any(guard.values()) and before==after
    out=dict(source_before=before,source_after=after,source_unchanged=before==after,selected_tests=selected,tests=result.testsRun if result else 0,failures=len(result.failures) if result else 0,errors=len(result.errors) if result else 0,skipped=len(result.skipped) if result else 0,runner_error=error,guard=guard,denied=denied,counts=ctx.counts,worker_seconds=time.monotonic()-started,exit_code=0 if success else 1,dependencies={x:importlib.metadata.version(x) for x in ("numpy","gymnasium","torch","stable-baselines3","scipy")},interpreter=dict(path=sys.executable,sha256=sha(pathlib.Path(sys.executable))),artifacts={str(p):sha(p) for k,p in paths.items() if k!=".json"},scope="R03 small predictive LP and R04 real fresh-policy artificial-driver runner contracts; no actual annual world or training")
    encoded=json.dumps(out,ensure_ascii=False,indent=2)+"\n"
    if len(encoded.encode("utf-8"))>512*1024:raise RuntimeError("receipt byte cap")
    with paths[".json"].open("x",encoding="utf-8",newline="\n") as f:f.write(encoded)
    return out["exit_code"]

if __name__=="__main__":
    if len(sys.argv)==1:sys.exit(supervisor())
    if sys.argv[1:]==["--prepare-dependency"]:sys.exit(supervisor(True))
    if len(sys.argv)==3 and sys.argv[1]=="--worker":sys.exit(worker(sys.argv[2]))
    raise PermissionError("unsupported invocation")
