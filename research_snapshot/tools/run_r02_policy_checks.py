"""Explicitly authorized R02 neural-interface unit scope; never creates an env."""
import sys
sys.dont_write_bytecode=True
import os,pathlib,json,hashlib,time,importlib.util,importlib.metadata,platform,tempfile
from contextlib import ExitStack
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[3]
CANDIDATE=pathlib.Path(__file__).resolve().parents[1]
SRC=CANDIDATE/"src"
sys.path.insert(0,str(SRC))
stdlib_zip=pathlib.Path(sys.base_prefix)/("python"+str(sys.version_info.major)+str(sys.version_info.minor)+".zip")
if stdlib_zip.exists():raise RuntimeError("unexpected real stdlib archive")
removed_zip=[p for p in sys.path if pathlib.Path(p).resolve()==stdlib_zip.resolve()]
sys.path[:]=[p for p in sys.path if pathlib.Path(p).resolve()!=stdlib_zip.resolve()]
RECORD=ROOT/"TASK_RECORD.json"
record=json.loads(RECORD.read_text(encoding="utf-8"))
scope=record["policy_interface_scope"]
if scope["approval_state"]!="APPROVED" or scope["state"]!="IN_PROGRESS":
    raise PermissionError("policy scope not active")
budget=scope["budgets"];actual=scope["actual"]
if actual["verification_invocations"]>=budget["verification_invocations_max"]:
    raise RuntimeError("scope invocation allowance exhausted")
if actual["constructor_attempts"]>=budget["new_untrained_policies_max_total"]:
    raise RuntimeError("two-policy total allowance already consumed; no automatic recreation")
number=actual["verification_invocations"]+1
prefix=ROOT/"logs"/("r02_policy_interface_"+str(number))
log_path=prefix.with_suffix(".log")
receipt_path=prefix.with_suffix(".json")
artifact_path=ROOT/"logs"/("r02_policy_interface_"+str(number)+"_artifacts.json")
if any(p.exists() for p in (log_path,receipt_path,artifact_path)):raise FileExistsError("preserve old results")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity():
    paths=sorted(list(SRC.rglob("*.py"))+list((CANDIDATE/"tools").glob("*.py")))
    files={p.relative_to(CANDIDATE).as_posix():sha(p) for p in paths}
    return dict(sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest(),files=files)
before=identity()
package_root=pathlib.Path(sys.prefix)/"Lib/site-packages"
dependency_paths=[package_root/"stable_baselines3"/p for p in
    ("common/policies.py","common/buffers.py","common/distributions.py","common/torch_layers.py","common/preprocessing.py")]
dependency_paths.append(package_root/"torch/testing/_comparison.py")
dependency_sources={str(p):sha(p) for p in dependency_paths}
dependencies={n:importlib.metadata.version(n) for n in ("numpy","gymnasium","torch","stable-baselines3")}
interpreter=dict(path=sys.executable,version=sys.version,sha256=sha(pathlib.Path(sys.executable)))
actual["verification_invocations"]=number
record["actual"]["test_invocations"]+=1
RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
runtime=ROOT/"logs/policy_runtime"
runtime.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(runtime/"matplotlib")
os.environ["TMP"]=os.environ["TEMP"]=str(runtime)
os.environ["TORCHINDUCTOR_CACHE_DIR"]=str(runtime/"torch_cache")
tempfile.tempdir=str(runtime)
native_winver=sys.getwindowsversion().platform_version
platform._syscmd_ver=lambda *a,**k:("Windows","", ".".join(map(str,native_winver)))
source=pathlib.Path(record["source_repository"]).resolve()
venv=(source/".venv-ppo").resolve()
counts=dict(forbidden_reads=0,forbidden_writes=0,child_process_attempts=0,
    world_construction_attempts=0,forbidden_model_or_update_attempts=0)
denied_events=[];phase="dependency_import"
write_files={p.resolve() for p in (RECORD,log_path,receipt_path,artifact_path)}
def deny(reason,path):
    frames=[];frame=sys._getframe(2)
    for _ in range(12):
        if frame is None:break
        frames.append(dict(file=frame.f_code.co_filename,line=frame.f_lineno,function=frame.f_code.co_name))
        frame=frame.f_back
    denied_events.append(dict(reason=reason,path=str(path),phase=phase,stack=frames))
def audit(event,args):
    if event in ("subprocess.Popen","os.system","os.posix_spawn"):
        counts["child_process_attempts"]+=1;deny("child_process",event)
        raise RuntimeError("no child or background process")
    if event in ("os.mkdir","os.remove","os.rmdir","os.rename"):
        targets=args[:2] if event=="os.rename" else args[:1]
        for target in targets:
            if isinstance(target,(str,bytes)):
                p=pathlib.Path(target).resolve()
                if not p.is_relative_to(runtime.resolve()):
                    counts["forbidden_writes"]+=1;deny(event,p)
                    raise RuntimeError("filesystem mutation outside current ephemeral runtime")
        return
    if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
    p=pathlib.Path(args[0]).resolve()
    if p==pathlib.Path(os.devnull).resolve():return
    mode=args[1];flags=args[2] if len(args)>2 else 0
    writing=(isinstance(mode,str) and any(c in mode for c in "wax+"))
    writing=writing or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
    if writing and p not in write_files and not p.is_relative_to(runtime.resolve()):
        counts["forbidden_writes"]+=1;deny("unapproved_write",p)
        raise RuntimeError("unapproved write")
    if p.is_relative_to(venv):return
    if p.is_relative_to(source) or p.suffix.lower() in (".csv",".npz",".npy",".pkl",".pickle",".zip",".pt",".pth",".ckpt"):
        counts["forbidden_reads"]+=1;deny("data_model_or_original_source",p)
        raise RuntimeError("no original project/data/model file access")
    if any(x.lower() in ("data","outputs","runs","sealed","review_bundles") for x in p.parts):
        counts["forbidden_reads"]+=1;deny("data_directory",p)
        raise RuntimeError("historical data directory denied")
sys.addaudithook(audit)
def refuse_world(*args,**kwargs):
    counts["world_construction_attempts"]+=1
    raise AssertionError("world/environment construction not authorized")
def refuse_model_or_update(*args,**kwargs):
    counts["forbidden_model_or_update_attempts"]+=1
    raise AssertionError("no PPO agent, model files, learn or optimizer updates")
start=time.monotonic()
def persist():
    record["policy_interface_scope"]["actual"]=actual
    RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
def wall():
    if time.monotonic()-start>=budget["per_invocation_wall_seconds_max"]:
        raise TimeoutError("policy scope wall allowance")
class Context:
    def __init__(self,torch,np,classes):
        self.torch=torch;self.np=np;self.classes=classes
        self.policies={};self.initial={};self.artifacts={"api_events":[],"state_checks":[]}
    def tensor_hash(self,t):
        t=t.detach().cpu().contiguous()
        return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
    def freeze(self,value):
        if self.torch.is_tensor(value):return self.tensor_hash(value)
        if isinstance(value,dict):return {str(k):self.freeze(v) for k,v in sorted(value.items(),key=lambda kv:str(kv[0]))}
        if isinstance(value,(list,tuple)):return [self.freeze(v) for v in value]
        return value
    def fingerprint(self,policy):
        weights={k:self.tensor_hash(v) for k,v in sorted(policy.state_dict().items())}
        opt=self.freeze(policy.optimizer.state_dict())
        digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        return dict(parameter_state_sha256=digest(weights),optimizer_state_sha256=digest(opt))
    def charge(self,layer,name,args):
        wall()
        if actual["policy_value_api_calls"]>=budget["policy_value_api_calls_max"]:
            raise RuntimeError("aggregate neural API allowance exhausted")
        if not args or type(args[0]) is not dict:raise ValueError("explicit Dict tensor observations required")
        rows={int(v.shape[0]) for v in args[0].values()}
        if len(rows)!=1 or not 1<=next(iter(rows))<=budget["batch_rows_max"]:
            raise ValueError("neural batch must have 1 or2 rows")
        if any(v.device.type!="cpu" for v in args[0].values()):
            raise PermissionError("CPU observations only")
        n=next(iter(rows))
        actual["policy_value_api_calls"]+=1;actual["api_rows"]+=n
        record["actual"]["model_inference_steps"]=actual["policy_value_api_calls"]
        record["actual"]["policy_value_api_calls"]=actual["policy_value_api_calls"]
        self.artifacts["api_events"].append(dict(layer=layer,method=name,rows=n,index=actual["policy_value_api_calls"]))
        persist()
    def create(self,layer,policy_cls,ospace,aspace):
        wall()
        if layer not in self.classes or policy_cls is not self.classes[layer] or layer in self.policies:
            raise PermissionError("only one explicitly selected upper/lower policy")
        if actual["constructor_attempts"]>=budget["new_untrained_policies_max_total"]:
            raise RuntimeError("aggregate policy creation allowance exhausted")
        actual["constructor_attempts"]+=1;persist()
        policy=policy_cls(ospace,aspace,lambda _: .0003,
            net_arch=dict(pi=[32,32],vf=[32,32]),ortho_init=True,
            normalize_images=True)
        if any(p.device.type!="cpu" or p.dtype!=self.torch.float32 for p in policy.parameters()):
            raise AssertionError("CPU float32 policy required")
        policy.set_training_mode(False)
        self.policies[layer]=policy
        actual["policies_created"]+=1
        record["actual"]["new_untrained_policies"]=actual["policies_created"]
        self.initial[layer]=self.fingerprint(policy)
        persist()
        for name in ("forward","evaluate_actions","get_distribution","predict_values"):
            original=getattr(policy,name)
            def guarded(*args,_original=original,_name=name,**kwargs):
                self.charge(layer,_name,args)
                return _original(*args,**kwargs)
            setattr(policy,name,guarded)
        self.artifacts.setdefault("policy_config",{})[layer]=dict(policy_class=type(policy).__name__,
            net_arch={"pi":[32,32],"vf":[32,32]},lr=.0003,ortho_init=True,
            dtype="float32",device="cpu",untrained=True,
            observation_dtype="already encoded manual fixture",normalize_images=True,
            parameter_count=sum(p.numel() for p in policy.parameters()),initial=self.initial[layer])
        return policy
    def verify_unchanged(self):
        for layer,policy in self.policies.items():
            current=self.fingerprint(policy)
            if current!=self.initial[layer] or any(p.grad is not None for p in policy.parameters()):
                raise AssertionError("fresh policy or optimizer state changed")
            self.artifacts["state_checks"].append(dict(layer=layer,unchanged=True,current=current))
result=None;failure=None;ctx=None;selected=[]
with log_path.open("x",encoding="utf-8") as log:
    try:
        import unittest,numpy as np,torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        from experiments.hybrid_upper_v3 import HybridUpperPolicy
        from v31 import lite_gym,lite_engine,lite_data
        from v3 import data,coordinator
        with ExitStack() as guards:
            for target in ("v31.lite_data.LiteStore.__init__","v3.data.YearStore.__init__",
                "v3.data.PublishedWorld.__init__","v3.coordinator.YearCoordinator.__init__",
                "v31.lite_gym.AnnualDecisionEnv.__init__","v31.lite_engine.LiteAnnualDriver.__init__"):
                guards.enter_context(patch(target,side_effect=refuse_world))
            for target in ("stable_baselines3.common.base_class.BaseAlgorithm.__init__",
                "stable_baselines3.common.base_class.BaseAlgorithm.load","stable_baselines3.common.base_class.BaseAlgorithm.save",
                "stable_baselines3.common.policies.BaseModel.load","stable_baselines3.common.policies.BaseModel.save",
                "stable_baselines3.PPO.learn","stable_baselines3.PPO.train","torch.load","torch.save",
                "torch.optim.Optimizer.step","torch.optim.Adam.step","torch.optim.SGD.step"):
                guards.enter_context(patch(target,new=refuse_model_or_update))
            ctx=Context(torch,np,{"upper":HybridUpperPolicy,"lower":MultiInputActorCriticPolicy})
            spec=importlib.util.spec_from_file_location("r02_policy_unit",SRC/"tests/test_research_policy_interface.py")
            tests=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tests;spec.loader.exec_module(tests)
            tests.CONTEXT=ctx
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.PolicyInterfaceTests)
            selected=[t.id() for t in suite]
            phase="policy_interface_checks"
            previous_np_state=np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20261024);np.random.seed(20261024)
                    result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
            finally:
                np.random.set_state(previous_np_state)
            ctx.verify_unchanged()
    except BaseException as error:
        failure=repr(error)
        log.write("\nRUNNER ERROR: "+failure+"\n")
        import traceback
        traceback.print_exc(file=log)
elapsed=time.monotonic()-start
after=identity()
if ctx is not None:
    with artifact_path.open("x",encoding="utf-8") as f:json.dump(ctx.artifacts,f,ensure_ascii=False,indent=2)
summary=dict(tests=result.testsRun if result else 0,failures=len(result.failures) if result else 0,
    errors=len(result.errors) if result else 0,skipped=len(result.skipped) if result else 0,
    seconds=elapsed,runner_error=failure,**counts,
    constructor_attempts=actual["constructor_attempts"],policies_created=actual["policies_created"],
    policy_value_api_calls=actual["policy_value_api_calls"],api_rows=actual["api_rows"])
passed=(result is not None and result.wasSuccessful() and result.testsRun==5 and not failure and before==after
        and not any(counts.values()) and elapsed<=budget["per_invocation_wall_seconds_max"]
        and actual["policies_created"]==2 and actual["policy_value_api_calls"]<=64)
code=0 if passed else 1
receipt=dict(command=[sys.executable,"-I","-B",str(pathlib.Path(__file__).resolve())],
    interpreter=interpreter,dependencies=dependencies,dependency_sources=dependency_sources,
    source_before=before,source_after=after,source_unchanged=before==after,
    scope_id=scope["id"],budgets=budget,selected_tests=selected,result=summary,
    denied_events=denied_events,log=str(log_path),log_sha256=sha(log_path),
    artifacts=str(artifact_path) if artifact_path.exists() else None,
    artifacts_sha256=sha(artifact_path) if artifact_path.exists() else None,
    seed=20261024,cpu_threads=1,tolerances=dict(float32_rtol=1.3e-6,float32_atol=1e-5,same_batch_repeat="bitwise equality"),
    search_path_cleanup=removed_zip,exit_code=code,
    limits="fresh untrained neural unit policies; no market/annual world, checkpoint read/write, optimizer step or learning; no economic result")
with receipt_path.open("x",encoding="utf-8") as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
record=json.loads(RECORD.read_text(encoding="utf-8"))
actual["seconds"]+=elapsed
record["policy_interface_scope"]["actual"]=actual
record["policy_interface_scope"]["state"]="EXECUTED_PASS_PENDING_REVIEW" if passed else "STOPPED_ON_DEFECT_OR_LIMIT"
record["actual"]["test_seconds"]+=elapsed
record["tasks"]["R02"]["checks"].append(dict(kind="current_policy_unit_execution",receipt=str(receipt_path),
    sha256=sha(receipt_path),run_source_identity=before["sha256"],exit_code=code,result=summary))
RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(dict(receipt=str(receipt_path),source_identity=before["sha256"],exit_code=code,result=summary)))
raise SystemExit(code)
