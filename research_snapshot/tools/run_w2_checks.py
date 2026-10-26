"""Execute only W2 artificial fixtures; no neural model/world construction."""
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,time,importlib.util,importlib.metadata,os
from contextlib import ExitStack
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[3]
CANDIDATE=pathlib.Path(__file__).resolve().parents[1]
SRC=CANDIDATE/"src"
sys.path.insert(0,str(SRC))
# A non-existent stdlib zip is a default search placeholder, not an input.
stdlib_zip=pathlib.Path(sys.base_prefix)/("python"+str(sys.version_info.major)+str(sys.version_info.minor)+".zip")
if stdlib_zip.exists():raise RuntimeError("unexpected real stdlib archive; stop rather than read it")
removed_missing_stdlib_zip=[p for p in sys.path if pathlib.Path(p).resolve()==stdlib_zip.resolve()]
sys.path[:]=[p for p in sys.path if pathlib.Path(p).resolve()!=stdlib_zip.resolve()]
RECORD=ROOT/"TASK_RECORD.json"
record=json.loads(RECORD.read_text(encoding="utf-8"))
if record["actual"]["test_invocations"]>=record["budget"]["test_invocations_max"]:
    raise RuntimeError("test invocation limit")
number=record["actual"]["test_invocations"]+1
log_path=ROOT/"logs"/("w2_deterministic_"+str(number)+".log")
receipt_path=log_path.with_suffix(".json")
if log_path.exists() or receipt_path.exists():raise FileExistsError("preserve old evidence")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity():
    paths=sorted(list(SRC.rglob("*.py"))+list((CANDIDATE/"tools").glob("*.py")))
    files={p.relative_to(CANDIDATE).as_posix():sha(p) for p in paths}
    return dict(sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest(),files=files)
before=identity()
dependencies={}
for name in ("numpy","gymnasium","torch","stable-baselines3","scipy","highspy","pulp","cvxpy","ortools"):
    try:dependencies[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:dependencies[name]="NOT_INSTALLED"
sb3=pathlib.Path(sys.prefix)/"Lib/site-packages/stable_baselines3"
dependency_sources={str(sb3/p):sha(sb3/p) for p in ("common/buffers.py","common/distributions.py","common/on_policy_algorithm.py","ppo/ppo.py")}
interpreter=dict(path=sys.executable,version=sys.version,sha256=sha(pathlib.Path(sys.executable)))
record["actual"]["test_invocations"]=number
RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
# Process-local locations only, not environment installation/configuration.
runtime=ROOT/"logs/runtime"
runtime.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(runtime/"matplotlib")
os.environ["TMP"]=os.environ["TEMP"]=str(runtime)
import tempfile
tempfile.tempdir=str(runtime)  # unittest.mock may have initialized the tempfile cache earlier
import platform
native_winver=sys.getwindowsversion().platform_version
# Python 3.11 platform.win32_ver otherwise launches cmd /c ver during Torch DLL import.
# Use the native version API for this read-only host probe; keep subprocess denial.
platform._syscmd_ver=lambda *a,**k:("Windows","", ".".join(map(str,native_winver)))
source=pathlib.Path(record["source_repository"]).resolve()
venv=(source/".venv-ppo").resolve()
counts=dict(forbidden_reads=0,forbidden_writes=0,world_construction_attempts=0,model_or_update_attempts=0)
denied_events=[]
phase="dependency_import"
def denied(reason,p):
    frames=[]
    frame=sys._getframe(2)
    for _ in range(12):
        if frame is None:break
        frames.append(dict(file=frame.f_code.co_filename,line=frame.f_lineno,function=frame.f_code.co_name))
        frame=frame.f_back
    denied_events.append(dict(reason=reason,path=str(p),phase=phase,stack=frames))
def audit(event,args):
    if event in ("subprocess.Popen","os.system","os.posix_spawn"):
        raise RuntimeError("child/background processes forbidden")
    if event!="open" or not args or not isinstance(args[0],(str,bytes)):return
    p=pathlib.Path(args[0]).resolve()
    if p == pathlib.Path(os.devnull).resolve():return  # Windows NUL discards bytes; no file mutation
    mode=args[1];flags=args[2] if len(args)>2 else 0
    writing=(isinstance(mode,str) and any(k in mode for k in "wax+"))
    writing=writing or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
    if writing and not p.is_relative_to(ROOT.resolve()):
        counts["forbidden_writes"]+=1
        denied("write_outside_workspace",p)
        raise RuntimeError("write outside W2: "+str(p))
    if p.is_relative_to(venv):return  # only existing interpreter/dependency read access
    if p.is_relative_to(source) or p.suffix.lower() in (".csv",".npz",".npy",".pkl",".zip",".pt",".pth",".ckpt"):
        counts["forbidden_reads"]+=1
        denied("source_or_artifact_suffix",p)
        raise RuntimeError("original source/data/model read forbidden during checks")
    if any(part.lower() in ("data","outputs","runs","sealed","review_bundles") for part in p.parts):
        counts["forbidden_reads"]+=1
        denied("data_directory_name",p)
        raise RuntimeError("historical/data directory read forbidden")
sys.addaudithook(audit)
def refuse_world(*args,**kwargs):
    counts["world_construction_attempts"]+=1
    raise AssertionError("no real world/environment construction")
def refuse_model(*args,**kwargs):
    counts["model_or_update_attempts"]+=1
    raise AssertionError("no neural model/load/learn/update")
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m
start=time.monotonic()
result=None;failure=None;selected=[]
with log_path.open("x",encoding="utf-8") as log:
    try:
        import unittest,torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        from stable_baselines3.common.base_class import BaseAlgorithm
        from v31 import lite_gym,lite_engine,lite_data
        from v3 import data,coordinator
        with ExitStack() as guards:
            for target in ("v31.lite_data.LiteStore.__init__","v3.data.YearStore.__init__",
                           "v3.data.PublishedWorld.__init__","v3.coordinator.YearCoordinator.__init__",
                           "v31.lite_gym.AnnualDecisionEnv.__init__",
                           "v31.lite_engine.LiteAnnualDriver.__init__"):
                guards.enter_context(patch(target,side_effect=refuse_world))
            for target in ("stable_baselines3.common.base_class.BaseAlgorithm.__init__",
                           "stable_baselines3.common.base_class.BaseAlgorithm.load",
                           "stable_baselines3.PPO.learn","stable_baselines3.PPO.train",
                           "torch.load","torch.optim.Adam.step","torch.optim.SGD.step"):
                guards.enter_context(patch(target,side_effect=refuse_model))
            new=module("w2_objective_tests",SRC/"tests/test_research_objective.py")
            ev=module("w2_existing_ev_tests",SRC/"tests/test_ev_fleet.py")
            prob=module("w2_existing_probability_tests",SRC/"tests/test_v3_hybrid_d8.py")
            suite=unittest.TestSuite()
            for cls in (new.ObjectiveTests,new.ProbabilityAndGAETests,new.TailTests,new.TailExtensionTests,new.IndustrialJointTests,new.TailInterruptFinalizationTests,ev.EVFleetTests):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
            for name in ("test_normalization_and_mask","test_joint_formula_independent",
                         "test_gradients_and_mask_finite_difference","test_invalid_actions_rejected",
                         "test_semantics_first_and_last"):
                suite.addTest(prob.ProbabilityTests(name))
            selected=[test.id() for test in suite]
            class TimedResult(unittest.TextTestResult):
                def startTest(self,test):
                    if time.monotonic()-start>record["budget"]["per_test_wall_seconds_max"]:
                        raise TimeoutError("test wall limit")
                    super().startTest(test)
            phase="selected_tests"
            result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=TimedResult).run(suite)
    except BaseException as error:
        failure=repr(error);log.write("\nRUNNER ERROR: "+failure+"\n")
        import traceback
        traceback.print_exc(file=log)
elapsed=time.monotonic()-start
after=identity()
summary=dict(tests=result.testsRun if result else 0,failures=len(result.failures) if result else 0,
    errors=len(result.errors) if result else 0,skipped=len(result.skipped) if result else 0,
    seconds=elapsed,runner_error=failure,**counts)
passed=(result is not None and result.wasSuccessful() and not failure and before==after
        and not any(counts.values()) and elapsed<=record["budget"]["per_test_wall_seconds_max"])
code=0 if passed else 1
receipt=dict(command=[sys.executable,"-I","-B",str(pathlib.Path(__file__).resolve())],interpreter=interpreter,
    dependencies=dependencies,dependency_sources=dependency_sources,source_before=before,source_after=after,
    source_unchanged=before==after,selected_tests=selected,denied_events=denied_events,log=str(log_path),log_sha256=sha(log_path),
    result=summary,exit_code=code,
    host_probe=dict(source="sys.getwindowsversion().platform_version, no shell",version=list(native_winver)),
    search_path_cleanup=dict(removed=removed_missing_stdlib_zip,reason="nonexistent standard-library archive placeholder; no archive permissions added"),
    scope="Actual formulas/ledger/devices/buffer/collector methods, artificial arrays and fake policy/driver only; model or world constructors and learning blocked. Not full-year execution.")
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
record=json.loads(RECORD.read_text(encoding="utf-8"))
record["actual"]["test_seconds"]+=elapsed
check=dict(kind="current_execution",receipt=str(receipt_path),sha256=sha(receipt_path),log_sha256=receipt["log_sha256"],
    run_source_identity=before["sha256"],command=receipt["command"],exit_code=code,result=summary)
record["tasks"]["R02"]["checks"].append(check)
record["tasks"]["R04"]["checks"].append(dict(reuses=check["receipt"],scope="same run, tail subset; not independent duplicate execution"))
RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(dict(receipt=str(receipt_path),source_identity=before["sha256"],exit_code=code,result=summary)))
raise SystemExit(code)
