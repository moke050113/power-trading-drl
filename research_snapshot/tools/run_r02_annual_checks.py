"""R02 single approved annual fixture worker, supervised by a hard timeout."""
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
PREFIX = ROOT / "logs/r02_annual_fixture_1"
OUTPUT_LIMITS = dict(raw=63*1024*1024, log=256*1024,
                     worker_receipt=512*1024, supervisor_receipt=128*1024)
assert sum(OUTPUT_LIMITS.values()) <= 64*1024*1024
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
    record = read(RECORD); scope = record["annual_fixture_scope"]
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
    remaining = 360 - (time.monotonic()-started)
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
    record = read(RECORD); scope = record["annual_fixture_scope"]
    scope["observed_supervised_wall_seconds"] = worker_seconds
    scope["observed_process_exit_code"] = code
    if status != "WORKER_EXITED" or code != 0:
        scope["state"] = "STOPPED_FAILURE_NO_AUTOMATIC_REPAIR"
    # A killed worker cannot finalize RAM counters. Recover durable attempts.
    counts = {}; journal = pathlib.Path(str(PREFIX)+"_events.jsonl")
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try: row = json.loads(line)
            except ValueError: continue
            if row.get("event") == "attempt": counts[row["kind"]] = row["count"]
    scope["actual"]["durable_attempt_counts"] = counts
    scope["actual"]["output_bytes"] = sum(p.stat().st_size for p in paths if p.exists())
    store_record(record)
    out = dict(command=command, status=status, worker_exit_code=code,
               supervised_wall_seconds=worker_seconds, error=error,
               counts=counts, limits=scope["budgets"],
               note="worker compute hard capped360s; post-stop receipt finalization separately measured")
    out["finalization_seconds_before_receipt"] = time.monotonic()-started-worker_seconds
    encoded=json.dumps(out,indent=2)+"\n"
    if len(encoded.encode("utf-8")) > OUTPUT_LIMITS["supervisor_receipt"]:
        raise RuntimeError("supervisor receipt cap before write")
    with paths[-1].open("x", encoding="utf-8", newline="\n") as f: f.write(encoded)
    record=read(RECORD)
    record["annual_fixture_scope"]["actual"]["output_bytes"]=sum(
        p.stat().st_size for p in paths if p.exists())
    store_record(record)
    print(json.dumps(dict(status=status, exit_code=code, seconds=worker_seconds,
                          counts=counts, receipt=str(PREFIX)+".json")))
    return 0 if code == 0 and status == "WORKER_EXITED" else 1

def worker(nonce):
    start = time.monotonic()
    record = read(RECORD); scope = record["annual_fixture_scope"]
    if scope.get("approval_state") != "APPROVED" or scope["state"] != "RUNNING" or scope["worker_nonce"] != nonce:
        raise PermissionError("unapproved worker")
    before = source_id()
    runtime = ROOT/"logs/annual_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(runtime)
    tempfile.tempdir = str(runtime)
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
    guard = dict(forbidden_reads=0, forbidden_writes=0, processes=0, neural_imports=0)
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
    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("torch","stable_baselines3","agents","experiments","diagnostics_v3"):
                guard["neural_imports"] += 1
                raise PermissionError("neural/legacy entry import denied")
    sys.addaudithook(audit); sys.meta_path.insert(0, NoNeural())
    class Context:
        def __init__(self):
            self.counts = {}; self.bytes = 0; self.environments=[]; self.current = None
            self.journal = paths["_events.jsonl"].open("x", encoding="utf-8", newline="\n")
            self.views = paths["_views.jsonl"].open("x", encoding="utf-8", newline="\n")
            self.sequence = 0
        def default(self, x):
            if hasattr(x,"isoformat"): return x.isoformat()
            if hasattr(x,"tolist"): return x.tolist()
            raise TypeError(type(x).__name__)
        def emit(self, stream, obj):
            line = json.dumps(obj, default=self.default, ensure_ascii=False, allow_nan=False)+"\n"
            size = len(line.encode("utf-8"))
            # Reserve1MiB for final receipt/log/metadata, never overwrite raw outputs.
            if self.bytes+size > OUTPUT_LIMITS["raw"]: raise RuntimeError("raw evidence cap")
            stream.write(line); stream.flush(); self.bytes += size
        def check_deadline(self):
            if time.monotonic()-start >= 355: raise TimeoutError("cooperative pre-stop; supervisor hard limit360")
        def trace(self, event, **extra):
            self.sequence += 1
            d = self.current
            row = dict(sequence=self.sequence, event=event, **extra)
            if d is not None:
                row.update(view=d._r02_level,t=d.c.t,at=d.c.now.isoformat(),
                           ledger_version=d.c.ledger.version,lot_count=len(d.c.ledger.lots))
            self.emit(self.journal,row)
        def charge(self, kind, cap):
            self.check_deadline()
            n = self.counts.get(kind,0)
            if n >= cap: raise RuntimeError("total cap reached: "+kind)
            self.counts[kind] = n+1
            self.trace("attempt",kind=kind,count=n+1)
        def new_environment(self, spec, level, partner):
            if spec != scope["spec"] or level != ("upper","lower")[len(self.environments)]:
                raise PermissionError("unique ordered fixture views required")
            policy = input_boundary.ResearchInputPolicy([spec])
            e = input_boundary.build_environment([spec],policy=policy,level=level,
                                                  seed=20261024,partner=partner)
            if e.driver is not None: raise RuntimeError("shell constructor unexpectedly reset")
            self.environments.append(e)
            return e
        def record_view(self, level, details, hours):
            d = self.current
            details["origins"] = d.context.origins
            details["final_state"] = dict(terminated=self.environments[-1].ended,
                pending=d.c.pending is not None, completed_hour=d.c.t,
                unfulfilled_fee=float(d.c.ledger.fee[d.c.t:].sum()))
            self.emit(self.views,dict(view=level,details=details,hours=hours,
                reused_hours_from="upper" if hours is None else None))
        def assert_total_usage(self, **expected):
            for k,v in {"stores":1,"replays":1,"drivers":2,"envs":2,
                        "coordinators":2,"published_worlds":4,"H_START":17520,"H_END":17520}.items():
                if self.counts.get(k)!=v: raise AssertionError((k,self.counts.get(k),v))
        def close(self):
            self.journal.close(); self.views.close()
    ctx = Context(); result = None; failure = None; tests = []
    log = CappedLog(paths[".log"])
    try:
        import unittest, numpy as np
        from research import input_boundary
        from v31 import lite_gym, lite_data, lite_engine, lite_rules
        from v3 import data, coordinator
        from v31 import history
        # Every replaced method is restored after the only run.
        with ExitStack() as stack:
            def blocked(*a,**kw): raise PermissionError("legacy persistence/data generator denied")
            for target in ("v3.data.YearStore.__init__","v3.data.generate",
                           "v31.history.save_new","v31.history.load_records"):
                stack.enter_context(patch(target,new=blocked))
            def install(cls, name, kind, cap, extra=None):
                original_method = getattr(cls,name)
                def wrapped(obj,*args,**kwargs):
                    ctx.charge(kind,cap)
                    if kind=="stores":
                        bound = dict(zip(("year","family","group"),args))
                        bound.update(kwargs)
                        if bound != scope["spec"]: raise PermissionError("store SPEC mismatch")
                    if kind=="replays":
                        if not obj.completed_year or len(obj.context.sealed)!=365:
                            raise PermissionError("only complete audited replay")
                    value = original_method(obj,*args,**kwargs)
                    if kind=="drivers":
                        obj._r02_level=("upper","lower")[ctx.counts[kind]-1];ctx.current=obj
                    if extra: extra(obj,value)
                    return value
                stack.enter_context(patch.object(cls,name,new=wrapped))
            install(lite_data.LiteStore,"__init__","stores",1)
            install(lite_data.LiteStore,"replay_copy","replays",1)
            install(lite_gym.AnnualDecisionEnv,"__init__","envs",2)
            install(lite_engine.LiteAnnualDriver,"__init__","drivers",2)
            install(coordinator.YearCoordinator,"__init__","coordinators",2)
            install(data.PublishedWorld,"__init__","published_worlds",4)
            def after_start(obj,value): ctx.trace("H_START")
            def after_end(obj,value): ctx.trace("H_END",settled_t=value["t"])
            def after_macro(obj,value): ctx.trace("MACRO",receipt=obj.macros[-1])
            install(coordinator.YearCoordinator,"prepare_hour","H_START",17520,after_start)
            install(coordinator.YearCoordinator,"settle_pending","H_END",17520,after_end)
            install(coordinator.YearCoordinator,"submit_macro","macros",196,after_macro)
            install(lite_gym.AnnualDecisionEnv,"step","env_steps",8858)
            # Instrument event order without changing scheduling or forecasts.
            orig_pub=lite_engine.LiteAnnualDriver._publication
            def publication(d,event,index):
                ctx.charge("publications",1656)
                value=orig_pub(d,event,index)
                tag=("op:" if event=="macro_publish" else "day:")+str(index)
                rows=d.context.records.get(tag,())
                origins=[o for o in d.context.origins if o["tag"]==tag]
                if len(origins)!=1: raise AssertionError("missing/duplicate publication origin")
                origin=origins[0]
                if event!="DA_seal":
                    if origin["published_at"]!=d.c.now.isoformat():
                        raise AssertionError("publication origin time mismatch")
                    if origin["forecasts"]!=[r.content_sha256 for r in rows]:
                        raise AssertionError("publication origin/vintage mismatch")
                ctx.trace(event,index=index,tag=tag,
                    vintages=[dict(sha=r.content_sha256,published_at=r.published_at,
                                   as_of=r.as_of) for r in rows],
                    origin=origin if event!="DA_seal" else None)
                return value
            stack.enter_context(patch.object(lite_engine.LiteAnnualDriver,"_publication",new=publication))
            original_rules=lite_rules.RULES["B2"]
            def rule(level, original_rule, cap):
                def act(obs):
                    ctx.charge("rule_"+level,cap)
                    if level=="lower":
                        h=int(obs["hour"][0])
                        expected=np.r_[np.ones(h),np.zeros(24-h)].astype(np.float32)
                        if not np.array_equal(obs["history_mask"],expected):
                            raise AssertionError("history mask not completed hours")
                    ctx.trace("RULE_"+level)
                    return original_rule(obs)
                return act
            stack.enter_context(patch.dict(lite_rules.RULES,{"B2":(
                rule("upper",original_rules[0],196),rule("lower",original_rules[1],17520))}))
            spec=importlib.util.spec_from_file_location("r02_annual_unit",SRC/"tests/test_research_annual_chain.py")
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            module.RUN_CONTEXT=ctx
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.AnnualChainTests)
            tests=[t.id() for t in suite]
            result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
            ctx.check_deadline()
    except BaseException as exc:
        failure=repr(exc)[:4096]
        try: traceback.print_exc(file=log)
        except BaseException: failure+="; exception log capped or unavailable"
    finally:
        ctx.close();log.close()
    after=source_id()
    success=(result is not None and result.wasSuccessful() and not failure and before==after
             and not any(guard.values()))
    receipt=dict(source_before=before,source_after=after,source_unchanged=before==after,
        selected_tests=tests,tests=result.testsRun if result else 0,
        failures=len(result.failures) if result else 0,errors=len(result.errors) if result else 0,
        skipped=len(result.skipped) if result else 0,runner_error=failure,guard=guard,
        denied=denied,counts=ctx.counts,worker_seconds=time.monotonic()-start,
        exit_code=0 if success else 1,
        artifacts={str(p):sha(p) for s,p in paths.items() if s!=".json"},
        interpreter=dict(path=sys.executable,version=sys.version,sha256=sha(pathlib.Path(sys.executable))),
        dependency_sources={str(p):sha(p) for p in (
            pathlib.Path(sys.prefix)/"Lib/site-packages/numpy/__init__.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/gymnasium/__init__.py",
            pathlib.Path(sys.prefix)/"Lib/site-packages/gymnasium/spaces/box.py")},
        dependencies={n:importlib.metadata.version(n) for n in ("numpy","gymnasium")},
        limits=scope["budgets"],
        claim_scope="one deterministic ordinary-year G00 fixture, two B2 decision views; no PPO learning/economic result")
    encoded=json.dumps(receipt,ensure_ascii=False,indent=2)
    if len((encoded+"\n").encode("utf-8"))>OUTPUT_LIMITS["worker_receipt"]:
        raise RuntimeError("worker receipt cap before write")
    with paths[".json"].open("x",encoding="utf-8",newline="\n") as f:f.write(encoded+"\n")
    scope["state"]="EXECUTED_PASS_PENDING_REVIEW" if success else "STOPPED_FAILURE_NO_AUTOMATIC_REPAIR"
    scope["actual"].update(counts=ctx.counts,worker_seconds=receipt["worker_seconds"],tests=receipt["tests"])
    record["annual_fixture_scope"]=scope;store_record(record)
    return receipt["exit_code"]

if __name__=="__main__":
    if len(sys.argv)==1:sys.exit(supervisor())
    if len(sys.argv)==3 and sys.argv[1]=="--worker":sys.exit(worker(sys.argv[2]))
    raise PermissionError("unsupported invocation")
