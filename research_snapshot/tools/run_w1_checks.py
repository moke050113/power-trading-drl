"""W1 artificial-only checks with a content-bound execution receipt."""
import sys
sys.dont_write_bytecode = True
import pathlib, unittest, importlib.util, importlib.metadata, json, time, hashlib
from contextlib import ExitStack
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[3]
CANDIDATE = pathlib.Path(__file__).resolve().parents[1]
SRC = CANDIDATE / "src"
sys.path.insert(0, str(SRC))
RECORD = ROOT / "TASK_RECORD.json"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_identity():
    paths = sorted(list(SRC.rglob("*.py")) + list((CANDIDATE / "tools").glob("*.py")))
    files = {p.relative_to(CANDIDATE).as_posix(): digest(p) for p in paths}
    key = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": key, "files": files}

before = source_identity()
dependencies = {}
for name in ("numpy", "gymnasium", "torch", "stable-baselines3"):
    try:
        dependencies[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        dependencies[name] = "NOT_INSTALLED"
interpreter = {"path": sys.executable, "version": sys.version, "sha256": digest(pathlib.Path(sys.executable))}
record = json.loads(RECORD.read_text(encoding="utf-8"))
if record["actual"]["test_invocations"] >= record["budget"]["test_invocations_max"]:
    raise RuntimeError("W1 test invocation budget consumed")
number = record["actual"]["test_invocations"] + 1
LOG = ROOT / "logs"
LOG.mkdir(exist_ok=True)
log_path = LOG / ("w1_boundary_" + str(number) + ".log")
receipt_path = LOG / ("w1_boundary_" + str(number) + ".json")
if log_path.exists() or receipt_path.exists():
    raise RuntimeError("refuse evidence overwrite")
record["actual"]["test_invocations"] = number
RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
counts = {"source_repository_read_attempts": 0, "data_or_model_read_attempts": 0,
          "environment_call_attempts": 0, "project_model_import_attempts": 0}
forbidden_names = ("torch", "stable_baselines3")
source_repository = pathlib.Path(record["source_repository"]).resolve()
allowed_write = ROOT.resolve()

def audit(event, args):
    if event == "import" and args and str(args[0]).split(".")[0] in forbidden_names:
        counts["project_model_import_attempts"] += 1
        raise RuntimeError("W1 forbids model library import")
    if event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
        raise RuntimeError("W1 tests cannot start child processes")
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    p = pathlib.Path(args[0]).resolve()
    mode, flags = args[1], args[2] if len(args) > 2 else 0
    writing = isinstance(mode, str) and any(c in mode for c in "wax+")
    if isinstance(flags, int):
        import os
        writing = writing or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if writing and not p.is_relative_to(allowed_write):
        raise RuntimeError("write outside W1 workspace")
    if p.is_relative_to(source_repository):
        counts["source_repository_read_attempts"] += 1
        raise RuntimeError("test tried to read original repository")
    if p.suffix.lower() in (".csv", ".npz", ".npy", ".pkl", ".zip", ".pt", ".pth", ".ckpt"):
        counts["data_or_model_read_attempts"] += 1
        raise RuntimeError("W1 tests forbid data/model reads")
    if any(c.lower() in ("data", "outputs", "runs", "sealed", "review_bundles") for c in p.parts):
        counts["data_or_model_read_attempts"] += 1
        raise RuntimeError("W1 tests forbid historical/data directories")

sys.addaudithook(audit)

def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def refuse_environment(*args, **kwargs):
    counts["environment_call_attempts"] += 1
    raise AssertionError("W1 forbids environment construction or steps")

start = time.monotonic()
failure = None
result = None
with log_path.open("x", encoding="utf-8") as log:
    try:
        new = module_from_file("w1_boundary_tests", SRC / "tests/test_research_input_boundary.py")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(new.ResearchInputBoundaryTests)
        old = module_from_file("w1_existing_lite_tests", SRC / "tests/test_v31_lite.py")
        suite.addTest(old.LiteTests("test_test_gate_before_history_or_arrays"))
        with ExitStack() as guards:
            for target in ("v31.lite_data.day_values", "v31.lite_engine.LiteAnnualDriver",
                           "v31.lite_gym.LiteAnnualDriver",
                           "v31.lite_gym.AnnualDecisionEnv.__init__",
                           "v31.lite_gym.AnnualDecisionEnv.step"):
                guards.enter_context(patch(target, side_effect=refuse_environment))
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    except BaseException as error:
        failure = repr(error)
        log.write("\nRUNNER ERROR: " + failure + "\n")
elapsed = time.monotonic() - start
after = source_identity()
summary = dict(tests=result.testsRun if result else 0, failures=len(result.failures) if result else 0,
               errors=len(result.errors) if result else 0, skipped=len(result.skipped) if result else 0,
               seconds=elapsed, runner_error=failure, **counts,
               imported_model_libraries=[n for n in sys.modules if n.split(".")[0] in forbidden_names])
passed = (result is not None and result.wasSuccessful() and not failure and before == after
          and elapsed <= record["budget"]["per_test_wall_seconds_max"]
          and not any(counts.values()) and not summary["imported_model_libraries"])
exit_code = 0 if passed else 1
receipt = dict(command=[sys.executable, "-I", "-B", str(pathlib.Path(__file__).resolve())],
               interpreter=interpreter, dependencies=dependencies, source_before=before,
               source_after=after, source_unchanged=before == after, result=summary,
               log=str(log_path), log_sha256=digest(log_path), exit_code=exit_code,
               limits="artificial and negative-access tests; environment zero also statically checked, not OS sandbox")
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
record = json.loads(RECORD.read_text(encoding="utf-8"))
record["actual"]["test_seconds"] += elapsed
record["tasks"]["R01"]["checks"].append(dict(kind="current_execution",
    command=receipt["command"], receipt=str(receipt_path), receipt_sha256=digest(receipt_path),
    log=str(log_path), log_sha256=receipt["log_sha256"], run_source_sha256=before["sha256"],
    exit_code=exit_code, result=summary))
RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(receipt=str(receipt_path), run_source_sha256=before["sha256"],
                     source_unchanged=before == after, exit_code=exit_code, result=summary)))
raise SystemExit(exit_code)
