"""W3 approved finite owner. Importing this file never imports project code."""
import argparse
import builtins
from contextlib import contextmanager
import copy
import ctypes
import datetime
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

CANDIDATE = Path(__file__).resolve().parents[1]
ROOT = CANDIDATE.parents[1]
RECORD = ROOT / "TASK_RECORD.json"
CONFIG_SHA = "3238304aa13ade077fd30ff6463d2d507ae8ee0c7b60ec8d5701b3bb7650a97e"
ANSWER = "\u6279\u51c6\u8be5R05\u5355\u6b21\u5148\u5bfc\u5305"
INTERPRETER = Path(r"D:\Project\power-trading-drl\.venv-ppo\Scripts\python.exe")
RESERVED_SUPERVISOR_BYTES = 256 * 1024
RESERVED_WORKER_RECEIPT_BYTES = 1024 * 1024

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def save_record(value):
    RECORD.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def source_identity():
    manifest = read(CANDIDATE / "MAIN_VERSION_MANIFEST.json")
    files = {name: sha(CANDIDATE / name) for name in manifest["candidate_files"]}
    if files != manifest["candidate_files"]:
        raise RuntimeError("fixed candidate content changed")
    return dict(files=files, sha256=hashlib.sha256(canonical(files).encode()).hexdigest())

def check_authority():
    record = read(RECORD)
    config_path = CANDIDATE / "configs/r05_pilot.json"
    if sha(config_path) != CONFIG_SHA:
        raise PermissionError("unapproved pilot configuration")
    cfg = read(config_path)
    auth, pkg = record["authorization"], record["r05_package"]
    if auth["approval_state"] != "APPROVED" or auth["approval_answer"] != ANSWER:
        raise PermissionError("explicit package approval required")
    if canonical(pkg["approved_budgets"]) != canonical(cfg["proposed_budgets"]):
        raise PermissionError("exact typed approved budgets required")
    if pkg["config_sha256"] != CONFIG_SHA:
        raise PermissionError("record/config binding mismatch")
    if record["candidate"] != CANDIDATE.name:
        raise PermissionError("wrong fixed candidate")
    mpath = CANDIDATE / "MAIN_VERSION_MANIFEST.json"
    if sha(mpath) != record["candidate_manifest_sha256"]:
        raise PermissionError("fixed manifest mismatch")
    fixed = source_identity()
    manifest = read(mpath)
    for name, expected in manifest["dependency_sources"].items():
        if sha(Path(name)) != expected:
            raise PermissionError("selected dependency changed: " + name)
    if sha(INTERPRETER) != "95e79a636ae976afe22672655b48189a4067a3da931b307072306ffcaf51f69c":
        raise PermissionError("approved interpreter changed")
    for item in pkg.get("code_review_records", []):
        p = Path(item["path"])
        if sha(p) != item["sha256"]:
            raise PermissionError("code review changed")
        review = read(p)
        if (review.get("verdict") != "PASS" or
            review.get("manifest_sha256") != record["candidate_manifest_sha256"] or
            review.get("config_sha256") != CONFIG_SHA or
            review.get("scope") != "R05_FIXED_CODE"):
            raise PermissionError("matching implementation review required")
    reviewers = {read(Path(x["path"]))["reviewer"] for x in pkg.get("code_review_records", [])}
    if reviewers != {"/root/w1_independent_review", "/root/w1_algorithm_review"}:
        raise PermissionError("two actual fixed-code reviews required")
    return record, cfg, fixed

class ProcessMemory:
    """Windows native live RSS measurement; no shell/process helpers."""
    def __init__(self, pid):
        from ctypes import wintypes
        class Counts(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (k, ctypes.c_size_t) for k in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        self.Counts = Counts
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counts), wintypes.DWORD]
        self.handle = self.kernel.OpenProcess(0x0410, False, pid)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess for RSS failed")
    def sample(self):
        c = self.Counts()
        c.cb = ctypes.sizeof(c)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(c), c.cb):
            raise OSError(ctypes.get_last_error(), "RSS query failed")
        return dict(rss_bytes=int(c.WorkingSetSize), peak_rss_bytes=int(c.PeakWorkingSetSize))
    def close(self):
        self.kernel.CloseHandle(self.handle)

def supervisor(mode):
    record, cfg, fixed = check_authority()
    pkg, budget = record["r05_package"], cfg["proposed_budgets"]
    actual = pkg["actual"]
    key = mode + "_invocations"
    if actual.get(key, 0) != 0:
        raise PermissionError("single attempt already consumed; no retries")
    if mode == "pilot":
        if pkg["state"] != "PREFLIGHT_PASS":
            raise PermissionError("supervised preflight PASS required")
        supref = pkg.get("supervisor_preflight")
        if not supref or sha(Path(supref["path"])) != supref["sha256"]:
            raise PermissionError("fixed preflight supervisor required")
        supervision = read(Path(supref["path"]))
        if supervision["status"] != "PASS" or supervision["exit_code"] != 0:
            raise PermissionError("preflight supervision rejected")
        ref = pkg.get("preflight_receipt")
        if not ref or sha(Path(ref["path"])) != ref["sha256"]:
            raise PermissionError("exact preflight evidence required")
        previous = read(Path(ref["path"]))
        if (previous["status"] != "PASS" or previous["source_before"] != fixed
                or previous["source_after"] != fixed or previous["mode"] != "preflight"):
            raise PermissionError("same-source preflight PASS required")
    elif pkg["state"] != "READY_FOR_PREFLIGHT":
        raise PermissionError("preflight fixed candidate not released")
    attempt = ROOT / "logs" / (mode + "_1")
    attempt.mkdir(exist_ok=False)
    (attempt / "evidence").mkdir()
    (attempt / "cache").mkdir()
    nonce = os.urandom(16).hex()
    cache_existing = sum(p.stat().st_size for name in ("preflight_1", "pilot_1")
                         for p in (ROOT / "logs" / name / "cache").rglob("*") if p.is_file())
    if cache_existing >= budget["dependency_temp_cache_bytes_max"]:
        raise PermissionError("cumulative cache budget exhausted")
    actual[key] = 1
    pkg["cache_remaining"] = budget["dependency_temp_cache_bytes_max"] - cache_existing
    pkg["cache_bytes_before_attempt"] = cache_existing
    pkg.update(state="RUNNING_" + mode.upper(), worker_nonce=nonce,
               worker_mode=mode, attempt=str(attempt),
               output_remaining=budget["evidence_output_bytes_max"] -
                                actual.get("evidence_output_bytes", 0))
    if pkg["output_remaining"] <= RESERVED_WORKER_RECEIPT_BYTES + RESERVED_SUPERVISOR_BYTES:
        raise PermissionError("insufficient reserved output budget")
    record["run_version"] = dict(candidate=CANDIDATE.name,
        candidate_identity=record["candidate_identity"],
        manifest_sha256=record["candidate_manifest_sha256"],
        config_sha256=CONFIG_SHA, source_before=fixed, mode=mode)
    save_record(record)  # Consume BEFORE launching, including failed launches.
    command = [str(INTERPRETER), "-I", "-B", str(Path(__file__).resolve()),
               "--worker", mode, nonce]
    seconds_limit = budget["preflight_hard_seconds_max"] if mode == "preflight" else budget["pilot_worker_hard_seconds_max"]
    start = time.monotonic()
    process = None
    memory = None
    peak = 0
    stopped = None
    first_control = None
    stdout = stderr = b""
    code = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=subprocess.CREATE_NO_WINDOW, cwd=str(CANDIDATE))
        memory = ProcessMemory(process.pid)
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                break
            current = memory.sample()
            peak = max(peak, current["peak_rss_bytes"])
            if max(current["rss_bytes"], current["peak_rss_bytes"]) > budget["pilot_rss_bytes_max"]:
                stopped = "RSS_CAP"
                process.kill()
                break
            if time.monotonic() - start >= seconds_limit:
                stopped = "HARD_TIME_CAP"
                process.kill()
                break
            try:
                remaining = seconds_limit - (time.monotonic() - start)
                if remaining <= 0:
                    stopped = "HARD_TIME_CAP"
                    process.kill()
                    break
                stdout, stderr = process.communicate(timeout=min(.25, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
        if stopped:
            stdout, stderr = process.communicate()
        code = process.returncode
    except BaseException as error:
        if not isinstance(error, Exception) and first_control is None:
            first_control = type(error).__name__ + ": " + str(error)[:2048]
        stopped = first_control or (type(error).__name__ + ": " + str(error)[:2048])
        if process is not None:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate()
            code = process.returncode
    finally:
        if memory is not None:
            try:
                memory.close()
            except BaseException as error:
                if not isinstance(error, Exception) and first_control is None:
                    first_control = type(error).__name__ + ": " + str(error)[:2048]
                stopped = first_control or stopped or ("MONITOR_CLEANUP: " + repr(error)[:2048])
    elapsed = time.monotonic() - start
    if elapsed > seconds_limit:
        stopped = stopped or "FINAL_WALL_CAP"
    if peak > budget["pilot_rss_bytes_max"]:
        stopped = stopped or "FINAL_PEAK_RSS_CAP"
    # stdout/stderr are not scientific records; never persist an unbounded stream.
    sup = dict(mode=mode, status="STOPPED", command=command, exit_code=code,
        seconds=elapsed, observed_peak_rss_bytes=peak, stop_reason=stopped,
        stdout=stdout[:32768].decode("utf-8", errors="replace"),
        stderr=stderr[:32768].decode("utf-8", errors="replace"))
    if len(stdout) > 32768 or len(stderr) > 32768:
        sup["stop_reason"] = "UNEXPECTED_PROCESS_OUTPUT_CAP"
    receipt_path = attempt / "evidence/worker_receipt.json"
    result = None
    if receipt_path.exists():
        try:
            result = read(receipt_path)
        except (ValueError, OSError) as exc:
            sup["stop_reason"] = "UNREADABLE_PARTIAL_WORKER_RECEIPT: " + repr(exc)[:2048]
    if result is not None:
        peak = max(peak, result.get("observed_peak_rss_bytes", 0))
        sup["observed_peak_rss_bytes"] = peak
        if peak > budget["pilot_rss_bytes_max"]:
            sup["stop_reason"] = "FINAL_WORKER_PEAK_RSS_CAP"
    if (not sup["stop_reason"] and code == 0 and result is not None
            and result["status"] == "PASS" and result["source_before"] == fixed
            and result["source_after"] == fixed and
            all(v == 0 for v in result["violations"].values())):
        sup["status"] = "PASS"
    if sup["status"] != "PASS":
        sup["liability_note"] = "Use last durable boundary; state beyond it UNKNOWN, never assume zero."
    outputs = [p for p in (attempt / "evidence").rglob("*") if p.is_file()]
    cache_bytes = sum(p.stat().st_size for p in (attempt / "cache").rglob("*") if p.is_file())
    sup["evidence_bytes_before_supervisor"] = sum(p.stat().st_size for p in outputs)
    sup["cache_bytes"] = cache_bytes
    # Recover only durable pre-call attempt reservations; never infer completion.
    durable_counts = {}
    journal_path = attempt / "evidence/attempts.jsonl"
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                durable_counts[row["kind"]] = row["count"]
            except (ValueError, KeyError):
                break
    sup["durable_attempt_counts"] = durable_counts
    sup["counts_completion_scope"] = "worker final receipt" if result else "last durable pre-call reservations; actual completion UNKNOWN"
    sup["cumulative_cache_bytes"] = pkg["cache_bytes_before_attempt"] + cache_bytes
    if (sup["evidence_bytes_before_supervisor"] + RESERVED_SUPERVISOR_BYTES > pkg["output_remaining"]
            or sup["cumulative_cache_bytes"] > budget["dependency_temp_cache_bytes_max"]):
        sup["status"] = "STOPPED"
        sup["stop_reason"] = "FINAL_STORAGE_CAP"
    if sup["status"] != "PASS":
        sup["liability_note"] = "State beyond last durable boundary UNKNOWN; no automatic tail, repair, retry or reset."
    # Freeze status only after all cap checks. No contradictory PASS receipt.
    encoded = (json.dumps(sup, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(encoded) > RESERVED_SUPERVISOR_BYTES:
        raise RuntimeError("supervisor reserve exhausted")
    sup_path = attempt / "supervisor.json"
    with sup_path.open("xb") as f:
        f.write(encoded)
    evidence_bytes = sum(p.stat().st_size for p in outputs) + len(encoded)
    latest = read(RECORD)
    target = latest["r05_package"]
    target["actual"]["evidence_output_bytes"] = target["actual"].get("evidence_output_bytes", 0) + evidence_bytes
    target["actual"]["cache_bytes"] = sup["cumulative_cache_bytes"]
    target["actual"]["durable_counts_" + mode] = durable_counts
    target["actual"]["process_seconds"] = target["actual"].get("process_seconds", 0) + elapsed
    target["actual"]["observed_peak_rss_bytes"] = max(target["actual"].get("observed_peak_rss_bytes", 0), peak)
    target["actual"].setdefault("attempts", []).append(dict(mode=mode, exit_code=code,
        status=sup["status"], seconds=elapsed, evidence_bytes=evidence_bytes,
        receipt=str(receipt_path) if result else None))
    if result:
        target["actual"]["last_worker_counts"] = result["counts"]
        target["actual"]["last_worker_state"] = result["status"]
        counts = result["counts"]
        target["actual"].setdefault("counts_by_mode", {})[mode] = counts
        target["actual"]["project_entry_invocations"] = sum(
            1 for x in target["actual"]["attempts"] if x["receipt"] is not None)
        target["actual"].pop("project_imports", None)
        target["actual"]["project_import_scope"] = "approved entry invocations, not a count of Python import statements"
        aliases = {"new_worlds": "new_native_worlds", "replay_copies": "audited_world_replay_copies",
                   "models": "PPO_controllers", "policy_API_calls": "policy_value_API_calls",
                   "model_file_reads": "new_checkpoint_loads", "model_file_writes": "new_checkpoint_saves",
                   "delivered_hours": "delivered_hours"}
        if mode == "pilot":
            for public, key in aliases.items():
                target["actual"][public + "_attempts"] = counts.get(key, 0)
                target["actual"][public] = counts.get(key, 0) if result["status"] == "PASS" else "UNKNOWN_COMPLETION; see attempts and durable boundary"
            target["actual"]["optimizer_updates"] = counts.get("optimizer_steps_completed", 0)
            target["actual"]["training_updates"] = counts.get("optimizer_steps_completed", 0)
            target["actual"]["update_scope"] = "completed Adam steps; PPO.train calls and epochs separately counted"
    ref = dict(path=str(receipt_path), sha256=sha(receipt_path)) if result else None
    target[mode + "_receipt"] = ref
    target["supervisor_" + mode] = dict(path=str(sup_path), sha256=sha(sup_path))
    target["state"] = ("PREFLIGHT_PASS" if mode == "preflight" else "PILOT_EXECUTED_PENDING_REVIEW") if sup["status"] == "PASS" else "STOPPED_NO_REPAIR_OR_RETRY"
    latest["queue_state"] = target["state"]
    save_record(latest)
    print(json.dumps(dict(status=sup["status"], mode=mode, seconds=elapsed,
                          worker_receipt=ref, peak_rss_bytes=peak)))
    return 0 if sup["status"] == "PASS" else 1

def worker(mode, nonce):
    record, cfg, fixed = check_authority()
    pkg = record["r05_package"]
    if (pkg.get("state") != "RUNNING_" + mode.upper()
            or pkg.get("worker_nonce") != nonce or pkg.get("worker_mode") != mode):
        raise PermissionError("matching consumed owner launch required")
    attempt = Path(pkg["attempt"])
    evidence, cache = attempt / "evidence", attempt / "cache"
    if attempt != ROOT / "logs" / (mode + "_1"):
        raise PermissionError("wrong isolated attempt path")
    budget = cfg["proposed_budgets"]
    started = time.monotonic()
    hard = budget["preflight_hard_seconds_max"] if mode == "preflight" else budget["pilot_worker_hard_seconds_max"]
    os.environ["TMP"] = os.environ["TEMP"] = str(cache)
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torch")
    tempfile.tempdir = str(cache)
    pv = sys.getwindowsversion().platform_version
    platform._syscmd_ver = lambda *a, **k: ("Windows", "", ".".join(map(str, pv)))
    missing_zip = Path(sys.base_prefix) / ("python" + str(sys.version_info.major) + str(sys.version_info.minor) + ".zip")
    if missing_zip.exists():
        raise RuntimeError("unexpected real stdlib archive")
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != missing_zip.resolve()]
    sys.path.insert(0, str(CANDIDATE / "src"))
    envroot = Path(sys.prefix).resolve()
    original = Path(record["source_repository"]).resolve()
    violations = dict(reads=0, writes=0, processes=0, network=0, legacy_imports=0)
    denied = []
    checkpoint = evidence / "model.zip"
    checkpoint_access = {"read": False, "write": False}
    def reject(kind, path):
        violations[kind] += 1
        denied.append(str(path))
        raise PermissionError(kind + " denied: " + str(path))
    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
            reject("processes", event)
        if event in ("socket.connect", "socket.bind", "socket.getaddrinfo"):
            reject("network", event)
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
            for value in (args[:2] if event == "os.rename" else args[:1]):
                if isinstance(value, (str, bytes)):
                    path = Path(os.fsdecode(value)).resolve()
                    if not path.is_relative_to(cache):
                        reject("writes", path)
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path == Path(os.devnull).resolve():
            return
        value, flags = args[1], args[2] if len(args) > 2 else 0
        writing = (isinstance(value, str) and any(x in value for x in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            if path == checkpoint and checkpoint_access["write"]:
                return
            if path.suffix.lower() in (".zip", ".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"):
                reject("writes", path)
            if not (path.is_relative_to(evidence) or path.is_relative_to(cache)):
                reject("writes", path)
            return
        if path == checkpoint:
            if checkpoint_access["read"]:
                return
            reject("reads", path)
        if path.is_relative_to(envroot):
            return
        if path.is_relative_to(original) or path.suffix.lower() in (
                ".zip", ".npz", ".npy", ".csv", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"):
            reject("reads", path)
        if any(x.lower() in ("data", "runs", "outputs", "sealed", "review_bundles") for x in path.parts):
            reject("reads", path)
    class NoLegacy(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if mode == "preflight" and fullname.split(".")[0] in (
                    "torch", "numpy", "gymnasium", "stable_baselines3", "v3", "v31", "experiments"):
                reject("legacy_imports", "real-compute import during fake preflight: " + fullname)
            if (fullname in ("v31.lite_training", "v31.r_training", "v31.s_training",
                             "v31.r_env", "v31.lite_stage_a", "agents.train_ppo")
                    or fullname.startswith("experiments.train_")):
                reject("legacy_imports", fullname)
    sys.addaudithook(audit)
    sys.meta_path.insert(0, NoLegacy())
    raw_io, raw_open, raw_os_open, raw_os_write = io.open, builtins.open, os.open, os.write
    fd_paths = {}
    pool = dict(evidence=0, cache=0)
    lock = threading.RLock()
    def reserve(path, size):
        kind = "evidence" if path.is_relative_to(evidence) else "cache"
        limit = (pkg["output_remaining"] - RESERVED_SUPERVISOR_BYTES if
                 path.name == "worker_receipt.json" else pkg["output_remaining"] -
                 RESERVED_SUPERVISOR_BYTES - RESERVED_WORKER_RECEIPT_BYTES) if kind == "evidence" else pkg["cache_remaining"]
        with lock:
            if pool[kind] + size > limit:
                raise RuntimeError("prewrite " + kind + " budget reached")
            pool[kind] += size
    class RuntimeFile:
        def __init__(self, f, path, newline):
            self.file, self.path, self.newline = f, path, newline
            self.end = max(f.tell(), os.fstat(f.fileno()).st_size)
        def __getattr__(self, key):
            return getattr(self.file, key)
        def __enter__(self):
            self.file.__enter__()
            return self
        def __exit__(self, *args):
            return self.file.__exit__(*args)
        def write(self, value):
            if isinstance(value, str):
                translated = value.replace("\n", os.linesep) if self.newline is None else value
                size = len(translated.encode(self.file.encoding or "utf-8"))
            else:
                size = len(value)
            if self.file.tell() > max(self.end, os.fstat(self.file.fileno()).st_size):
                raise PermissionError("sparse writes denied")
            reserve(self.path, size)
            written = self.file.write(value)
            self.end = max(self.end, self.file.tell())
            return written
        def writelines(self, values):
            for value in values:
                self.write(value)
        def truncate(self, size=None):
            size = self.file.tell() if size is None else size
            if size > max(self.end, os.fstat(self.file.fileno()).st_size):
                raise PermissionError("growing truncate denied")
            self.end = size
            return self.file.truncate(size)
    def wrapped_open(function, file, *args, **kwargs):
        f = function(file, *args, **kwargs)
        filemode = args[0] if args else kwargs.get("mode", "r")
        path = fd_paths.get(file) if type(file) is int else Path(file).resolve()
        if path is not None and (path.is_relative_to(evidence) or path.is_relative_to(cache)) and any(c in filemode for c in "wax+"):
            newline = kwargs.get("newline", args[4] if len(args) > 4 else None)
            return RuntimeFile(f, path, newline)
        return f
    def os_open(file, flags, *args, **kwargs):
        fd = raw_os_open(file, flags, *args, **kwargs)
        path = Path(file).resolve()
        fd_paths[fd] = path
        return fd
    def os_write(fd, value):
        path = fd_paths.get(fd)
        if path is not None and (path.is_relative_to(evidence) or path.is_relative_to(cache)):
            reserve(path, len(value))
        return raw_os_write(fd, value)
    io.open = lambda file, *a, **k: wrapped_open(raw_io, file, *a, **k)
    builtins.open = lambda file, *a, **k: wrapped_open(raw_open, file, *a, **k)
    os.open, os.write = os_open, os_write
    memory = ProcessMemory(os.getpid())
    class Context:
        def __init__(self):
            self.runtime, self.cache = evidence, cache
            self.candidate_path = CANDIDATE
            self.source_identity = fixed
            self.counts = {}
            self.guard = dict(approved=True, config_sha256=CONFIG_SHA, phase=mode,
                owner=object(), config=copy.deepcopy(cfg), source_identity=fixed)
            self.violations = violations
            self.denied = denied
            self.artifacts = {}
            self.open_streams = {}
            self.stage_seconds = {}
            self.finalization_errors = []
            self.peak_rss = 0
            self.last_memory_check = 0.
            self.checkpoint_written = self.checkpoint_read = False
        def check(self):
            if time.monotonic() - started >= hard - min(5., hard / 10):
                raise TimeoutError("cooperative stop before hard cap")
            if time.monotonic() - self.last_memory_check >= .25:
                mem = memory.sample()
                self.last_memory_check = time.monotonic()
                self.peak_rss = max(self.peak_rss, mem["peak_rss_bytes"])
                if max(mem["rss_bytes"], mem["peak_rss_bytes"]) > budget["pilot_rss_bytes_max"]:
                    raise MemoryError("RSS cap")
        def check_sources(self):
            self.check()
            if source_identity() != fixed:
                raise RuntimeError("fixed source changed")
        def charge(self, kind, cap_key, amount=1):
            self.check()
            if type(amount) is not int or amount < 1 or type(budget[cap_key]) is not int:
                raise ValueError("typed positive attempt charge required")
            value = self.counts.get(kind, 0) + amount
            if value > budget[cap_key]:
                raise RuntimeError("attempt cap before " + kind)
            self.counts[kind] = value
            # Append/flush BEFORE the charged operation. Hard-kill recovery uses this.
            self.emit("attempts", dict(kind=kind, count=value, amount=amount))
        def path(self, name):
            if type(name) is not str or not name or Path(name).name != name or name in (".", ".."):
                raise PermissionError("flat own evidence names only")
            return evidence / name
        def emit(self, name, obj):
            self.check()
            p = self.path(name if name.endswith(".jsonl") else name + ".jsonl")
            if p == checkpoint:
                raise PermissionError("checkpoint is not a JSONL stream")
            if p.name not in self.open_streams:
                self.open_streams[p.name] = p.open("x", encoding="utf-8", newline="\n")
            f = self.open_streams[p.name]
            f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            f.flush()
        def write_json(self, name, obj):
            p = self.path(name)
            encoded = (json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
            with p.open("xb") as f:
                f.write(encoded)
            self.artifacts[name] = dict(sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))
        def write_bytes(self, name, payload):
            if name != "model.zip" or type(payload) is not bytes or self.checkpoint_written:
                raise PermissionError("single same-task checkpoint only")
            if len(payload) > budget["single_checkpoint_bytes_max"]:
                raise RuntimeError("checkpoint byte cap before write")
            self.checkpoint_written = True
            checkpoint_access["write"] = True
            try:
                with checkpoint.open("xb") as f:
                    f.write(payload)
            finally:
                checkpoint_access["write"] = False
            self.artifacts[name] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        def read_own_checkpoint(self, name):
            if name != "model.zip" or not self.checkpoint_written or self.checkpoint_read:
                raise PermissionError("single readback of owned new checkpoint only")
            self.checkpoint_read = True
            checkpoint_access["read"] = True
            try:
                payload = checkpoint.read_bytes()
            finally:
                checkpoint_access["read"] = False
            if hashlib.sha256(payload).hexdigest() != self.artifacts[name]["sha256"]:
                raise RuntimeError("new checkpoint bytes changed")
            return payload
        @contextmanager
        def stage(self, name):
            self.check_sources()
            self.emit("events", dict(event="stage_start", stage=name, counts=dict(self.counts)))
            tick = time.monotonic()
            primary = None
            try:
                yield
            except BaseException as exc:
                primary = exc
                raise
            finally:
                duration = time.monotonic() - tick
                self.stage_seconds[name] = self.stage_seconds.get(name, 0.) + duration
                try:
                    self.emit("events", dict(event="stage_end", stage=name, seconds=duration, counts=dict(self.counts)))
                    self.check_sources()
                except BaseException as exc:
                    if primary is None or (isinstance(primary, Exception) and not isinstance(exc, Exception)):
                        if primary is not None:
                            self.finalization_errors.append(dict(stage=name+":ordinary_primary", error=repr(primary)[:2048]))
                        raise
                    self.finalization_errors.append(dict(stage=name, error=repr(exc)[:2048]))
        def boundary(self, env, label):
            if env is None or getattr(env, "driver", None) is None:
                current = dict(state="UNKNOWN", note="no assigned driver")
            else:
                from research.runner_contract import environment_receipt
                current = environment_receipt(env)
                current["driver_phase"] = env.driver.phase
            self.emit("boundaries", dict(label=label, receipt=current, counts=dict(self.counts)))
        def close_streams(self):
            errors = []
            for name, f in self.open_streams.items():
                try:
                    f.close()
                    p = evidence / name
                    self.artifacts[name] = dict(sha256=sha(p), bytes=p.stat().st_size)
                except BaseException as exc:
                    errors.append(exc)
                    self.finalization_errors.append(dict(stage="stream:" + name, error=repr(exc)[:2048]))
            if errors:
                raise next((x for x in errors if not isinstance(x, Exception)), errors[0])
    ctx = Context()
    result = None
    status, error = "STOPPED", None
    primary_error = None
    try:
        from research.r05_runner import run_pilot, run_preflight
        ctx.check_sources()
        result = run_preflight(ctx, cfg) if mode == "preflight" else run_pilot(ctx, cfg)
        ctx.check_sources()
        if any(violations.values()):
            raise PermissionError("a guard refused an operation")
        status = "PASS"
    except BaseException as exc:
        primary_error = exc
        error = dict(type=type(exc).__name__, message=str(exc)[:4096])
    finally:
        for cleanup_name, cleanup in (("streams", ctx.close_streams), ("memory", memory.close)):
            try:
                cleanup()
            except BaseException as exc:
                status = "STOPPED"
                ctx.finalization_errors.append(dict(stage=cleanup_name, error=repr(exc)[:2048]))
                if primary_error is None or (isinstance(primary_error, Exception) and not isinstance(exc, Exception)):
                    if error is not None:
                        ctx.finalization_errors.append(dict(stage="superseded_ordinary_primary", error=error))
                    primary_error = exc
                    error = dict(type=type(exc).__name__, message=str(exc)[:4096])
    try:
        after = source_identity()
    except BaseException as exc:
        after = dict(state="UNKNOWN", error=repr(exc)[:2048])
        status = "STOPPED"
        if primary_error is None or (isinstance(primary_error, Exception) and not isinstance(exc, Exception)):
            primary_error = exc
            error = dict(type=type(exc).__name__, message=str(exc)[:4096])
    if error is not None and primary_error is not None:
        error["related_cleanup_errors"] = getattr(primary_error, "r05_cleanup_errors", [])
        error["related_restore_errors"] = getattr(primary_error, "r05_restore_errors", [])
    receipt = dict(status=status, error=error, mode=mode, config_sha256=CONFIG_SHA,
        source_before=fixed, source_after=after, counts=ctx.counts,
        violations=violations, denied=denied, result=result, stage_seconds=ctx.stage_seconds,
        worker_seconds=time.monotonic() - started, observed_peak_rss_bytes=ctx.peak_rss,
        written_byte_pools=pool.copy(), artifacts=ctx.artifacts,
        finalization_errors=ctx.finalization_errors,
        checkpoint_IO_attempts=dict(file_write=int(ctx.checkpoint_written), file_read=int(ctx.checkpoint_read)),
        limitation="one training world/seed exploratory; no R05 full acceptance or R06 authorization")
    ctx.write_json("worker_receipt.json", receipt)
    print(json.dumps(dict(status=status, mode=mode, counts=ctx.counts, error=error)))
    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "pilot"))
    parser.add_argument("--worker", nargs=2, metavar=("MODE", "NONCE"))
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(*args.worker))
    if not args.mode:
        parser.error("--mode required")
    raise SystemExit(supervisor(args.mode))
