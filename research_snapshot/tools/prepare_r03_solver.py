"""Install only the approved, hash-pinned SciPy wheel into a new isolated directory.

This is a bounded dependency operation, never a project/solver/test invocation.
No setup scripts or dependency resolver are executed; original environments stay unchanged.
"""
import sys
sys.dont_write_bytecode = True
import json, hashlib, pathlib, urllib.request, urllib.parse, zipfile, time
ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORD = ROOT / "TASK_RECORD.json"
FILENAME = "scipy-1.16.2-cp311-cp311-win_amd64.whl"
EXPECTED = "91e9e8a37befa5a69e9cacbe0bcb79ae5afb4a0b130fd6db6ee6cc0d491695fa"
TARGET = ROOT / "dependencies/scipy_1_16_2"
EVIDENCE = ROOT / "logs/r03_dependency_1"
MAX_DOWNLOAD = 42 * 1024 * 1024
MAX_EXTRACT = 200 * 1024 * 1024

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(record): RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def bounded_get(url, limit):
    if urllib.parse.urlparse(url).scheme != "https": raise ValueError("HTTPS required")
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(limit+1)
    if len(data)>limit: raise RuntimeError("download byte cap")
    return data

def main():
    start=time.monotonic(); record=json.loads(RECORD.read_text(encoding="utf-8"))
    scope=record["r03_r04_package"]
    if len(sys.argv)!=3 or sys.argv[1]!="--worker" or sys.argv[2]!=scope.get("dependency_worker_nonce"):
        raise PermissionError("approved hard-limited supervisor required")
    if scope.get("approval_state")!="APPROVED" or not scope.get("budgets",{}).get("solver_installations_max"):
        raise PermissionError("explicit dependency installation approval required")
    if scope.get("actual",{}).get("dependency_installation_attempts",0):
        raise RuntimeError("installation already attempted; preserve partial evidence")
    if sys.version_info[:2]!=(3,11) or sys.platform!="win32" or sys.maxsize<=2**32:
        raise RuntimeError("approved CPython3.11 Windows64 required")
    manifest=pathlib.Path(__file__).resolve().parents[1]/"MAIN_VERSION_MANIFEST.json"
    if digest(manifest)!=scope["fixed_manifest_sha256"]: raise RuntimeError("unreviewed candidate")
    expected=next(x["proposed_budgets"] for x in json.loads((manifest.parent/"EVIDENCE_INDEX.json").read_text(encoding="utf-8"))["entries"] if x["id"]=="E45")
    for actual in (scope.get("budgets"),scope.get("approved_budgets")):
        if type(actual) is not dict or set(actual)!=set(expected) or any(type(actual[k]) is not type(v) or actual[k]!=v for k,v in expected.items()):
            raise PermissionError("dependency budgets differ from approved fixed scope")
    if TARGET.exists() or EVIDENCE.exists(): raise FileExistsError("new dependency target/evidence only")
    scope.setdefault("actual",{})["dependency_installation_attempts"]=1;save(record)
    EVIDENCE.mkdir(parents=True); receipt={"status":"STOPPED","wheel":FILENAME,"expected_sha256":EXPECTED,"target":str(TARGET),"original_environments_modified":False,"executed_setup_scripts":0}
    try:
        metadata=bounded_get("https://pypi.org/pypi/scipy/1.16.2/json",2*1024*1024)
        (EVIDENCE/"pypi_metadata.json").write_bytes(metadata)
        rows=[x for x in json.loads(metadata)["urls"] if x["filename"]==FILENAME and x["digests"]["sha256"]==EXPECTED]
        if len(rows)!=1: raise RuntimeError("official wheel identity mismatch")
        url=rows[0]["url"]
        if urllib.parse.urlparse(url).hostname!="files.pythonhosted.org": raise RuntimeError("unapproved wheel origin")
        wheel=EVIDENCE/FILENAME;wheel.write_bytes(bounded_get(url,MAX_DOWNLOAD))
        if digest(wheel)!=EXPECTED: raise RuntimeError("wheel SHA256 mismatch")
        if time.monotonic()-start>=150: raise TimeoutError("dependency cooperative deadline")
        with zipfile.ZipFile(wheel) as archive:
            members=archive.infolist()
            if len(members)>10000 or sum(x.file_size for x in members)>MAX_EXTRACT:
                raise RuntimeError("extracted dependency cap")
            for item in members:
                path=pathlib.PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in item.filename or ":" in item.filename:
                    raise RuntimeError("wheel path escape")
                if ((item.external_attr>>16)&0o170000)==0o120000: raise RuntimeError("no symlink extraction")
            TARGET.mkdir(parents=True)
            for item in members:
                if time.monotonic()-start>=165: raise TimeoutError("dependency extraction deadline")
                dest=TARGET/pathlib.PurePosixPath(item.filename)
                if item.is_dir(): dest.mkdir(parents=True,exist_ok=True);continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source,dest.open("xb") as output:
                    while True:
                        block=source.read(1024*1024)
                        if not block:break
                        output.write(block)
        metadata_path=TARGET/"scipy-1.16.2.dist-info/METADATA"
        text=metadata_path.read_text(encoding="utf-8")
        if "Version: 1.16.2" not in text:raise RuntimeError("installed metadata mismatch")
        files={q.relative_to(TARGET).as_posix():digest(q) for q in TARGET.rglob("*") if q.is_file()}
        receipt.update(status="INSTALLED_NOT_IMPORTED",wheel_sha256=digest(wheel),wheel_bytes=wheel.stat().st_size,extracted_bytes=sum((TARGET/k).stat().st_size for k in files),files=files,setup_or_solver_calls=0)
    except BaseException as error:
        receipt["error"]=repr(error)
        raise
    finally:
        receipt["seconds"]=time.monotonic()-start
        encoded=(json.dumps(receipt,indent=2)+"\n").encode("utf-8")
        if len(encoded)>2*1024*1024:
            receipt=dict(status="STOPPED",error="dependency receipt exceeds2MiB",seconds=receipt["seconds"])
            encoded=(json.dumps(receipt)+"\n").encode("utf-8")
        (EVIDENCE/"receipt.json").write_bytes(encoded)
        record=json.loads(RECORD.read_text(encoding="utf-8"));record["r03_r04_package"]["dependency_status"]=receipt["status"];save(record)
    print(json.dumps({k:v for k,v in receipt.items() if k!="files"}))

if __name__=="__main__":main()
