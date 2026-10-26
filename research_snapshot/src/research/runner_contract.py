"""R04 minimal, owner-authorized no-learning tail contract (not a training CLI).

This routes trusted capabilities, not hostile Python. Its wall clock is cooperative;
a separately approved supervisor owns hard timeouts. State hashes are RAM identities,
not checkpoints or proof of training origin. Only explicitly listed source files are
read by the caller's source_reader; no world or historical-model imports occur here.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time

from .tail_budget import TailBudget, TailStopped, finish_tail


SCALE_ID = "v31-fixed-physical-divisors-and-raw-cost-div-1e6"
_STATE_KEYS = frozenset(("parameters", "optimizer", "gradients", "rollout_buffer",
                        "num_timesteps", "n_updates"))


def state_digest(value):
    """Deterministic framed identity of supported in-memory values; never pickle."""
    h = sha256()
    def emit(tag, payload=b""):
        h.update(tag.encode("ascii") + str(len(payload)).encode("ascii") + b":" + payload)
    def visit(v):
        if v is None: emit("none")
        elif type(v) is bool: emit("bool", str(v).encode("ascii"))
        elif type(v) is int: emit("int", str(v).encode("ascii"))
        elif type(v) is float: emit("float", v.hex().encode("ascii"))
        elif type(v) is str: emit("str", v.encode("utf-8"))
        elif type(v) is bytes: emit("bytes", v)
        elif isinstance(v, dict):
            emit("map", str(len(v)).encode("ascii"))
            for k in sorted(v, key=lambda x: (type(x).__name__, repr(x))):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            emit(type(v).__name__, str(len(v)).encode("ascii"))
            for x in v: visit(x)
        elif hasattr(v, "detach"):
            a = v.detach().cpu().contiguous().numpy()
            emit("tensor", (str(v.dtype) + repr(tuple(v.shape))).encode("ascii"))
            emit("data", a.tobytes(order="C"))
        elif hasattr(v, "dtype") and hasattr(v, "tobytes"):
            if getattr(v.dtype, "hasobject", False):
                raise TypeError("object arrays are not stable learning-state identities")
            emit("array", (str(v.dtype) + repr(tuple(v.shape))).encode("ascii"))
            emit("data", v.tobytes(order="C"))
        else:
            raise TypeError("unsupported RAM state: " + type(v).__name__)
    visit(value)
    return h.hexdigest()


def snapshot_learning_state(model):
    """Hash actual model weights, optimizer, gradients and complete SB3 Buffer RAM.

    Buffer spaces/device have immutable configuration identity in the run metadata.
    Every other buffer attribute must be a supported value, rather than silently
    ignoring unknown arrays/new fields. Reading these values does not sample Buffer.
    """
    policy, buffer = model.policy, model.rollout_buffer
    buffer_state = {k: v for k, v in vars(buffer).items()
                    if k not in ("observation_space", "action_space", "device")}
    return dict(parameters=state_digest(dict(policy.state_dict())),
                optimizer=state_digest(policy.optimizer.state_dict()),
                gradients=state_digest({name: p.grad for name, p in policy.named_parameters()}),
                rollout_buffer=state_digest(buffer_state),
                num_timesteps=int(model.num_timesteps), n_updates=int(model._n_updates))


def model_config_reader(model):
    """Actual attributes for every declared minimal interface-fixture setting."""
    aliases={"optimizer_updates":"_n_updates"}
    return {key:getattr(model,aliases.get(key,key)) for key in
            ("gamma","gae_lambda","n_steps","batch_size","optimizer_updates")}


def rule_state_reader(partner):
    """Use only for explicitly approved stateless rule functions, never PPO closures."""
    if type(partner).__name__ != "function" or partner.__closure__:
        raise TypeError("only a stateless function rule may use this reader")
    return dict(kind="stateless_rule", implementation=state_digest(dict(
        module=partner.__module__, name=partner.__qualname__,
        bytecode=partner.__code__.co_code,
        constants=repr(partner.__code__.co_consts), defaults=partner.__defaults__)))


def environment_receipt(env):
    """Accounting projection; advancing economic state is expected during a tail."""
    c, store = env.driver.c, env.driver.store
    t = int(c.t)
    if type(c.t) is not int or type(store.n) is not int or not 0 <= t <= store.n or len(c.hours) != t:
        raise RuntimeError("inconsistent delivered-hour accounting boundary")
    signed = math.fsum(math.fsum(l.quantities) * l.unit_price for l in c.ledger.lots)
    delivered = math.fsum(float(row["contract_cost"]) for row in c.hours)
    remaining = float(c.ledger.fee[t:].sum())
    tolerance = 1e-6 + 8 * max(1, int(store.n)) * 2.220446049250313e-16 * max(1., abs(signed))
    residual = signed - delivered - remaining
    if not all(math.isfinite(v) and v >= 0 for v in (signed, delivered, remaining, float(c.total_purchase), float(c.total_flex))):
        raise RuntimeError("nonfinite or negative economic accounting boundary")
    if abs(residual) > tolerance:
        raise RuntimeError("contract liability accounting does not reconcile")
    return dict(completed_hours=t, settled_hour_rows=len(c.hours),
                remaining_hours=int(store.n)-t, pending=c.pending is not None,
                purchase_cost=float(c.total_purchase), flex_cost=float(c.total_flex),
                signed_contract_fee=signed, delivered_contract_fee=delivered,
                remaining_contract_fee=remaining, liability_residual=residual,
                liability_tolerance=tolerance, ended=bool(env.ended))


@dataclass(frozen=True)
class TailRunApproval:
    owner: object
    level: str
    identities_sha256: str
    max_hours: int
    max_decisions: int
    max_policy_calls: int
    wall_seconds: float
    max_journal_bytes: int
    journal_root: str
    approved: bool = False

    def validate(self, owner, level, identities, budget, max_bytes):
        if (self.approved is not True or owner is None or self.owner is not owner
                or level != self.level or level not in ("upper", "lower")):
            raise PermissionError("an explicit matching approved owner is required")
        if state_digest(identities) != self.identities_sha256:
            raise PermissionError("run identities differ from approved identities")
        if (budget.max_hours > self.max_hours or budget.max_decisions > self.max_decisions
                or budget.max_policy_calls > self.max_policy_calls
                or budget.wall_seconds > self.wall_seconds
                or max_bytes > self.max_journal_bytes):
            raise PermissionError("requested work exceeds approved tail budget")


def _identities(value):
    if type(value) is not dict or set(value) != {
            "source_sha256", "config", "observation_scale", "model", "partner", "scope"}:
        raise ValueError("explicit source/config/scales/model/partner/scope identities required")
    sources = value["source_sha256"]
    if not sources or not isinstance(sources, dict) or not all(
            type(k) is str and re.fullmatch(r"[0-9a-f]{64}", v)
            for k, v in sources.items()):
        raise ValueError("fixed source SHA256 map required")
    if value["observation_scale"] != SCALE_ID:
        raise PermissionError("only the frozen physical observation scales are approved")
    if value["model"] != {"origin": "fresh_untrained", "checkpoint": None}:
        raise PermissionError("this stage permits only a fresh untrained model")
    if not value["partner"] or not value["scope"] or not value["config"]:
        raise ValueError("empty configuration, partner or scope identity")
    if value["config"] != dict(gamma=1.,gae_lambda=.95,n_steps=2,batch_size=2,optimizer_updates=0):
        raise PermissionError("only explicit two-step no-update interface-fixture configuration is approved")


def run_tail_checked(*, approval=None, owner=None, level, env, make_model,
                     partner_owner, partner_state_reader, identities, source_reader,
                     journal_root, run_id, budget, max_journal_bytes=32768,
                     state_reader=snapshot_learning_state, clock=time.monotonic):
    """Authorize before factory; call the existing tail; persist a bounded receipt.

    Journal root is an already approved directory, run_id a basename. Exclusive
    creation prevents overwriting older runs. A STOPPED record has reserved bytes;
    oversized success receipts become a budget failure. No policy is called again.
    Control interruptions retain object identity across cleanup and persistence.
    """
    if not isinstance(budget, TailBudget):
        raise TypeError("TailBudget required")
    if type(max_journal_bytes) is not int or max_journal_bytes < 8192:
        raise ValueError("journal requires at least 8192 explicitly approved bytes")
    if not isinstance(approval, TailRunApproval):
        raise PermissionError("no approved RUN_CONTEXT owner")
    approval.validate(owner, level, identities, budget, max_journal_bytes)
    _identities(identities)
    if getattr(env,"level",None)!=level or getattr(env,"partner",None) is not partner_owner:
        raise PermissionError("actual environment role or partner differs from approval")
    actual_partner_state=partner_state_reader(partner_owner)
    if "state" not in identities["partner"] or state_digest(actual_partner_state)!=state_digest(identities["partner"]["state"]):
        raise PermissionError("actual partner state differs from approved identity")
    # Freeze metadata against caller-side mutation during factory/tail callbacks.
    identities = json.loads(json.dumps(identities, allow_nan=False))
    if not all(callable(f) for f in (make_model, partner_state_reader, state_reader, source_reader)):
        raise TypeError("explicit factory and state/source readers are required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must be a single safe basename")
    root = Path(journal_root).resolve(strict=True)
    if not root.is_dir(): raise ValueError("approved journal root must be an existing directory")
    if root != Path(approval.journal_root).resolve(strict=True):
        raise PermissionError("journal root differs from explicitly approved directory")
    path = root / (run_id + ".jsonl")
    if path.is_symlink(): raise PermissionError("journal symlink denied")
    # This check precedes model construction and any tail calls.
    if source_reader() != identities["source_sha256"]:
        raise PermissionError("source content differs from approved input")
    record = dict(status="RUNNING", level=level, run_id=run_id,
                  identities=identities, identities_sha256=state_digest(identities),
                  model_factory_attempts=0, learning_state_scope=sorted(_STATE_KEYS),
                  journal=str(path), scope=identities["scope"])
    written = 0
    started = clock()
    stream = path.open("xb")
    def append(row, *, reserve=0):
        nonlocal written
        payload=(json.dumps(row, ensure_ascii=False, sort_keys=True,
                            allow_nan=False, separators=(",", ":"))+"\n").encode("utf-8")
        if written + len(payload) + reserve > max_journal_bytes:
            raise TailStopped("journal byte budget reached", record)
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        # Persist each bounded line; abrupt kills beyond the last line remain UNKNOWN.
        written += len(payload)
    first_error = None
    first_control = None
    model = None
    def remember(error, stage):
        nonlocal first_error, first_control
        if first_error is None: first_error = error
        if not isinstance(error, Exception) and first_control is None: first_control = error
        primary=first_control if first_control is not None else first_error
        record.update(status="STOPPED", error_type=type(primary).__name__, error=repr(primary)[:512])
        record.setdefault("errors", []).append(dict(stage=stage, type=type(error).__name__,
                                                     message=repr(error)[:512]))
    try:
        append(dict(status="STARTED", run_id=run_id, level=level,
                    identities_sha256=record["identities_sha256"]), reserve=4096)
        record["environment_before"] = environment_receipt(env)
        record["initial_pending"] = record["environment_before"]["pending"]
        record["partner_before"] = actual_partner_state
        record["model_factory_attempts"] += 1
        model = make_model(level, env)
        if env.level!=level or env.partner is not partner_owner:
            raise PermissionError("model factory replaced actual environment role/partner")
        before = state_reader(model)
        if type(before) is not dict or set(before) != _STATE_KEYS:
            raise TypeError("model state reader must cover all learning state fields")
        record["model_before"] = before
        record["config_before"] = model_config_reader(model)
        if record["config_before"] != identities["config"]:
            raise PermissionError("actual fresh model configuration differs from approved fixture")
        if before["n_updates"] != 0:
            raise PermissionError("fresh-untrained model has existing optimizer updates")
        if ("kind" not in record["partner_before"]
                and set(record["partner_before"]) != _STATE_KEYS):
            raise TypeError("partner state reader must cover learning state or stateless rule")
        if "kind" in record["partner_before"] and record["partner_before"].get("kind") != "stateless_rule":
            raise TypeError("unknown partner-state exemption")
        remaining_wall = budget.wall_seconds - (clock() - started)
        if remaining_wall <= 0:
            raise TailStopped("runner wall budget reached before tail", record)
        append(dict(record, status="READY_FOR_TAIL"), reserve=4096)
        bounded_tail = TailBudget(budget.max_hours, budget.max_decisions, remaining_wall,
                                  max_policy_calls=budget.max_policy_calls)
        record["tail"] = finish_tail(env, model, bounded_tail, clock=clock)
    except BaseException as error:
        if isinstance(error, TailStopped) and error.receipt is not record:
            record["tail"] = error.receipt
        elif hasattr(error, "tail_receipt"): record["tail"] = error.tail_receipt
        remember(error, "tail_or_setup")
    finally:
        for name, reader in (
                ("model_after", lambda: state_reader(model) if model is not None else None),
                ("config_after", lambda: model_config_reader(model) if model is not None else None),
                ("partner_after", lambda: partner_state_reader(partner_owner)),
                ("environment_after", lambda: environment_receipt(env)),
                ("source_after", source_reader),
                ("cooperative_seconds", lambda: clock() - started)):
            try: record[name] = reader()
            except BaseException as error: remember(error, name)
        try:
            if model is not None and record.get("config_before") != record.get("config_after"):
                raise RuntimeError("fixed model configuration changed")
            if model is not None and record.get("model_before") != record.get("model_after"):
                raise RuntimeError("no-learning tail mutated model/optimizer/gradient/Buffer/counters")
            if env.level!=level or env.partner is not partner_owner:
                raise RuntimeError("actual environment role/partner replaced during tail")
            if record.get("partner_before") != record.get("partner_after"):
                raise RuntimeError("frozen partner state changed")
            if record.get("source_after") != identities["source_sha256"]:
                raise RuntimeError("fixed source changed during the tail")
            if "environment_before" in record and "environment_after" in record:
                before, after = record["environment_before"], record["environment_after"]
                record["settled_hours_delta"] = after["completed_hours"]-before["completed_hours"]
                record["new_H_START_attempts"] = record.get("tail", {}).get("hours_started", 0)
                if record["settled_hours_delta"] < 0:
                    raise RuntimeError("economic clock regressed")
                expected_delta=record["new_H_START_attempts"]+int(before["pending"])-int(after["pending"])
                if record["settled_hours_delta"] != expected_delta:
                    raise RuntimeError("initial pending/new H_START/settlement counts do not reconcile")
                if first_error is None:
                    result=record.get("tail",{}).get("tail")
                    empty=isinstance(result,dict) and result.get("empty_auto_reset") is True
                    if empty:
                        if not (level=="upper" and env.actions==0 and not env.driver.c.macros and not before["pending"] and record["settled_hours_delta"]==0 and before==after):
                            raise RuntimeError("invalid empty-upper-auto-reset exemption")
                    elif after["remaining_hours"]!=0 or not after["ended"] or after["pending"]:
                        raise RuntimeError("nonempty tail did not discharge its declared driver scope")
            if first_error is None: record["status"] = "COMPLETE"
        except BaseException as error: remember(error, "invariance")
        try:
            append(record)
        except BaseException as error:
            remember(error, "journal_final")
            # Reserved compact failure line; never write an oversized success.
            stopped = dict(status="STOPPED", run_id=run_id,
                           identities_sha256=record["identities_sha256"],
                           error_type=record.get("error_type"), error=record.get("error"),
                           learning_state_match=record.get("model_before")==record.get("model_after"),
                           initial_pending=record.get("initial_pending"),
                           environment_after=record.get("environment_after"))
            try: append(stopped)
            except BaseException as fallback_error: remember(fallback_error, "journal_fallback")
        try: stream.close()
        except BaseException as error: remember(error, "journal_close")
    record["journal_bytes"] = written
    if first_control is not None:
        first_control.tail_receipt = record
        raise first_control
    if first_error is not None:
        raise TailStopped(str(first_error), record) from first_error
    return record
