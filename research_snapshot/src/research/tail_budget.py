"""Budget the existing tail; caller separately authorizes all real computation."""
from dataclasses import dataclass
import math
import time


class TailStopped(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class TailBudget:
    max_hours: int
    max_decisions: int  # maximum submission attempts, retained for call compatibility
    wall_seconds: float
    max_policy_calls: int | None = None

    def __post_init__(self):
        if self.max_policy_calls is None:
            object.__setattr__(self, "max_policy_calls", self.max_decisions)
        for value in (self.max_hours, self.max_decisions, self.max_policy_calls):
            if type(value) is not int or value < 0:
                raise ValueError("nonnegative integer tail limits required")
        if (isinstance(self.wall_seconds, bool)
                or not isinstance(self.wall_seconds, (int, float))
                or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0):
            raise ValueError("positive finite wall limit required")


def finish_tail(env, model, budget, *, clock=time.monotonic, receipt_sink=None):
    """Check before H_START, submissions, model.predict and explicit partner calls.

    Policy-call limits conservatively count rule partners too. An active upper
    tail requires an explicit partner, rather than an uninstrumented default.
    Wall checks are cooperative, not OS preemption. This adapter neither resets
    economic liability nor proves weights/optimizer/real rollout-buffer invariance.
    A real runner must supply its authorized receipt sink and state checks.
    The first control interruption retains identity across cleanup, final metrics and sink errors.
    """
    if not isinstance(budget, TailBudget):
        raise TypeError("TailBudget required")
    if receipt_sink is not None and not callable(receipt_sink):
        raise TypeError("receipt_sink must be callable")
    start = clock()
    learned = model.num_timesteps
    rows = len(env.steps)
    receipt = dict(status="RUNNING", hours_started=0, submission_attempts=0,
                   policy_call_attempts=0, learning_steps_before=learned,
                   env_record_rows_before=rows,
                   learning_state_scope="counter and environment records only")
    saved = []

    def check():
        if clock() - start >= budget.wall_seconds:
            raise TailStopped("tail wall budget reached", receipt)

    def wrap(owner, name, count, limit, *, policy=False):
        original = getattr(owner, name)
        had_local = name in vars(owner)
        local_value = vars(owner).get(name)

        def guarded(*args, **kwargs):
            check()
            if policy:
                # Prevent an otherwise wasted prediction before the next rejected action.
                if receipt["hours_started"] >= budget.max_hours:
                    raise TailStopped("tail hours budget reached before policy call", receipt)
                if receipt["submission_attempts"] >= budget.max_decisions:
                    raise TailStopped("tail submission budget reached before policy call", receipt)
            if receipt[count] >= limit:
                raise TailStopped("tail " + count + " budget reached", receipt)
            receipt[count] += 1  # attempted calls that raise still consume this allowance
            return original(*args, **kwargs)

        saved.append((owner, name, had_local, local_value))
        setattr(owner, name, guarded)

    first_error = None
    first_control = None

    def remember(error, stage):
        nonlocal first_error, first_control
        if not isinstance(error, Exception) and first_control is None:
            first_control = error
        if first_error is None:
            first_error = error
        primary = first_control if first_control is not None else first_error
        receipt.update(status="STOPPED", error=repr(primary), error_type=type(primary).__name__)
        receipt.setdefault("errors", []).append(dict(stage=stage, type=type(error).__name__,
                                                    message=repr(error)))

    try:
        check()
        if not env.ended:
            driver = env.driver
            if driver is None or getattr(driver, "aborted", False):
                raise TailStopped("missing or aborted driver", receipt)
            empty_upper = (getattr(env, "level", None) == "upper"
                           and env.actions == 0 and not driver.c.macros)
            if (getattr(env, "level", None) == "upper" and not empty_upper
                    and not callable(getattr(env, "partner", None))):
                raise TailStopped("explicit frozen partner required for upper tail", receipt)
            wrap(driver.c, "prepare_hour", "hours_started", budget.max_hours)
            wrap(driver, "submit_upper", "submission_attempts", budget.max_decisions)
            wrap(driver, "submit_lower", "submission_attempts", budget.max_decisions)
            if callable(getattr(model, "predict", None)):
                wrap(model, "predict", "policy_call_attempts", budget.max_policy_calls, policy=True)
            if callable(getattr(env, "partner", None)):
                wrap(env, "partner", "policy_call_attempts", budget.max_policy_calls, policy=True)
        result = env.finish_without_learning(model)
        check()
        if model.num_timesteps != learned or len(env.steps) != rows:
            raise TailStopped("tail changed learning counter or environment records", receipt)
        if not env.ended and not (isinstance(result, dict) and result.get("empty_auto_reset")):
            raise TailStopped("tail did not complete", receipt)
        receipt.update(status="COMPLETE", tail=result)
    except BaseException as error:
        remember(error, "tail")

    # Defer propagation until restoration and receipt finalization have been tried.
    # One interrupted restoration gets one bounded retry; no policy/action is retried.
    for owner, name, had_local, local_value in reversed(saved):
        for attempt in range(2):
            try:
                if had_local:
                    setattr(owner, name, local_value)
                else:
                    vars(owner).pop(name, None)
                break
            except BaseException as error:
                remember(error, "restore:" + name)
                if isinstance(error, Exception) or attempt == 1:
                    break
    for key, supplier in (
            ("seconds", lambda: clock() - start),
            ("learning_steps_after", lambda: model.num_timesteps),
            ("env_record_rows_after", lambda: len(env.steps))):
        try:
            receipt[key] = supplier()
        except BaseException as error:
            receipt[key] = "UNKNOWN"
            remember(error, "finalize:" + key)
    if receipt_sink is not None:
        try:
            receipt_sink(dict(receipt))
        except BaseException as error:
            receipt["receipt_sink_error"] = repr(error)
            remember(error, "receipt_sink")
    if first_control is not None:
        first_control.tail_receipt = receipt
        raise first_control
    if first_error is not None:
        raise TailStopped(str(first_error), receipt) from first_error
    return receipt
