"""R01 approved in-memory research input boundary; no file/data/model imports.

The trusted owner supplies the allowlist and factory. A PPO policy never receives
this object. This is a capability-routing contract, not an OS security sandbox.
"""
from dataclasses import dataclass

_FIELDS = frozenset(("year", "family", "group", "split", "fixture"))
_GROUPS = frozenset(("G00", "G01", "G10", "G11"))


def _key(spec):
    if type(spec) is not dict or set(spec) != _FIELDS:
        raise PermissionError("only an explicit in-memory world specification is accepted")
    year, family, group, split, fixture = (spec[k] for k in
                                         ("year", "family", "group", "split", "fixture"))
    if type(year) is not int or not 1900 <= year <= 2100:
        raise PermissionError("invalid explicit year")
    if type(family) is not str or not family or any(c in family for c in "/\\:"):
        raise PermissionError("invalid world family")
    if type(group) is not str or group not in _GROUPS:
        raise PermissionError("invalid condition")
    if type(split) is not str or split not in ("fixture", "train", "validation"):
        raise PermissionError("test/confirmation split is not accessible")
    if type(fixture) is not bool:
        raise PermissionError("fixture must be an explicit boolean")
    return year, family, group, split, fixture


@dataclass(frozen=True, init=False)
class ResearchInputPolicy:
    """Immutable allowlist. W1 permits artificial fixtures, never research worlds."""
    approved_keys: frozenset

    def __init__(self, approved_specs):
        keys = frozenset(_key(spec) for spec in approved_specs)
        if not keys:
            raise ValueError("an explicit approved allowlist is required")
        if any(not k[4] or k[3] == "validation" for k in keys):
            raise PermissionError("W1 authority covers artificial fixtures only")
        object.__setattr__(self, "approved_keys", keys)

    def validate(self, request):
        key = _key(request)
        if key not in self.approved_keys:
            raise PermissionError("world is outside the approved W1 allowlist")
        return dict(zip(("year", "family", "group", "split", "fixture"), key))

    def create_store(self, request, *, factory):
        approved = self.validate(request)
        # No factory invocation, array construction or file read before validation.
        return factory(**approved)


def build_environment(requests, *, policy, level, seed=0, partner=None):
    """Selected environment port; legacy training CLIs are not research entries.

    W1 tests reject requests here before importing the existing environment.
    Positive environment construction/reset is reserved for W2 approval.
    """
    specs = [policy.validate(request) for request in requests]
    if not specs:
        raise ValueError("at least one explicitly approved world is required")
    if level not in ("upper", "lower"):
        raise ValueError("unknown decision level")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(spec["split"] != "train" for spec in specs):
        raise PermissionError("training cache requires a train-labelled fixture")
    from v31.lite_gym import WorldCache, AnnualDecisionEnv

    class ApprovedWorldCache(WorldCache):
        def make(self, index):
            if type(index) is not int or not 0 <= index < len(self.specs):
                raise ValueError("invalid world index")
            # Check again even on a cache hit and after trusted-side list changes.
            policy.validate(self.specs[index])
            return super().make(index)

    return AnnualDecisionEnv(ApprovedWorldCache(specs), level, seed=seed, partner=partner)
