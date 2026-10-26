"""Same-information rolling LP for the frozen synthetic V3.1 hourly model.

Only lower-observation arrays enter this module.  No driver/world/model access.
SciPy is imported only after an explicit, precharged SolveBudget reservation.
A plan is not a device trace: verify_execution compares the trusted hourly row.
This is a fixed-contract predictive comparator, never a joint-policy lower bound.
"""
from dataclasses import dataclass, field
import math
import time


PRIMAL_ATOL = 1e-7
DUAL_ATOL = 1e-7
GAP_RTOL = 1e-8
# Observations are float32, while the executor retains float64 task balances.
# 24 * eps32 * 7 MWh is 2.003e-5 MWh. The allowance concerns projection /
# execution comparisons only, NEVER LP feasibility or dropping service energy.
# This also exceeds two float32 ULPs at remaining=33.3333 (2*2**-18),
# one SOC ULP transformed to grid energy ((60/.9)*2**-24), and one
# power/minimum ULP near 7 (2**-21), combined < 1.22e-5 MWh.
EXECUTION_ATOL = 24 * (2 ** -23) * 7


class BaselineError(RuntimeError):
    """Explicit failed comparison; callers must stop, never substitute a rule."""


class SolveBudgetExceeded(BaselineError):
    pass


@dataclass
class SolveBudget:
    """Caller-owned, finite authorization. Defaults deny any solver invocation.

    Reserve before dependency import / solver entry. Exceptions and failed solves
    consume one attempt. A caller must not recreate this ledger to retry for free.
    time_limit is passed to HiGHS; a process hard limit belongs to the owner runner.
    """
    max_attempts: int = 0
    time_limit: float = 5.0
    attempts: list = field(default_factory=list)

    def __post_init__(self):
        if type(self.max_attempts) is not int or not 0 <= self.max_attempts <= 64:
            raise ValueError('solver attempts must be an integer in [0,64]')
        if not math.isfinite(self.time_limit) or not 0 < self.time_limit <= 5:
            raise ValueError('per-solve time_limit must be in (0,5] seconds')

    def reserve(self, variables, inequalities, equalities):
        self.__post_init__()  # revalidate mutable caller-owned limits before use
        if len(self.attempts) >= self.max_attempts:
            raise SolveBudgetExceeded('no authorized solver attempts remain')
        if not (1 <= variables <= 192 and inequalities <= 256 and equalities <= 2):
            raise SolveBudgetExceeded('problem exceeds authorized shape')
        row = dict(attempt=len(self.attempts) + 1, variables=variables,
                   inequalities=inequalities, equalities=equalities,
                   time_limit_seconds=self.time_limit, state='RESERVED')
        self.attempts.append(row)
        return row


def _vector(obs, key, count, *, nonnegative=True):
    try:
        value = obs[key]
        if len(value) != count:
            raise ValueError('wrong length')
        out = tuple(float(x) for x in value)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BaselineError('missing/invalid observation field: ' + key) from exc
    if any(not math.isfinite(x) or (nonnegative and x < 0) for x in out):
        raise BaselineError('nonfinite/negative observation field: ' + key)
    return out


def _integer(value, name):
    if not math.isfinite(value) or value != int(value):
        raise BaselineError(name + ' must be an exact integer')
    return int(value)


@dataclass(frozen=True)
class RollingProblem:
    hour: int
    hours: tuple
    objective: tuple
    inequalities: tuple
    inequality_rhs: tuple
    equalities: tuple
    equality_rhs: tuple
    bounds: tuple
    contract_fee_constant: float
    current_capacity: float
    device_snapshot: tuple
    nominal_ev: tuple
    nominal_industrial: tuple


def build_problem(observation):
    """Compile only the authorized projected fields, never future realizations.

    Forecast load / prices are the sealed 24-hour predictions. The current offer
    overrides its forecast because it is public at H_START. Task amounts before
    arrival are fixed public reservations, not future actual SOC. Do not replace
    task equalities with inequalities or reduce demand to make a solve succeed.
    """
    h = _integer(_vector(observation, 'hour', 1)[0], 'hour')
    if not 0 <= h < 24:
        raise BaselineError('hour outside [0,23]')
    load = _vector(observation, 'forecast_load', 24)
    balance = _vector(observation, 'forecast_price', 24)
    offer_prediction = _vector(observation, 'forecast_offer_price', 24)
    q = _vector(observation, 'contract_plan', 24)
    fee = _vector(observation, 'contract_fee', 24)
    offer, capacity = _vector(observation, 'offer', 2)
    if capacity != 160:
        raise BaselineError('frozen model requires public spot capacity 160 MWh')
    d = _vector(observation, 'devices', 9)
    mask = _vector(observation, 'actual_state_mask', 2)
    initial_soc = _vector(observation, 'declared_initial_soc', 1)[0]
    es, ee, ins, ine = [_integer(d[j], 'device window') for j in (3, 4, 7, 8)]
    if es not in (18, 19) or ee != 24 or ins not in (8, 12) or ine != 16:
        raise BaselineError('window differs from frozen normal/limited sessions')
    if (es == 19) != (ins == 12):
        raise BaselineError('inconsistent normal/limited session pair')
    if mask != (float(h >= es), float(h >= ins)):
        raise BaselineError('actual-state availability mask inconsistent with clock')
    if abs(initial_soc - .2) > 2 ** -23:
        raise BaselineError('declared initial SOC differs from frozen session')
    if h < es:
        if d[0] != 0 or abs(d[1] - 100 / 3) > EXECUTION_ATOL:
            raise BaselineError('pre-arrival EV projection differs from public reservation')
    elif not .2 - 2 ** -23 <= d[0] <= .7 + 2 ** -23:
        raise BaselineError('active EV SOC outside frozen range')
    elif abs((.7 - d[0]) * 60 / .9 - d[1]) > EXECUTION_ATOL:
        raise BaselineError('SOC and remaining grid demand disagree')
    if h < ins and abs(d[5] - 12) > EXECUTION_ATOL:
        raise BaselineError('pre-arrival industrial task differs from reservation')
    if d[1] > 100 / 3 + EXECUTION_ATOL or d[5] > 12 + EXECUTION_ATOL:
        raise BaselineError('remaining task exceeds declared daily service')
    n = 24 - h
    size = 6 * n
    c = [0.] * size
    bounds = []
    minimums = []
    for remaining, maximum, start, end, supplied_minimum in (
        (d[1], 7., es, ee, d[2]), (d[5], 4., ins, ine, d[6])):
        slots = max(0, end - max(start, h))
        if remaining > slots * maximum + PRIMAL_ATOL:
            raise BaselineError('task infeasible: remaining demand exceeds window capacity')
        must = max(0., remaining - maximum * (slots - 1)) if start <= h < end else 0.
        if abs(must - supplied_minimum) > EXECUTION_ATOL:
            raise BaselineError('published current device minimum is inconsistent')
        minimums.append(must)
        for t in range(h, 24):
            if start <= t < end:
                bounds.append((must if t == h else 0., min(maximum, remaining)))
            else:
                bounds.append((0., 0.))
    bounds += [(0., 160.)] * n + [(0., None)] * (3 * n)
    rows, rhs = [], []
    nominal_e, nominal_i = [], []
    for j, t in enumerate(range(h, 24)):
        c[2 * n + j] = offer if j == 0 else offer_prediction[t]
        c[3 * n + j] = balance[t]
        c[4 * n + j] = c[5 * n + j] = 1.
        row = [0.] * size
        row[j] = row[n + j] = 1.
        row[2 * n + j] = row[3 * n + j] = -1.
        rows.append(tuple(row)); rhs.append(q[t] - load[t])
        ne = (100 / 3) / (24 - es) if es <= t < 24 else 0.
        ni = 12 / (16 - ins) if ins <= t < 16 else 0.
        nominal_e.append(ne); nominal_i.append(ni)
        for group, aux, nominal in ((0, 4, ne), (1, 5, ni)):
            plus = [0.] * size; minus = [0.] * size
            plus[group * n + j] = 1.; plus[aux * n + j] = -1.
            minus[group * n + j] = -1.; minus[aux * n + j] = -1.
            rows.extend((tuple(plus), tuple(minus)))
            rhs.extend((nominal, -nominal))
    eq = []
    for group in (0, 1):
        row = [0.] * size
        row[group * n:(group + 1) * n] = [1.] * n
        eq.append(tuple(row))
    if not math.isfinite(sum(fee[h:])):
        raise BaselineError('remaining fixed contract fee sum is nonfinite')
    return RollingProblem(h, tuple(range(h, 24)), tuple(c), tuple(rows), tuple(rhs),
                          tuple(eq), (d[1], d[5]), tuple(bounds), sum(fee[h:]),
                          capacity, d, tuple(nominal_e), tuple(nominal_i))


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _certificate(problem, result):
    """Independently recompute primal / dual identities from HiGHS marginals.

    All arithmetic here is Python float64. Absolute 1e-7 MWh tolerates LP
    feasibility only; currency gap is 1e-7 + 1e-8 * max(1, |primal|, |dual|).
    Float32 observation uncertainty is kept separate in EXECUTION_ATOL.
    """
    n = len(problem.objective)
    x = tuple(float(v) for v in result.x)
    try:
        iy = tuple(float(v) for v in result.ineqlin.marginals)
        ey = tuple(float(v) for v in result.eqlin.marginals)
        ly = tuple(float(v) for v in result.lower.marginals)
        uy = tuple(float(v) for v in result.upper.marginals)
    except (AttributeError, TypeError, ValueError) as exc:
        raise BaselineError('solver returned no usable dual certificate') from exc
    if (len(x) != n or len(iy) != len(problem.inequalities) or
            len(ey) != len(problem.equalities) or len(ly) != n or len(uy) != n or
            not all(math.isfinite(v) for vector in (x, iy, ey, ly, uy) for v in vector)):
        raise BaselineError('solver certificate has invalid shape/nonfinite entries')
    ineq = max([0.] + [_dot(a, x) - b for a, b in zip(problem.inequalities, problem.inequality_rhs)])
    eq = max([0.] + [abs(_dot(a, x) - b) for a, b in zip(problem.equalities, problem.equality_rhs)])
    bound_error = max([0.] + [max(lo - v, 0. if hi is None else v - hi)
                             for v, (lo, hi) in zip(x, problem.bounds)])
    dual = _dot(problem.inequality_rhs, iy) + _dot(problem.equality_rhs, ey)
    dual += math.fsum(lo * l + (0. if hi is None else hi * u)
                      for (lo, hi), l, u in zip(problem.bounds, ly, uy))
    primal = _dot(problem.objective, x)
    stationarity = max(abs(problem.objective[j] - math.fsum(a[j] * y for a, y in zip(problem.inequalities, iy))
                           - math.fsum(a[j] * y for a, y in zip(problem.equalities, ey)) - ly[j] - uy[j])
                         for j in range(n))
    sign_error = max([0.] + list(iy) + [-v for v in ly] + list(uy) +
                     [abs(uy[j]) for j, (_, hi) in enumerate(problem.bounds) if hi is None])
    gap = abs(primal - dual)
    gap_limit = DUAL_ATOL + GAP_RTOL * max(1., abs(primal), abs(dual))
    if not all(math.isfinite(v) for v in (primal, dual, gap, gap_limit, stationarity, sign_error)):
        raise BaselineError('certificate arithmetic is nonfinite')
    if max(ineq, eq, bound_error) > PRIMAL_ATOL or max(stationarity, sign_error) > DUAL_ATOL or gap > gap_limit:
        raise BaselineError('LP primal/dual certificate exceeds frozen tolerance')
    return dict(primal_violation=max(ineq, eq, bound_error), equality_residual=eq,
                dual_stationarity_residual=stationarity, dual_sign_violation=sign_error,
                primal_objective_cny=primal, dual_objective_cny=dual,
                absolute_duality_gap_cny=gap, gap_limit_cny=gap_limit,
                contract_fee_constant_cny=problem.contract_fee_constant,
                total_predictive_objective_cny=primal + problem.contract_fee_constant)


@dataclass(frozen=True)
class RollingDecision:
    action: tuple
    plan: dict
    certificate: dict
    problem: RollingProblem


def solve_dispatch(observation, *, budget):
    """Solve once, execute only first hour; failures consume the attempt and raise.

    Action order: spot_ratio, EV_power_ratio, industrial_power_ratio. Values are
    float64 requests; no silent float32 conversion can lower the hard task goal.
    """
    if not isinstance(budget, SolveBudget):
        raise BaselineError('explicit SolveBudget required')
    problem = build_problem(observation)
    attempt = budget.reserve(len(problem.objective), len(problem.inequalities), len(problem.equalities))
    started = time.monotonic()
    try:
        from scipy.optimize import linprog
        result = linprog(problem.objective, A_ub=problem.inequalities,
                         b_ub=problem.inequality_rhs, A_eq=problem.equalities,
                         b_eq=problem.equality_rhs, bounds=problem.bounds,
                         method='highs', options=dict(time_limit=budget.time_limit,
                            primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        attempt.update(solver_status=int(result.status), solver_message=str(result.message),
                       solver_success=bool(result.success))
        if not result.success or result.status != 0:
            raise BaselineError('LP did not return an optimal solution: ' + str(result.message))
        certificate = _certificate(problem, result)
        n = len(problem.hours)
        groups = [tuple(float(v) for v in result.x[j * n:(j + 1) * n]) for j in range(6)]
        ev, ind, spot = groups[0][0], groups[1][0], groups[2][0]
        # Clip solver-level 1e-9 bound roundoff only; fail rather than hide a
        # material request/execution discrepancy. Mapping includes forced floors.
        action = (min(1., max(0., spot / problem.current_capacity)),
                  min(1., max(0., ev / 7)), min(1., max(0., ind / 4)))
        mapped = predicted_execution(problem, action)
        if max(abs(mapped['ev_grid_mwh'] - ev), abs(mapped['industrial_mwh'] - ind),
               abs(action[0] * problem.current_capacity - spot)) > EXECUTION_ATOL:
            raise BaselineError('LP first hour cannot be executed by existing actuator')
        certificate.update(expected_first_hour_execution=mapped,
                           execution_projection=dict(mapped, status='PREDICTED_NOT_OBSERVED',
                               task_equalities_preserved=True, observed_state_dtype='float32',
                               scope='current-hour unchanged actuator mapping'),
                           execution_projection_atol_mwh=EXECUTION_ATOL)
        attempt.update(state='PASS', certificate=certificate)
        plan = dict(zip(('ev_grid_mwh', 'industrial_mwh', 'spot_mwh', 'balance_mwh',
                         'ev_absolute_deviation_mwh', 'industrial_absolute_deviation_mwh'), groups))
        return RollingDecision(action, plan, certificate, problem)
    except BaseException as exc:
        attempt.update(state='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        attempt['elapsed_seconds'] = time.monotonic() - started


def predicted_execution(problem, action):
    if len(action) != 3 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in action):
        raise BaselineError('action must be finite ratios in [0,1]')
    d, h = problem.device_snapshot, problem.hour
    er, ir = float(action[1]) * 7, float(action[2]) * 4
    ev = min(7., d[1], max(er, d[2])) if d[3] <= h < d[4] else 0.
    ind = min(4., d[5], max(ir, d[6])) if d[7] <= h < d[8] else 0.
    return dict(ev_grid_mwh=ev, industrial_mwh=ind,
                ev_request_mw=er, industrial_request_mw=ir,
                ev_forced_mwh=max(0., ev - er), industrial_forced_mwh=max(0., ind - ir),
                flex_cost=abs(ev - problem.nominal_ev[0]) + abs(ind - problem.nominal_industrial[0]))


def verify_execution(decision, trusted_hour_row):
    """Call after the unchanged trusted H_END settlement. No forecasting of truth.

    Deviations stop the comparison. This certifies the actuation mapping only;
    forecast-vs-realized load and balancing prices legitimately differ.
    """
    expected = decision.certificate['expected_first_hour_execution']
    try:
        actual_hour = float(trusted_hour_row['hour'])
        if actual_hour != decision.problem.hour:
            raise BaselineError('trusted execution row belongs to a different hour')
        errors = {}
        for name in ('ev_grid_mwh', 'industrial_mwh', 'ev_request_mw', 'industrial_request_mw',
                     'ev_forced_mwh', 'industrial_forced_mwh', 'flex_cost'):
            value = float(trusted_hour_row[name])
            if not math.isfinite(value):
                raise BaselineError('nonfinite trusted actuator result')
            errors[name] = abs(value - expected[name])
        spot = float(trusted_hour_row['spot_mwh'])
        errors['spot_mwh'] = abs(spot - decision.action[0] * decision.problem.current_capacity)
        if not math.isfinite(spot) or max(errors.values()) > EXECUTION_ATOL:
            raise BaselineError('actual actuator execution differs from LP first-hour mapping')
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineError('missing/invalid trusted execution evidence') from exc
    return dict(state='PASS',
                maximum_energy_error_mwh=max(v for k, v in errors.items() if k != 'flex_cost'),
                flex_cost_error_cny=errors['flex_cost'], field_errors=errors)
