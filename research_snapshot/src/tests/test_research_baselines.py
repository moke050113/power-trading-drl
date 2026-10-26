"""Bounded R03 LP checks. Importing this module never executes a solver.

The owner must inject RUN_CONTEXT before unittest execution. No context is an
error, not a skipped/passed validation. These use synthetic observations and one
25-prepare DailyDevices case; no world, model, file dataset or training is used.
"""
from copy import deepcopy
from types import SimpleNamespace
import builtins
import math
import unittest
from unittest.mock import patch

from research.prediction_optimization_baseline import (
    BaselineError, EXECUTION_ATOL, SolveBudget, SolveBudgetExceeded,
    build_problem, predicted_execution, solve_dispatch, verify_execution,
)


RUN_CONTEXT = None


def observation(hour=23, ev_remaining=1., *, offer=100., balancing=200., contract=0.):
    # Fixed public daily task; earlier consumption is summarized by current SOC.
    slots = 24 - hour
    return dict(hour=[hour], forecast_load=[1.] * 24,
                forecast_price=[balancing] * 24, forecast_offer_price=[offer] * 24,
                contract_plan=[contract] * 24, contract_fee=[0.] * 24,
                offer=[offer, 160.],
                devices=[.7 - .9 * ev_remaining / 60, ev_remaining,
                         max(0., ev_remaining - 7 * (slots - 1)), 18., 24.,
                         0., 0., 8., 16.],
                actual_state_mask=[1., 1.], declared_initial_soc=[.2])


class PredictionOptimizationBaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if RUN_CONTEXT is None:
            raise RuntimeError('R03 checks require explicit bounded RUN_CONTEXT authorization')
        if not isinstance(RUN_CONTEXT.get('solver_budget'), SolveBudget):
            raise RuntimeError('RUN_CONTEXT must contain the shared approved SolveBudget')
        if not isinstance(RUN_CONTEXT.get('results'), list):
            raise RuntimeError('RUN_CONTEXT results ledger missing')

    def solve(self, obs):
        result = solve_dispatch(obs, budget=RUN_CONTEXT['solver_budget'])
        RUN_CONTEXT['results'].append(dict(test=self.id(), action=list(result.action),
                                           certificate=result.certificate,
                                           plan={k: list(v) for k, v in result.plan.items()}))
        return result

    def test_one_hour_known_optimum_and_fixed_fee(self):
        obs = observation()
        obs['contract_fee'][23] = 50.
        result = self.solve(obs)
        self.assertAlmostEqual(result.plan['ev_grid_mwh'][0], 1., places=7)
        self.assertAlmostEqual(result.plan['industrial_mwh'][0], 0., places=7)
        self.assertAlmostEqual(result.plan['spot_mwh'][0], 2., places=7)
        self.assertAlmostEqual(result.plan['balance_mwh'][0], 0., places=7)
        expected_flex = 100 / 18 - 1
        self.assertAlmostEqual(result.certificate['primal_objective_cny'], 200 + expected_flex, places=7)
        self.assertAlmostEqual(result.certificate['total_predictive_objective_cny'], 250 + expected_flex, places=7)
        self.assertLessEqual(result.certificate['primal_violation'], 1e-7)
        self.assertLessEqual(result.certificate['dual_stationarity_residual'], 1e-7)
        self.assertLessEqual(result.certificate['absolute_duality_gap_cny'], result.certificate['gap_limit_cny'])

    def test_two_hour_known_optimum_keeps_task_equality(self):
        obs = observation(hour=22, ev_remaining=7., offer=100., balancing=400.)
        obs['forecast_load'] = [0.] * 24
        obs['forecast_offer_price'][23] = 300.
        result = self.solve(obs)
        self.assertAlmostEqual(sum(result.plan['ev_grid_mwh']), 7., places=7)
        self.assertAlmostEqual(result.plan['ev_grid_mwh'][0], 7., places=7)
        self.assertAlmostEqual(result.plan['ev_grid_mwh'][1], 0., places=7)
        self.assertAlmostEqual(result.certificate['primal_objective_cny'], 707., places=7)

    def test_current_public_offer_overrides_older_prediction(self):
        obs = observation(offer=300., balancing=200.)
        obs['forecast_offer_price'][23] = 100.
        result = self.solve(obs)
        self.assertAlmostEqual(result.plan['spot_mwh'][0], 0., places=7)
        self.assertAlmostEqual(result.plan['balance_mwh'][0], 2., places=7)
        self.assertAlmostEqual(result.certificate['primal_objective_cny'], 400 + 100 / 18 - 1, places=7)
        # A truly cheaper currently published quote changes the optimum.
        obs['offer'][0] = 100.
        cheaper = self.solve(obs)
        self.assertAlmostEqual(cheaper.plan['spot_mwh'][0], 2., places=7)

    def test_problem_only_reads_legal_projection_fields(self):
        obs = observation(hour=22, ev_remaining=7.)
        class LegalView(dict):
            def __getitem__(self, key):
                if key in ('realized_load', 'future_offer', 'balancing_price', 'driver', 'world', 'history'):
                    raise AssertionError('forbidden field accessed: ' + key)
                return super().__getitem__(key)
        clean = build_problem(obs)
        projected = LegalView(deepcopy(obs))
        projected.update(realized_load=object(), future_offer=object(), balancing_price=object(),
                         driver=object(), world=object(), history=object())
        self.assertEqual(clean, build_problem(projected))
        self.assertEqual(len(clean.objective), 12)
        self.assertEqual(len(clean.inequalities), 10)
        self.assertEqual(clean.equality_rhs, (7., 0.))
        # Future *published forecast* is legitimate and changes coefficients.
        projected['forecast_offer_price'][23] += 17.
        self.assertNotEqual(clean.objective, build_problem(projected).objective)

    def test_infeasible_task_is_rejected_without_reducing_service(self):
        obs = observation(ev_remaining=7.5)
        before = len(RUN_CONTEXT['solver_budget'].attempts)
        with self.assertRaisesRegex(BaselineError, 'infeasible'):
            solve_dispatch(obs, budget=RUN_CONTEXT['solver_budget'])
        self.assertEqual(obs['devices'][1], 7.5)
        self.assertEqual(len(RUN_CONTEXT['solver_budget'].attempts), before)

    def test_default_budget_denies_before_scipy_import(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'scipy.optimize':
                raise AssertionError('dependency import happened before budget rejection')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            with self.assertRaises(SolveBudgetExceeded):
                solve_dispatch(observation(), budget=SolveBudget())

    def test_solver_failure_consumes_attempt_without_fallback(self):
        budget = RUN_CONTEXT['solver_budget']
        before = len(budget.attempts)
        # This failed mocked solver entry consumes the SAME approved ledger.
        with patch('scipy.optimize.linprog', return_value=SimpleNamespace(
                status=2, success=False, message='synthetic infeasible result')) as solver:
            with self.assertRaisesRegex(BaselineError, 'did not return an optimal'):
                solve_dispatch(observation(), budget=budget)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(len(budget.attempts), before + 1)
        self.assertEqual(budget.attempts[-1]['state'], 'FAILED')
        self.assertEqual(budget.attempts[-1]['solver_status'], 2)
        RUN_CONTEXT['results'].append(dict(test=self.id(), failed_attempt=dict(budget.attempts[-1]), mocked=True))

    def test_missing_nonfinite_or_inconsistent_input_fails_closed(self):
        for key in ('forecast_offer_price', 'actual_state_mask', 'contract_fee'):
            obs = observation(); del obs[key]
            with self.subTest(missing=key), self.assertRaises(BaselineError):
                build_problem(obs)
        for bad in (float('nan'), float('inf'), -1.):
            obs = observation(); obs['forecast_load'][23] = bad
            with self.subTest(bad=bad), self.assertRaises(BaselineError):
                build_problem(obs)
        obs = observation(); obs['devices'][2] = 0.
        with self.assertRaisesRegex(BaselineError, 'minimum'):
            build_problem(obs)

    def test_prearrival_uses_public_task_not_hidden_actual_soc(self):
        obs = observation(hour=0)
        obs['devices'] = [0., 100 / 3, 0., 18., 24., 12., 0., 8., 16.]
        obs['actual_state_mask'] = [0., 0.]
        problem = build_problem(obs)
        self.assertEqual(problem.equality_rhs, (100 / 3, 12.))
        self.assertEqual(len(problem.objective), 144)
        self.assertEqual(len(problem.inequalities), 120)
        self.assertEqual(problem.bounds[0], (0., 0.))
        self.assertEqual(problem.bounds[24], (0., 0.))
        self.assertEqual(predicted_execution(problem, (0., 1., 1.))['ev_grid_mwh'], 0.)
        self.assertEqual(predicted_execution(problem, (0., 1., 1.))['industrial_mwh'], 0.)

    def test_real_daily_executor_matches_plan_and_protection_mapping(self):
        import numpy as np
        from v3.devices import DailyDevices
        devices = DailyDevices()
        # Legal chronological prefix only: zero requests cause the unchanged
        # executor to provide compulsory service. No clock or SOC is assigned.
        for _ in range(23):
            devices, _row = devices.prepare(0., 0.)
        state = devices.state()
        obs = observation(ev_remaining=state['ev_remaining'])
        obs['devices'] = [state[k] for k in ('ev_soc', 'ev_remaining', 'ev_min', 'ev_start',
                           'ev_end', 'industrial_remaining', 'industrial_min',
                           'industrial_start', 'industrial_end')]
        # Match LiteAnnualDriver.lower_observation's float32 projection while
        # the trusted DailyDevices instance retains its original float64 state.
        # No executor state, clock, or task target is rounded or reassigned.
        obs = {key: np.asarray(values, dtype=np.float32) for key, values in obs.items()}
        input_dtypes = {key: str(values.dtype) for key, values in obs.items()}
        self.assertTrue(all(dtype == 'float32' for dtype in input_dtypes.values()))
        result = self.solve(obs)
        _, row = devices.prepare(result.action[1] * 7, result.action[2] * 4)
        row.update(hour=23, ev_grid_mwh=row['ev_actual_mw'], industrial_mwh=row['industrial_actual_mw'],
                   spot_mwh=result.action[0] * 160)
        receipt = verify_execution(result, row)
        self.assertEqual(receipt['state'], 'PASS')
        # Separate counterfactual requests on the same input state reveal the
        # compulsory minimum; they are not labeled the LP action or another year.
        _, forced = devices.prepare(0., 0.)
        projection = predicted_execution(result.problem, (0., 0., 0.))
        for key, actual in (('ev_grid_mwh', forced['ev_actual_mw']),
                            ('industrial_mwh', forced['industrial_actual_mw']),
                            ('ev_forced_mwh', forced['ev_forced_mwh']),
                            ('industrial_forced_mwh', forced['industrial_forced_mwh']),
                            ('flex_cost', forced['flex_cost'])):
            self.assertLessEqual(abs(projection[key] - actual), EXECUTION_ATOL)
        self.assertAlmostEqual(forced['ev_forced_mwh'], 7., places=7)
        bad = dict(row, ev_grid_mwh=0.)
        with self.assertRaisesRegex(BaselineError, 'differs'):
            verify_execution(result, bad)
        RUN_CONTEXT['results'].append(dict(test=self.id(), device_prepare_calls=25,
                                           actual=row, compulsory_counterfactual=forced,
                                           execution_check=receipt, projection=projection,
                                           input_dtypes=input_dtypes,
                                           execution_projection_residuals=receipt['field_errors'],
                                           compulsory_projection_residuals={
                                               'ev_grid_mwh': abs(projection['ev_grid_mwh'] - forced['ev_actual_mw']),
                                               'industrial_mwh': abs(projection['industrial_mwh'] - forced['industrial_actual_mw']),
                                               'ev_forced_mwh': abs(projection['ev_forced_mwh'] - forced['ev_forced_mwh']),
                                               'industrial_forced_mwh': abs(projection['industrial_forced_mwh'] - forced['industrial_forced_mwh']),
                                               'flex_cost_cny': abs(projection['flex_cost'] - forced['flex_cost'])}))
