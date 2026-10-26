"""R02 annual fixture integration draft. Execute only through an approved runner.

No trained policy, PPO controller, disk dataset or historical result is used.
The runner must enforce the total construction/hour/time/output budgets.
"""
import math
from dataclasses import asdict
import unittest

import numpy as np

from v31.lite_gym import AnnualDecisionEnv
from v31.lite_rules import RULES
from v31.schema import instant

# No direct unittest/discover entry may materialize a world without its owner.
RUN_CONTEXT = None
SPEC = dict(year=2033, family="r02_annual_fixture", group="G00",
            split="train", fixture=True)


def annual_close(test, actual, expected, terms):
    # Frozen float64 accumulation bound: eight rounding operations per term,
    # plus1e-6 CNY absolute floor. It is not chosen from observed differences.
    scale = max(1.0, math.fsum(abs(float(x)) for x in terms))
    tolerance = 1e-6 + 8 * len(terms) * np.finfo(np.float64).eps * scale
    test.assertLessEqual(abs(float(actual) - float(expected)), tolerance)
    return tolerance


class AnnualChainTests(unittest.TestCase):
    def test_one_fixture_two_decision_views_full_delivery(self):
        if RUN_CONTEXT is None:
            raise RuntimeError("explicitly approved bounded annual runner required")
        ctx = RUN_CONTEXT
        upper_rule, lower_rule = RULES["B2"]
        environments = []
        records = {}
        for level in ("upper", "lower"):
            env = ctx.new_environment(SPEC.copy(), level,
                                      lower_rule if level == "upper" else upper_rule)
            self.assertIsInstance(env, AnnualDecisionEnv)
            if environments:
                # Cache only exogenous vintages already audited by the first pass.
                # No unverified cache, clock override, shortened year or fake audit.
                self.assertTrue(environments[0].driver.store.completed_year)
                env.cache = environments[0].cache
            env.reset(seed=20261024)
            self.assertEqual(env.driver.store.n, 8760)
            self.assertEqual(env.driver.store.manifest()["fixture"], True)
            self.assertEqual(env.driver.store.family, SPEC["family"])
            driver = env.driver
            # A narrowly scoped counterfactual projection check at the first
            # observation only. Owner-only future arrays never reach the policy.
            original_arrays = driver.c.world._arrays
            before_projection = {k: v.copy() for k, v in env.raw.items()}
            try:
                changed = dict(original_arrays)
                for name in ("realized_load", "balancing_price"):
                    changed[name] = original_arrays[name] + 123.0
                driver.c.world._arrays = changed
                projection = (driver.upper_observation() if level == "upper"
                              else driver.lower_observation())
                self.assertTrue(all(np.array_equal(v, projection[k])
                                    for k, v in before_projection.items()))
            finally:
                driver.c.world._arrays = original_arrays
            step_boundaries = []
            initial_cost = driver.c.total_purchase + driver.c.total_flex
            self.assertEqual(initial_cost, 0.0)
            # The guard, installed before reset, counts lower partner decisions,
            # publications and H_END as well as chosen-layer step calls.
            while not env.ended:
                ctx.check_deadline()
                if level == "upper":
                    verb, ratios = upper_rule(env.raw)
                    action = np.r_[("WAIT", "SKIP", "COMMIT").index(verb), ratios]
                else:
                    action = lower_rule(env.raw)
                _, reward, terminated, truncated, info = env.step(action)
                step_boundaries.append(dict(step_index=len(env.steps)-1,
                    now=driver.c.now.isoformat(), completed_hour=driver.c.t,
                    lot_count=len(driver.c.ledger.lots),
                    ledger_version=driver.c.ledger.version))
                self.assertFalse(truncated)
                self.assertEqual(reward, info["raw_reward"] / 1e6)
                self.assertEqual(terminated, env.ended)
                self.assertEqual(info["purchase_cost"], driver.c.total_purchase)
                self.assertEqual(info["flex_cost"], driver.c.total_flex)
            self.assertEqual(driver.c.t, 8760)
            self.assertEqual(len(driver.c.hours), 8760)
            self.assertIsNone(driver.c.pending)
            self.assertEqual(len(driver.daily_receipts), 365)
            self.assertEqual(len(driver.context.sealed), 365)
            self.assertIsNone(driver.c.next_op())
            self.assertEqual(float(driver.c.ledger.fee[driver.c.t:].sum()), 0.0)
            self.assertEqual(float(driver.c.ledger.quantity[driver.c.t:].sum()), 0.0)
            # Recompute through the existing independent auditor, not settlement().
            # step() already calls the unchanged independent audit_year().
            # Reuse that successful record instead of running its full loop twice.
            self.assertEqual(len(env.episodes), 1)
            audit = env.episodes[0]
            hours = driver.c.hours
            row_costs = [r["purchase_cost"] + r["flex_cost"] for r in hours]
            objective = math.fsum(row_costs)
            tolerances = {}
            tolerances["coordinator"] = annual_close(
                self, driver.c.total_purchase + driver.c.total_flex, objective, row_costs)
            tolerances["auditor"] = annual_close(
                self, audit["objective_cost"], objective, row_costs)
            raw_rewards = [s["raw_reward"] for s in env.steps]
            tolerances["raw_reward"] = annual_close(
                self, -math.fsum(raw_rewards), objective, row_costs + raw_rewards)
            tolerances["scaled_reward"] = annual_close(
                self, -math.fsum(s["training_reward"] for s in env.steps) * 1e6,
                objective, row_costs + raw_rewards)
            tolerances["hourly_reward"] = annual_close(
                self, -math.fsum(r["reward"] for r in hours), objective, row_costs)
            prefix_contract_fee = np.r_[0.0, np.cumsum(
                [r["contract_cost"] for r in hours], dtype=np.float64)]
            for step in env.steps:
                t = step["completed_hour"]
                # At this decision boundary use signed fees only; don't include
                # contracts that will be signed after this time in prefix liability.
                paid = float(prefix_contract_fee[t])
                self.assertLessEqual(
                    abs(paid + step["unsettled_contract_fee"] - step["signed_contract_fee"]),
                    1e-6 + 8 * 8760 * np.finfo(np.float64).eps *
                    max(1.0, abs(step["signed_contract_fee"])))
            publication_times = [instant(e["at"]) for e in driver.events]
            self.assertEqual(publication_times, sorted(publication_times))
            self.assertEqual(sum(e["event"] == "DA_seal" for e in driver.events), 365)
            for macro in driver.macro_receipts:
                self.assertLessEqual(instant(macro["published_at"]), instant(macro["at"]))
                self.assertLessEqual(instant(macro["at"]), instant(macro["expires_at"]))
            # Sample lineage at Y/M/I and first/final DA; the complete causal
            # publication trace remains subject to the dedicated runner guards.
            tags = {"op:0", "day:0", "day:364"}
            for stage in ("M", "I"):
                tags.add("op:" + str(next(i for i, op in enumerate(driver.c.world.ops)
                                         if op.stage == stage)))
            lineage = []
            for tag in sorted(tags):
                for forecast in driver.context.records[tag]:
                    self.assertLessEqual(instant(forecast.as_of), instant(forecast.published_at))
                    self.assertLess(instant(forecast.published_at), forecast.grid.start)
                    for ref in forecast.input_vintages:
                        h = driver.context.history.resolve(ref, forecast.as_of)
                        self.assertLessEqual(instant(h.available_at), instant(forecast.as_of))
                    lineage.append(dict(tag=tag, forecast=forecast.content_sha256,
                                        as_of=forecast.as_of, published_at=forecast.published_at,
                                        input_count=len(forecast.input_vintages)))
            if level == "upper":
                self.assertTrue(any(s["completed_hour"] == 0 and s["raw_reward"] == 0.0
                                    for s in env.steps))
            # Full-year contract costs reconstructed from original lot quantities.
            lot_fees = [math.fsum(l.quantities) * l.unit_price for l in driver.c.ledger.lots]
            fee_rows = [r["contract_cost"] for r in hours]
            tolerances["contract_once"] = annual_close(
                self, math.fsum(fee_rows), math.fsum(lot_fees), fee_rows + lot_fees)
            self.assertEqual(
                [(r["t"], r["hour"]) for r in hours],
                [(t, t % 24) for t in range(8760)])
            self.assertEqual(len(env.episodes), 1)
            self.assertEqual(env.episodes[0]["row_sha256"], audit["row_sha256"])
            # Never reset to a third instance merely for archive/cleanup.
            env._archive()
            self.assertEqual(len(env.recorded_hours), 8760)
            if environments:
                other = environments[0]
                self.assertIsNot(env.driver, other.driver)
                self.assertIsNot(env.driver.c.ledger, other.driver.c.ledger)
                self.assertIsNot(env.driver.c.devices, other.driver.c.devices)
                self.assertEqual(audit["row_sha256"], records["upper"]["audit"]["row_sha256"])
                self.assertEqual(audit["market_sha256"], records["upper"]["audit"]["market_sha256"])
                self.assertEqual(driver.store.manifest()["arrays_sha256"],
                                 other.driver.store.manifest()["arrays_sha256"])
                annual_close(self, objective, records["upper"]["objective"], row_costs)
            records[level] = dict(
                audit=audit, objective=objective, tolerances=tolerances,
                steps=env.steps, step_boundaries=step_boundaries,
                contracts=[asdict(lot) for lot in driver.c.ledger.lots],
                macro_receipts=driver.macro_receipts,
                daily_receipts=driver.daily_receipts, events=driver.events,
                fixture_manifest=driver.store.manifest(), sampled_lineage=lineage,
                first_observation_future_projection_unchanged=True)
            # Owner writes bounded, new-only raw evidence. First view stores hourly
            # rows once; the second reuses those rows only after full hash equality.
            ctx.record_view(level, records[level], hours if level == "upper" else None)
            environments.append(env)
        ctx.assert_total_usage(max_new_stores=1, max_replays=1,
                               drivers=2, settled_hours=17520)
