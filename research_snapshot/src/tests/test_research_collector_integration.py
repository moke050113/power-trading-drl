"""R02 real PPO collector + real policy + real annual environment prefixes.
No optimization, checkpoints, economic evaluation or annual completion.
"""
import math
import unittest
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

CONTEXT = None
RTOL, ATOL = 1.3e-6, 1e-5  # pre-fixed Torch float32 limits, reused from E39


def copied(value):
    return {k:v.copy() for k,v in value.items()}


class CapturedRollout(BaseCallback):
    def __init__(self):
        super().__init__(); self.rows=[]
    def _on_step(self):
        x=self.locals
        self.rows.append(dict(
            observations=copied(self.model._last_obs),
            raw_actions=x["actions"].copy(),
            clipped_env_actions=x["clipped_actions"].copy(),
            values=x["values"].detach().cpu().numpy().copy(),
            log_prob=x["log_probs"].detach().cpu().numpy().copy(),
            rewards=x["rewards"].copy(),dones=x["dones"].copy(),
            infos=[dict(v) for v in x["infos"]]))
        return True


class CollectorIntegrationTests(unittest.TestCase):
    def test_both_real_collectors_two_transitions_without_updates(self):
        if CONTEXT is None:
            raise RuntimeError("use the separately authorized bounded collector runner")
        ctx=CONTEXT
        for layer in ("upper","lower"):
            model,env=ctx.new_model(layer)
            initial=ctx.fingerprint(model)
            callback=CapturedRollout()
            # Real SB3 lifecycle setup only. learn/train remain guarded out.
            total,callback=model._setup_learn(2,callback=callback,reset_num_timesteps=True)
            self.assertEqual(total,2)
            self.assertEqual(model.num_timesteps,0)
            ok=model.collect_rollouts(model.get_env(),callback,model.rollout_buffer,2)
            self.assertTrue(ok)
            self.assertEqual(model.num_timesteps,2)  # collected transitions, not updates
            self.assertEqual(model._n_updates,0)
            self.assertEqual(callback.n_calls,2)
            self.assertEqual(len(callback.rows),2)
            buffer=model.rollout_buffer
            self.assertTrue(buffer.full)
            self.assertEqual(buffer.pos,2)
            self.assertEqual(len(env.steps),2)
            self.assertFalse(env.ended)
            self.assertFalse(env.driver.store.completed_year)
            self.assertIsNone(env.driver.c.pending)
            self.assertEqual(env.driver.c.t,0 if layer=="upper" else 2)
            np.testing.assert_array_equal(buffer.episode_starts[:,0],[1,0])
            for i,row in enumerate(callback.rows):
                self.assertFalse(bool(row["dones"][0]))
                self.assertFalse(row["infos"][0].get("TimeLimit.truncated",False))
                np.testing.assert_array_equal(buffer.actions[i],row["raw_actions"])
                np.testing.assert_array_equal(buffer.log_probs[i],row["log_prob"])
                for k in buffer.observations:
                    np.testing.assert_array_equal(buffer.observations[k][i],row["observations"][k])
                np.testing.assert_array_equal(row["clipped_env_actions"],
                    np.clip(row["raw_actions"],model.action_space.low,model.action_space.high))
                np.testing.assert_array_equal(
                    np.asarray(env.steps[i]["action"]),row["clipped_env_actions"][0])
                if layer=="upper":
                    np.testing.assert_array_equal(row["raw_actions"],row["clipped_env_actions"])
                    code=float(row["raw_actions"][0,0])
                    self.assertIn(code,(0.,1.,2.))
                    if row["observations"]["calendar"][0,2]==1:
                        self.assertNotEqual(code,0.)
                expected=np.float32(env.steps[i]["raw_reward"]/1e6)
                self.assertEqual(float(buffer.rewards[i,0]),float(expected))
                self.assertEqual(float(row["rewards"][0]),float(expected))
            rewards=buffer.rewards[:,0].astype(np.float64).copy()
            old_values=buffer.values[:,0].astype(np.float64).copy()
            bootstrap=float(ctx.last_bootstrap[layer].reshape(-1)[0])
            next_value,acc=bootstrap,0.0
            independent=np.zeros(2,np.float64)
            for i in (1,0):
                delta=rewards[i]+next_value-old_values[i]
                acc=delta+.95*acc
                independent[i]=acc
                next_value=old_values[i]
            np.testing.assert_allclose(buffer.advantages[:,0],independent,rtol=RTOL,atol=ATOL)
            np.testing.assert_allclose(buffer.returns[:,0],independent+old_values,rtol=RTOL,atol=ATOL)
            stored_advantages=buffer.advantages.copy()
            stored_returns=buffer.returns.copy()
            sample=next(buffer.get(batch_size=2))
            joined=[]
            for i in range(2):
                hits=[j for j,row in enumerate(callback.rows) if all(
                    np.array_equal(sample.observations[k][i].cpu().numpy(),row["observations"][k][0])
                    for k in row["observations"])]
                self.assertEqual(len(hits),1)
                j=hits[0];joined.append(j)
                np.testing.assert_array_equal(sample.actions[i].cpu().numpy(),callback.rows[j]["raw_actions"][0])
                self.assertEqual(float(sample.old_log_prob[i]),float(callback.rows[j]["log_prob"][0]))
            with torch.no_grad():
                value,new_log_prob,_=model.policy.evaluate_actions(sample.observations,sample.actions)
            torch.testing.assert_close(new_log_prob,sample.old_log_prob,rtol=RTOL,atol=ATOL)
            torch.testing.assert_close(value.flatten(),sample.old_values,rtol=RTOL,atol=ATOL)
            ratio=(new_log_prob-sample.old_log_prob).exp()
            torch.testing.assert_close(ratio,torch.ones_like(ratio),rtol=RTOL,atol=ATOL)
            final=ctx.fingerprint(model)
            self.assertEqual(initial,final)
            self.assertFalse(model.policy.optimizer.state)
            self.assertTrue(all(p.grad is None for p in model.policy.parameters()))
            self.assertEqual(model._n_updates,0)
            c=env.driver.c
            signed=math.fsum(math.fsum(x.quantities)*x.unit_price for x in c.ledger.lots)
            delivered=math.fsum(x["contract_cost"] for x in c.hours)
            remaining=float(c.ledger.fee[c.t:].sum())
            liability_tolerance=1e-6+8*8760*np.finfo(np.float64).eps*max(1.,abs(signed))
            self.assertLessEqual(abs(signed-delivered-remaining),liability_tolerance)
            # A permitted artificial open-year snapshot, NOT a restart checkpoint.
            ctx.record(layer,dict(
                rows=callback.rows,buffer_advantages=stored_advantages,
                buffer_returns=stored_returns,independent_gae=independent,
                bootstrap=bootstrap,sample_order=joined,ratio=ratio.cpu().numpy(),
                initial_state=initial,final_state=final,
                collected_transitions=model.num_timesteps,optimizer_updates=model._n_updates,
                encoded_next_observations=copied(model._last_obs),
                environment_steps=env.steps,hour_rows=c.hours,
                contracts=[asdict(x) for x in c.ledger.lots],
                completed_hours=c.t,remaining_hours=env.driver.store.n-c.t,
                signed_contract_fee=signed,delivered_contract_fee=delivered,
                liability_residual_cny=signed-delivered-remaining,
                liability_tolerance_cny=liability_tolerance,
                clipped_dimensions=sum(int(np.count_nonzero(
                    row["raw_actions"]!=row["clipped_env_actions"])) for row in callback.rows),
                remaining_signed_contract_fee=remaining,
                remaining_signed_contract_mwh=float(c.ledger.quantity[c.t:].sum()),
                next_phase=env.driver.phase,now=c.now.isoformat(),
                pending=c.pending is not None,annual_complete=False,
                fixture_manifest=env.driver.store.manifest(),
                limitation="two nonterminal transitions only; no tail or replay recovery claim"))
        ctx.check_final_counts()
