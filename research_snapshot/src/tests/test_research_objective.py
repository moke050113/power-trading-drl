"""Artificial-only R02/R04 checks. No world generation or neural policy forward."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.distributions import DiagGaussianDistribution
from v3.core import PositionLedger, opportunities, settlement, year_axis
from v31.lite_gym import AnnualDecisionEnv, decode_hybrid, encode, UPPER_SHAPES
from experiments.hybrid_upper_v3 import HybridDistribution
from research.tail_budget import TailBudget, TailStopped, finish_tail


class ObjectiveTests(unittest.TestCase):
    def test_three_two_two_prices_and_liability_partition(self):
        # Existing BoundaryTests.test_three_two_two_and_price_handcalculation,
        # reused here because importing that original module probes old outputs.
        ledger = PositionLedger(2033, np.full(8760, 10.))
        for scope, target, price in (("Y", .3, 400), ("M1", .5, 420), ("I1-1", .7, 450)):
            op = next(o for o in opportunities(2033) if o.scope == scope and o.index == 0)
            ledger.decide(op, "COMMIT", np.full(op.end-op.start, target), price,
                now=op.decision_at, published_at=op.published_at, expires_at=op.decision_at,
                expected_version=ledger.version, trade_id=scope)
        self.assertEqual([lot.quantities[0] for lot in ledger.lots], [3., 2., 2.])
        self.assertEqual(ledger.fee[0], 2940.)
        self.assertEqual(settlement(9, 0, 0, 7, 2940, 1, 500, 600)["purchase_cost"], 4040.)
        for t in (0, 1, 743, 8759, 8760):
            self.assertAlmostEqual(ledger.fee.sum(), ledger.fee[:t].sum()+ledger.fee[t:].sum(), places=7)
        self.assertEqual(sum(sum(l.quantities)*l.unit_price for l in ledger.lots), ledger.fee.sum())

    def test_overbuy_is_charged_and_no_refund(self):
        row = settlement(1, 0, 0, 3, 1200, 2, 500, 600)
        self.assertEqual((row["unused_contract_mwh"], row["unused_spot_mwh"]), (2, 2))
        self.assertEqual(row["purchase_cost"], 2200)
        self.assertEqual(row["balance_mwh"], 0)

    def test_valid_reward_signs_and_invalid_negative_economics(self):
        for cost in (0., 1., 1e9):
            row=settlement(0, 0, 0, 0, cost, 0, 0, 0)
            reward=-row["purchase_cost"]
            self.assertEqual(reward, -cost)
            self.assertLessEqual(reward, 0)
        with self.assertRaises(ValueError):
            settlement(1, 0, 0, 0, 0, 0, -1, 600)
        # Positive raw reward is impossible under the frozen nonnegative cost model.

    def test_interior_spot_optimum_not_always_buy_max(self):
        def cost(spot, price):
            return settlement(9, 0, 0, 2, 800, spot, price, 600)["purchase_cost"]
        self.assertEqual([cost(s, 400) for s in (0, 7, 8)], [5000, 3600, 4000])
        self.assertLess(cost(0, 700), cost(7, 700))  # expensive offer reverses ranking

    def test_annual_and_leap_hour_boundaries(self):
        for year, n in ((2033, 8760), (2036, 8784)):
            self.assertEqual(year_axis(year)[1], n)
            ops=opportunities(year)
            self.assertEqual(len(ops), 98)
            self.assertTrue(all(0 <= o.start < o.end <= n for o in ops))
            ledger=PositionLedger(year, np.zeros(n))
            ledger._started_through=n-2;ledger._settled=n-2  # unit-only terminal fixture
            ledger.start_hour(n-1);ledger.commit_hour(n-1)
            with self.assertRaises(RuntimeError): ledger.start_hour(n)

    def test_actual_step_cost_difference_zero_macro_and_terminal(self):
        # Exercise actual step function with artificial driver; never construct env/world.
        env=object.__new__(AnnualDecisionEnv)
        ledger=SimpleNamespace(fee=np.array([4., 6.]))
        c=SimpleNamespace(total_purchase=0.,total_flex=0.,ledger=ledger,t=0)
        driver=SimpleNamespace(c=c,store=SimpleNamespace(key="artificial"),submit_upper=Mock())
        env.driver=driver;env.ended=False;env.level="upper";env.actions=0
        env.resets=[{}];env.steps=[];env.episodes=[]
        env.observation_space=spaces.Dict({"fake":spaces.Box(-np.inf,np.inf,(1,),dtype=np.float32)})
        env.cache=SimpleNamespace(complete=Mock())
        env.raw={"calendar":np.array([0,0,0])}
        rewards=[]
        for j, delta in enumerate((0.,7.,3.)):
            env.raw={"calendar":np.array([0,0,0])}
            def advance(d=delta, last=j==2):
                c.total_purchase+=d;c.t+=int(d>0)
                return None if last else {"fake":np.zeros(1,np.float32)}
            env._advance=advance
            with patch("v31.lite_gym.encode",return_value={"fake":np.zeros(1,np.float32)}), \
                 patch("v31.lite_gym.audit_year",return_value={"artificial":True}):
                _,r,done,truncated,info=env.step(np.r_[1.,np.full(12,.5)])
            rewards.append(r)
            self.assertEqual(info["raw_reward"],-delta)
            self.assertFalse(truncated)
            self.assertEqual(done,j==2)
        self.assertEqual(rewards,[0.,-7e-6,-3e-6])
        self.assertAlmostEqual(sum(s["raw_reward"] for s in env.steps),-c.total_purchase)
        self.assertEqual(env.steps[-1]["unsettled_contract_fee"],0.)
        self.assertEqual(env.actions,3)
        with self.assertRaises(RuntimeError):env.step(np.zeros(13))


class ProbabilityAndGAETests(unittest.TestCase):
    def buffer(self, rewards, values, starts, last, done):
        b=RolloutBuffer(len(rewards),spaces.Box(-1,1,(1,),np.float32),
            spaces.Box(0,1,(3,),np.float32),device="cpu",gamma=1.,gae_lambda=.95)
        b.rewards[:,0]=rewards;b.values[:,0]=values;b.episode_starts[:,0]=starts
        b.compute_returns_and_advantage(torch.tensor([last]),np.array([done]))
        return b

    def test_gae_terminal_and_rollout_bootstrap(self):
        end=self.buffer([-2,-3],[1,2],[1,0],4,True)
        np.testing.assert_allclose(end.advantages[:,0],[-5.75,-5],rtol=0,atol=2e-6)
        np.testing.assert_allclose(end.returns[:,0],[-4.75,-3],rtol=0,atol=2e-6)
        continuing=self.buffer([-2,-3],[1,2],[1,0],4,False)
        np.testing.assert_allclose(continuing.advantages[:,0],[-1.95,-1],rtol=0,atol=2e-6)
        np.testing.assert_allclose(continuing.returns[:,0],[-.95,1],rtol=0,atol=2e-6)

    def test_gae_episode_boundary_zero_reward_and_positive_advantage(self):
        b=self.buffer([0,-3],[1,2],[1,1],0,True)
        np.testing.assert_allclose(b.advantages[:,0],[-1,-5],rtol=0,atol=2e-6)
        b=self.buffer([0,-7,-3],[0,0,0],[1,0,0],0,True)
        np.testing.assert_allclose(b.advantages[:,0],[-9.3575,-9.85,-3],rtol=0,atol=2e-6)
        b=self.buffer([0],[ -2],[1],0,True)
        self.assertEqual(float(b.advantages[0,0]),2.)  # positive advantage, not positive raw cost reward

    def test_joint_ratio_and_inactive_beta_clipping_counterexample(self):
        p=torch.zeros((1,27),dtype=torch.float64)
        q=p.clone();q[0,3]=.4
        action=torch.tensor([[1.]+[.3]*12],dtype=torch.float64)  # SKIP: no ratio affects execution
        old=HybridDistribution().proba_distribution(p,torch.tensor([False]))
        new=HybridDistribution().proba_distribution(q,torch.tensor([False]))
        actual=(new.log_prob(action)-old.log_prob(action)).exp()
        expected=(new.category.log_prob(action[:,0].long())-old.category.log_prob(action[:,0].long())).exp()
        expected*=((new.ratios.log_prob(action[:,1:])-old.ratios.log_prob(action[:,1:])).sum(-1)).exp()
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)
        torch.testing.assert_close((old.log_prob(action)-old.log_prob(action)).exp(),torch.ones(1,dtype=torch.float64))
        # Beta(2,2)->Beta(3,2), r=2u; integrate piecewise exactly by Gauss-Legendre.
        x,w=np.polynomial.legendre.leggauss(8)
        def integral(fn,a,b):
            u=(b-a)*(x+1)/2+a
            return float(np.dot(w,fn(u))*(b-a)/2)
        uncut=integral(lambda u:6*u*(1-u)*(2*u),0,1)
        clipped=integral(lambda u:6*u*(1-u)*(2*u),0,.6)+integral(lambda u:6*u*(1-u)*1.2,.6,1)
        self.assertAlmostEqual(uncut,1.,places=13)
        self.assertAlmostEqual(clipped,.8976,places=13)
        self.assertNotAlmostEqual(clipped,1.,places=4)

    def test_encoded_mask_and_final_wait_refusal(self):
        obs={k:np.zeros(n,np.float32) for k,n in UPPER_SHAPES.items()}
        for last in (0,1):
            obs["calendar"][2]=last
            encoded=encode(obs,"upper")
            np.testing.assert_array_equal(encoded["calendar"],obs["calendar"])
            d=HybridDistribution().proba_distribution(torch.zeros((1,27)),torch.tensor([last==1]))
            self.assertEqual(float(d.category.probs[0,0]),0. if last else float(torch.tensor(1/3)))
            if last:
                with self.assertRaises(ValueError):decode_hybrid(obs,np.r_[0.,np.full(12,.5)])

    def collector_case(self, truncated):
        raw=torch.tensor([[-.2,.4,1.3]])
        gaussian=DiagGaussianDistribution(3).proba_distribution(torch.zeros((1,3)),torch.zeros(3))
        lp=gaussian.log_prob(raw)
        class FakePolicy:
            squash_output=False
            def set_training_mode(self,mode):pass
            def __call__(self,obs):return raw.clone(),torch.zeros(1),lp.clone()
            def predict_values(self,obs):return torch.tensor([4.])
            def obs_to_tensor(self,obs):return torch.as_tensor(obs).reshape(1,1),False
        class FakeVec:
            num_envs=1
            def step(self,action):
                self.executed=action.copy()
                info={"terminal_observation":np.array([0.],np.float32),"TimeLimit.truncated":truncated}
                return np.zeros((1,1),np.float32),np.array([-2.],np.float32),np.array([True]),[info]
        callback=SimpleNamespace(on_rollout_start=lambda:None,update_locals=lambda v:None,
            on_step=lambda:True,on_rollout_end=lambda:None)
        space=spaces.Box(0,1,(3,),np.float32)
        owner=SimpleNamespace(policy=FakePolicy(),_last_obs=np.zeros((1,1),np.float32),
            _last_episode_starts=np.ones(1),use_sde=False,sde_sample_freq=-1,
            device=torch.device("cpu"),action_space=space,num_timesteps=0,gamma=1.,
            _update_info_buffer=lambda infos,dones:None)
        buffer=RolloutBuffer(1,spaces.Box(-1,1,(1,),np.float32),space,device="cpu",gamma=1.,gae_lambda=.95)
        vec=FakeVec()
        self.assertTrue(OnPolicyAlgorithm.collect_rollouts(owner,vec,callback,buffer,1))
        np.testing.assert_array_equal(vec.executed,np.clip(raw.numpy(),0,1))
        np.testing.assert_array_equal(buffer.actions[0],raw.numpy())
        np.testing.assert_allclose(buffer.log_probs[0],lp.numpy(),rtol=0,atol=1e-6)
        self.assertNotAlmostEqual(float(lp[0]),float(gaussian.log_prob(torch.tensor(vec.executed))[0]),places=4)
        self.assertEqual(float(buffer.rewards[0,0]),2. if truncated else -2.)
        self.assertEqual(owner.num_timesteps,1)  # fake-only collector count; no model/real environment
        return buffer

    def test_actual_collector_keeps_gaussian_sample_and_terminal_reward(self):
        self.collector_case(False)

    def test_actual_collector_timeout_bootstrap(self):
        self.collector_case(True)


class TailTests(unittest.TestCase):
    def fake(self, stop=False):
        c=SimpleNamespace(t=0)
        c.prepare_hour=Mock()
        driver=SimpleNamespace(c=c,aborted=False,submit_upper=Mock(),submit_lower=Mock())
        model=SimpleNamespace(num_timesteps=32)
        class Env:
            ended=False
            steps=[{"artificial_learning_row":True}]
            def finish_without_learning(self,m):
                for h in range(2):
                    self.driver.submit_lower([0,0,0]);self.driver.c.prepare_hour([0,0,0])
                    self.driver.c.t+=1
                if stop:raise ValueError("fake source failure")
                self.ended=True
                return {"hours":2}
        e=Env();e.steps=e.steps.copy();e.driver=driver
        return e,model

    def test_bounded_tail_preserves_counter_and_environment_records(self):
        env,model=self.fake()
        original=env.driver.c.prepare_hour
        receipt=finish_tail(env,model,TailBudget(2,2,10),clock=lambda:0.)
        self.assertEqual(receipt["status"],"COMPLETE")
        self.assertEqual((receipt["hours_started"],receipt["submission_attempts"]),(2,2))
        self.assertEqual((model.num_timesteps,len(env.steps)),(32,1))
        self.assertIs(env.driver.c.prepare_hour,original)

    def test_limit_stops_before_next_hour_and_preserves_open_driver(self):
        env,model=self.fake()
        with self.assertRaises(TailStopped) as error:
            finish_tail(env,model,TailBudget(1,4,10),clock=lambda:0.)
        self.assertEqual(env.driver.c.t,1)
        self.assertFalse(env.ended)
        self.assertEqual(error.exception.receipt["status"],"STOPPED")
        self.assertEqual(error.exception.receipt["hours_started"],1)
        self.assertEqual((model.num_timesteps,len(env.steps)),(32,1))

    def test_deadline_driver_failure_and_learning_mutation_fail_closed(self):
        for mode in ("deadline","failure","counter"):
            env,model=self.fake(stop=mode=="failure")
            if mode=="counter":
                def bad(m):m.num_timesteps+=1;env.ended=True
                env.finish_without_learning=bad
            ticks=iter((0.,11.,11.,11.)) if mode=="deadline" else None
            clock=(lambda:next(ticks)) if ticks else (lambda:0.)
            with self.subTest(mode=mode),self.assertRaises(TailStopped):
                finish_tail(env,model,TailBudget(3,4,10),clock=clock)

    def test_bad_limits_are_rejected(self):
        for args in ((True,1,1),(1,-1,1),(1,1,0),(1,1,float("inf"))):
            with self.assertRaises(ValueError):TailBudget(*args)

    def test_actual_tail_fake_driver_empty_ended_and_active_roles(self):
        for level, actions, ended in (("upper",0,False),("upper",1,False),("lower",0,False),("upper",1,True)):
            with self.subTest(level=level,actions=actions,ended=ended):
                env=object.__new__(AnnualDecisionEnv)
                env.level=level;env.actions=actions;env.ended=ended;env.steps=[]
                env.partner=lambda raw:np.zeros(3) if level=="upper" else ("SKIP",np.zeros(12))
                c=SimpleNamespace(t=0,macros=[],prepare_hour=Mock())
                driver=SimpleNamespace(c=c,store=SimpleNamespace(n=2,key="artificial"),phase="lower",aborted=False)
                driver.lower_observation=lambda:{}
                driver.upper_observation=lambda:{}
                def submit_lower(a):c.prepare_hour(a)
                driver.submit_lower=submit_lower
                driver.submit_upper=Mock()
                def advance(**kwargs):
                    c.t+=1
                    return None if c.t==2 else {}
                driver.advance=advance
                env.driver=driver;env.cache=SimpleNamespace(complete=Mock());env._archive=Mock()
                model=SimpleNamespace(num_timesteps=32)
                with patch("v31.lite_gym.upper_policy",return_value=lambda o:("SKIP",np.zeros(12))), \
                     patch("v31.lite_gym.lower_policy",return_value=lambda o:np.zeros(3)), \
                     patch("v31.lite_gym.audit_year",return_value={"objective_cost":10.}):
                    result=finish_tail(env,model,TailBudget(2,4,10),clock=lambda:0.)
                self.assertEqual(result["status"],"COMPLETE")
                if ended or (level=="upper" and actions==0):
                    self.assertEqual(c.t,0)
                else:
                    self.assertTrue(env.ended);self.assertEqual(c.t,2)
                    self.assertEqual(result["hours_started"],2)
                    env._archive.assert_called_once()
                self.assertEqual(model.num_timesteps,32);self.assertEqual(env.steps,[])


class TailExtensionTests(unittest.TestCase):
    def fake(self, *, partner=False):
        c=SimpleNamespace(t=0,macros=[{"fixture":True}],prepare_hour=Mock())
        driver=SimpleNamespace(c=c,aborted=False,submit_upper=Mock(),submit_lower=Mock())
        model=SimpleNamespace(num_timesteps=32,predict=Mock(return_value=np.zeros(3)))
        class Env:
            level="lower"
            actions=1
            ended=False
            def finish_without_learning(self,m):
                for _ in range(2):
                    action=self.partner({}) if partner else m.predict({})
                    self.driver.submit_lower(action)
                    self.driver.c.prepare_hour(action)
                    self.driver.c.t+=1
                self.ended=True
                return {"hours":2}
        env=Env();env.steps=[{"fixture":True}];env.driver=driver
        env.partner=Mock(return_value=np.zeros(3))
        return env,model

    def test_zero_submission_hour_or_policy_budget_prevents_both_policy_ports(self):
        limits=(TailBudget(2,0,10),TailBudget(0,3,10),TailBudget(2,3,10,0))
        for use_partner in (False,True):
            for budget in limits:
                with self.subTest(partner=use_partner,budget=budget):
                    env,model=self.fake(partner=use_partner)
                    main=model.predict;other=env.partner
                    with self.assertRaises(TailStopped) as stopped:
                        finish_tail(env,model,budget,clock=lambda:0.)
                    main.assert_not_called();other.assert_not_called()
                    self.assertEqual(stopped.exception.receipt["policy_call_attempts"],0)
                    self.assertEqual(env.driver.c.t,0)
                    self.assertIs(model.predict,main);self.assertIs(env.partner,other)

    def test_policy_call_cap_and_failures_count_before_call(self):
        for use_partner in (False,True):
            env,model=self.fake(partner=use_partner)
            main=model.predict;other=env.partner
            with self.assertRaises(TailStopped) as stopped:
                finish_tail(env,model,TailBudget(3,3,10,1),clock=lambda:0.)
            self.assertEqual(stopped.exception.receipt["policy_call_attempts"],1)
            self.assertEqual((other if use_partner else main).call_count,1)
            self.assertEqual(env.driver.c.t,1)
        env,model=self.fake()
        predict=model.predict;predict.side_effect=ValueError("fake prediction failed")
        with self.assertRaises(TailStopped) as stopped:
            finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.)
        self.assertEqual(stopped.exception.receipt["policy_call_attempts"],1)
        self.assertIs(model.predict,predict)

    def test_expired_clock_rejects_before_prediction(self):
        env,model=self.fake()
        predict=model.predict
        ticks=iter((0.,0.,1.,1.))
        with self.assertRaises(TailStopped):
            finish_tail(env,model,TailBudget(3,3,1),clock=lambda:next(ticks))
        predict.assert_not_called()

    def test_interrupt_receipt_is_finalized_and_same_interrupt_propagates(self):
        for error in (KeyboardInterrupt("fake user stop"),SystemExit(17)):
            with self.subTest(kind=type(error).__name__):
                env,model=self.fake();original=model.predict;original.side_effect=error
                saved=[]
                with self.assertRaises(type(error)) as stopped:
                    finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.,receipt_sink=saved.append)
                self.assertIs(stopped.exception,error)
                self.assertIs(model.predict,original)
                receipt=error.tail_receipt
                self.assertEqual(receipt["status"],"STOPPED")
                self.assertEqual(receipt["error_type"],type(error).__name__)
                self.assertEqual(receipt["seconds"],0.)
                self.assertEqual(receipt["env_record_rows_before"],1)
                self.assertEqual(receipt["env_record_rows_after"],1)
                self.assertEqual(saved,[receipt])
                self.assertNotIn("buffer_rows_after",receipt)

    def test_sink_failure_cannot_hide_interrupt_or_report_success(self):
        def bad_sink(receipt):raise OSError("fake log sink failure")
        env,model=self.fake()
        with self.assertRaises(TailStopped) as stopped:
            finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.,receipt_sink=bad_sink)
        self.assertEqual(stopped.exception.receipt["status"],"STOPPED")
        self.assertIn("receipt_sink_error",stopped.exception.receipt)
        env,model=self.fake();error=KeyboardInterrupt("fake user stop");model.predict.side_effect=error
        with self.assertRaises(KeyboardInterrupt) as stopped:
            finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.,receipt_sink=bad_sink)
        self.assertIs(stopped.exception,error)
        self.assertIn("receipt_sink_error",error.tail_receipt)

    def test_upper_partner_must_be_explicit_and_limits_strict(self):
        env,model=self.fake();env.level="upper";env.partner=None
        with self.assertRaises(TailStopped):
            finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.)
        for value in (True,-1,1.2):
            with self.subTest(value=value),self.assertRaises(ValueError):
                TailBudget(3,3,10,value)


class IndustrialJointTests(unittest.TestCase):
    def run_day(self, limited, uniform):
        from v3.devices import DailyDevices
        device=DailyDevices(limited)
        rows=[]
        for h in range(24):
            state=device.state()
            er=(100/3)/(24-state["ev_start"]) if uniform and state["ev_start"]<=h<24 else 0.
            ir=12/(16-state["industrial_start"]) if uniform and state["industrial_start"]<=h<16 else 0.
            candidate,row=device.prepare(er,ir)
            self.assertEqual(device.state(),state)  # candidate has not changed committed state
            device=candidate;rows.append(row)
        return device,rows

    def test_forced_joint_schedule_flex_and_grid_settlement_once(self):
        for limited in (False,True):
            with self.subTest(limited=limited):
                device,rows=self.run_day(limited,False)
                ev=np.array([0.]*19+[16/3]+[7.]*4)
                ind=np.array([0.]*13+[4.]*3+[0.]*8)
                np.testing.assert_allclose([r["ev_actual_mw"] for r in rows],ev,rtol=0,atol=1e-10)
                np.testing.assert_allclose([r["industrial_actual_mw"] for r in rows],ind,rtol=0,atol=1e-10)
                self.assertAlmostEqual(device.ev_total,100/3,places=8)
                self.assertAlmostEqual(device.ind_total,12,places=8)
                self.assertAlmostEqual(device.ev_forced,100/3,places=8)
                self.assertAlmostEqual(device.ind_forced,12,places=8)
                self.assertAlmostEqual(sum(r["ev_battery_mwh"] for r in rows),30,places=8)
                self.assertAlmostEqual(device.ev.soc,.7,places=10)
                es=19 if limited else 18;is_=12 if limited else 8
                ref_e=np.r_[np.zeros(es),np.full(24-es,(100/3)/(24-es))]
                ref_i=np.array([12/(16-is_) if is_<=h<16 else 0. for h in range(24)])
                flex=float(np.abs(ev-ref_e).sum()+np.abs(ind-ref_i).sum())
                self.assertAlmostEqual(sum(r["flex_cost"] for r in rows),flex,places=8)
                purchase=load=0.
                for row in rows:
                    paid=settlement(10,row["industrial_actual_mw"],row["ev_actual_mw"],12,1200,2,400,600)
                    purchase+=paid["purchase_cost"];load+=paid["total_load_mwh"]
                expected=24*(1200+2*400)+600*np.maximum(10+ev+ind-12-2,0).sum()
                self.assertAlmostEqual(purchase,float(expected),places=7)
                self.assertAlmostEqual(load,240+100/3+12,places=8)
                self.assertAlmostEqual(sum(r["industrial_forced_mwh"] for r in rows),12,places=8)

    def test_uniform_joint_requests_complete_with_zero_flex_and_no_forcing(self):
        for limited in (False,True):
            device,rows=self.run_day(limited,True)
            self.assertAlmostEqual(sum(r["flex_cost"] for r in rows),0,places=8)
            self.assertAlmostEqual(device.ev_forced,0,places=8)
            self.assertAlmostEqual(device.ind_forced,0,places=8)
            self.assertAlmostEqual(device.ev_total,100/3,places=8)
            self.assertAlmostEqual(device.ind_total,12,places=8)

    def test_industrial_infeasible_task_is_rejected_without_lowering_target(self):
        from load_models.industrial_load import IndustrialLoad
        with self.assertRaises(ValueError):
            IndustrialLoad(task_energy_mwh=17,max_power_mw=4,start_hour=12,end_hour=16)


class TailInterruptFinalizationTests(unittest.TestCase):
    fake = TailExtensionTests.fake

    def test_sink_control_interrupt_after_success_keeps_identity(self):
        for error in (KeyboardInterrupt("sink interrupt"),SystemExit(21)):
            env,model=self.fake();original=model.predict
            def sink(receipt):raise error
            with self.subTest(kind=type(error).__name__),self.assertRaises(type(error)) as stopped:
                finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.,receipt_sink=sink)
            self.assertIs(stopped.exception,error)
            self.assertIs(model.predict,original)
            self.assertEqual(error.tail_receipt["status"],"STOPPED")
            self.assertEqual(error.tail_receipt["errors"][-1]["stage"],"receipt_sink")

    def test_restore_interrupt_priority_and_bounded_restoration(self):
        for mode in ("success","ordinary_failure","control_failure"):
            env,_=self.fake()
            cleanup_error=KeyboardInterrupt("restore interrupt")
            flow_error=KeyboardInterrupt("first user interrupt") if mode=="control_failure" else ValueError("ordinary failure")
            original=Mock(return_value=np.zeros(3))
            class Model:
                def __init__(self):
                    self.num_timesteps=32;self.armed=False;self.predict=original
                def __setattr__(self,name,value):
                    if name=="predict" and value is original and getattr(self,"armed",False):
                        object.__setattr__(self,"armed",False)
                        raise cleanup_error
                    object.__setattr__(self,name,value)
            model=Model()
            def predict(*args,**kwargs):
                model.armed=True
                if mode!="success":raise flow_error
                return np.zeros(3)
            original.side_effect=predict
            saved=[]
            def sink(receipt):
                saved.append(receipt)
                if mode=="control_failure":raise SystemExit(99)
            expected=flow_error if mode=="control_failure" else cleanup_error
            with self.subTest(mode=mode),self.assertRaises(type(expected)) as stopped:
                finish_tail(env,model,TailBudget(3,3,10),clock=lambda:0.,receipt_sink=sink)
            self.assertIs(stopped.exception,expected)
            self.assertIs(model.predict,original)  # one bounded interrupted-restore retry succeeded
            self.assertEqual(expected.tail_receipt["status"],"STOPPED")
            self.assertTrue(any(e["stage"]=="restore:predict" for e in expected.tail_receipt["errors"]))
            self.assertEqual(len(saved),1)

    def test_final_clock_interrupt_retains_receipt_and_does_not_skip_sink(self):
        env,model=self.fake();original=model.predict
        error=KeyboardInterrupt("final clock interrupt")
        def clock():
            if env.ended and model.predict is original:raise error
            return 0.
        saved=[]
        with self.assertRaises(KeyboardInterrupt) as stopped:
            finish_tail(env,model,TailBudget(3,3,10),clock=clock,receipt_sink=saved.append)
        self.assertIs(stopped.exception,error)
        self.assertEqual(error.tail_receipt["seconds"],"UNKNOWN")
        self.assertEqual(error.tail_receipt["env_record_rows_after"],1)
        self.assertEqual(len(saved),1)
        self.assertIs(model.predict,original)
