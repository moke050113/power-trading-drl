"""New R04 durable-runner gaps only; invoke with the approved bounded CONTEXT.

The real-policy check uses two synthetic hours and an artificial driver, NOT a
native annual world or independent performance test. It calls the actual existing
AnnualDecisionEnv.finish_without_learning, via run_tail_checked and finish_tail.
"""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from research.runner_contract import (
    SCALE_ID, TailRunApproval, environment_receipt, rule_state_reader,
    run_tail_checked, snapshot_learning_state, state_digest)
from research.tail_budget import TailBudget, TailStopped
from v31.lite_gym import AnnualDecisionEnv, LOWER_SHAPES, UPPER_SHAPES

CONTEXT = None


def artificial_rule(raw):
    # Pure stateless rule only: no model closure or unobserved training state.
    if "calendar" in raw: return "SKIP", np.full(12, .5)
    return np.zeros(3, np.float32)


def synthetic_raw(level):
    shapes=UPPER_SHAPES if level=="upper" else LOWER_SHAPES
    raw={k:np.zeros(n, np.float32) for k,n in shapes.items()}
    if level=="upper": raw["calendar"][2]=1  # last opportunity: WAIT forbidden
    return raw


class ArtificialDriver:
    """Two-hour accounting fixture; no LiteStore/YearCoordinator construction."""
    def __init__(self, level):
        ledger=SimpleNamespace(lots=[SimpleNamespace(quantities=(1.,1.),unit_price=100.)],
                               fee=np.array([100.,100.]))
        self.c=SimpleNamespace(t=0,pending=(object() if level=="lower" else None),
            macros=[{"artificial_signed_reset_contract":True}],hours=[],ledger=ledger,
            total_purchase=0.,total_flex=0.)
        self.c.prepare_hour=self.prepare_hour
        self.phase=level; self.aborted=False
        self.store=SimpleNamespace(n=2,key="artificial_two_hour_driver",completed_year=False)
        self.macro_receipts=[];self.daily_receipts=[]
    def prepare_hour(self, action): self.c.pending=object()
    def upper_observation(self): return synthetic_raw("upper")
    def lower_observation(self): return synthetic_raw("lower")
    def submit_upper(self, verb, ratios):
        if verb not in ("WAIT","SKIP","COMMIT"): raise ValueError("bad artificial upper action")
        self.phase="lower"
    def submit_lower(self, action):
        if self.c.pending is None: self.c.prepare_hour(action)
        self.c.hours.append(dict(t=self.c.t,contract_cost=100.,purchase_cost=100.,flex_cost=0.))
        self.c.total_purchase+=100.;self.c.t+=1;self.c.pending=None;self.phase="upper"
    def advance(self, *, stop, lower=None):
        if self.phase=="lower": self.submit_lower(lower(self.lower_observation()))
        if self.c.t>=self.store.n:
            self.store.completed_year=True
            return None
        return self.upper_observation()


def artificial_env(level):
    from gymnasium import spaces
    env=object.__new__(AnnualDecisionEnv)
    env.level=level;env.driver=ArtificialDriver(level);env.ended=False
    # Upper is active; lower reset already has contract/pending H_START liability.
    env.actions=1 if level=="upper" else 0
    env.steps=[];env.resets=[{"artificial":True}];env.episodes=[]
    env.partner=artificial_rule;env.raw=synthetic_raw(level)
    shapes=UPPER_SHAPES if level=="upper" else LOWER_SHAPES
    env.observation_space=spaces.Dict({k:spaces.Box(-np.inf,np.inf,(n,),np.float32)
                                       for k,n in shapes.items()})
    env.action_space=(spaces.Box(np.zeros(13,np.float32),np.array([2]+[1]*12,np.float32),dtype=np.float32)
                      if level=="upper" else spaces.Box(0,1,(3,),np.float32))
    env.cache=SimpleNamespace(complete=lambda store:None)
    env._archive=lambda:None
    return env


def fake_model(failure=None):
    param=SimpleNamespace(grad=None)
    weights={"weight":np.array([1.,2.],np.float32)}
    optimizer=SimpleNamespace(state_dict=lambda:{"state":{},"param_groups":[{"params":[0]}]})
    policy=SimpleNamespace(state_dict=lambda:weights,optimizer=optimizer,
                           named_parameters=lambda:[("weight",param)])
    buffer=SimpleNamespace(observations={"x":np.array([[3.]],np.float32)},
        actions=np.zeros((1,3),np.float32),rewards=np.array([4.],np.float32),
        pos=1,full=True,generator_ready=False)
    model=SimpleNamespace(policy=policy,rollout_buffer=buffer,num_timesteps=7,_n_updates=0,
        gamma=1.,gae_lambda=.95,n_steps=2,batch_size=2)
    def predict(obs=None, deterministic=True):
        if failure is not None: raise failure
        return np.zeros(3,np.float32),None
    model.predict=predict
    model.fake_weights=weights;model.fake_parameter=param
    return model


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        if CONTEXT is None or getattr(CONTEXT,"owner",None) is None:
            raise RuntimeError("approved bounded runner must inject CONTEXT.owner")
        self.ctx=CONTEXT
        self.ctx.check_deadline()
        self.root=Path(self.ctx.runtime)/"r04_contract"
        self.root.mkdir(exist_ok=True)
        src=Path(__file__).resolve().parents[1]
        files=[src/"research"/"runner_contract.py",src/"research"/"tail_budget.py",src/"v31"/"lite_gym.py"]
        self.source_reader=lambda:{p.relative_to(src).as_posix():sha256(p.read_bytes()).hexdigest() for p in files}
        self.identities=dict(source_sha256=self.source_reader(),
            config=dict(gamma=1.,gae_lambda=.95,n_steps=2,batch_size=2,optimizer_updates=0),
            observation_scale=SCALE_ID,model=dict(origin="fresh_untrained",checkpoint=None),
            partner=dict(kind="stateless_rule",name="artificial_rule",
                         state=rule_state_reader(artificial_rule)),
            scope="artificial_two_hour_driver_NOT_native_annual")
        self.budget=TailBudget(2,4,30.,max_policy_calls=8)
    def approval(self, identities=None, max_bytes=32768):
        identities=self.identities if identities is None else identities
        return TailRunApproval(owner=self.ctx.owner,level="lower",
            identities_sha256=state_digest(identities),max_hours=2,max_decisions=4,
            max_policy_calls=8,wall_seconds=30.,max_journal_bytes=max_bytes,
            journal_root=str(self.root),approved=True)
    def arguments(self, run_id, model=None, env=None, **changes):
        args=dict(approval=self.approval(),owner=self.ctx.owner,level="lower",
            env=artificial_env("lower") if env is None else env,
            make_model=(lambda level,env:fake_model()) if model is None else (lambda level,env:model),
            partner_owner=artificial_rule,partner_state_reader=rule_state_reader,
            identities=self.identities,source_reader=self.source_reader,journal_root=self.root,
            run_id=run_id,budget=self.budget)
        args.update(changes);return args
    def persisted(self, run_id):
        p=self.root/(run_id+".jsonl")
        return p,[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]

    def test_owner_identity_source_and_budget_denied_before_model_factory(self):
        factory=Mock(side_effect=AssertionError("factory must not execute"))
        for label,changes in (
                ("owner",dict(owner=None)),("approval",dict(approval=None)),
                ("source",dict(source_reader=lambda:{})),
                ("budget",dict(budget=TailBudget(3,4,30.,max_policy_calls=8))),
                ("scale",dict(identities={**self.identities,"observation_scale":"unapproved"}))):
            with self.subTest(label=label),self.assertRaises((PermissionError,ValueError)):
                run_tail_checked(**self.arguments("denied_"+label,make_model=factory,**changes))
        wrong_env=artificial_env("upper")
        with self.assertRaises(PermissionError):
            run_tail_checked(**self.arguments("denied_level",make_model=factory,env=wrong_env))
        wrong_partner=artificial_env("lower");wrong_partner.partner=lambda raw:np.zeros(3)
        with self.assertRaises(PermissionError):
            run_tail_checked(**self.arguments("denied_partner",make_model=factory,env=wrong_partner))
        mismatched={**self.identities,"partner":dict(kind="stateless_rule",state={"not":"approved_actual_partner"})}
        with self.assertRaises(PermissionError):
            run_tail_checked(**self.arguments("denied_partner_state",make_model=factory,identities=mismatched,approval=self.approval(mismatched)))
        factory.assert_not_called()
        self.assertFalse(any(self.root.glob("denied_*.jsonl")))

    def test_failed_attempts_and_each_learning_state_mutation_are_stopped_and_persisted(self):
        # Source failure consumes one predict attempt; no retry, no settlement.
        m=fake_model(ValueError("synthetic policy failure"))
        with self.assertRaises(TailStopped) as raised:
            run_tail_checked(**self.arguments("failed_attempt",model=m))
        r=raised.exception.receipt
        self.assertEqual(r["tail"]["policy_call_attempts"],1)
        self.assertEqual(r["environment_after"]["completed_hours"],0)
        self.assertEqual(self.persisted("failed_attempt")[1][-1]["status"],"STOPPED")
        # Mutations below are fake RAM only, never actual PPO weights/optimizer.
        def mutate_parameter(m):m.fake_weights["weight"][0]+=1
        def mutate_gradient(m):m.fake_parameter.grad=np.array([1.],np.float32)
        def mutate_optimizer(m):m.policy.optimizer.state_dict=lambda:{"state":{0:{"step":1}},"param_groups":[]}
        def mutate_buffer(m):m.rollout_buffer.observations["x"][0,0]+=1
        def mutate_time(m):m.num_timesteps+=1
        def mutate_updates(m):m._n_updates+=1
        for name,change in (("parameter",mutate_parameter),("gradient",mutate_gradient),
                ("optimizer",mutate_optimizer),("buffer",mutate_buffer),
                ("timesteps",mutate_time),("updates",mutate_updates)):
            m=fake_model();e=artificial_env("lower")
            def fake_finish(model, change=change):
                with patch("v31.lite_gym.audit_year",lambda driver:{"artificial":True}):
                    result=AnnualDecisionEnv.finish_without_learning(e,model)
                change(model)
                return result
            e.finish_without_learning=fake_finish
            with self.subTest(name=name),self.assertRaises(TailStopped) as changed:
                run_tail_checked(**self.arguments("mutation_"+name,model=m,env=e))
            receipt=changed.exception.receipt
            field=dict(parameter="parameters",gradient="gradients",optimizer="optimizer",buffer="rollout_buffer",timesteps="num_timesteps",updates="n_updates")[name]
            self.assertNotEqual(receipt["model_before"][field],receipt["model_after"][field])
            for other in receipt["model_before"]:
                if other!=field:self.assertEqual(receipt["model_before"][other],receipt["model_after"][other])
            self.assertEqual(receipt["status"],"STOPPED")
            if name=="updates":
                self.assertEqual(receipt["config_before"]["optimizer_updates"],0)
                self.assertEqual(receipt["config_after"]["optimizer_updates"],1)
            self.assertEqual(self.persisted("mutation_"+name)[1][-1]["status"],"STOPPED")
        class FakePartner:
            state=0
            def __call__(self,raw):return artificial_rule(raw)
        partner=FakePartner()
        reader=lambda owner:dict(parameters=str(owner.state),optimizer="constant",
            gradients="constant",rollout_buffer="constant",num_timesteps=0,n_updates=0)
        e=artificial_env("lower");e.partner=partner;m=fake_model()
        partner_identity={**self.identities,"partner":dict(kind="fake_policy",state=reader(partner))}
        def changed_partner(model):partner.state+=1;e.ended=True;return {"artificial":True}
        e.finish_without_learning=changed_partner
        with self.assertRaises(TailStopped) as partner_changed:
            run_tail_checked(**self.arguments("partner_mutation",model=m,env=e,
                partner_owner=partner,partner_state_reader=reader,identities=partner_identity,approval=self.approval(partner_identity)))
        self.assertIn("frozen partner state changed",str(partner_changed.exception))
        self.assertEqual(self.persisted("partner_mutation")[1][-1]["status"],"STOPPED")
        # Bind fixed source before/after; a mutation must never return COMPLETE.
        source_calls=[]
        def changed_source():
            source_calls.append(1)
            return self.source_reader() if len(source_calls)==1 else {}
        e=artificial_env("lower");e.finish_without_learning=lambda model:{"empty_auto_reset":True}
        with self.assertRaises(TailStopped):
            run_tail_checked(**self.arguments("source_mutation",env=e,source_reader=changed_source))
        self.assertEqual(self.persisted("source_mutation")[1][-1]["status"],"STOPPED")
        e=artificial_env("lower");e.finish_without_learning=lambda model:{"empty_auto_reset":True}
        with self.assertRaises(TailStopped) as invalid:
            run_tail_checked(**self.arguments("invalid_lower_empty",env=e))
        self.assertIn("invalid empty",str(invalid.exception))
        self.assertEqual(self.persisted("invalid_lower_empty")[1][-1]["status"],"STOPPED")

    def test_control_identity_restoration_exclusive_journal_and_byte_stop(self):
        interrupt=KeyboardInterrupt("artificial control interruption")
        m=fake_model(interrupt);e=artificial_env("lower");original=m.predict
        partner_reads=[]
        def interrupted_cleanup_reader(partner):
            partner_reads.append(1)
            if len(partner_reads)>1:raise ValueError("synthetic secondary cleanup failure")
            return rule_state_reader(partner)
        with self.assertRaises(KeyboardInterrupt) as raised:
            run_tail_checked(**self.arguments("interrupted",model=m,env=e,
                partner_state_reader=interrupted_cleanup_reader))
        self.assertIs(raised.exception,interrupt)
        self.assertIs(m.predict,original)
        self.assertEqual(self.persisted("interrupted")[1][-1]["status"],"STOPPED")
        factory=Mock(side_effect=AssertionError("exclusive name must refuse before factory"))
        with self.assertRaises(FileExistsError):
            run_tail_checked(**self.arguments("interrupted",make_model=factory))
        factory.assert_not_called()
        # Oversize terminal payload becomes a compact STOPPED record within cap.
        e=artificial_env("lower")
        def oversized(model):e.ended=True;return {"artificial_only":"x"*20000}
        e.finish_without_learning=oversized
        with self.assertRaises(TailStopped) as byte_stopped:
            run_tail_checked(**self.arguments("byte_stop",env=e,
                approval=self.approval(max_bytes=8192),max_journal_bytes=8192))
        p,rows=self.persisted("byte_stop")
        self.assertLessEqual(p.stat().st_size,8192)
        self.assertEqual(rows[-1]["status"],"STOPPED")
        self.assertNotIn("COMPLETE",[r["status"] for r in rows])
        self.assertTrue(any(x["stage"]=="journal_final" for x in byte_stopped.exception.receipt["errors"]))

    def test_real_fresh_ppo_tail_both_layers_ram_unchanged_and_pending_accounted(self):
        for level in ("upper","lower"):
            self.ctx.check_deadline()
            env=artificial_env(level)
            approval=TailRunApproval(owner=self.ctx.owner,level=level,
                identities_sha256=state_digest(self.identities),max_hours=2,max_decisions=4,
                max_policy_calls=8,wall_seconds=30.,max_journal_bytes=32768,
                journal_root=str(self.root),approved=True)
            captured=[]
            def factory(layer, selected_env):
                model=self.ctx.make_model(layer,selected_env)
                # Nonzero actual buffer RAM without optimizer or collector execution.
                # This tests preserving real contents, not only empty-buffer counters.
                for value in vars(model.rollout_buffer).values():
                    arrays=value.values() if isinstance(value,dict) else (value,)
                    for a in arrays:
                        if isinstance(a,np.ndarray):a.fill(.125)
                captured.append(model)
                return model
            def fixture_audit(driver):
                return dict(artificial_driver=True,native_annual=False,
                    purchase_cost=driver.c.total_purchase,flex_cost=driver.c.total_flex,
                    completed_hours=driver.c.t,scope="two synthetic hours only")
            with patch("v31.lite_gym.audit_year",fixture_audit):
                receipt=run_tail_checked(approval=approval,owner=self.ctx.owner,level=level,
                    env=env,make_model=factory,partner_owner=artificial_rule,
                    partner_state_reader=rule_state_reader,identities=self.identities,
                    source_reader=self.source_reader,journal_root=self.root,run_id="real_"+level,
                    budget=self.budget)
            self.assertEqual(receipt["status"],"COMPLETE")
            self.assertEqual(receipt["model_factory_attempts"],1)
            self.assertEqual(receipt["model_before"],receipt["model_after"])
            self.assertEqual(receipt["config_before"],receipt["config_after"])
            self.assertEqual(receipt["config_before"],self.identities["config"])
            self.assertEqual(receipt["partner_before"],receipt["partner_after"])
            self.assertEqual(receipt["settled_hours_delta"],2)
            self.assertEqual(receipt["initial_pending"],level=="lower")
            self.assertEqual(receipt["new_H_START_attempts"],2 if level=="upper" else 1)
            self.assertEqual(receipt["environment_after"]["remaining_hours"],0)
            self.assertEqual(receipt["environment_after"]["liability_residual"],0.)
            self.assertEqual(receipt["environment_after"]["purchase_cost"],200.)
            self.assertEqual(receipt["model_after"]["num_timesteps"],0)
            self.assertEqual(receipt["model_after"]["n_updates"],0)
            self.assertEqual(captured[0].rollout_buffer.pos,0)
            self.assertFalse(captured[0].rollout_buffer.full)
            self.assertEqual(len(env.steps),0)
            self.assertTrue(receipt["tail"]["tail"]["artificial_driver"])
            self.assertFalse(receipt["tail"]["tail"]["native_annual"])
            self.assertEqual(self.persisted("real_"+level)[1][-1]["status"],"COMPLETE")
            self.ctx.emit(dict(kind="r04_real_tail",level=level,receipt=receipt,
                actual_learning_state=snapshot_learning_state(captured[0]),
                limitation="real PPO/existing tail method, artificial driver only; no native annual proof"))
