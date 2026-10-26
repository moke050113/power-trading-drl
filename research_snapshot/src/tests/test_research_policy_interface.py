"""R02: two fresh untrained policies, artificial observations, no env or learning."""
import hashlib
import json
import unittest
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.buffers import DictRolloutBuffer
from experiments.hybrid_upper_v3 import HybridUpperPolicy
from v31.lite_gym import UPPER_SHAPES, LOWER_SHAPES, decode_hybrid

CONTEXT = None
RTOL = 1.3e-6
ATOL = 1e-5  # installed Torch float32 default; frozen before execution


def observation_fixture(shapes, layer):
    """Schema fixtures only; no world, history or forecasting claim."""
    obs = {k: np.stack([np.linspace(.01,.09,n,dtype=np.float32),
                       np.linspace(.02,.10,n,dtype=np.float32)])
           for k,n in shapes.items()}
    if layer == "upper":
        obs["stage"] = np.array([[1,0,0],[0,1,0]],np.float32)
        obs["calendar"] = np.array([[0,0,0],[1,0,1]],np.float32)
        obs["coverage"][:] = .2
        obs["reference"][:] = 1.
        obs["remaining_capacity"][:] = .8
    else:
        obs["hour"] = np.array([[0.],[23/24]],np.float32)
        obs["history_mask"][:] = 0.
        obs["actual_state_mask"][:] = 0.
        obs["declared_initial_soc"][:] = .2
    return obs


def dict_space(shapes):
    return spaces.Dict({k:spaces.Box(-np.inf,np.inf,(n,),np.float32) for k,n in shapes.items()})


class PolicyInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CONTEXT is None:
            raise RuntimeError("use the specifically authorized guarded runner")
        cls.ctx = CONTEXT
        cls.policies = {}
        cls.observations = {}
        cls.samples = {}
        for layer,shapes,policy_cls in (("upper",UPPER_SHAPES,HybridUpperPolicy),
                                        ("lower",LOWER_SHAPES,MultiInputActorCriticPolicy)):
            ospace=dict_space(shapes)
            aspace=(spaces.Box(np.zeros(13,np.float32),np.array([2]+[1]*12,np.float32),dtype=np.float32)
                    if layer=="upper" else spaces.Box(0,1,(3,),np.float32))
            policy=cls.ctx.create(layer,policy_cls,ospace,aspace)
            obs=observation_fixture(shapes,layer)
            tensor,_=policy.obs_to_tensor(obs)
            with torch.no_grad():
                action,value,lp=policy.forward(tensor,deterministic=False)
            cls.policies[layer]=policy;cls.observations[layer]=(obs,tensor)
            cls.samples[layer]=(action.detach().clone(),value.detach().clone(),lp.detach().clone())
            cls.ctx.artifacts.setdefault("fixtures",{})[layer]={
                "description":"two artificial encoded Dict observations; no simulated world",
                "observations":{k:v.tolist() for k,v in obs.items()},
                "raw_actions":action.cpu().tolist(),"values":value.cpu().tolist(),"log_prob":lp.cpu().tolist()}
        cls.ctx.verify_unchanged()

    def tearDown(self):
        self.ctx.verify_unchanged()

    def test_upper_forward_evaluate_values_mask_and_prediction(self):
        policy=self.policies["upper"];obs,tensor=self.observations["upper"]
        action,value,lp=self.samples["upper"]
        v2,lp2,entropy=policy.evaluate_actions(tensor,action)
        torch.testing.assert_close(value,v2,rtol=0,atol=0)
        torch.testing.assert_close(lp,lp2,rtol=0,atol=0)
        dist=policy.get_distribution(tensor)
        self.assertEqual(float(dist.category.probs[1,0]),0.)
        torch.testing.assert_close(dist.category.probs.sum(-1),torch.ones(2),rtol=RTOL,atol=ATOL)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(((action[:,1:]>0)&(action[:,1:]<1)).all())
        self.assertTrue(((action[:,0]>=0)&(action[:,0]<=2)&(action[:,0]==action[:,0].round())).all())
        for row in range(2):
            verb,_=decode_hybrid({"calendar":obs["calendar"][row]},action[row].cpu().numpy())
            if row==1:self.assertNotEqual(verb,"WAIT")
        expected=dist.mode().detach().cpu().numpy()
        p1,_=policy.predict(obs,deterministic=True)
        p2,_=policy.predict(obs,deterministic=True)
        np.testing.assert_array_equal(p1,p2)
        np.testing.assert_array_equal(p1,expected)
        torch.testing.assert_close(policy.predict_values(tensor),value,rtol=RTOL,atol=ATOL)
        self.ctx.artifacts["upper_math"]={"same_batch_log_prob_exact":True,"final_wait_probability":0.,
            "prediction_repeat_exact":True,"sample_ratio":(lp2-lp).exp().tolist()}

    def test_lower_forward_evaluate_gaussian_and_sample_execution_coordinates(self):
        policy=self.policies["lower"];obs,tensor=self.observations["lower"]
        action,value,lp=self.samples["lower"]
        v2,lp2,entropy=policy.evaluate_actions(tensor,action)
        torch.testing.assert_close(value,v2,rtol=0,atol=0)
        torch.testing.assert_close(lp,lp2,rtol=0,atol=0)
        dist=policy.get_distribution(tensor)
        independent=torch.distributions.Normal(dist.distribution.mean,dist.distribution.stddev)
        torch.testing.assert_close(lp,independent.log_prob(action).sum(-1),rtol=RTOL,atol=ATOL)
        self.assertTrue(torch.isfinite(entropy).all())
        raw=torch.tensor([[-.2,.4,1.3],[-.3,.5,1.1]],dtype=torch.float32)
        clipped=raw.clamp(0,1)
        _,raw_lp,_=policy.evaluate_actions(tensor,raw)
        torch.testing.assert_close(raw_lp,independent.log_prob(raw).sum(-1),rtol=RTOL,atol=ATOL)
        self.assertTrue((torch.abs(raw_lp-independent.log_prob(clipped).sum(-1))>ATOL).all())
        expected=np.clip(dist.mode().detach().cpu().numpy(),0,1)
        p1,_=policy.predict(obs,deterministic=True)
        p2,_=policy.predict(obs,deterministic=True)
        np.testing.assert_array_equal(p1,p2)
        np.testing.assert_array_equal(p1,expected)
        torch.testing.assert_close(policy.predict_values(tensor),value,rtol=RTOL,atol=ATOL)
        self.ctx.artifacts["lower_math"]={"same_batch_log_prob_exact":True,
            "raw_manual_action":raw.tolist(),"executed_manual_action":clipped.tolist(),
            "raw_log_prob":raw_lp.tolist(),"clipped_point_log_prob":independent.log_prob(clipped).sum(-1).tolist(),
            "prediction_repeat_exact":True}

    def buffer_roundtrip(self,layer):
        policy=self.policies[layer];obs,_=self.observations[layer]
        actions,values,lp=self.samples[layer]
        buffer=DictRolloutBuffer(2,policy.observation_space,policy.action_space,
            device="cpu",gamma=1.,gae_lambda=.95,n_envs=1)
        for row in range(2):
            buffer.add({k:v[row:row+1].copy() for k,v in obs.items()},
                actions[row:row+1].cpu().numpy().copy(),np.array([-.1],np.float32),
                np.array([row==0]),values[row:row+1].clone(),lp[row:row+1].clone())
        buffer.compute_returns_and_advantage(torch.zeros(1),np.array([True]))
        self.assertTrue(buffer.full)
        sample=next(buffer.get(batch_size=2))
        # Join against the original observation identity, not current env.raw or row order.
        tag=sample.observations["calendar"][:,2] if layer=="upper" else sample.observations["hour"][:,0]
        original_tag=obs["calendar"][:,2] if layer=="upper" else obs["hour"][:,0]
        for i in range(2):
            row=int(np.flatnonzero(original_tag==float(tag[i]))[0])
            torch.testing.assert_close(sample.actions[i],actions[row],rtol=0,atol=0)
            torch.testing.assert_close(sample.old_log_prob[i],lp[row],rtol=0,atol=0)
            for key in obs:
                np.testing.assert_array_equal(sample.observations[key][i].cpu().numpy(),obs[key][row])
        _,new_lp,_=policy.evaluate_actions(sample.observations,sample.actions)
        torch.testing.assert_close(new_lp,sample.old_log_prob,rtol=RTOL,atol=ATOL)
        ratio=(new_lp-sample.old_log_prob).exp()
        torch.testing.assert_close(ratio,torch.ones_like(ratio),rtol=RTOL,atol=ATOL)
        self.ctx.artifacts.setdefault("dict_buffer_roundtrip",{})[layer]={
            "transitions":2,"same_stored_observation_action_log_prob":True,
            "ratio":ratio.tolist(),"max_log_prob_difference":float((new_lp-sample.old_log_prob).abs().max()),
            "scope":"actual SB3 DictRolloutBuffer on artificial observations; no rollout/environment"}
        self.ctx.verify_unchanged()

    def test_upper_actual_dict_buffer_roundtrip(self):
        self.buffer_roundtrip("upper")

    def test_lower_actual_dict_buffer_roundtrip(self):
        self.buffer_roundtrip("lower")

    def test_fresh_policy_and_optimizer_states_unchanged(self):
        self.ctx.verify_unchanged()
        for policy in self.policies.values():
            self.assertFalse(policy.optimizer.state)
            self.assertTrue(all(p.grad is None for p in policy.parameters()))
        self.ctx.artifacts["learning"]={"updates":0,"all_parameter_grads_none":True,
            "optimizer_states_empty":True,"scope":"fresh unit-test policy state only"}
