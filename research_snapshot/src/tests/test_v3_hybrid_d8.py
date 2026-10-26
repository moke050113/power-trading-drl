"""D8 probability/accounting gates. No test market access and no PPO.learn in this file."""
import unittest
import tempfile
import json, csv
from pathlib import Path
import numpy as np
try:
    import torch as th
    from stable_baselines3 import PPO
    from experiments.hybrid_upper_v3 import (HybridDistribution, HybridUpperPolicy,
        HybridUpperEnv, decode_hybrid, canonical_action)
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'optional SB3/Torch')
class ProbabilityTests(unittest.TestCase):
    def dist(self,last=False):
        return HybridDistribution().proba_distribution(th.zeros((2,27),dtype=th.float64),th.tensor([last,last]))

    def test_normalization_and_mask(self):
        for last in (False,True):
            d=self.dist(last)
            th.testing.assert_close(d.category.probs.sum(-1),th.ones(2,dtype=th.float64))
            self.assertEqual(float(d.category.probs[0,0]),0 if last else 1/3)
            self.assertTrue(th.isfinite(d.entropy()).all())

    def test_sampling_support(self):
        with th.random.fork_rng():
            th.manual_seed(301)
            d=self.dist(True)
            a=th.stack([d.sample() for _ in range(3000)])
            self.assertFalse((a[:,:,0]==0).any())
            self.assertTrue((a[:,:,1:]>0).all() and (a[:,:,1:]<1).all())
            self.assertTrue(th.isfinite(d.log_prob(a[0])).all())

    def test_joint_formula_independent(self):
        p=th.linspace(-.8,.9,54,dtype=th.float64).reshape(2,27)
        d=HybridDistribution().proba_distribution(p,th.tensor([False,True]))
        a=d.mode()
        cat=th.distributions.Categorical(logits=p[:,:3].masked_fill(th.tensor([[False]*3,[True,False,False]]),-th.inf))
        beta=th.distributions.Beta(1.0001+th.nn.functional.softplus(p[:,3:15]),1.0001+th.nn.functional.softplus(p[:,15:]))
        th.testing.assert_close(d.log_prob(a),cat.log_prob(a[:,0])+beta.log_prob(a[:,1:]).sum(-1))
        th.testing.assert_close(d.entropy(),cat.entropy()+beta.entropy().sum(-1))

    def test_density_integrates_to_one(self):
        # Independent Gauss-Legendre quadrature of one-dimensional Beta;
        # independence makes product of 12 normalized integrals the joint normalization.
        x,w=np.polynomial.legendre.leggauss(512);u=th.tensor((x+1)/2,dtype=th.float64)
        for last in (False,True):
            d=self.dist(last);beta=th.distributions.Beta(d.a[0,0],d.b[0,0])
            integral=float(np.dot(w/2,beta.log_prob(u).exp().numpy()))
            self.assertAlmostEqual(integral**12*float(d.category.probs[0].sum()),1,places=6)

    def test_entropy_monte_carlo(self):
        with th.random.fork_rng():
            th.manual_seed(312)
            p=th.zeros((12000,27));d=HybridDistribution().proba_distribution(p,th.ones(12000,dtype=th.bool))
            self.assertLess(abs(float(-d.log_prob(d.sample()).mean()-d.entropy().mean())),.045)

    def test_gradients_and_mask_finite_difference(self):
        p=th.zeros((1,27),dtype=th.float64,requires_grad=True)
        a=th.tensor([[2.]+[.3]*12],dtype=th.float64)
        def f(x):return HybridDistribution().proba_distribution(x,th.tensor([True])).log_prob(a).sum()
        f(p).backward();self.assertTrue(th.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[0,0]),0.)
        for j in (1,2,3,15):
            self.assertNotEqual(float(p.grad[0,j]),0.)
            plus=p.detach().clone();minus=plus.clone();plus[0,j]+=1e-5;minus[0,j]-=1e-5
            self.assertAlmostEqual(float(p.grad[0,j]),float((f(plus)-f(minus))/2e-5),places=7)

    def test_invalid_actions_rejected(self):
        obs={'calendar':np.array([0,0,1],np.float32)}
        for a in ([0]+[.5]*12,[1.3]+[.5]*12,[2]+[float('nan')]*12,[2]+[-.1]*12):
            with self.assertRaises(ValueError):decode_hybrid(obs,a)
        with self.assertRaises(ValueError):self.dist().log_prob(th.ones((2,13)))

    def test_semantics_first_and_last(self):
        from v3.environments import decode_upper
        for last in (0,1):
            obs={'calendar':np.array([0,0,last],np.float32)}
            for c in ((0,1,2) if not last else (1,2)):
                a=np.array([c]+[.3]*12,np.float32)
                self.assertEqual(decode_upper(obs,canonical_action(obs,a))[0],('WAIT','SKIP','COMMIT')[c])

    def test_no_test_constructor(self):
        from unittest.mock import patch
        with patch('v3.environments.YearStore',side_effect=AssertionError('must not construct')):
            with self.assertRaises(PermissionError):HybridUpperEnv(split='test')

    def test_policy_forward_evaluate_reload(self):
        env=HybridUpperEnv(seed=0);obs,_=env.reset(seed=0)
        model=PPO(HybridUpperPolicy,env,seed=0,n_steps=32,batch_size=32,n_epochs=2,
            policy_kwargs={'net_arch':[32,32]},device='cpu')
        ot,_=model.policy.obs_to_tensor(obs)
        with th.no_grad():
            a,v,lp=model.policy(ot);v2,lp2,ent=model.policy.evaluate_actions(ot,a)
        th.testing.assert_close(lp,lp2,rtol=0,atol=0)
        th.testing.assert_close(v,v2,rtol=0,atol=0)
        th.testing.assert_close((lp2-lp).exp(),th.ones_like(lp))
        self.assertTrue(th.isfinite(ent).all())
        self.assertTrue(env.action_space.contains(a[0].numpy()))
        self.assertTrue(th.equal(a,th.maximum(th.minimum(a,th.tensor(env.action_space.high)),th.tensor(env.action_space.low))))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime-tmp') as tmp:
            path=Path(tmp)/'policy.zip';model.save(path);loaded=PPO.load(path)
            np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
        self.assertEqual(model.num_timesteps,0)

    def test_real_contract_and_last_skip(self):
        env=HybridUpperEnv(seed=0);obs,_=env.reset(seed=0)
        env.step(np.array([0]+[.5]*12,np.float32)) # first WAIT -> final quote
        self.assertEqual(env.c.next_op().index,1)
        quantity=env.c.ledger.quantity.copy();fee=env.c.ledger.fee.copy();lots=len(env.c.ledger.lots)
        env.step(np.array([1]+[.5]*12,np.float32)) # final SKIP must not turn into COMMIT
        np.testing.assert_array_equal(env.c.ledger.quantity,quantity)
        np.testing.assert_array_equal(env.c.ledger.fee,fee)
        self.assertEqual(len(env.c.ledger.lots),lots)
        env.step(np.array([2]+[.5]*12,np.float32))
        self.assertEqual(len(env.c.ledger.lots),lots+1)
        self.assertGreater(sum(env.c.ledger.lots[-1].quantities),0)
        self.assertTrue((env.c.ledger.quantity<=env.c.world.reference).all())

@unittest.skipUnless(AVAILABLE,'optional SB3/Torch')
class ProtocolTests(unittest.TestCase):
    def test_sources_and_history(self):
        from experiments.d8_common import verify_protection,check_sources
        verify_protection();check_sources()

    def test_only_action_representation_changed(self):
        from experiments.d8_common import PARAMS,PROTOCOL,ROOT
        old=json.loads((ROOT/'outputs/v3/d7b_upper_budget_20261015/experiment_config.json').read_text())
        self.assertEqual(PARAMS,old)
        self.assertEqual(PROTOCOL['seeds'],[0,1])
        self.assertEqual(PROTOCOL['hybrid']['smoke_steps'],32)
        self.assertFalse(PROTOCOL['test_authorized'])

    def test_evaluation_gate_before_model_or_storage(self):
        from unittest.mock import patch
        from experiments.evaluate_upper_hybrid_d8 import evaluate_new
        with patch('experiments.evaluate_upper_hybrid_d8.verify_models',side_effect=AssertionError('too late')):
            with self.assertRaises(PermissionError):evaluate_new('unused',{},split='test')

    def test_existing_output_refused(self):
        from unittest.mock import patch
        from experiments.train_upper_hybrid_d8 import main,OUT
        if not OUT.exists():self.skipTest('not started')
        with patch('sys.argv',['entry']),patch('experiments.train_upper_hybrid_d8.preflight',side_effect=AssertionError('too late')):
            with self.assertRaises(FileExistsError):main()

@unittest.skipUnless(AVAILABLE,'optional SB3/Torch')
class ArtifactTests(unittest.TestCase):
    def setUp(self):
        from experiments.d8_common import OUT
        self.out=OUT
        if not (OUT/'completion.json').exists():self.skipTest('formal artifacts not completed')

    def read(self,path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

    def test_budgets_and_real_final_update(self):
        for seed in (0,1):
            p=self.out/f'seed_{seed}'
            s=json.loads((p/'training_summary.json').read_text())
            self.assertEqual(s['steps'],1536);self.assertEqual(s['updates'],96)
            self.assertEqual(set(s['optimizer_steps'].values()),{96})
            opt=self.read(p/'optimizer_updates.csv');self.assertEqual(len(opt),96)
            self.assertEqual(int(opt[-1]['update']),96)
            self.assertTrue(all(np.isfinite(float(r['category_grad_l2'])) and np.isfinite(float(r['beta_grad_l2'])) for r in opt))
            signals=self.read(p/'update_signals.csv')
            self.assertEqual(max(int(r['timesteps']) for r in signals),1536)
            self.assertTrue(all(np.isfinite(float(r['value'])) for r in signals))
        smoke=json.loads((self.out/'engineering_smoke/training_summary.json').read_text())
        self.assertEqual(smoke['steps'],32);self.assertEqual(smoke['updates'],2)

    def test_categories_probabilities_and_samples(self):
        for seed in (0,1):
            p=self.out/f'seed_{seed}'
            records=self.read(p/'sampled_action_probabilities.csv')
            self.assertEqual(len(records),1536)
            for r in records:
                probs=np.array([float(r['p_'+v]) for v in ('WAIT','SKIP','COMMIT')])
                self.assertAlmostEqual(float(probs.sum()),1,places=6)
                self.assertGreater(probs[2],0)
                if int(r['opportunity'])==1:
                    self.assertEqual(probs[0],0);self.assertNotEqual(r['verb'],'WAIT')
                a=np.array(json.loads(r['action']))
                self.assertIn(a[0],(0,1,2));self.assertTrue(((a[1:]>0)&(a[1:]<1)).all())
                self.assertTrue(np.isfinite(float(r['joint_log_prob'])))
                if r['verb']!='COMMIT':self.assertEqual(float(r['increment_mwh']),0)

    def test_saved_models_and_no_update(self):
        from unittest.mock import patch
        from v3.data import sha
        from v3.training import policy_hash
        lock=json.loads((self.out/'prevalidation_model_lock.json').read_text())
        self.assertEqual(len(lock['models']),2)
        with patch.object(PPO,'learn',side_effect=AssertionError('forbidden')):
            for m in lock['models']:
                path=Path(m['model_path']);self.assertEqual(sha(path),m['model_sha256'])
                model=PPO.load(path,device='cpu');digest=policy_hash(model)
                self.assertEqual(digest,m['policy_sha256'])
                env=HybridUpperEnv(seed=m['seed']);obs,_=env.reset(seed=m['seed'])
                a=model.predict(obs,deterministic=True)[0];b=model.predict(obs,deterministic=True)[0]
                np.testing.assert_array_equal(a,b);self.assertEqual(digest,policy_hash(model))
                self.assertIsInstance(model.policy,HybridUpperPolicy)

    def test_evaluation_costs_tasks_and_scenario_fairness(self):
        from experiments.d8_common import CANDIDATES
        rr=self.read(self.out/'validation/combined_annual.csv');self.assertEqual(len(rr),48)
        for world in {x['world'] for x in rr}:
            same=[r for r in rr if r['world']==world]
            self.assertEqual({x['strategy'] for x in same},set(CANDIDATES))
            self.assertEqual(len({x['scenario_sha256'] for x in same}),1)
            self.assertTrue(world.startswith('validation_'))
            for row in same:
                self.assertAlmostEqual(float(row['purchase_cost'])+float(row['flex_cost']),float(row['objective_cost']),places=5)
                self.assertAlmostEqual(float(row['ev_grid_mwh']),365*100/3,places=7)
                self.assertAlmostEqual(float(row['industrial_mwh']),365*12,places=7)
        hourly=self.out/'validation/hourly.csv'
        sums={}
        with hourly.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                key=(row['strategy'],row['world']);v=sums.setdefault(key,[0.,0.,0])
                cost=float(row['purchase_cost']);flex=float(row['flex_cost'])
                self.assertAlmostEqual(cost,float(row['contract_cost'])+float(row['spot_cost'])+float(row['balance_cost']),places=7)
                self.assertAlmostEqual(float(row['reward']),-cost-flex,places=7)
                v[0]+=cost;v[1]+=flex;v[2]+=1
        for key,(cost,flex,n) in sums.items():
            annual=next(r for r in rr if (r['strategy'],r['world'])==key)
            self.assertEqual(n,8760);self.assertAlmostEqual(cost,float(annual['purchase_cost']),places=5)
            self.assertAlmostEqual(flex,float(annual['flex_cost']),places=6)

if __name__=='__main__':unittest.main()
