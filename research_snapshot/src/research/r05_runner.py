"""One explicitly approved R05 pilot; no legacy CLI, discovery or retry path.

Top-level imports and run_preflight are standard-library only. Real dependencies
are imported after the fixed approval/config/spec contract has passed. This is a
trusted capability router, with OS IO/process/RSS controls owned by its supervisor.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import inspect
import importlib.util
import json
import math
from pathlib import Path
import unittest

CONFIG_SHA256 = "3238304aa13ade077fd30ff6463d2d507ae8ee0c7b60ec8d5701b3bb7650a97e"
APPROVAL_ANSWER = "批准该R05单次先导包"
SPEC = dict(year=2033,family="w3_pilot_20261025",group="G00",split="train",fixture=False)


def same_json(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False,separators=(",",":"))==json.dumps(right,sort_keys=True,allow_nan=False,separators=(",",":"))


def validate_launch(config, approved_budgets, approval_answer):
    """Pure pre-construction checks; approval text is supplied by the trusted owner."""
    if approval_answer != APPROVAL_ANSWER:
        raise PermissionError("explicit R05 pilot package approval required")
    if not isinstance(config,dict) or config.get("task_id")!="R05" or config.get("work_package")!="W3":
        raise PermissionError("wrong work package")
    if not same_json(config.get("input_spec"),SPEC):
        raise PermissionError("only the approved synthetic development world is accessible")
    if not same_json(config.get("proposed_budgets"),approved_budgets):
        raise PermissionError("approved numeric budgets differ")
    p=config.get("parameters",{})
    exact=dict(level="upper",seed=0,device="cpu",torch_threads=1,learning_rate=.0003,
        gamma=1.,gae_lambda=.95,n_steps=32,batch_size=32,n_epochs=2,total_timesteps=32,
        n_envs=1,reset_num_timesteps=True,progress_bar=False,tensorboard_log=None,
        verbose=0,normalize_advantage=True,clip_range=.2,clip_range_vf=None,
        ent_coef=0.,vf_coef=.5,max_grad_norm=.5,use_sde=False,target_kl=None)
    if any(p.get(k)!=v for k,v in exact.items()):
        raise PermissionError("frozen PPO configuration changed")
    if p.get("net_arch")!={"pi":[32,32],"vf":[32,32]} or p.get("optimizer_kwargs")!={"eps":1e-5}:
        raise PermissionError("network/optimizer configuration changed")
    if p.get("reward_divisor")!=1e6 or p.get("lr_schedule")!="constant":
        raise PermissionError("economic scale/schedule changed")
    if not config.get("exposure",{}).get("training_and_evaluation_same_world"):
        raise PermissionError("this is one exposed exploratory world, not confirmation")
    return deepcopy(SPEC)


def authorize(ctx, config, phase):
    guard=getattr(ctx,"guard",{})
    if (guard.get("approved") is not True or guard.get("owner") is None
            or guard.get("phase")!=phase or guard.get("config_sha256")!=CONFIG_SHA256
            or not same_json(guard.get("config"),config)
            or guard.get("source_identity")!=getattr(ctx,"source_identity",None)
            or not getattr(ctx,"source_identity",None)):
        raise PermissionError("fixed owner/config/source approval is missing")
    validate_launch(config,config["proposed_budgets"],APPROVAL_ANSWER)
    ctx.check()
    return deepcopy(config)


def require_spec(request, config):
    if type(request) is not dict or not same_json(request,config["input_spec"]) or not same_json(request,SPEC):
        raise PermissionError("deny before world factory: unapproved spec")
    return dict(request)


class Hooks:
    """Temporary methods, restored even after failures; never retry an operation."""
    def __init__(self):self.saved=[]
    def replace(self, owner, name, replacement):
        own=name in vars(owner);value=vars(owner).get(name)
        self.saved.append((owner,name,own,value));setattr(owner,name,replacement)
    def wrap(self, owner, name, before, after=None):
        original=getattr(owner,name)
        def wrapped(*args,**kwargs):
            before(*args,**kwargs)
            result=original(*args,**kwargs)
            if after is not None:after(result,*args,**kwargs)
            return result
        self.replace(owner,name,wrapped)
        return original
    def restore(self):
        errors=[]
        for owner,name,own,value in reversed(self.saved):
            try:
                if own:setattr(owner,name,value)
                else:delattr(owner,name)
            except BaseException as error:errors.append(error)
        self.saved=[]
        if errors:
            chosen=prioritize_error(errors)
            chosen.r05_restore_errors=[dict(type=type(e).__name__,message=repr(e)) for e in errors]
            raise chosen


def prioritize_error(errors):
    """First control interruption wins; otherwise preserve the first ordinary error."""
    return next((e for e in errors if not isinstance(e,Exception)),errors[0])


@contextmanager
def preserving_cleanup(ctx,stage,cleanup):
    """Always attempt cleanup; a later ordinary error cannot hide a control signal."""
    errors=[]
    try:yield
    except BaseException as primary:errors.append(primary)
    try:cleanup()
    except BaseException as cleanup_error:
        errors.append(cleanup_error)
        note=dict(stage=stage,type=type(cleanup_error).__name__,message=repr(cleanup_error))
        if hasattr(ctx,"finalization_errors"):ctx.finalization_errors.append(note)
        try:ctx.emit("failures",note)
        except BaseException as journal_error:
            errors.append(journal_error)
            if hasattr(ctx,"finalization_errors"):
                ctx.finalization_errors.append(dict(stage=stage+":journal",type=type(journal_error).__name__,message=repr(journal_error)))
    if errors:
        chosen=prioritize_error(errors)
        chosen.r05_cleanup_errors=[dict(type=type(e).__name__,message=repr(e)) for e in errors if e is not chosen]
        raise chosen


@contextmanager
def installed_hooks(ctx):
    hooks=Hooks()
    with preserving_cleanup(ctx,"hook_restore",hooks.restore):yield hooks


def plain(value):
    """JSON copy; only data conversion, never model calls or file discovery."""
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:
        if not math.isfinite(value):raise ValueError("nonfinite numeric evidence")
        return value
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,"detach"):return plain(value.detach().cpu().numpy())
    if hasattr(value,"tolist"):return plain(value.tolist())
    if hasattr(value,"isoformat"):return value.isoformat()
    raise TypeError("unsupported evidence value: "+type(value).__name__)


def run_preflight(ctx, config):
    """Exactly the new fake-only tests; never import NumPy/Torch/project models."""
    frozen=authorize(ctx,config,"preflight")
    file=Path(__file__).resolve().parents[1]/"tests"/"test_r05_pilot_contract.py"
    spec=importlib.util.spec_from_file_location("r05_pilot_contract_checks",file)
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    checks.CONTEXT=ctx;checks.CONFIG=frozen
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(checks.PilotContractTests)
    from io import StringIO
    output=StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    passed=result.wasSuccessful() and result.testsRun==6 and not result.skipped
    receipt=dict(tests=result.testsRun,tests_run=result.testsRun,
                 failures=[dict(test=str(t),traceback=tb) for t,tb in result.failures],
                 errors=[dict(test=str(t),traceback=tb) for t,tb in result.errors],
                 skipped=[dict(test=str(t),reason=reason) for t,reason in result.skipped],
                 unittest_output=output.getvalue(),status="PASS" if passed else "STOPPED",
                 fake_calls=ctx.counts.get("preflight_fake_calls",0),
                 project_models_worlds_updates=0)
    ctx.write_json("preflight_result.json",receipt)
    if not passed:
        raise RuntimeError("R05 preflight failed/incomplete/skipped; no repair/retry")
    return receipt


def pack_rule_action(verb, fractions):
    """Preserve the original rule's binary64 fractions through hybrid transport."""
    code = {"WAIT": 0., "SKIP": 1., "COMMIT": 2.}[verb]
    return [code] + [float(value) for value in fractions]


def validate_checkpoint_payload(payload, expected_sha256, max_bytes):
    if type(payload) is not bytes or not 0<len(payload)<=max_bytes:
        raise PermissionError("owned checkpoint byte/type limit")
    if sha256(payload).hexdigest()!=expected_sha256:
        raise PermissionError("owned checkpoint content identity mismatch")
    return payload


def validate_checkpoint_state(saved,reloaded,saved_config,reloaded_config):
    persisted=("parameters","optimizer","num_timesteps","n_updates")
    if saved.get("num_timesteps")!=32 or saved.get("n_updates")!=2:
        raise RuntimeError("saved state is not exact32/two-update pilot")
    if any(saved[k]!=reloaded[k] for k in persisted) or not same_json(saved_config,reloaded_config):
        raise RuntimeError("checkpoint persisted fields/config do not roundtrip")
    return persisted


def run_pilot(ctx, config):
    frozen=authorize(ctx,config,"pilot")
    # Everything above this point is dependency-free and precedes all model/world constructors.
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.policies import BasePolicy
    from experiments.hybrid_upper_v3 import HybridUpperPolicy
    from v31 import lite_gym as gymmod
    from v31.lite_data import LiteStore
    from v31.lite_engine import LiteAnnualDriver
    from v31.lite_rules import RULES
    from v3.coordinator import YearCoordinator
    from v3.devices import DailyDevices
    from v3.data import PublishedWorld
    from research.tail_budget import TailBudget,finish_tail
    from research.runner_contract import snapshot_learning_state,environment_receipt,rule_state_reader,state_digest
    p=frozen["parameters"];budgets=frozen["proposed_budgets"]
    envs=[];labels={};coordinators={};state=dict(phase="prepare",allow_reset=False,
        allow_train=False,allow_save=False,allow_load=False,allow_constructor=False,
        store_created=False,probe=False)
    def charge(kind,key,amount=1):ctx.charge(kind,key,amount);ctx.check()
    def deny(message):raise PermissionError(message)
    def emit(name,obj):ctx.emit(name,plain(obj))
    def check_finite(value):
        if isinstance(value,(tuple,list)):
            for x in value:check_finite(x)
        elif isinstance(value,dict):
            for x in value.values():check_finite(x)
        elif hasattr(value,"detach"):
            if not bool(torch.isfinite(value).all()):raise ValueError("nonfinite policy tensor")
        elif isinstance(value,np.ndarray) and not np.isfinite(value).all():
            raise ValueError("nonfinite policy array")
    def batch_rows(obs):
        first=next(iter(obs.values())) if isinstance(obs,dict) else obs
        if len(first.shape)<2:raise ValueError("instrumented policy API requires a batched observation")
        n=int(first.shape[0])
        if not 1<=n<=budgets["policy_value_batch_rows_max"]:raise RuntimeError("per-API batch row cap exceeded")
        return n
    def api_before(model, obs, *args, **kwargs):
        n=batch_rows(obs);charge("policy_value_API_calls","policy_value_API_calls_max")
        charge("policy_value_total_batch_rows","policy_value_total_batch_rows_max",n)
        check_finite(obs)
    def api_after(result,*args,**kwargs):
        if hasattr(result,"category"):
            check_finite((result.category.probs,result.a,result.b))
        else:check_finite(result)
    def config_identity(model):
        keys=("seed","gamma","gae_lambda","n_steps","batch_size","n_epochs",
              "normalize_advantage","ent_coef","vf_coef","max_grad_norm","use_sde","target_kl","n_envs")
        values={k:getattr(model,k) for k in keys}
        values.update(device=str(model.device),learning_rate=float(model.learning_rate),
            clip_range=float(model.clip_range(1.)),clip_range_vf=model.clip_range_vf,
            net_arch=model.policy.net_arch,activation_fn=model.policy.activation_fn.__module__+"."+model.policy.activation_fn.__name__,
            ortho_init=model.policy.ortho_init,normalize_images=model.policy.normalize_images,
            share_features_extractor=model.policy.share_features_extractor,
            optimizer_class=type(model.policy.optimizer).__module__+"."+type(model.policy.optimizer).__name__,
            optimizer_eps=[g["eps"] for g in model.policy.optimizer.param_groups])
        for key in keys:
            if values[key]!=p[key]:raise RuntimeError("actual config mismatch: "+key)
        if (values["device"]!="cpu" or values["learning_rate"]!=p["learning_rate"]
                or values["clip_range"]!=p["clip_range"] or values["clip_range_vf"] is not None
                or values["net_arch"]!=p["net_arch"] or not values["ortho_init"]
                or not values["normalize_images"] or not values["share_features_extractor"]
                or values["optimizer_eps"]!=[1e-5]
                or model.policy.activation_fn is not torch.nn.Tanh
                or type(model.policy.optimizer) is not torch.optim.Adam):
            raise RuntimeError("actual policy/optimizer configuration mismatch")
        values["spaces_sha256"]=state_digest(dict(
            observation={k:dict(shape=list(s.shape),dtype=str(s.dtype),low=s.low,high=s.high)
                         for k,s in model.observation_space.spaces.items()},
            action=dict(shape=list(model.action_space.shape),dtype=str(model.action_space.dtype),
                        low=model.action_space.low,high=model.action_space.high)))
        return plain(values)
    def diag_call(name,rows,fn):
        charge("additional_diagnostic_API_calls","additional_diagnostic_API_calls_max")
        charge("additional_diagnostic_batch_rows","additional_diagnostic_batch_rows_max",rows)
        with torch.no_grad():result=fn()
        emit("diagnostics",dict(name=name,rows=rows))
        return result
    def boundary(env,label):ctx.boundary(env,label)
    def dump_finish(env,label,metric):
        for row in env.driver.macro_receipts:emit(label+"_macro",row)
        for row in env.driver.daily_receipts:emit(label+"_daily",row)
        for row in env.driver.events:emit(label+"_events",row)
        for row in env.steps:emit(label+"_steps",row)
        for row in env.resets:emit(label+"_resets",row)
        emit(label+"_metrics",dict(metrics=metric,world=env.driver.store.manifest(),
              liability=environment_receipt(env),source_identity=ctx.source_identity,
              config_sha256=CONFIG_SHA256,role=label))
    def make_env(cache,label):
        state["new_env_label"]=label
        env=gymmod.AnnualDecisionEnv(cache,"upper",seed=0,partner=rule_lower)
        envs.append(env);labels[id(env)]=label
        return env
    def diagnose_distribution(dist):
        return dict(category_probs=plain(dist.category.probs),beta_a=plain(dist.a),beta_b=plain(dist.b))

    try:
        with installed_hooks(ctx) as hooks:
            def counted(owner,name,kind,key):
                hooks.wrap(owner,name,lambda *a,**kw:charge(kind,key))
            def world_before(store,*args,**kwargs):
                bound=inspect.signature(original_store).bind(store,*args,**kwargs);bound.apply_defaults()
                spec={k:bound.arguments[k] for k in SPEC};require_spec(spec,frozen)
                if state["store_created"]:deny("second native world denied before constructor")
                charge("new_native_worlds","new_native_worlds_max");state["store_created"]=True
            original_store=hooks.wrap(LiteStore,"__init__",world_before)
            counted(LiteStore,"replay_copy","audited_world_replay_copies","audited_world_replay_copies_max")
            counted(LiteAnnualDriver,"__init__","drivers","drivers_max")
            counted(YearCoordinator,"__init__","coordinators","coordinators_max")
            counted(PublishedWorld,"__init__","published_worlds","published_worlds_max")
            counted(DailyDevices,"prepare","device_prepare_hours","device_prepare_hours_max")
            counted(YearCoordinator,"prepare_hour","hour_prepare_attempts","hour_prepare_attempts_max")
            counted(LiteAnnualDriver,"_publication","publication_and_DA_seal_events","publication_and_DA_seal_events_max")
            counted(LiteAnnualDriver,"submit_upper","upper_submission_attempts","upper_submission_attempts_max")
            counted(LiteAnnualDriver,"submit_lower","lower_submission_attempts","lower_submission_attempts_max")
            counted(gymmod.AnnualDecisionEnv,"step","native_env_step_calls","native_env_step_calls_max")
            def settled(row,c,*args,**kwargs):
                charge("delivered_hours","delivered_hours_max")
                env=coordinators.get(id(c))
                if env is None:raise RuntimeError("settlement has no bound native environment")
                emit(labels[id(env)]+"_hourly",row)
                if c.t%24==0:boundary(env,labels[id(env)]+"_day_"+str(c.t//24))
            hooks.wrap(YearCoordinator,"settle_pending",
                lambda *a,**kw:charge("hour_settlement_attempts","hour_settlement_attempts_max"),settled)
            def env_before(env,cache,level,*,seed=0,partner=None):
                require_spec(cache.specs[0],frozen)
                if level!="upper" or seed!=0 or partner is not rule_lower:deny("unapproved environment construction")
                charge("envs","envs_max")
            hooks.wrap(gymmod.AnnualDecisionEnv,"__init__",env_before)
            def reset_before(env,*args,**kwargs):
                if not state["allow_reset"] or env.driver is not None:deny("automatic/extra reset denied")
                if kwargs.get("seed",0) not in (None,0):deny("unapproved reset seed")
                state["allow_reset"]=False
                charge("native_resets","native_resets_max")
            def reset_after(result,env,*args,**kwargs):
                coordinators[id(env.driver.c)]=env
                boundary(env,labels[id(env)]+"_initial")
            hooks.wrap(gymmod.AnnualDecisionEnv,"reset",reset_before,reset_after)
            original_upper,original_lower=RULES["B2"]
            rule_identity=dict(upper=rule_state_reader(original_upper),lower=rule_state_reader(original_lower))
            def rule_lower(raw):
                charge("rule_lower_calls","rule_lower_calls_max")
                return original_lower(raw)
            def rule_upper(raw):
                charge("rule_upper_calls","rule_upper_calls_max")
                return original_upper(raw)
            def audit_before(driver):charge("native_annual_deliveries","native_annual_deliveries_max")
            hooks.wrap(gymmod,"audit_year",audit_before)
            def constructor_before(model,*args,**kwargs):
                if not state["allow_constructor"]:deny("unapproved PPO constructor")
                charge("PPO_controllers","PPO_controllers_max")
                if state["allow_load"]:
                    if kwargs.get("_init_setup_model") is not False:deny("reload must be native SB3 reconstruction")
                else:
                    if kwargs.get("seed")!=0 or str(kwargs.get("device"))!="cpu":deny("constructor seed/device mismatch")
                    for name in ("n_steps","batch_size","n_epochs","gamma","gae_lambda"):
                        if kwargs.get(name)!=p[name]:deny("constructor config mismatch: "+name)
            hooks.wrap(PPO,"__init__",constructor_before)
            def policy_before(policy,*args,**kwargs):
                if not state["allow_constructor"]:deny("unapproved policy construction")
                charge("policy_objects","policy_objects_max")
            hooks.wrap(HybridUpperPolicy,"__init__",policy_before)
            for name in ("forward","evaluate_actions","predict_values","get_distribution"):
                hooks.wrap(HybridUpperPolicy,name,api_before,api_after)
            def predict_before(model,*args,**kwargs):
                if state["phase"] not in ("tail","checkpoint_probe","C10"):deny("unlisted predict phase")
                charge("model_predict_wrapper_calls","model_predict_wrapper_calls_max")
            hooks.wrap(PPO,"predict",predict_before,lambda result,*a,**kw:check_finite(result))
            def train_before(model,*args,**kwargs):
                if not state["allow_train"]:deny("optimization outside one approved update stage")
                charge("PPO_train_calls","PPO_train_calls_max")
            original_train=hooks.wrap(PPO,"train",train_before)
            def scalar(value):
                if hasattr(value,"detach"):value=value.detach().cpu().item()
                elif hasattr(value,"item"):value=value.item()
                number=float(value)
                if not math.isfinite(number):raise ValueError("nonfinite actual update diagnostic")
                return number
            def actual_update_diagnostic(loss):
                # Inspect only the already executing fixed SB3 train frame. No forward.
                frame=inspect.currentframe();values=None
                try:
                    frame=frame.f_back
                    while frame is not None and frame.f_code is not original_train.__code__:
                        frame=frame.f_back
                    if frame is None:raise RuntimeError("backward has no original approved PPO.train frame")
                    values=frame.f_locals
                    if values.get("loss") is not loss:raise RuntimeError("backward tensor differs from actual PPO loss")
                    if values.get("self") is not model:raise RuntimeError("update diagnostics refer to another model")
                    epoch=int(values["epoch"]);data=values["rollout_data"]
                    if epoch not in (0,1) or len(data.actions)!=32:raise RuntimeError("unexpected real update epoch/batch")
                    return plain(dict(epoch=epoch,batch_rows=len(data.actions),
                        ppo_update_counter_before=int(model._n_updates),
                        loss=scalar(values["loss"]),policy_loss=scalar(values["policy_loss"]),
                        value_loss=scalar(values["value_loss"]),entropy_loss=scalar(values["entropy_loss"]),
                        approx_kl=scalar(values["approx_kl_div"]),clip_fraction=scalar(values["clip_fraction"]),
                        clip_range=scalar(values["clip_range"]),
                        actual_values=values["values"],target_returns=data.returns,
                        old_values=data.old_values,new_log_prob=values["log_prob"],old_log_prob=data.old_log_prob,
                        normalized_advantages=values["advantages"],raw_advantages=data.advantages,
                        ratio=values["ratio"],entropy=values["entropy"]))
                finally:
                    # Do not retain train frames/autograd graphs in the evidence state.
                    del frame
                    if values is not None:del values
            def update_before(optimizer,*args,**kwargs):
                if not state["allow_train"] or optimizer is not model.policy.optimizer:
                    deny("optimizer operation outside approved model/stage")
                charge("optimizer_step_attempts","optimizer_step_attempts_max")
                diagnostic=state.get("current_step_diag")
                if diagnostic is None:raise RuntimeError("Adam has no captured actual backward diagnostics")
                norm_squared=0.;gradient_tensors=0
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        check_finite(parameter)
                        if parameter.grad is not None:
                            check_finite(parameter.grad);gradient_tensors+=1
                            gradient=parameter.grad.detach().to(dtype=torch.float64)
                            norm_squared+=float(torch.sum(gradient*gradient).item())
                norm=math.sqrt(norm_squared)
                if not gradient_tensors or not math.isfinite(norm):raise ValueError("invalid gradients before Adam")
                if norm>p["max_grad_norm"]+2e-6:raise RuntimeError("actual clipped gradient norm exceeds frozen setting")
                diagnostic.update(optimizer_attempt=ctx.counts["optimizer_step_attempts"],
                    gradient_norm_after_clip=norm,gradient_tensors_checked=gradient_tensors,
                    gradients_checked_before_adam=True)
            def update_after(result,optimizer,*args,**kwargs):
                ctx.counts["optimizer_steps_completed"]=ctx.counts.get("optimizer_steps_completed",0)+1
                diagnostic=state.pop("current_step_diag")
                for group in optimizer.param_groups:
                    for parameter in group["params"]:check_finite((parameter,parameter.grad))
                emit("optimizer",dict(completed=ctx.counts["optimizer_steps_completed"],
                    attempt=ctx.counts["optimizer_step_attempts"],actual_update=diagnostic,
                    parameter_steps=[float(v["step"]) for v in optimizer.state.values() if "step" in v]))
            hooks.wrap(torch.optim.Adam,"step",update_before,update_after)
            def backward_before(loss,*args,**kwargs):
                if not state["allow_train"]:deny("backward outside approved stage")
                charge("backward_attempts","backward_attempts_max");check_finite(loss)
                if "current_step_diag" in state:raise RuntimeError("previous backward has no completed Adam step")
                state["current_step_diag"]=actual_update_diagnostic(loss)
            def backward_after(result,*args,**kwargs):
                ctx.counts["backward_completed"]=ctx.counts.get("backward_completed",0)+1
            hooks.wrap(torch.Tensor,"backward",backward_before,backward_after)
            def collect_before(model,*args,**kwargs):
                if state["phase"]!="collect":deny("unapproved collection")
                charge("rollout_collection_calls","rollout_collection_calls_max")
            hooks.wrap(PPO,"collect_rollouts",collect_before)
            hooks.replace(PPO,"learn",lambda *a,**kw:deny("learn CLI is not an approved pilot entry"))
            original_seed=getattr(BaseAlgorithm,"set_random_seed")
            def checked_seed(model,seed=None):
                if seed!=0:deny("unapproved training/reload seed")
                return original_seed(model,seed)
            hooks.replace(BaseAlgorithm,"set_random_seed",checked_seed)
            def save_before(model,path,*args,**kwargs):
                if not state["allow_save"] or not isinstance(path,BytesIO):deny("only one in-RAM own checkpoint save")
                charge("new_checkpoint_saves","new_checkpoint_saves_max")
            hooks.wrap(PPO,"save",save_before)
            def load_before(path,*args,**kwargs):
                if not state["allow_load"] or not isinstance(path,BytesIO):deny("historical/path checkpoint load denied")
                charge("new_checkpoint_loads","new_checkpoint_loads_max")
            original_load=PPO.load
            def checked_load(*args,**kwargs):load_before(*args,**kwargs);return original_load(*args,**kwargs)
            hooks.replace(PPO,"load",checked_load)

            class OwnedCache(gymmod.WorldCache):
                def make(self,index):
                    if type(index) is not int or index!=0:deny("unapproved world index")
                    require_spec(self.specs[0],frozen)
                    key=tuple(sorted(self.specs[0].items()))
                    if state["store_created"] and key not in self.compiled:deny("audited cache miss before world factory")
                    return super().make(index)
                def complete(self,store):
                    require_spec(dict(year=store.year,family=store.family,group=store.group,
                                      split=store.split,fixture=store.fixture),frozen)
                    if not store.completed_year:deny("unaudited cache completion")
                    return super().complete(store)
            class Capture(BaseCallback):
                def __init__(self):super().__init__();self.rows=[]
                def _on_step(self):
                    charge("gradient_collection_steps","gradient_collection_steps_max")
                    x=self.locals
                    row=dict(index=len(self.rows),observations={k:v.copy() for k,v in self.model._last_obs.items()},
                        actions=x["actions"].copy(),clipped_env_actions=x["clipped_actions"].copy(),
                        values=x["values"].detach().cpu().numpy().copy(),log_prob=x["log_probs"].detach().cpu().numpy().copy(),
                        rewards=x["rewards"].copy(),dones=x["dones"].copy(),
                        macro_decision=dict(envs[0].driver.macro_receipts[-1]))
                    if bool(np.asarray(row["dones"]).any()):raise RuntimeError("unexpected annual termination inside rollout32")
                    self.rows.append(row);emit("rollout",row)
                    boundary(envs[0],"training_prefix_"+str(len(self.rows)))
                    return True

            with ctx.stage("prepare_and_collect"):
                charge("world_families","world_families_max")
                charge("condition_groups","condition_groups_max")
                charge("training_seeds","training_seeds_max")
                torch.set_num_threads(1)
                cache=OwnedCache([require_spec(SPEC,frozen)])
                training=make_env(cache,"training_trajectory")
                state["allow_constructor"]=True
                try:
                    model=PPO(HybridUpperPolicy,training,learning_rate=p["learning_rate"],
                        n_steps=32,batch_size=32,n_epochs=2,gamma=1.,gae_lambda=.95,
                        normalize_advantage=True,clip_range=.2,clip_range_vf=None,
                        ent_coef=0.,vf_coef=.5,max_grad_norm=.5,use_sde=False,target_kl=None,
                        policy_kwargs=dict(net_arch=deepcopy(p["net_arch"]),activation_fn=torch.nn.Tanh,
                            ortho_init=True,normalize_images=True,share_features_extractor=True,
                            optimizer_class=torch.optim.Adam,optimizer_kwargs=dict(eps=1e-5)),
                        seed=0,device="cpu",verbose=0,tensorboard_log=None)
                finally:state["allow_constructor"]=False
                actual_config=config_identity(model)
                initial=snapshot_learning_state(model)
                model.set_logger(Logger(None,[]))
                callback=Capture();state["allow_reset"]=True
                total,callback=model._setup_learn(32,callback=callback,reset_num_timesteps=True)
                if total!=32:raise RuntimeError("unexpected setup collection budget")
                source_world=training.driver.store.manifest()
                state["phase"]="collect"
                if not model.collect_rollouts(model.get_env(),callback,model.rollout_buffer,32):
                    raise RuntimeError("collector did not finish exact rollout32")
                if model.num_timesteps!=32 or model._n_updates!=0 or len(callback.rows)!=32:
                    raise RuntimeError("actual collection counters differ")
                buffer=model.rollout_buffer
                if not buffer.full or buffer.pos!=32:raise RuntimeError("actual Buffer incomplete")
                raw_rewards=math.fsum(float(r["raw_reward"]) for r in training.steps)
                prefix=environment_receipt(training)
                if not math.isclose(-raw_rewards,prefix["purchase_cost"]+prefix["flex_cost"],rel_tol=1e-11,abs_tol=1e-5):
                    raise RuntimeError("prefix rewards do not match prefix economic objective")
                ctx.write_json("prefix.json",plain(dict(receipt=prefix,raw_reward_sum=raw_rewards,
                    training_reward_sum=float(buffer.rewards.sum()),actual_config=actual_config,
                    collection_macro_stages=[r["macro_decision"]["stage"] for r in callback.rows],
                    collection_macro_times=[r["macro_decision"]["at"] for r in callback.rows],
                    scope="32 macro decisions; partial calendar coverage, not whole-year learning",
                    source_world=source_world,initial_learning_state=initial)))
                obs={k:torch.as_tensor(v[:,0],device="cpu") for k,v in buffer.observations.items()}
                actions=torch.as_tensor(buffer.actions[:,0],device="cpu")
                one={k:v[:1] for k,v in obs.items()}
                observation_one={k:v[0].detach().cpu().numpy().copy() for k,v in one.items()}
                pre_values,pre_log_prob,pre_entropy=diag_call("pre_update_evaluate",32,lambda:model.policy.evaluate_actions(obs,actions))
                pre_values=pre_values.detach().clone();pre_log_prob=pre_log_prob.detach().clone()
                np.testing.assert_allclose(pre_log_prob.cpu().numpy(),buffer.log_probs[:,0],rtol=1.3e-6,atol=1e-5)
                pre_value_probe=diag_call("pre_update_value",1,lambda:model.policy.predict_values(one)).detach().clone()
                pre_distribution=diagnose_distribution(diag_call("pre_update_distribution",1,lambda:model.policy.get_distribution(one)))
                rollout_snapshot=dict(observations=buffer.observations,actions=buffer.actions,rewards=buffer.rewards,
                    values=buffer.values,log_probs=buffer.log_probs,advantages=buffer.advantages,returns=buffer.returns,
                    episode_starts=buffer.episode_starts,pos=buffer.pos,full=buffer.full)
                ctx.write_json("rollout_buffer.json",plain(rollout_snapshot))

            with ctx.stage("two_epoch_update"):
                state["phase"]="update";state["allow_train"]=True
                try:
                    model._update_current_progress_remaining(32,32)
                    model.train()
                finally:state["allow_train"]=False
                if model._n_updates!=2 or model.num_timesteps!=32:
                    raise RuntimeError("PPO counters must be32 collected /2 updates")
                if ctx.counts.get("optimizer_step_attempts")!=2 or ctx.counts.get("backward_attempts")!=2:
                    raise RuntimeError("exactly two Adam/backward attempts required")
                post_values,post_log_prob,post_entropy=diag_call("post_update_evaluate",32,lambda:model.policy.evaluate_actions(obs,actions))
                post_value_probe=diag_call("post_update_value",1,lambda:model.policy.predict_values(one))
                post_distribution=diagnose_distribution(diag_call("post_update_distribution",1,lambda:model.policy.get_distribution(one)))
                log_ratio=(post_log_prob-pre_log_prob).detach();ratio=log_ratio.exp()
                check_finite((ratio,post_values,post_entropy))
                diagnostics=dict(pre_values=pre_values,post_values=post_values,pre_log_prob=pre_log_prob,
                    post_log_prob=post_log_prob,ratio=ratio,approx_kl=((ratio-1)-log_ratio).mean(),
                    clip_fraction=((ratio-1).abs()>.2).float().mean(),
                    pre_value_probe=pre_value_probe,post_value_probe=post_value_probe,
                    pre_distribution=pre_distribution,post_distribution=post_distribution,
                    logger=dict(model.logger.name_to_value),parameters_changed=initial["parameters"]!=snapshot_learning_state(model)["parameters"])
                ctx.write_json("learning_diagnostics.json",plain(diagnostics))

            with ctx.stage("training_no_learning_tail"):
                state["phase"]="tail"
                tail_before=snapshot_learning_state(model)
                partner_before=rule_state_reader(original_lower)
                remaining=training.driver.store.n-training.driver.c.t
                tail_limit=TailBudget(remaining,budgets["upper_submission_attempts_max"]+budgets["lower_submission_attempts_max"],
                    float(budgets["pilot_worker_hard_seconds_max"]),
                    max_policy_calls=budgets["rule_lower_calls_max"]+budgets["model_predict_wrapper_calls_max"])
                with preserving_cleanup(ctx,"training_tail_final",
                        lambda:boundary(training,"training_tail_final")):
                    tail=finish_tail(training,model,tail_limit,
                        receipt_sink=lambda receipt:emit("training_tail_receipt",receipt))
                tail_after=snapshot_learning_state(model)
                if tail_before!=tail_after or partner_before!=rule_state_reader(original_lower):
                    raise RuntimeError("trained model six-RAM state or frozen rule changed in no-learning tail")
                if config_identity(model)!=actual_config:raise RuntimeError("actual constructor/config changed in tail")
                if training.driver.store.manifest()!=source_world:raise RuntimeError("source arrays changed during training/tail")
                liability=environment_receipt(training)
                if not training.ended or liability["remaining_hours"] or liability["pending"]:
                    raise RuntimeError("training annual liability not discharged")
                training_metric=tail["tail"]
                input_tape=state_digest(plain(dict(origins=training.driver.store.context.origins,
                    validation_stamps=training.driver.store.context.validation_stamps)))
                ctx.write_json("tail_invariance.json",plain(dict(before=tail_before,after=tail_after,
                    config=actual_config,rule_state=partner_before,source_world=source_world,
                    published_input_tape_sha256=input_tape,receipt=tail,
                    scope="same trained object; excludes checkpoint reload semantics")))
                dump_finish(training,"training_trajectory",training_metric)

            with ctx.stage("own_checkpoint_roundtrip"):
                state["phase"]="checkpoint_probe"
                original_probe=diag_call("original_checkpoint_action",1,
                    lambda:model.predict(observation_one,deterministic=True))[0]
                save_state=snapshot_learning_state(model)
                payload=BytesIO();state["allow_save"]=True
                try:model.save(payload)
                finally:state["allow_save"]=False
                checkpoint=payload.getvalue()
                if not 0<len(checkpoint)<=budgets["single_checkpoint_bytes_max"]:
                    raise RuntimeError("own checkpoint byte cap")
                checkpoint_sha=sha256(checkpoint).hexdigest()
                ctx.write_bytes("model.zip",checkpoint)
                own_bytes=validate_checkpoint_payload(ctx.read_own_checkpoint("model.zip"),
                    checkpoint_sha,budgets["single_checkpoint_bytes_max"])
                if len(own_bytes)!=len(checkpoint):raise RuntimeError("own checkpoint length changed")
                state["allow_load"]=True;state["allow_constructor"]=True
                try:reloaded=PPO.load(BytesIO(own_bytes),device="cpu",force_reset=True,print_system_info=False)
                finally:state["allow_load"]=False;state["allow_constructor"]=False
                if snapshot_learning_state(model)!=save_state:
                    raise RuntimeError("saving changed original model RAM")
                reloaded_state=snapshot_learning_state(reloaded)
                # SB3 does not persist gradients/Buffer. Compare only saved fields.
                persisted=validate_checkpoint_state(save_state,reloaded_state,actual_config,config_identity(reloaded))
                reloaded_probe=diag_call("reloaded_checkpoint_action",1,
                    lambda:reloaded.predict(observation_one,deterministic=True))[0]
                np.testing.assert_array_equal(original_probe,reloaded_probe)
                ctx.write_json("checkpoint_identity.json",plain(dict(
                    checkpoint_sha256=checkpoint_sha,bytes=len(checkpoint),config_sha256=CONFIG_SHA256,
                    source_identity=ctx.source_identity,world=source_world,
                    saved_field_names=persisted,saved_fields={k:save_state[k] for k in persisted},
                    reloaded_fields={k:reloaded_state[k] for k in persisted},actual_config=actual_config,
                    original_action=original_probe,reloaded_action=reloaded_probe,
                    not_compared_across_reload=["gradients","rollout_buffer"],
                    origin="created by this same approved attempt; not historical provenance inference")))

            def evaluate(label,policy_model=None):
                state["phase"]=label
                env=make_env(cache,label)
                state["allow_reset"]=True
                obs,_=env.reset(seed=0)
                frozen_before=snapshot_learning_state(policy_model) if policy_model is not None else None
                rule_before=dict(upper=rule_state_reader(original_upper),lower=rule_state_reader(original_lower))
                while not env.ended:
                    ctx.check()
                    if policy_model is None:
                        verb,fraction=rule_upper(env.raw)
                        action=pack_rule_action(verb,fraction)
                    else:action=policy_model.predict(obs,deterministic=True)[0]
                    obs,reward,terminated,truncated,info=env.step(action)
                    boundary(env,label+"_macro_"+str(env.actions))
                    if truncated:raise RuntimeError("native annual evaluation unexpectedly truncated")
                    if terminated:break  # no automatic vector reset at year end
                if not env.ended or len(env.episodes)!=1:raise RuntimeError("native annual evaluation missing one audit")
                after=environment_receipt(env)
                if after["remaining_hours"] or after["pending"]:raise RuntimeError("evaluation liability remains")
                if policy_model is not None and snapshot_learning_state(policy_model)!=frozen_before:
                    raise RuntimeError("frozen checkpoint RAM changed during evaluation")
                if rule_before!=dict(upper=rule_state_reader(original_upper),lower=rule_state_reader(original_lower)):
                    raise RuntimeError("frozen rule identity changed")
                if env.driver.store.manifest()!=source_world:raise RuntimeError("replay source world identity mismatch")
                tape=state_digest(plain(dict(origins=env.driver.store.context.origins,
                    validation_stamps=env.driver.store.context.validation_stamps)))
                if tape!=input_tape:raise RuntimeError("audited exogenous publication replay identity mismatch")
                metric=env.episodes[0]
                if metric["market_sha256"]!=training_metric["market_sha256"]:
                    raise RuntimeError("same-world comparisons have different exogenous hourly observations")
                dump_finish(env,label,metric)
                emit(label+"_frozen_state",dict(before=frozen_before,
                    after=snapshot_learning_state(policy_model) if policy_model is not None else None,
                    checkpoint_sha256=checkpoint_sha if policy_model is not None else None,
                    source_world=source_world,published_input_tape_sha256=tape))
                return metric
            with ctx.stage("frozen_C10_same_world"):
                c10=evaluate("C10",reloaded)
            with ctx.stage("rule_C00_same_world"):
                c00=evaluate("C00")
            with ctx.stage("derive_exploratory_metrics"):
                exact=dict(new_native_worlds=1,audited_world_replay_copies=2,envs=3,
                    native_resets=3,drivers=3,coordinators=3,published_worlds=6,
                    native_annual_deliveries=3,delivered_hours=26280,
                    hour_prepare_attempts=26280,hour_settlement_attempts=26280,
                    device_prepare_hours=26280,publication_and_DA_seal_events=2484,
                    PPO_controllers=2,policy_objects=2,rollout_collection_calls=1,
                    gradient_collection_steps=32,PPO_train_calls=1,optimizer_step_attempts=2,
                    backward_attempts=2,optimizer_steps_completed=2,backward_completed=2,new_checkpoint_saves=1,new_checkpoint_loads=1,
                    additional_diagnostic_API_calls=8,additional_diagnostic_batch_rows=70)
                for kind,amount in exact.items():
                    if ctx.counts.get(kind,0)!=amount:raise RuntimeError("actual operation count mismatch: "+kind)
                if source_world!=training.driver.store.manifest():raise RuntimeError("source scenario changed at final boundary")
                summary=dict(status="SCOPED_PILOT_COMPLETE",task_R05_complete=False,
                    independent_world_N=1,training_seeds=1,same_training_evaluation_world=True,
                    costs=dict(training_trajectory=training_metric["objective_cost"],
                               C10=c10["objective_cost"],C00=c00["objective_cost"]),
                    C10_minus_C00=c10["objective_cost"]-c00["objective_cost"],
                    statistics="descriptive exposed N1; no CI, significance, CVaR or independent advantage",
                    actual_config=actual_config,checkpoint_sha256=checkpoint_sha,
                    config_sha256=CONFIG_SHA256,source_identity=ctx.source_identity,
                    world=source_world,rule_identity=rule_identity,counts=dict(ctx.counts),
                    remaining_scope="full multi-seed/independent-world R05 and R06 are not authorized",
                    auto_retry=0,auto_repair=0)
                ctx.write_json("pilot_metrics.json",plain(summary))
                return summary
    except BaseException as primary:
        errors=[primary]
        for env in envs:
            try:boundary(env,labels.get(id(env),"UNKNOWN")+"_failed_final")
            except BaseException as cleanup:
                errors.append(cleanup)
                try:emit("failures",dict(stage="failed_boundary",type=type(cleanup).__name__,message=repr(cleanup)))
                except BaseException as journal_error:errors.append(journal_error)
        chosen=prioritize_error(errors)
        try:emit("failures",dict(status="STOPPED",type=type(chosen).__name__,message=repr(chosen),
                                 counts=dict(ctx.counts),automatic_retry=0,automatic_repair=0,
                                 secondary_errors=[dict(type=type(e).__name__,message=repr(e)) for e in errors if e is not chosen]))
        except BaseException as journal_error:errors.append(journal_error)
        chosen=prioritize_error(errors)
        if chosen is primary:raise
        raise chosen
