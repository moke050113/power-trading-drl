"""Only new R05 fake pre-construction/ownership contract checks; no third-party imports."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
import unittest
import struct
from research.r05_runner import (APPROVAL_ANSWER,CONFIG_SHA256,SPEC,Hooks,authorize,
    installed_hooks,pack_rule_action,plain,require_spec,same_json,validate_checkpoint_payload,
    validate_checkpoint_state,validate_launch)

CONTEXT=None
CONFIG=None


class PilotContractTests(unittest.TestCase):
    def setUp(self):
        if CONTEXT is None or CONFIG is None:
            raise RuntimeError("approved preflight CONTEXT/config injection required")
        self.ctx=CONTEXT;self.config=deepcopy(CONFIG)
        authorize(self.ctx,self.config,"preflight")
    def call(self,fn,*args,**kwargs):
        self.ctx.charge("preflight_fake_calls","preflight_fake_calls_max")
        return fn(*args,**kwargs)

    def test_exact_owner_config_and_phase_precede_any_constructor(self):
        constructed=[]
        def launch(ctx,cfg):
            self.call(authorize,ctx,cfg,"pilot")
            constructed.append("forbidden factory")
        variants=[{},dict(approved=False),dict(owner=None),dict(config_sha256="0"*64),
                  dict(phase="preflight"),dict(source_identity={})]
        for change in variants:
            guard=deepcopy(self.ctx.guard)
            guard["phase"]="pilot"
            guard.update(change)
            if not change:guard={}
            fake=SimpleNamespace(guard=guard,source_identity=self.ctx.source_identity,check=lambda:None)
            with self.subTest(change=change),self.assertRaises(PermissionError):launch(fake,self.config)
        self.assertEqual(constructed,[])
        source=deepcopy(self.ctx.guard)
        source.update(phase="pilot",owner=object())
        valid=SimpleNamespace(guard=source,source_identity=self.ctx.source_identity,check=lambda:None)
        self.assertEqual(self.call(authorize,valid,self.config,"pilot"),self.config)

    def test_fixed_development_spec_and_typed_budgets_reject_before_factory(self):
        self.assertEqual(self.call(require_spec,dict(SPEC),self.config),SPEC)
        for change in (dict(split="validation"),dict(split="test"),dict(family="sealed_40"),
                       dict(fixture=True),dict(fixture=0),dict(year=2036),dict(group="G01")):
            request={**SPEC,**change}
            with self.subTest(change=change),self.assertRaises(PermissionError):
                self.call(require_spec,request,self.config)
        bad=deepcopy(self.config);bad["parameters"]["n_epochs"]=3
        with self.assertRaises(PermissionError):
            self.call(validate_launch,bad,self.config["proposed_budgets"],APPROVAL_ANSWER)
        changed=deepcopy(self.config["proposed_budgets"]);changed["new_native_worlds_max"]=True
        with self.assertRaises(PermissionError):
            self.call(validate_launch,self.config,changed,APPROVAL_ANSWER)
        with self.assertRaises(PermissionError):
            self.call(validate_launch,self.config,self.config["proposed_budgets"],"continue")

    def test_failed_operation_consumes_attempt_and_never_retries(self):
        called=[];attempts=[]
        def operation():called.append(1);raise ValueError("synthetic original failure")
        obj=SimpleNamespace(operation=operation)
        hooks=Hooks()
        self.call(hooks.wrap,obj,"operation",lambda:attempts.append("before"))
        with self.assertRaisesRegex(ValueError,"synthetic original failure"):
            self.call(obj.operation)
        self.assertEqual((len(attempts),len(called)),(1,1))
        self.call(hooks.restore)
        self.assertIs(obj.operation,operation)
        hooks=Hooks();calls=[]
        self.call(hooks.wrap,obj,"operation",lambda:(_ for _ in ()).throw(PermissionError("before factory")))
        with self.assertRaises(PermissionError):self.call(obj.operation)
        self.assertEqual(len(called),1)
        self.call(hooks.restore)

    def test_control_interruption_retains_identity_and_restores_exact_attribute(self):
        interrupt=KeyboardInterrupt("synthetic preflight control")
        obj=SimpleNamespace(value="original")
        fake=SimpleNamespace(emit=lambda *args:None)
        with self.assertRaises(KeyboardInterrupt) as raised:
            with installed_hooks(fake) as hooks:
                self.call(hooks.replace,obj,"value","temporary")
                self.call(lambda:(_ for _ in ()).throw(interrupt))
        self.assertIs(raised.exception,interrupt)
        self.assertEqual(obj.value,"original")
        class RestoreFault:
            def __init__(self):object.__setattr__(self,"value","original")
            def __setattr__(self,key,value):
                if value=="original":raise ValueError("synthetic restoration failure")
                object.__setattr__(self,key,value)
        fault=RestoreFault();events=[]
        fake=SimpleNamespace(emit=lambda *args:events.append(args))
        with self.assertRaises(KeyboardInterrupt) as second:
            with installed_hooks(fake) as hooks:
                self.call(hooks.replace,fault,"value","temporary")
                self.call(lambda:(_ for _ in ()).throw(interrupt))
        self.assertIs(second.exception,interrupt)
        self.assertEqual(len(events),1)
        # An ordinary main error must not swallow a cleanup control interruption.
        cleanup_control=KeyboardInterrupt("synthetic cleanup control")
        attempts=[]
        class ErrorOnRestore:
            def __init__(self,label,error):
                object.__setattr__(self,"value","original")
                object.__setattr__(self,"label",label);object.__setattr__(self,"error",error)
            def __setattr__(self,key,value):
                if key=="value" and value=="original":
                    attempts.append(self.label);raise self.error
                object.__setattr__(self,key,value)
        controlled=ErrorOnRestore("ordinary-main-cleanup-control",cleanup_control)
        with self.assertRaises(KeyboardInterrupt) as third:
            with installed_hooks(fake) as hooks:
                self.call(hooks.replace,controlled,"value","temporary")
                self.call(lambda:(_ for _ in ()).throw(ValueError("synthetic ordinary main")))
        self.assertIs(third.exception,cleanup_control)
        self.assertEqual(attempts,["ordinary-main-cleanup-control"])
        # Try every restoration: ordinary first, then first control, then another control.
        attempts.clear();first_control=KeyboardInterrupt("first restoration control")
        second_control=SystemExit("later restoration control")
        hooks=Hooks()
        self.call(hooks.replace,ErrorOnRestore("last",second_control),"value","temporary")
        self.call(hooks.replace,ErrorOnRestore("first-control",first_control),"value","temporary")
        self.call(hooks.replace,ErrorOnRestore("ordinary",ValueError("first ordinary restoration")),"value","temporary")
        with self.assertRaises(KeyboardInterrupt) as fourth:self.call(hooks.restore)
        self.assertIs(fourth.exception,first_control)
        self.assertEqual(attempts,["ordinary","first-control","last"])
        self.assertEqual(hooks.saved,[])
        self.assertEqual(len(first_control.r05_restore_errors),3)

    def test_owned_checkpoint_bytes_and_saved_field_scope(self):
        payload=b"artificial archive identity, not a real model"
        digest=sha256(payload).hexdigest()
        self.assertEqual(self.call(validate_checkpoint_payload,payload,digest,1024),payload)
        for bad,want,cap in ((payload,"0"*64,1024),(payload,digest,1),("path.zip",digest,1024)):
            with self.subTest(bad=bad),self.assertRaises(PermissionError):
                self.call(validate_checkpoint_payload,bad,want,cap)
        saved=dict(parameters="p",optimizer="o",num_timesteps=32,n_updates=2,
                   gradients="trained_gradient",rollout_buffer="collected32")
        restored={**saved,"gradients":"not_saved","rollout_buffer":"new_empty"}
        fields=self.call(validate_checkpoint_state,saved,restored,{"space":"s"},{"space":"s"})
        self.assertEqual(fields,("parameters","optimizer","num_timesteps","n_updates"))
        for key,value in (("parameters","wrong"),("optimizer","wrong"),("num_timesteps",0),("n_updates",0)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                self.call(validate_checkpoint_state,saved,{**restored,key:value},{"space":"s"},{"space":"s"})

    def test_plain_evidence_rejects_nonfinite_and_typed_identity_is_not_coerced(self):
        native=[.3]*12
        packed=self.call(pack_rule_action,"COMMIT",native)
        self.assertEqual(packed,[2.]+native)
        narrowed=struct.unpack("f",struct.pack("f",.3))[0]
        self.assertNotEqual(packed[1],narrowed)
        self.assertEqual(self.call(plain,{"cost":1.5,"rows":[1,2],"scope":"fake"}),
                         {"cost":1.5,"rows":[1,2],"scope":"fake"})
        for value in (float("nan"),float("inf"),float("-inf")):
            with self.subTest(value=value),self.assertRaises(ValueError):self.call(plain,value)
        self.assertFalse(self.call(same_json,{"count":1},{"count":True}))
        self.assertFalse(self.call(same_json,{"fixture":False},{"fixture":0}))
