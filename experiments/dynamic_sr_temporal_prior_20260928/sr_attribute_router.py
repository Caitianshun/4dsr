"""Reusable policy facade over the already audited, unchanged SR derivative arithmetic."""
import importlib.util
from dv_common import ROOT

_path = ROOT / 'experiments/dynamic_sr_dynamic_validation_20260927/routing.py'
_spec = importlib.util.spec_from_file_location('validated_sr_routing_20260927', _path)
_validated = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validated)
render_model = _validated.render_model
POLICIES = {'joint': 'J_joint', 'appearance': 'A_sh', 'appearance_covariance': 'S_cov'}
ARM_TO_POLICY = {v: k for k, v in POLICIES.items()}

class SRAttributeRouter:
    def __init__(self, policy, model):
        if policy not in POLICIES: raise ValueError(policy)
        self.policy, self.model, self.arm = policy, model, POLICIES[policy]
        self.parameters = _validated.selected(model, self.arm)

    def backward_sr(self, loss, audit=False):
        return _validated.backward_prior(loss, self.model, self.arm, audit=audit)

    def describe(self):
        d = _validated.policy_info(self.model, self.arm)
        d.update(policy=self.policy, adds_learnable_parameters=0, inference_operations=0,
            parameter_ids={n: id(p) for n, p in _validated.all_named(self.model).items()},
            arithmetic_implementation=str(_path.relative_to(ROOT)))
        return d
