"""Synthetic CPU readiness contracts; never production deployment evidence.

All data, metadata, metric assets and extension files live in a temporary
directory. Extension-shaped files contain JSON fixture bytes and are never
loaded. Distribution versions are mocked only while validating these fixtures.
No tensors, images, scientific checkpoints, GPUs, SSH or services are opened.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('readiness_contract_operator', HERE / 'full_operator_registered_v2.py')
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)


class FixtureEvidence:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, name):
        return self.root / name

    def entry(self, name):
        return dict(path=name, sha256=op.sha(self.path(name)))

    def bound(self, reference):
        path = self.path(reference['path'])
        assert op.sha(path) == reference['sha256']
        return path

    def read(self, reference):
        return json.loads(self.bound(reference).read_text())

    def asset(self, name, value):
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return self.entry(name)


def refused(cases, label, call):
    try:
        call()
    except ValueError:
        cases.append(label)
        return
    raise AssertionError('Invalid synthetic fixture accepted: ' + label)


def inventory_platform_contracts(e, cases):
    asset = e.asset
    source = asset('source.json', dict(CPU_fixture=True))
    lr = asset('full_LR.json', dict(CPU_fixture='LR metadata'))
    teacher_file = asset('full_teacher.json', dict(CPU_fixture='teacher metadata'))
    artifact = asset('artifact.json', dict(CPU_fixture='initialization artifact'))
    initialization = asset('init.json', dict(artifacts=[artifact]))
    readiness = asset('readiness.json', dict(original_Wu=dict(sources=[source])))
    teacher = asset('teacher.json', dict(producer_config=source, producer_summary=artifact, plan=initialization))
    parent = asset('parent.json', dict(author_sources=dict(native=source)))
    protocol = asset('protocol.json', dict(sources={source['path']: source['sha256']},
                                         ROI=dict(original_protocol=artifact), test_keys=[],
                                         train_teacher_diagnostics=dict(keys=[])))
    manifest = asset('manifest.json', dict(observations=[]))
    external = asset('upstream/native.json', dict(CPU_fixture='runtime metadata'))
    absolute_external = dict(path=str(e.path(external['path'])), sha256=external['sha256'])
    upstream = str(e.root / 'upstream')
    extensions = []
    for package in ('diff_gaussian_rasterization', 'simple_knn'):
        reference = asset(package + '/_C.fixture.so', dict(CPU_fixture='NOT an actual binary'))
        extensions.append(dict(path=str(e.path(reference['path'])), sha256=reference['sha256']))
    evaluation = dict(host='CPU_fixture', GPU='GPU_fixture', GPU_model='RTX_fixture',
                      runtime_python=sys.executable, upstream=upstream, transport='local',
                      workspace='/home/cai_tianshun/Project/4dsr', wait_for_shared_physical_lock=True,
                      cpu_threads=1, idle_max_used_MiB=1024, idle_max_utilization_percent=10,
                      native_extensions=extensions)
    platform = asset('platform.json', dict(status='registered_same_platform_native_full_uniform_evaluation',
                     host=evaluation['host'], GPU=evaluation['GPU'], GPU_model=evaluation['GPU_model'],
                     runtime_python=evaluation['runtime_python'], upstream=upstream,
                     torch='2.7.1+cu128', cuda='12.8', runtime_asset_refs=[source],
                     metric_runtime=dict(LPIPS=dict(learned_weights=absolute_external,
                                         AlexNet_pretrained_weights=absolute_external,
                                         source_files={absolute_external['path']: absolute_external['sha256']}))))
    plan = dict(source_files=dict(project={source['path']: source['sha256']},
                                  upstream={'native.json': external['sha256']}),
                training_files=dict(LR=[lr], teacher=[teacher_file]),
                data=dict(initialization_receipt=initialization, initial_points=artifact,
                          domain_protocol=dict(readiness=readiness)))
    plan_reference = asset('plan.json', plan)
    s = dict(host='CPU_fixture', workspace=str(e.root), sources=dict(source=source),
             production_core_source=source, CPU_temporal=dict(source=source),
             assets=dict(plan=plan_reference, teacher=teacher, parent_plan=parent,
                         evaluation_protocol=protocol, manifest=manifest,
                         uniform_evaluation_platform=platform),
             training=dict(runtime_python=sys.executable, upstream=upstream, native_extensions=extensions),
             evaluation=evaluation)
    original_version = op.importlib.metadata.version
    op.importlib.metadata.version = lambda key: {'torch': '2.7.1+cu128', 'numpy': '1.26.4'}[key]
    try:
        for role in ('manager', 'callback'):
            files = [dict(reference, bytes=e.bound(reference).stat().st_size)
                     for reference in op.required_role_files(s, e, plan, role)]
            assert {v['path'] for v in (lr, teacher_file, artifact, initialization, readiness)} <= {v['path'] for v in files}
            deployment = dict(schema='actual_native_full_v2_role_CPU_all_SHA', role=role,
                         host=s['host'], workspace=s['workspace'] if role == 'manager' else evaluation['workspace'],
                         full_inventory_verified=True, GPU_calls=0, model_imports=0, image_decodes=0,
                         formal_updates=0, verified_files=files, files=len(files),
                         bytes=sum(v['bytes'] for v in files), python=sys.executable,
                         python_binary_sha256=op.sha(Path(sys.executable).resolve()),
                         versions=dict(torch='2.7.1+cu128', numpy='1.26.4'), min_free_bytes=1,
                         observed_free_bytes=1, external_runtime_files=[absolute_external, *extensions])
            good = copy.deepcopy(s)
            good['role_deployments'] = {role: deployment}
            op.validate_role_deployment(good, e, plan, role)
            cases.append(role + ' complete transitive synthetic inventory accepted')
            for reference in (lr, teacher_file, artifact, initialization, readiness):
                wrong = copy.deepcopy(good)
                d = wrong['role_deployments'][role]
                d['verified_files'] = [v for v in d['verified_files'] if v['path'] != reference['path']]
                d['files'] = len(d['verified_files'])
                d['bytes'] = sum(v['bytes'] for v in d['verified_files'])
                refused(cases, role + ' omission ' + reference['path'],
                        lambda wrong=wrong: op.validate_role_deployment(wrong, e, plan, role))
            wrong = copy.deepcopy(good)
            d = wrong['role_deployments'][role]
            d['verified_files'][0]['bytes'] = 0
            d['bytes'] = sum(v['bytes'] for v in d['verified_files'])
            refused(cases, role + ' changed actual file bytes rejected',
                    lambda: op.validate_role_deployment(wrong, e, plan, role))
    finally:
        op.importlib.metadata.version = original_version
    op.validate_platform(s, e)
    cases.append('same registered synthetic platform accepted')
    for key, value in (('GPU', 'other'), ('runtime_python', '/other/python'), ('workspace', '/other'),
                       ('idle_max_used_MiB', 1025), ('idle_max_utilization_percent', 11)):
        wrong = copy.deepcopy(s)
        wrong['evaluation'][key] = value
        refused(cases, 'platform changed ' + key,
                lambda wrong=wrong: op.validate_platform(wrong, e))


def release_contracts(e, cases):
    source = e.asset('release_source.json', dict(synthetic=True))
    spec_reference = e.asset('release_spec.json', dict(synthetic='current'))
    other = e.asset('other_spec.json', dict(synthetic='old'))
    s = dict(scene='meetroom_discussion', seed=20261007, method='B0', host='fixture-remote',
             workspace='/fixture/remote', manager_unit='manager.service', callback_unit='callback.service',
             training=dict(runtime_python='/remote/python'),
             evaluation=dict(host='fixture-local', workspace='/fixture/local', runtime_python='/local/python'),
             assets=dict(runtime_adapter_source=source),
             release_path=dict(manager='/remote/release.json', callback='/local/release.json'))

    def registration(role):
        workspace = s['workspace'] if role == 'manager' else s['evaluation']['workspace']
        host = s['host'] if role == 'manager' else s['evaluation']['host']
        unit = op.services(s, role)
        return dict(unit=unit, source=source, spec=spec_reference,
                    command=op.role_command(s, e, role, spec_reference), host=host, workspace=workspace,
                    invocation_id='a' * 32, pid=100, start_ticks='100', start_monotonic='100',
                    owned_cgroup='/user.slice/' + unit)

    manager, callback = registration('manager'), registration('callback')
    manager_reference = e.asset('release_manager.json', manager)

    def release(counterpart, name):
        callback_reference = e.asset('release_callback_' + name + '.json', counterpart)
        reference = e.asset('release_' + name + '.json',
                    dict(schema='root_released_source_aware_native_full_v2', spec=spec_reference,
                         task_key=op.task_key(s), root_only_registration=True, no_scientific_asset_changes=True,
                         registrations=dict(manager=manager_reference, callback=callback_reference)))
        path = e.bound(reference)
        path.chmod(0o444)
        return path

    legacy = SimpleNamespace(wait_immutable_binding=lambda path: path)
    op.resolve_release(s, e, spec_reference, manager, 'manager', release(callback, 'valid'), legacy)
    cases.append('complete exact two-role synthetic release accepted')
    for field, value in (('spec', other), ('source', other), ('command', ['different']), ('host', 'other'),
                         ('workspace', '/other'), ('invocation_id', 'invalid'), ('pid', 0), ('start_ticks', '0'),
                         ('start_monotonic', '0'), ('owned_cgroup', '/other.service')):
        wrong = copy.deepcopy(callback)
        wrong[field] = value
        refused(cases, 'changed counterpart ' + field + ' rejected',
                lambda wrong=wrong, field=field: op.resolve_release(
                    s, e, spec_reference, manager, 'manager', release(wrong, field), legacy))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only visibility required'
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'scene', 'gaussian_renderer'))
    cases = []
    with tempfile.TemporaryDirectory(prefix='full_operator_v2_readiness_CPU_fixture_') as directory:
        evidence = FixtureEvidence(directory)
        inventory_platform_contracts(evidence, cases)
        release_contracts(evidence, cases)
    assert len(cases) == 31
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'scene', 'gaussian_renderer'))
    print(json.dumps(dict(status='passed_synthetic_CPU_operator_v2_readiness_contracts', fixture_only=True,
                          cases=cases, checks=len(cases), operator_source_sha256=op.sha(HERE / 'full_operator_registered_v2.py'),
                          checks_source_sha256=op.sha(Path(__file__)), GPU_calls=0, model_imports=0,
                          real_image_or_checkpoint_reads=0, actual_native_extension_loads=0,
                          actual_role_deployment_proved=False, actual_release_proved=False,
                          source_mutations=0, global_state_modified=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
