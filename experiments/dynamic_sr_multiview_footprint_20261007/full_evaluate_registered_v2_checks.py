"""CPU contracts for the versioned full-native evaluator source adapter.

No image bytes are opened or decoded, no checkpoint is restored, and no torch
module is imported. The fixtures below are explicitly synthetic metadata.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_evaluate_registered as old
import full_evaluate_registered_v2 as ev
import full_refine_registered_v2 as native
from fp_common import ROOT, OUT, read, sha

OLD_EVALUATOR_SHA256 = '87fd5f4d29453be980704ce7030d8cf1d7d3d35205ea978d1a81e1b7d0d4ab05'
NATIVE_V2_SHA256 = 'c278928f56b0515adedf15eef9a30850e8bd2a4c41e6f6e0f828068ca774b75d'
SELECTION = OUT / 'summary_recovery_20261007/registered_full_development_selection.json'


def refused(call, message):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(message)


class ExpectedAdapter(ast.NodeTransformer):
    """The whole evaluator is identical after these two identity substitutions."""
    def __init__(self):
        self.native_imports = 0
        self.native_paths = 0

    def visit_Import(self, node):
        for item in node.names:
            if item.name == 'full_refine_registered' and item.asname == 'native':
                item.name = 'full_refine_registered_v2'
                self.native_imports += 1
        return node

    def visit_Constant(self, node):
        if node.value == 'full_refine_registered.py':
            node.value = 'full_refine_registered_v2.py'
            self.native_paths += 1
        return node


def ast_contract():
    assert sha(HERE / 'full_evaluate_registered.py') == OLD_EVALUATOR_SHA256
    assert sha(HERE / 'full_refine_registered_v2.py') == NATIVE_V2_SHA256
    adapter = ExpectedAdapter()
    expected = adapter.visit(ast.parse((HERE / 'full_evaluate_registered.py').read_text()))
    actual = ast.parse((HERE / 'full_evaluate_registered_v2.py').read_text())
    assert adapter.native_imports == 5 and adapter.native_paths == 1
    assert ast.dump(expected, include_attributes=False) == ast.dump(actual, include_attributes=False), 'Metric/render/registration AST changed beyond explicit source adaptation'
    return dict(native_import_substitutions=adapter.native_imports,
                native_source_path_substitutions=adapter.native_paths,
                entire_normalized_AST_equal=True,
                metric_render_math_AST_unchanged=True)


def source_contract(upstream):
    before, after = old.sources(), ev.sources()
    old_paths = {str((HERE / name).relative_to(ROOT)) for name in
                 ('full_evaluate_registered.py', 'full_refine_registered.py')}
    new_paths = {str((HERE / name).relative_to(ROOT)) for name in
                 ('full_evaluate_registered_v2.py', 'full_refine_registered_v2.py')}
    assert set(before) - set(after) == old_paths
    assert set(after) - set(before) == new_paths
    assert all(before[key] == after[key] for key in set(before) & set(after))
    ev.verify_sources(dict(sources=after))
    refused(lambda: ev.verify_sources(dict(sources=before)), 'Old evaluator protocol must require a new registration')
    changed = dict(after)
    changed[str(Path(ev.__file__).relative_to(ROOT))] = '0' * 64
    refused(lambda: ev.verify_sources(dict(sources=changed)), 'Changed evaluator source must be rejected')
    # Exercise the actual native source verifier through the evaluator, without
    # importing CUDA or reading any scientific checkpoint/image bytes.
    plan = dict(source_files=native.source_files(upstream), dependencies={}, parent=None)
    runtime = ev.validate_author_runtime(plan, upstream)
    assert runtime == plan['source_files']['upstream']
    wrong_native = copy.deepcopy(plan)
    project = wrong_native['source_files']['project']
    project.pop(str((HERE / 'full_refine_registered_v2.py').relative_to(ROOT)))
    project[str((HERE / 'full_refine_registered.py').relative_to(ROOT))] = sha(HERE / 'full_refine_registered.py')
    refused(lambda: ev.validate_author_runtime(wrong_native, upstream), 'V1-native source identity must be rejected for a V2 plan')
    wrong_runtime = copy.deepcopy(plan)
    wrong_runtime['source_files']['upstream']['scene/gaussian_model.py'] = '0' * 64
    refused(lambda: ev.validate_author_runtime(wrong_runtime, upstream), 'Changed author runtime must be rejected')
    return dict(v2_evaluator_and_native_explicitly_bound=True,
                shared_metric_sources_equal=True,
                legacy_protocol_sources_rejected=True,
                mismatched_native_and_author_sources_rejected=True)


def protocol_contract():
    manifest_entry = dict(path='explicit_synthetic_fixture/manifest.json', sha256='1' * 64)
    data = dict(manifest=manifest_entry)
    manifest = dict(initialization=dict(seed=20261007), scene='synthetic_metadata_only',
                    role='already_used_development', resolutions=dict(hr=[8, 8], lr=[2, 2]),
                    cameras=dict(cam00=dict(K_lr=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
                    observations=[dict(camera_id='cam00', frame_index=f, split='test',
                                       hr_sha256='2' * 64, lr_sha256='3' * 64) for f in range(300)] +
                                 [dict(camera_id='cam01', frame_index=0, split='train')])
    definitions = dict(scope='Explicit synthetic metadata only; not scientific evidence')
    protocol = dict(schema=ev.SCHEMA, status='registered_full_native_evaluation_before_prediction_reads',
                    manifest=manifest_entry, seed=20261007, scene=manifest['scene'], role=manifest['role'],
                    data=data, resolutions=manifest['resolutions'],
                    test_input_SHA256=ev.prefix.digest([('cam00', f, '2' * 64, '3' * 64) for f in range(300)]),
                    measurement_definitions=definitions, measurement_definition_SHA256=ev.prefix.digest(definitions),
                    test_keys=[['cam00', f] for f in range(300)],
                    camera_interface_SHA256=ev.prefix.digest(manifest['cameras']), selection=None,
                    train_teacher_diagnostics=dict(keys=[], teacher_index=None))
    assert ev.validate_protocol(protocol, manifest, data)
    cases = []
    for field, value in (('schema', 'wrong_schema'), ('manifest', dict(path='other', sha256='4' * 64)),
                         ('seed', 20261008), ('test_keys', [['cam00', 0]]),
                         ('measurement_definition_SHA256', '0' * 64), ('camera_interface_SHA256', '0' * 64)):
        changed = copy.deepcopy(protocol)
        changed[field] = value
        refused(lambda: ev.validate_protocol(changed, manifest, data), 'Changed protocol field accepted: ' + field)
        cases.append(field)
    changed = copy.deepcopy(protocol)
    changed['train_teacher_diagnostics'] = dict(keys=[['cam00', 0]], teacher_index=manifest_entry)
    refused(lambda: ev.validate_protocol(changed, manifest, data), 'Held-out teacher key must be rejected')
    changed = copy.deepcopy(protocol)
    changed['train_teacher_diagnostics'] = dict(keys=[['cam01', 0], ['cam01', 0]], teacher_index=manifest_entry)
    refused(lambda: ev.validate_protocol(changed, manifest, data), 'Duplicate train diagnostic key must be rejected')
    wrong_plan = dict(schema=ev.SR_SCHEMA, scene=manifest['scene'], seed=20261008, method='Bsync')
    refused(lambda: ev.validate_protocol(protocol, manifest, data, wrong_plan), 'Checkpoint/protocol seed mismatch must be rejected')
    held_manifest = copy.deepcopy(manifest)
    held_manifest['role'] = 'held_confirmation'
    held_protocol = copy.deepcopy(protocol)
    held_protocol['role'] = 'held_confirmation'
    refused(lambda: ev.validate_protocol(held_protocol, held_manifest, data), 'Unfrozen held confirmation must be rejected')
    return dict(synthetic_metadata_valid=True, rejected_protocol_fields=cases,
                heldout_or_duplicate_diagnostics_rejected=True,
                checkpoint_seed_and_unfrozen_confirmation_rejected=True)


def selection_contract():
    # These are previously frozen JSON identities, not prediction pixels.
    value = read(SELECTION)
    assert ev.validate_selection_gate(value, 'B0')
    assert ev.validate_selection_gate(value, 'Bsync')
    refused(lambda: ev.validate_selection_gate(value, 'M'), 'Unselected candidate must be rejected')
    overlap = dict(value, selected_candidates=['B0'])
    refused(lambda: ev.validate_selection_gate(overlap, 'B0'), 'Baseline/candidate overlap must be rejected')
    too_many = dict(value, selected_candidates=['Bsync', 'M', 'X'])
    refused(lambda: ev.validate_selection_gate(too_many, 'Bsync'), 'Unnecessary full grid must be rejected')
    return dict(B0_and_Bsync_legal=True, unselected_or_overlap_or_expanded_grid_rejected=True,
                selection=dict(path=str(SELECTION.relative_to(ROOT)), sha256=sha(SELECTION)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path(os.environ.get('FOURDSR_UPSTREAM', '/home/cai_tianshun/Project/4dgs')))
    args = parser.parse_args()
    results = dict(AST=ast_contract(), sources=source_contract(args.upstream),
                   protocol=protocol_contract(), selection=selection_contract())
    assert 'torch' not in sys.modules, 'CPU metadata contracts unexpectedly imported torch'
    print(json.dumps(dict(status='passed_full_evaluate_registered_v2_CPU_contract', contracts=results,
                          sources={name:sha(HERE / name) for name in
                                   ('full_evaluate_registered.py', 'full_evaluate_registered_v2.py',
                                    'full_evaluate_registered_v2_checks.py', 'full_refine_registered_v2.py')},
                          CUDA_initialized=False, GPU_forwards=0, formal_updates=0,
                          HR_image_bytes_opened=0, checkpoint_payloads_opened=0,
                          global_state_modified=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
