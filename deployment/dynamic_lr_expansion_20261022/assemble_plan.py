"""Join the six native-method preparation records into an inert command ledger.

This standard-library tool does not execute the commands it records. It freezes
the exact train/test workload, source identities, and remaining acceptance gates.
All paths refer to the local research checkout; host mapping is a later step.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from prepare_suite import ROOT, HERE, dump, sha


def read(path):
    return json.loads(Path(path).read_text())


def local(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def identity(path):
    p=local(path)
    return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,default=ROOT/'output/dynamic_lr_expansion_20261022/preparation')
    args=parser.parse_args();out=args.preparation.resolve()
    suite=read(out/'suite_plan.json')
    strict=read(HERE/'notes/strict4d_CPU_static_acceptance.json')
    strict_views={(r['method'],r['scene'],r['scope']):r for r in strict['records']}
    strict_plans={(r['method'],r['scene']):r for r in read(HERE/'strict4d_preparation_index.json')['entries']}
    local_audit=read(HERE/'notes/local_cem_audit.json')
    local_rows={(r['method'],r['scene']):r for r in local_audit['rows']}
    local_views={(r['method'],r['scene'],r['scope']):r for r in read(HERE/'notes/local_cem_cpu_acceptance.json')['plans']}
    labels={s['scene']:s['label'] for s in suite['scenes']}
    entries=[]
    for task in suite['tasks']:
        method,scene=task['method'],task['scene'];manifest=ROOT/next(s['manifest'] for s in suite['scenes'] if s['scene']==scene)
        key=(method,scene);entry=dict(task)
        entry['render_plans']={};entry['later_training_argv']=[]
        if method in ['wu','multi4d']:
            spec_path=HERE/f'specs/wu_multi/{method}_{scene}_bicubic.json';spec=read(spec_path)
            entry['method_preparation']=identity(spec_path)
            entry['remaining_acceptance']=spec.get('remaining_TODO',[])
            if task['training']!='reuse_completed':
                entry['later_training_argv']=[['<ADMITTED_METHOD_PYTHON>',str(ROOT/'deployment/dynamic_lr_baselines_20261020/methods/wu_multi/run_multi_native_lr.py'),
                    '--spec',str(spec_path),'--out',str(local(spec['planned_training_output']))]]
            for scope in ['train','test']:
                expected=HERE/'plans'/method/labels[scene]/scope/'expected_views.json'
                argv=[str(HERE/'render_wu_multi.py'),'--spec',str(spec_path),'--scope',scope,
                      '--out',str(ROOT/f'output/dynamic_lr_expansion_20261022/rendered/{method}/{scene}/{scope}')]
                entry['render_plans'][scope]={'expected_views':identity(expected),'later_render_argv':argv}
        elif method in ['yang','4c4d']:
            plan_path=local(strict_plans[key]['plan']);plan=read(plan_path)
            entry['method_preparation']=identity(plan_path)
            entry['remaining_acceptance']=plan['CPU_requirements_pending']+plan['GPU_requirements_HOLD']
            if task['training']!='reuse_completed':
                runtime=local(plan['native_runtime']['project_relative_path']);run=local(plan['expected_run_project_relative_path'])
                entry['later_training_cwd']=str(runtime)
                entry['later_training_argv']=[['<ADMITTED_METHOD_PYTHON>',str(ROOT/'deployment/dynamic_lr_baselines_20261020/methods/strict4d/train_exec.py'),
                    '--script',str(runtime/'train.py'),'--record',str(run/'train_resources.json'),'--','--config',plan['configuration']['path'],
                    *plan['configuration']['author_extra_CLI']]]
            for scope in ['train','test']:
                record=strict_views[method,scene,scope];argv=list(record['argv_plan_only'])
                argv[argv.index('--output')+1]=str(ROOT/f'output/dynamic_lr_expansion_20261022/rendered/{method}/{scene}/{scope}')
                entry['render_plans'][scope]={'expected_views':identity(record['expected_views']),'later_render_argv':argv}
        else:
            row=local_rows[key];later=row['later_execution_plan']
            entry['method_preparation']={'audit':identity(HERE/'notes/local_cem_audit.json'),'row_key':[method,scene]}
            entry['remaining_acceptance']=row['missing_acceptance']
            entry['later_training_argv']=later['inert_author_argv']
            entry['initialization_preparation']=row['initialization']
            for scope in ['train','test']:
                expected=local_views[method,scene,scope]['expected_views']['path']
                argv=[str(HERE/'render_local_cem.py'),'--method',method,'--manifest',str(manifest),
                    '--runtime',later['runtime'],'--model',later['model'],'--source',later['source'],
                    '--iteration',str(row['author_config']['endpoint']),'--scope',scope,'--output',
                    str(ROOT/f'output/dynamic_lr_expansion_20261022/rendered/{method}/{scene}/{scope}')]
                if method=='localdygs':argv+=['--config',later['config'][0]]
                if row['existing_endpoint']:argv+=['--checkpoint-index',str(local(row['existing_endpoint']['result_index']['path']))]
                if scene.startswith('meetroom_'):argv+=['--runtime-acceptance','<FUTURE_VERIFIED_RUNTIME_ACCEPTANCE_JSON>']
                entry['render_plans'][scope]={'expected_views':identity(expected),'later_render_argv':argv}
        for scope,record in entry['render_plans'].items():
            plan=read(local(record['expected_views']['path']))
            expected_keys={(r['camera_id'],int(r['frame'])) for r in plan['records']}
            common=read(ROOT/task['evaluation_scopes'][scope]['view_list'])
            if expected_keys!={(r['camera_id'],r['frame']) for r in common['records']} or len(plan['records'])!=len(expected_keys):
                raise ValueError(f'Native/common camera-frame contract differs: {method}/{scene}/{scope}')
            if plan['manifest_sha256']!=sha(manifest) or plan['resolution']!=common['resolution']:
                raise ValueError('Native/common manifest or HR dimensions differ')
            argv=record['later_render_argv']
            if '--execute' in argv:raise ValueError('Only inert, default plan commands may be recorded')
            record['execution_requires_explicit_execute_flag']=True
            render_dir=Path(argv[argv.index('--out')+1] if '--out' in argv else argv[argv.index('--output')+1])
            score_argv=[str(HERE/'score_views.py'),'--manifest',str(manifest),'--scope',scope,'--method',method,
                        '--render-dir',str(render_dir),'--out',str(ROOT/f'output/dynamic_lr_expansion_20261022/metrics/{method}/{scene}/{scope}')]
            record['later_score_argv']=score_argv
            record['render_and_score_required']=scope=='train' or task['test_metrics']!='reuse_completed'
            record['images']=len(expected_keys)
        entries.append(entry)
    counts={'method_scene_tasks':len(entries),'native_render_plans':sum(len(e['render_plans']) for e in entries),
        'missing_training':sum(e['training']!='reuse_completed' for e in entries),
        'missing_metric_scopes':sum(r['render_and_score_required'] for e in entries for r in e['render_plans'].values()),
        'missing_train_metric_images':sum(e['render_plans']['train']['images'] for e in entries),
        'missing_test_metric_images':sum(e['render_plans']['test']['images'] for e in entries if e['render_plans']['test']['render_and_score_required'])}
    if counts!={'method_scene_tasks':36,'native_render_plans':72,'missing_training':20,'missing_metric_scopes':60,
                'missing_train_metric_images':18000,'missing_test_metric_images':7200}:raise ValueError(counts)
    result={'schema':'dynamic_lr_held_completion_ledger_v1','status':'HELD_NOT_SUBMITTED','execution_enabled':False,
        'not_a_queue':True,'no_watcher_or_autorun':True,'counts':counts,'suite_identity':identity(out/'suite_plan.json'),
        'future_completion_contract':['fresh user resumption and per-GPU resource admission','verify complete transfer hashes, native data loader and camera contract',
            'prepare original method initialization','execute native training only when missing','on successful process exit immediately render and score missing scopes',
            'on failed exit write failure state without scoring','verify output and checkpoint hashes, return local provenance and cost, update one combined train/test table per dataset'],
        'commands_are_metadata_only':'No command is run by this tool. Placeholder Python, GPU, acceptance receipt and host paths require future validated binding.',
        'entries':entries}
    dump(out/'completion_ledger.json',result)
    print(json.dumps({'status':result['status'],**counts}))


if __name__=='__main__':main()
