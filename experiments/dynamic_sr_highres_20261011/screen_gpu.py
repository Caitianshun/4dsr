"""Own one GPU for a short calibrated Z5 screen, with exit-driven receipts."""
import argparse,csv,datetime,json,os,subprocess,sys,time,traceback
from common import ROOT,OUT,entry,mkdir,read,sha,write,local

def query():
    def rows(fields,kind):
        text=subprocess.check_output(['nvidia-smi',f'--query-{kind}='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
        return [dict(zip(fields,[s.strip() for s in row])) for row in csv.reader(text.splitlines())]
    return dict(gpus=rows(['index','uuid','name','memory.used','memory.total','utilization.gpu'],'gpu'),
                processes=rows(['gpu_uuid','pid','process_name','used_gpu_memory'],'compute-apps'),
                captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--uuid',required=True)
    p.add_argument('--samples',type=local,required=True);p.add_argument('--job-out',type=local,required=True)
    p.add_argument('--allow-desktop',action='store_true');a=p.parse_args()
    if not a.job_out.is_relative_to(OUT/'zero_checks') or a.job_out.exists():raise ValueError('Fresh zero_checks output required')
    samples=read(a.samples)
    if samples['test_views_used'] or samples['teacher_cache'] or samples['rung']!='half':raise ValueError('Expected preregistered half train diagnostics')
    state=query();g=[x for x in state['gpus'] if int(x['index'])==a.gpu]
    if len(g)!=1 or g[0]['uuid']!=a.uuid:raise ValueError('GPU identity changed')
    processes=[x for x in state['processes'] if x['gpu_uuid']==a.uuid]
    unknown=[x for x in processes if not(a.allow_desktop and x['process_name']=='/usr/libexec/gnome-remote-desktop-daemon' and float(x['used_gpu_memory'])<=600)]
    if unknown or float(g[0]['memory.used'])>1024 or float(g[0]['utilization.gpu'])>10:raise RuntimeError('GPU no longer idle; existing jobs protected')
    mkdir(a.job_out);claims=mkdir(OUT/'claims');lock=claims/(a.uuid+'.lock');held=False
    record=dict(kind='calibrated_Z5_screen',gpu=a.gpu,gpu_uuid=a.uuid,resources_before=state,
                samples=entry(a.samples),source=entry(__file__),inference_source=entry(ROOT/'experiments/dynamic_sr_highres_20261011/zero_checks.py'),
                host=os.uname().nodename,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=os.getpid())
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o664)
        with os.fdopen(fd,'w') as f:json.dump(record,f)
        held=True;write(a.job_out/'allocation.json',record)
        result=a.job_out/'z5_half.json';argv=[sys.executable,'-B',str(ROOT/'experiments/dynamic_sr_highres_20261011/zero_checks.py'),'infer','--samples',str(a.samples),'--device','cuda:0','--out',str(result)]
        begin=time.monotonic()
        with (a.job_out/'inference.log').open('x') as log:
            child=subprocess.run(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:raise RuntimeError(f'Inference exit {child.returncode}; preserved inference.log')
        report=read(result)
        if report['samples_identity']['sha256']!=sha(a.samples) or report['source']!=record['inference_source']:raise ValueError('Result identities changed')
        record.update(status='completed_passed_Z5' if report['Z5_pass'] else 'completed_rejected_Z5',seconds=time.monotonic()-begin,result=entry(result),aggregate=report['aggregate'],resources_after=query())
        write(a.job_out/'complete.json',record)
        print(json.dumps({k:record[k] for k in ('status','gpu_uuid','seconds','aggregate','result')},ensure_ascii=False))
    except BaseException:
        write(a.job_out/f'failure_{time.time_ns()}.json',dict(record,status='failed_preserved',error=traceback.format_exc()))
        raise
    finally:
        if held:lock.unlink()

if __name__=='__main__':main()
