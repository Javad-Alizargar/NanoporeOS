"""User-invoked bounded campaign resumption. Never self-relaunches.

Each invocation is a new explicitly requested run window; spending and scientific
records persist. Checkpoints are complete cases, not in-flight Newton iterates.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import check_stop,StopRequested,write_json,source_identity,sha256
from nanoporeos.datasets import load_result


def current(path,**expected):
    try:data=json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return False
    return all(data.get(k)==v for k,v in expected.items())


def main():
    p=argparse.ArgumentParser();p.add_argument('--minutes',type=float,default=120);p.add_argument('--check',action='store_true')
    args=p.parse_args();minutes=max(0,min(args.minutes,120))
    (ROOT/'orchestration').mkdir(exist_ok=True)
    owner=(ROOT/'orchestration/run.lock').open('a+b')
    owner.seek(0);owner.write(b'0');owner.flush();owner.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(owner.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        owner.close();print('Another project scheduler owns this workspace');return 3
    started=time.monotonic();deadline=started+minutes*60
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    state=dict(status='CHECKING' if args.check else 'RUNNING',run_id=runid,minutes=minutes,
        started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_identity(),
        checkpoint_granularity='completed case; interrupted case restarts from initial state',tasks=[])
    (ROOT/'logs').mkdir(exist_ok=True)
    try:
        check_stop()
        if args.check:
            count=0
            for pth in (ROOT/'results').glob('*/[a-f0-9]*'):
                if pth.is_dir() and (pth/'manifest.json').exists():load_result(pth);count+=1
            if (ROOT/'tools/verify_handoff.py').exists() and (ROOT/'HANDOFF_MANIFEST.json').exists():
                subprocess.run([sys.executable,'tools/verify_handoff.py'],cwd=ROOT,check=True)
                handoff='LOCAL_VERIFIER_EXECUTED'
            else:handoff='EXTERNAL_PRIVATE_HANDOFF_NOT_INCLUDED_NOT_VERIFIED'
            state.update(status='CHECK_PASS',verified_case_exports=count,handoff_integrity=handoff)
        else:
            tasks=[
                ('matrix',['scripts/campaign.py','--mode','matrix','--legacy','--minutes','20'],
                    current('results/matrix/campaign.json',status='FINISHED',source=source_identity())),
                ('stress',['scripts/campaign.py','--mode','stress','--minutes','5'],
                    current('results/stress/campaign.json',status='FINISHED',source=source_identity())),
                ('references',['scripts/verify_references.py'],current('results/references/independent_pb.json',
                    solver_source_sha256=source_identity(),reference_driver_sha256=sha256(ROOT/'scripts/verify_references.py'))),
                ('stability',['scripts/gummel_stability.py'],current('results/gummel_stability.json',source_sha256=source_identity())),
                ('independent_nonequilibrium_audit',['scripts/noneq_reference.py','--check'],False),
                ('dataset',['scripts/dataset_factory.py'],False),
                ('performance',['scripts/performance.py'],current('results/performance.json',status='FINISHED',source_sha256=source_identity())),
                ('performance_analysis',['scripts/analyze_performance.py'],False),
                ('summary',['scripts/summarize.py'],False),
                ('figures',['scripts/make_figures.py'],False),
                ('tables',['scripts/make_tables.py'],False)]
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
            for name,command,skip in tasks:
                check_stop(deadline)
                if skip:
                    state['tasks'].append(dict(task=name,status='SOURCE_MATCHED_CHECKPOINT'));continue
                log=ROOT/'logs'/f'{runid}-{name}.txt'
                with log.open('w',encoding='utf-8') as stream:
                    proc=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    state['active_pid']=proc.pid;state['active_task']=name
                    write_json(ROOT/'orchestration/task_state.json',state)
                    try:
                        while proc.poll() is None:
                            check_stop(deadline);time.sleep(.2)
                    except StopRequested:
                        # This process handle was created here; never kill unrelated Python.
                        proc.terminate();proc.wait(timeout=10);raise
                state['tasks'].append(dict(task=name,status='PASS' if proc.returncode==0 else 'FAIL',
                    command=[sys.executable,*command],exit_code=proc.returncode,log=str(log.relative_to(ROOT))))
                if proc.returncode:raise RuntimeError(f'{name} failed; inspect {log.name}')
            state['status']='COMPUTATIONAL_BATCH_FINISHED_SCIENTIFIC_LIMITATIONS_REMAIN'
    except (StopRequested,RuntimeError,subprocess.CalledProcessError) as exc:
        state.update(status='CHECKPOINTED',reason=str(exc))
    state.pop('active_pid',None);state.pop('active_task',None)
    state.update(elapsed_seconds=time.monotonic()-started,
        next_command=r'.\.venv\Scripts\python.exe scripts\resume.py --minutes 120')
    write_json(ROOT/'orchestration/task_state.json',state)
    write_json(ROOT/'logs'/f'{runid}-state.json',state)
    print(json.dumps(state,indent=2))
    owner.close()
    return 0 if state['status'] in ['CHECK_PASS','COMPUTATIONAL_BATCH_FINISHED_SCIENTIFIC_LIMITATIONS_REMAIN'] else 2


if __name__=='__main__':raise SystemExit(main())
