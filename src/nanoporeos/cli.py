from pathlib import Path
import argparse
import json
import time
from .core import PNP,Scenario,StopRequested
from .datasets import save_result,load_result


def main():
    p=argparse.ArgumentParser(description='NanoporeOS: steady PNP, no experimental validation')
    sub=p.add_subparsers(dest='command',required=True)
    solve=sub.add_parser('solve');solve.add_argument('--config',type=Path);solve.add_argument('--output',type=Path,required=True)
    solve.add_argument('--seconds',type=float,default=60)
    verify=sub.add_parser('verify-export');verify.add_argument('directory',type=Path)
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8765)
    serve.add_argument('--workspace',type=Path,default=Path('runs/local_api'))
    args=p.parse_args()
    if args.command=='solve':
        s=Scenario(**(json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}))
        model=PNP(s)
        try:result=model.solve(deadline=time.monotonic()+max(.1,min(args.seconds,1200)))
        except StopRequested as exc:print(str(exc));return 2
        save_result(args.output,model,result)
        print(json.dumps(dict(status=result['status'],case_id=result['identity'],diagnostics=result['diagnostics']),indent=2))
        return 0 if result['status']=='PASS' else 1
    if args.command=='verify-export':
        summary,_=load_result(args.directory);print(json.dumps(dict(integrity='PASS',numerical_status=summary['status'],
            training_qualified=summary['training_qualified'])));return 0
    if args.command=='serve':
        from .api import serve
        serve(args.port,args.workspace)
    return 0


if __name__=='__main__':raise SystemExit(main())
