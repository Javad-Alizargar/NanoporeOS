from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import PNP,Scenario,write_json,source_identity,sha256,check_stop,StopRequested


def main():
    spec=importlib.util.spec_from_file_location('legacy',ROOT/'benchmarks/legacy_kernel.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    protocol=ROOT/'config/performance_protocol.json'
    cfg=json.loads(protocol.read_text());records=[];target=ROOT/'results/performance.json'
    fingerprint=hashlib.sha256((source_identity()+sha256(protocol)+sha256(ROOT/'benchmarks/legacy_kernel.py')).encode()).hexdigest()
    if target.exists():
        prior=json.loads(target.read_text())
        if prior['identity']!=fingerprint:raise RuntimeError('Performance source/protocol changed; retain output and choose a new version')
        records=prior['records']
    done={(r['repeat'],r['case'],r['method']) for r in records}
    cases=[Scenario(nx=n,nr=m,radius_nm=r,salt_mM=c,voltage_mV=v,wall_offset_mV=w)
        for r in (5.,10.) for c in (10.,100.) for v in (-100.,0.,100.) for w in (-15.,0.,15.)
        for n,m in [(16,8),(24,12),(36,18)]]
    def run(s,method):
        start=time.perf_counter()
        if method=='candidate':
            out=PNP(s).solve(deadline=deadline)
            success=out['status']=='PASS'
        else:
            m=legacy.AxisymPNP(s.nx,s.nr,Lnm=s.length_nm,Rnm=s.radius_nm,c_mM=s.salt_mM,
                V_mV=s.voltage_mV,zeta_mV=s.wall_offset_mV)
            out=m.solve();success=out['success']
        return time.perf_counter()-start,success
    deadline=time.monotonic()+900
    run(Scenario(nx=8,nr=4),'candidate');run(Scenario(nx=8,nr=4),'legacy')
    try:
        for repeat,seed in enumerate(cfg['seeds']):
            rng=np.random.default_rng(seed)
            for index in rng.permutation(len(cases)):
                for method in rng.permutation(['candidate','legacy']):
                    key=(repeat,int(index),str(method))
                    if key in done:continue
                    check_stop(deadline)
                    s=cases[index];seconds,success=run(s,method)
                    records.append(dict(repeat=repeat,seed=seed,case=int(index),method=str(method),
                        scenario=dataclasses.asdict(s),seconds=seconds,method_success=bool(success)))
                    write_json(target,dict(status='RUNNING',identity=fingerprint,records=records))
                if len(records)%36==0:print(f'{len(records)}/648 method timings',flush=True)
        status='FINISHED'
    except StopRequested as exc:status=str(exc)
    write_json(target,dict(status=status,identity=fingerprint,records=records,protocol_sha256=sha256(protocol),
        source_sha256=source_identity(),note='Different stopping criteria; attempted-compute timings, not equal-accuracy speedup'))


if __name__=='__main__':main()
