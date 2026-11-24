from pathlib import Path
import collections
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import write_json,sha256


def main():
    path=ROOT/'results/performance.json';data=json.loads(path.read_text())
    if data['status']!='FINISHED' or len(data['records'])!=648:raise RuntimeError('Repeated timing incomplete')
    rows=data['records'];groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        s={k:v for k,v in r['scenario'].items() if k not in ['nx','nr']}
        gid=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
        groups[gid][r['method']].append(r['seconds'])
    differences=np.array([np.mean(v['legacy'])-np.mean(v['candidate']) for v in groups.values()])
    rng=np.random.default_rng(20261109)
    bootstrap=differences[rng.integers(0,len(differences),size=(10000,len(differences)))].mean(axis=1)
    summary=dict(status='FINISHED',source_artifact_sha256=sha256(path),method_timings=len(rows),
        physical_scenario_groups=len(groups),repetitions=3,
        methods={method:dict(attempts=sum(r['method']==method for r in rows),
            successes=sum(r['method']==method and r['method_success'] for r in rows),
            median_seconds=float(np.median([r['seconds'] for r in rows if r['method']==method])),
            total_seconds=float(sum(r['seconds'] for r in rows if r['method']==method))) for method in ['legacy','candidate']},
        mean_paired_attempt_time_difference_seconds=float(differences.mean()),
        scenario_bootstrap_95percent_CI_seconds=np.quantile(bootstrap,[.025,.975]).tolist(),
        bootstrap_seed=20261109,bootstrap_draws=10000,
        scope='Construction plus method call; candidate includes diagnostics, legacy retains original stopping rule; failures included',
        interpretation='Attempt cost/coverage comparison only. Different accuracy criteria and failure outcomes prevent an equal-accuracy solver speedup claim.',
        energy_or_full_system_peak_memory='NOT_MEASURED')
    write_json(ROOT/'results/performance_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
