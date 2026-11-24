"""POST HOC component audit on stored currents. Not a new held-out experiment."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def main():
    results=[]
    for family in ['verification_v3','verification_v4']:
        pointer=json.loads((ROOT/'results'/family/'latest.json').read_text())
        path=ROOT/pointer['summary'];summary=json.loads(path.read_text());directory=path.parent
        records=[]
        for case in summary['records']:
            if case['split']!='held_out':continue
            solves=[json.loads((directory/'cases'/identity/'result.json').read_text()) for identity in case['solve_identities']]
            selected=None;visited=[]
            for i,item in enumerate(solves):
                visited.append(i)
                if i==0 or item['status']!='PASS' or solves[i-1]['status']!='PASS':continue
                previous,current=solves[i-1]['mid_current_A'],item['mid_current_A']
                estimate=2/3*abs(previous-current)/max(abs(current),1e-15)
                if estimate<=.01:selected=i;break
            records.append(dict(case=case['case']['name'],selected_index=selected,
                accurate=None if selected is None else case['exact_accuracy_passes'][selected],visited=visited,
                attempted_cells=sum(np.prod(case['grids'][i]).item() for i in visited),
                solve_seconds=sum(solves[i]['construction_and_solve_seconds'] for i in visited)))
        results.append(dict(family=family,accurate=sum(r['accurate'] is True for r in records),
            inaccurate=sum(r['accurate'] is False for r in records),unresolved=sum(r['selected_index'] is None for r in records),
            attempted_cells=sum(r['attempted_cells'] for r in records),cases=records))
    output=dict(status='POST_HOC_ABLATION_NOT_NEW_HOLDOUT',policy='one-pair current-only with2/3Richardson factor and1%target',
        primary_protocols_and_results_unchanged=True,results=results,
        interpretation='Isolates the current-estimator convention more fairly; observed component evidence only, not prospective validation')
    (HERE/'matched_current_posthoc.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='cases'} for r in results],indent=2))


if __name__=='__main__':main()
