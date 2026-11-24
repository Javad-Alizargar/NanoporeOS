"""Audit frozen matched-family outputs; never select against exact truth."""
from pathlib import Path
import json,csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import sha256,write_json
from nanoporeos.accuracy import Snapshot
from nanoporeos.acceptance_matched import select

def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    pointer=json.loads((ROOT/'results/verification_v5/latest.json').read_text());path=ROOT/pointer['summary']
    assert sha256(path)==pointer['sha256'];data=json.loads(path.read_text());assert data['status']=='FINISHED_FIXED_HOLDOUT'
    directory=path.parent;protocol_path=directory/'provenance/verification_v5a_protocol.json'
    assert sha256(protocol_path)==data['protocol_sha256'];protocol=json.loads(protocol_path.read_text())
    names={'families_source_sha256':'manufactured_families.py','source_aware_audit_source_sha256':'manufactured.py',
        'controller_sha256':'accuracy.py','matched_controller_sha256':'acceptance_matched.py','driver_sha256':'verification_v5.py'}
    for k,v in names.items():assert sha256(directory/'provenance'/v)==data['source_hashes'][k]
    rows=[];case_rows=[];paired=[];orders=[];discrete=[]
    for record in data['records']:
        decisions=directory/(record['case']['name']+'_decisions.json');assert sha256(decisions)==record['decisions_before_scoring_sha256']
        blind=json.loads(decisions.read_text());assert all('actual_errors' not in d and 'exact_accuracy_pass' not in d for d in blind['policies'])
        snapshots=[]
        for i,identity in enumerate(record['solve_identities']):
            parent=directory/'cases'/identity;manifest=json.loads((parent/'manifest.json').read_text())
            assert manifest['identity']==identity and manifest['protocol_sha256']==data['protocol_sha256']
            assert all(sha256(parent/n)==h for n,h in manifest['files'].items())
            saved=json.loads((parent/'result.json').read_text());assert saved['manufactured_case']==record['case']
            assert saved['identity']==identity and saved['verification_only'] and not saved['training_qualified']
            with np.load(parent/'fields.npz',allow_pickle=False) as z:
                snapshots.append(Snapshot(saved['problem_id'],saved['model'],z['x_nm'],z['r_nm'],z['y'],
                    saved['mid_current_A'],saved['status'],saved['construction_and_solve_seconds']))
            discrete.append(saved['status'])
            case_rows.append(dict(case=record['case']['name'],family=record['case']['family'],split=record['split'],
                nx=record['grids'][i][0],nr=record['grids'][i][1],discrete_status=saved['status'],
                **record['errors'][i],all_budgets_pass=record['exact_accuracy_passes'][i]))
        for d in record['policies']:
            replay=select(d['policy'],snapshots,protocol['exact_evaluation_tolerances'],2,2,1)
            for k in ['selected_index','selected_grid','visited_indices','attempted_cells','indicators']:assert replay[k]==d[k]
            selected=d['selected_index'];expected=None if selected is None else record['exact_accuracy_passes'][selected]
            assert expected==d['exact_accuracy_pass']
        if record['split']=='development':orders.extend(record['observed_last_pair_orders'].values());continue
        current=next(d for d in record['policies'] if d['policy']=='current_matched')
        field=next(d for d in record['policies'] if d['policy']=='field_matched')
        paired.append(dict(case=record['case']['name'],family=record['case']['family'],
            current_accuracy=current['exact_accuracy_pass'],field_accuracy=field['exact_accuracy_pass'],
            current_grid=str(current['selected_grid']),field_grid=str(field['selected_grid']),
            current_cells=current['attempted_cells'],field_cells=field['attempted_cells']))
    evaluation=[r for r in data['records'] if r['split']=='held_out'];assert len(evaluation)==24
    for family in ['ALL','polynomial','oscillatory','localized']:
        group=[r for r in evaluation if family=='ALL' or r['case']['family']==family]
        for policy in protocol['policies']:
            ds=[next(d for d in r['policies'] if d['policy']==policy) for r in group]
            rows.append(dict(family=family,policy=policy,scenarios=len(ds),accurate=sum(d['exact_accuracy_pass'] is True for d in ds),
                inaccurate=sum(d['exact_accuracy_pass'] is False for d in ds),unresolved=sum(d['selected_index'] is None for d in ds),
                attempted_cells=sum(d['attempted_cells'] for d in ds),replayed_seconds=sum(d['solve_plus_selection_seconds'] for d in ds)))
    violations={k:0 for k in protocol['exact_evaluation_tolerances']}
    for record in evaluation:
        current=next(d for d in record['policies'] if d['policy']=='current_matched')
        if current['actual_errors'] is not None:
            for k,limit in protocol['exact_evaluation_tolerances'].items():violations[k]+=int(current['actual_errors'][k]>limit)
    summary=dict(status='PASS_FROZEN_RESULT_AUDIT',campaign_summary=str(path.relative_to(ROOT)),campaign_summary_sha256=sha256(path),
        protocol_sha256=data['protocol_sha256'],evaluation_combinations=24,function_families=3,unique_parameter_tuples=8,
        discrete_pass=sum(s=='PASS' for s in discrete),discrete_total=len(discrete),development_order_min=min(orders),
        development_order_max=max(orders),policies=rows,
        paired_current_inaccurate_field_accurate=sum(r['current_accuracy'] is False and r['field_accuracy'] is True for r in paired),
        paired_current_inaccurate_field_unresolved=sum(r['current_accuracy'] is False and r['field_accuracy'] is None for r in paired),
        paired_current_accurate_field_unresolved=sum(r['current_accuracy'] is True and r['field_accuracy'] is None for r in paired),
        current_selected_exact_budget_violations=violations,
        shared_estimator_factor=2/3,certified_bound=False,physical_qualification=False,
        statistical_scope=protocol['statistical_unit'],timing_scope=protocol['cost_estimand'])
    output=ROOT/'results/verification_v5_analysis';output.mkdir(exist_ok=True)
    write_json(output/'summary.json',summary);csvwrite(output/'policy_results.csv',rows);csvwrite(output/'case_errors.csv',case_rows);csvwrite(output/'paired_results.csv',paired)
    tables=ROOT/'manuscript/tables';figures=ROOT/'manuscript/figures'
    csvwrite(tables/'Supplemental_Table_S7_Distinct_Family_Policies.csv',rows)
    csvwrite(tables/'Supplemental_Table_S8_Distinct_Family_Pairs.csv',paired)
    csvwrite(tables/'Supplemental_Table_S9_Distinct_Family_Errors.csv',case_rows)
    short=[]
    labels={'residual_only':'Residual','current_matched':'Current','field_matched':'Fields','uniform_finest':'Finest'}
    for row in rows:
        if row['family']=='ALL':continue
        short.append({'Family':row['family'].capitalize(),'Policy':labels[row['policy']],'Accurate':row['accurate'],
            'Inaccurate':row['inaccurate'],'Unresolved':row['unresolved'],'Cells':f"{row['attempted_cells']:,}"})
    csvwrite(tables/'Table5_Matched_Function_Families.csv',short)
    fig,axes=plt.subplots(1,3,figsize=(6.5,3.6),layout='constrained')
    for ax,(i,family) in zip(axes,enumerate(['polynomial','oscillatory','localized'])):
        group=[r for r in rows if r['family']==family];positions=np.arange(4);bottom=np.zeros(4)
        for key,label,color in [('accurate','Within all budgets','#26968b'),('inaccurate','Exceeds a budget','#c47756'),('unresolved','Unresolved','#a8b2bb')]:
            values=np.array([r[key] for r in group]);ax.bar(positions,values,bottom=bottom,color=color,label=label)
            for x,v,b in zip(positions,values,bottom):
                if v:ax.text(x,b+v/2,str(v),ha='center',va='center',fontsize=9)
            bottom+=values
        ax.set_xticks(positions,[labels[r['policy']] for r in group],rotation=35,ha='right',fontsize=8.5)
        ax.set_ylim(0,10);ax.set_yticks([0,2,4,6,8]);ax.set_title(chr(65+i)+'  '+family.capitalize())
        if i==0:ax.set_ylabel('Fixed combinations per family (n = 8)',fontsize=9);ax.legend(fontsize=7.3,loc='upper center')
    fig.savefig(figures/'figure7.png',dpi=300);fig.savefig(figures/'figure7.pdf');plt.close(fig)
    write_json(figures/'verification_v5_source_data.json',dict(summary_sha256=sha256(output/'summary.json'),policy_results=rows,script_sha256=sha256(Path(__file__))))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
