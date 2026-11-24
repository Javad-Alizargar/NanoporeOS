"""Audit immutable verification campaigns and generate manuscript evidence."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import sha256,write_json
from nanoporeos.accuracy import Snapshot,select

def load_campaign(family):
    pointer=json.loads((ROOT/'results'/family/'latest.json').read_text(encoding='utf-8'))
    path=ROOT/pointer['summary'];assert sha256(path)==pointer['sha256']
    summary=json.loads(path.read_text(encoding='utf-8'));assert summary['status']=='FINISHED_FIXED_HOLDOUT'
    directory=path.parent;header=json.loads((directory/'header.json').read_text(encoding='utf-8'))
    names={'manufactured_source_sha256':'manufactured.py','controller_sha256':'accuracy.py','driver_sha256':'verification_v3.py'}
    for key,name in names.items():assert sha256(directory/'provenance'/name)==header['source_hashes'][key]
    protocol=next((directory/'provenance').glob('*_protocol.json'));assert sha256(protocol)==summary['protocol_sha256']
    for case in summary['records']:
        decision=directory/(case['case']['name']+'_decisions.json')
        assert sha256(decision)==case['decisions_before_scoring_sha256']
        blind=json.loads(decision.read_text(encoding='utf-8'))
        assert all('actual_errors' not in d and 'exact_accuracy_pass' not in d for d in blind['policies'])
        for identity in case['solve_identities']:
            parent=directory/'cases'/identity;m=json.loads((parent/'manifest.json').read_text())
            assert all(sha256(parent/name)==digest for name,digest in m['files'].items())
            with np.load(parent/'fields.npz',allow_pickle=False) as archive:assert np.isfinite(archive['y']).all()
    assert len([r for r in summary['records'] if r['split']=='held_out'])==8
    return summary,path

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    first,p1=load_campaign('verification_v3');second,p2=load_campaign('verification_v4')
    output=ROOT/'results/verification_analysis';output.mkdir(parents=True,exist_ok=True)
    first_keys={json.dumps({k:v for k,v in r['case'].items() if k!='name'},sort_keys=True) for r in first['records'] if r['split']=='held_out'}
    second_keys={json.dumps({k:v for k,v in r['case'].items() if k!='name'},sort_keys=True) for r in second['records'] if r['split']=='held_out'}
    assert first_keys.isdisjoint(second_keys),'Held-out parameter tuples were reused'
    rows=[];case_rows=[]
    labels={'residual_only':'Residual only','current_mesh':'Current mesh','field_controller':'Field controller','uniform_finest':'Finest uniform'}
    for name,summary in [('First set / two pairs',first),('New set / one pair',second)]:
        reference=next(r['total_attempted_cells'] for r in summary['held_out_policy_aggregate'] if r['policy']=='uniform_finest')
        for r in summary['held_out_policy_aggregate']:
            rows.append(dict(evaluation=name,policy=r['policy'],accurate=r['actual_accurate_accepts'],
                inaccurate=r['inaccurate_accepts'],unresolved=r['unresolved'],scenarios=r['scenarios'],
                attempted_cells=r['total_attempted_cells'],cells_relative_to_finest=r['total_attempted_cells']/reference,
                solve_plus_selection_seconds=r['total_solve_plus_selection_seconds']))
        for case in summary['records']:
            for grid,error,passed in zip(case['grids'],case['errors'],case['exact_accuracy_passes']):
                case_rows.append(dict(evaluation=name,case=case['case']['name'],split=case['split'],nx=grid[0],nr=grid[1],
                    **error,all_budgets_pass=passed))
    write_csv(output/'policy_results.csv',rows);write_csv(output/'case_errors.csv',case_rows)
    # Post hoc sensitivity check, not a new primary holdout or tuned policy.
    # Align the current-only estimator with the already-fixed factor 2/3.
    matched=[]
    for case in second['records']:
        if case['split']!='held_out':continue
        snapshots=[]
        for identity in case['solve_identities']:
            directory=p2.parent/'cases'/identity;saved=json.loads((directory/'result.json').read_text())
            with np.load(directory/'fields.npz',allow_pickle=False) as archive:
                snapshots.append(Snapshot(saved['problem_id'],saved['model'],archive['x_nm'],archive['r_nm'],archive['y'],
                    saved['mid_current_A'],saved['status'],saved['construction_and_solve_seconds']))
        decision=select('current_mesh',snapshots,{'mid_current_relative_error':.015})
        index=decision['selected_index']
        matched.append(dict(case=case['case']['name'],selected_grid=decision['selected_grid'],
            accurate=None if index is None else case['exact_accuracy_passes'][index],
            attempted_cells=decision['attempted_cells']))
    matched_summary=dict(status='POST_HOC_FIXED_ESTIMATOR_MATCHING_NOT_PRIMARY_HOLDOUT',
        definition='Current-only selection with2/3 times raw current difference <=1%, matching the frozen field rule current estimator',
        accurate=sum(r['accurate'] is True for r in matched),inaccurate=sum(r['accurate'] is False for r in matched),
        unresolved=sum(r['selected_grid'] is None for r in matched),attempted_cells=sum(r['attempted_cells'] for r in matched),records=matched,
        interpretation='Sensitivity/ablation only. Original four-policy outcomes remain unchanged. No causal field benefit inferred from differently scaled original current criteria.')
    write_json(output/'matched_current_ablation.json',matched_summary)
    development=[r for r in second['records'] if r['split']=='development']
    orders=[p for r in development for p in r['observed_last_pair_orders'].values()]
    summary=dict(status='AUDITED_BOTH_CAMPAIGNS',campaigns=[dict(path=str(p.relative_to(ROOT)),sha256=sha256(p)) for p in [p1,p2]],
        unique_evaluation_tuples=16,evaluation_sets_disjoint=True,policy_results=rows,
        development_observed_order_min=min(orders),development_observed_order_max=max(orders),
        development_cases=2,grids_per_case=4,reported_current='Midplane manufactured observable; prescribed sources permit axial current variation',
        matched_current_ablation=matched_summary,
        truth_separation='Saved pre-scoring policy records omit exact errors; exact fields score outputs afterwards',
        interpretation='Finite deterministic manufactured family; empirical qualification decisions, not certified error bounds, physical validation or generalization',
        legacy_charged_wall_gate='HOLD_UNCHANGED')
    write_json(output/'summary.json',summary)
    figures=ROOT/'manuscript/figures'
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.7),layout='constrained')
    colors=['#087e8b','#b95e36','#596ba4']
    for ax,record,panel in zip(axes,development,['A','B']):
        x=[g[0] for g in record['grids']]
        for key,label,color in zip(['max_phi_error_VT','max_plus_relative_error','max_minus_relative_error'],
            ['Potential / thermal voltage','Cation relative error','Anion relative error'],colors):
            ax.loglog(x,[r[key] for r in record['errors']],'o-',label=label,color=color,markersize=4)
        guide=record['errors'][1]['max_phi_error_VT']*(np.asarray(x)/x[1])**-2
        ax.loglog(x,guide,':',color='#5b6168',label='Second-order slope guide')
        ax.set_title(panel+'  '+record['case']['name'].replace('_',' '));ax.set_xlabel('Axial cells (radial = axial / 2)')
        ax.set_xticks(x,[str(n) for n in x]);ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylabel('Maximum dimensionless error');ax.legend(fontsize=6.5)
    fig.savefig(figures/'figure5.png',dpi=300);fig.savefig(figures/'figure5.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.5,4.2),layout='constrained')
    for ax,name,panel in zip(axes,['First set / two pairs','New set / one pair'],['A','B']):
        values=[r for r in rows if r['evaluation']==name];positions=np.arange(4);bottom=np.zeros(4)
        for key,label,color in [('accurate','Within all budgets','#2a9685'),('inaccurate','Exceeds a budget','#c57655'),('unresolved','Unresolved','#abb4bd')]:
            height=np.array([r[key] for r in values]);ax.bar(positions,height,bottom=bottom,label=label,color=color)
            for x,y,h in zip(positions,bottom,height):
                if h:ax.text(x,y+h/2,str(h),ha='center',va='center',fontsize=9)
            bottom+=height
        ax.set_xticks(positions,['Residual\nonly','Current\nmesh','Field\ncontroller','Finest\nuniform'],fontsize=7.5)
        ax.set_ylim(0,9.5);ax.set_ylabel('Fixed evaluation scenarios (n = 8)');ax.set_title(panel+'  '+name.replace(' / ','\n'))
        ax.legend(fontsize=6.7,loc='upper center',frameon=False)
    fig.savefig(figures/'figure6.png',dpi=300);fig.savefig(figures/'figure6.pdf');plt.close(fig)
    displayed=[]
    for row in rows:
        displayed.append({'Set':'First' if row['evaluation'].startswith('First') else 'New','Policy':labels[row['policy']],
            'Accurate (n/8)':row['accurate'],'Inaccurate (n/8)':row['inaccurate'],'Unresolved (n/8)':row['unresolved'],
            'Cells / finest':f"{row['cells_relative_to_finest']:.3f}"})
    write_csv(ROOT/'manuscript/tables/Table4_Controlled_Verification.csv',displayed)
    write_csv(ROOT/'manuscript/tables/Supplemental_Table_S5_Manufactured_Case_Errors.csv',case_rows)
    write_csv(ROOT/'manuscript/tables/Supplemental_Table_S6_Controller_Costs.csv',rows)
    source=dict(analysis_summary_sha256=sha256(output/'summary.json'),policy_results=rows,
        development_cases=development,script_sha256=sha256(Path(__file__)))
    write_json(figures/'verification_source_data.json',source)
    write_json(output/'manifest.json',dict(files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps({k:v for k,v in summary.items() if k!='policy_results'},indent=2))

if __name__=='__main__':main()
