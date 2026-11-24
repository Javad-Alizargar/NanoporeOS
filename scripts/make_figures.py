from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import make_workflow_figures as workflow
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import write_json,sha256


def main():
    stats=json.loads((ROOT/'results/statistics.json').read_text())
    ref=json.loads((ROOT/'results/references/independent_pb.json').read_text())
    out=ROOT/'manuscript/figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
    labels=['Legacy\nreported','Legacy\npost-hoc check','Candidate\ncoupled check']
    values=[stats['legacy_reported_converged'],stats['legacy_strict_coupled_gate_pass'],stats['candidate_discrete_pass']]
    ax[0].bar(labels,values,color=['#8798aa','#b78858','#168f83']);ax[0].set_ylim(0,120);ax[0].set_ylabel('Mesh evaluations passing (of 108)')
    for i,v in enumerate(values):ax[0].text(i,v+2,str(v),ha='center')
    ax[0].set_title('A  Discrete convergence and solver checks')
    ax[1].bar(['Current\nspread','Potential and\nconcentration fields'],[stats['mesh_current_pass'],stats['mesh_field_pass']],color=['#168f83','#b78858'])
    ax[1].set_ylim(0,40);ax[1].set_ylabel('Physical scenarios passing (of 36)');ax[1].set_title('B  Original three-mesh criteria')
    for i,v in enumerate([stats['mesh_current_pass'],stats['mesh_field_pass']]):ax[1].text(i,v+1,str(v),ha='center')
    fig.savefig(out/'Supplemental_Figure_S2.png',dpi=300);fig.savefig(out/'Supplemental_Figure_S2.pdf');plt.close(fig)
    records=ref['records'][:5]
    fig,ax=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
    cells=[r['scenario']['nr'] for r in records];error=[r['reference_volume_L2_relative'] for r in records]
    ax[0].loglog(cells,error,'o-',color='#168f83');ax[0].set_xlabel('Radial cells (axial cells = 2 × radial)');ax[0].set_ylabel('Volume-weighted relative potential L2 error')
    ax[0].set_title('A  Independent finite-cylinder PB reference')
    comparisons=ref['mesh_comparisons'];names=['Both axes\n36×18 to\n72×36','Both axes\n72×36 to\n144×72','Radial only\n72×36 to\n72×72','Axial only\n72×36 to\n144×36']
    values=[r['fields'][0]['max_absolute'] for r in comparisons]
    ax[1].bar(names,values,color='#8798aa');ax[1].axhline(.02,color='#b25d4a',ls='--',label='Predeclared 0.02 VT limit')
    ax[1].set_ylabel('Maximum potential difference / VT');ax[1].set_title('B  Mesh direction and rim sensitivity');ax[1].tick_params(axis='x',labelsize=7.5);ax[1].legend(fontsize=8)
    fig.savefig(out/'figure3.png',dpi=300);fig.savefig(out/'figure3.pdf');plt.close(fig)
    source=dict(figure1=dict(labels=labels,passes=values if False else [stats['legacy_reported_converged'],stats['legacy_strict_coupled_gate_pass'],stats['candidate_discrete_pass']],
        mesh_current_pass=stats['mesh_current_pass'],mesh_field_pass=stats['mesh_field_pass']),figure2=dict(radial_cells=cells,potential_relative_L2=error,
        directional_max_phi_VT=values),statistics_sha256=sha256(ROOT/'results/statistics.json'),reference_sha256=sha256(ROOT/'results/references/independent_pb.json'))
    comparison=json.loads((ROOT/'results/noneq_reference/comparison.json').read_text())
    audit=json.loads((ROOT/'results/noneq_reference/array_audit.json').read_text())
    selected=next(r for r in comparison['records'] if r['voltage_mV']==100)
    grids=[r['candidate_grid'][0] for r in selected['comparisons']]
    difference=[r['max_phi_difference_VT'] for r in selected['comparisons']]
    ref_change=[r['max_refinement_difference_VT'] for r in audit['reference_resolution_at_each_candidate_grid'] if r['voltage_mV']==100]
    current=[100*r['current_relative_error'] for r in selected['comparisons']]
    fig,ax=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
    ax[0].loglog(grids,difference,'o-',color='#168f83',label='Candidate / 256×128 reference')
    ax[0].loglog(grids,ref_change,'s-',color='#b78858',label='Reference 128×64 to 256×128')
    ax[0].axhline(.02,color='#b25d4a',ls='--',label='Original 0.02 VT criterion')
    ax[0].set_xlabel('Candidate axial cells (radial = axial / 2)');ax[0].set_ylabel('Maximum potential difference / VT')
    ax[0].set_title('A  Reference refinement at evaluation points');ax[0].legend(fontsize=7)
    ax[1].loglog(grids,current,'o-',color='#168f83');ax[1].set_xlabel('Candidate axial cells (radial = axial / 2)')
    ax[1].set_ylabel('Current difference (%)');ax[1].set_title('B  Current agreement')
    fig.savefig(out/'figure4.png',dpi=300);fig.savefig(out/'figure4.pdf');plt.close(fig)
    source['figure3']=dict(candidate_axial_cells=grids,max_candidate_phi_difference_VT=difference,
        max_reference_refinement_difference_VT=ref_change,current_relative_discrepancy_percent=current,
        comparison_sha256=sha256(ROOT/'results/noneq_reference/comparison.json'),audit_sha256=sha256(ROOT/'results/noneq_reference/array_audit.json'))
    source['Supplemental_Figure_S2']=source.pop('figure1')
    source['figure4']=source.pop('figure3');source['figure3']=source.pop('figure2')
    manifest,scenarios,policies,performance,statistics=workflow.build_sources(ROOT)
    architecture=dict(neutral_records=36,neutral_scenarios=12,exploratory_records=72,
        experimental_validation='NOT_PERFORMED',figure_type='workflow schematic')
    workflow.workflow(out,architecture);workflow.scenario_map(out,scenarios)
    source.update(figure1=architecture,figure2=scenarios,workflow_input_hashes=manifest)
    write_json(out/'source_data.json',source)
    write_json(out/'manifest.json',dict(files={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('Created four main figures and one supplemental figure, vector PDF and PNG, with source data')


if __name__=='__main__':main()
