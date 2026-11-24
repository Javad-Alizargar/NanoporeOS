"""Manuscript figures/tables from saved NanoporeOS results; no solves or API calls.

Default root follows this source checkout. Outputs are manuscript figure assets.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PASS = '#d3e9e3'
FAIL = '#f0d8cf'
INK = '#172b37'
EDGE = '#71828b'
PALE = '#edf1f3'
BLUE = '#dae6ee'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)


def scenario_key(s):
    return tuple(float(s[name]) for name in ('radius_nm','salt_mM','voltage_mV','wall_offset_mV'))


def build_sources(root):
    relative = [
        'results/matrix/campaign.json', 'results/statistics.json',
        'results/manuscript_v2/qualification_audit.json',
        'results/manuscript_v2/acceptance_policies.csv',
        'results/performance_summary.json', 'results/performance.json',
        'config/verification_protocol.json', 'config/manuscript_v2_analysis_plan.json',
        'config/performance_protocol.json',
    ]
    manifest = {name: digest(root/name) for name in relative}
    matrix = read_json(root/relative[0])
    statistics = read_json(root/relative[1])
    audit = read_json(root/relative[2])
    performance = read_json(root/'results/performance_summary.json')
    assert matrix['status'] == 'FINISHED' and len(matrix['records']) == 108
    assert audit['status'] == 'PASS_RETROSPECTIVE_AUDIT'
    assert audit['matrix_sha256'] == manifest['results/matrix/campaign.json']
    assert performance['status'] == 'FINISHED'
    assert performance['source_artifact_sha256'] == manifest['results/performance.json']
    by_key = defaultdict(list)
    for row in matrix['records']:
        by_key[scenario_key(row['scenario'])].append(row)
    audit_by_id = {row['case_id']: row for row in audit['records']}
    assert len(by_key) == 36 and len(audit_by_id) == 108
    scenario_rows = []
    for number, (key, records) in enumerate(sorted(by_key.items()), 1):
        assert len(records) == 3
        assert sorted((r['scenario']['nx'],r['scenario']['nr']) for r in records) == [(16,8),(24,12),(36,18)]
        audited = [audit_by_id[r['case_id']] for r in records]
        flags = {r['mesh_field_gate'] for r in audited}
        assert len(flags) == 1
        candidate = sum(r['status']=='PASS' for r in records)
        assert candidate == sum(r['candidate_discrete'] for r in audited)
        legacy = sum(bool(r['legacy_reported_converged']) for r in records)
        assert legacy == sum(r['legacy_reported'] for r in audited)
        scenario_rows.append(dict(scenario_number=number, radius_nm=key[0], salt_mM=key[1],
            voltage_mV=key[2], wall_offset_mV=key[3],legacy_reported_mesh_pass=legacy,
            candidate_strict_mesh_pass=candidate, mesh_count=3,field_mesh_gate=bool(next(iter(flags))),
            analytically_qualified_mesh_records=sum(r['qualified'] for r in audited)))
    assert sum(r['legacy_reported_mesh_pass'] for r in scenario_rows) == statistics['legacy_reported_converged']
    assert sum(r['candidate_strict_mesh_pass'] for r in scenario_rows) == statistics['candidate_discrete_pass']
    assert sum(r['field_mesh_gate'] for r in scenario_rows) == statistics['mesh_field_pass']
    with (root/'results/manuscript_v2/acceptance_policies.csv').open(encoding='utf-8-sig',newline='') as stream:
        policy_rows = list(csv.DictReader(stream))
    policy_functions = [
        lambda r:r['legacy_reported'],
        lambda r:r['candidate_discrete'],
        lambda r:r['candidate_discrete'] and r['mesh_field_gate'],
        lambda r:r['qualified'],
    ]
    assert len(policy_rows) == len(policy_functions) == 4
    for policy, select in zip(policy_rows,policy_functions):
        selected = [r for r in audit['records'] if select(r)]
        expected = dict(selected_mesh_records=len(selected),total_mesh_records=108,
            selected_from_field_gate_failed_scenarios=sum(not r['mesh_field_gate'] for r in selected),
            selected_non_neutral_records=sum(r['scenario']['wall_offset_mV']!=0 for r in selected))
        for name, value in expected.items():
            assert int(policy[name]) == value, (policy['policy'],name)
            policy[name] = value
    return manifest, scenario_rows, policy_rows, performance, statistics


def save_figure(fig, out, stem):
    fig.savefig(out/f'{stem}.png', dpi=300, facecolor='white')
    fig.savefig(out/f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def workflow(out, source):
    fig = plt.figure(figsize=(6.5,5.35))
    ax = fig.add_axes([0.025,0.02,.95,.96])
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    def box(x,y,w,h,title,body='',color=PALE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',
            linewidth=.8,edgecolor=EDGE,facecolor=color))
        if body:
            ax.text(x+w/2,y+h-.20,title,ha='center',va='top',fontsize=9.4,fontweight='bold',color=INK)
            ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=8.3,linespacing=1.35,color=INK)
        else:
            ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=9.4,color=INK)
    def arrow(start,end,color=EDGE,style='-'):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
            linewidth=1,color=color,linestyle=style,connectionstyle='arc3'))
    box(.15,8.68,4.25,1.1,'Declared problem','Geometry · electrolyte · units\nEquations and named boundary conditions',BLUE)
    box(5.6,8.68,4.25,1.1,'Numerical configuration','Mesh · tolerances · bounded run\nEnvironment and source identities',BLUE)
    box(.15,6.98,2.12,1.0,'CLI / API','Shared inputs')
    box(3.02,6.75,4.03,1.38,'Canonical steady PNP engine','Log-density coupled Newton\nSame final legacy boundary problem')
    box(7.78,6.98,2.07,1.0,'Job queue','Cancel / restart')
    arrow((2.27,7.48),(3.02,7.48)); arrow((7.78,7.48),(7.05,7.48))
    arrow((2.27,8.68),(4.03,8.13)); arrow((7.73,8.68),(6.06,8.13))
    box(.48,4.63,4.12,1.32,'Discrete equation checks','Coupled residuals · ion positivity\nSpecies balance · section currents',PASS)
    box(5.40,4.63,4.12,1.32,'Spatial / reference checks','Mesh and directional refinement\nAnalytical / independent comparisons',BLUE)
    arrow((4.34,6.75),(2.60,5.95)); arrow((5.67,6.75),(7.47,5.95))
    arrow((4.60,5.30),(5.40,5.30))
    box(.48,4.01,9.04,.43,'Qualification is a separate recorded decision',color=PALE)
    box(.48,2.54,4.12,1.30,'Neutral analytical limit',
        f"{source['neutral_records']} mesh records / {source['neutral_scenarios']} scenarios\nEligible for the bounded baseline view",PASS)
    box(5.40,2.54,4.12,1.30,'Charged-wall HOLD',
        f"{source['exploratory_records']} mesh records remain exploratory\nNo qualified charged-wall training targets",FAIL)
    arrow((2.60,4.63),(2.60,4.44)); arrow((7.47,4.63),(7.47,4.44))
    arrow((2.60,4.01),(2.60,3.84)); arrow((7.47,4.01),(7.47,3.84))
    box(1.38,.64,7.24,1.18,'Documented exports retain the evidence',
        'Field arrays · coordinates and units · QA and failure records\nConfiguration · source / environment lineage · integrity hashes',PALE)
    arrow((2.60,2.54),(3.30,1.82)); arrow((7.47,2.54),(6.70,1.82))
    ax.text(5,.15,'Stationary synthetic fields. Numerical qualification does not establish experimental validity.',
        ha='center',va='center',fontsize=7.9,color=INK)
    save_figure(fig,out,'figure1')


def scenario_map(out, rows):
    fig, axes = plt.subplots(2,2,figsize=(6.5,7.6))
    fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.12,wspace=.36,hspace=.38)
    facet_keys = [(5.,10.),(5.,100.),(10.,10.),(10.,100.)]
    for index, (ax,(radius,salt)) in enumerate(zip(axes.ravel(),facet_keys)):
        selected = sorted([r for r in rows if r['radius_nm']==radius and r['salt_mM']==salt],
                          key=lambda r:(r['voltage_mV'],r['wall_offset_mV']))
        assert len(selected)==9
        ax.set_xlim(-1.34,3); ax.set_ylim(9.05,-1.05); ax.axis('off')
        ax.set_title(f"{'ABCD'[index]}   Radius {radius:g} nm · salt {salt:g} mM",
            loc='left',fontsize=10.2,fontweight='bold',pad=10,color=INK)
        column_titles=['Legacy\nreported','Candidate\nstrict','Field\ncriterion']
        for j,label in enumerate(column_titles):
            ax.text(j+.5,-.67,label,ha='center',va='center',fontsize=8.2,color=INK,linespacing=1.2)
        ax.text(-.10,-.67,'V / offset\n(mV)',ha='right',va='center',fontsize=8.2,color=INK)
        for i,row in enumerate(selected):
            ax.text(-.10,i+.5,f"{row['voltage_mV']:+.0f} / {row['wall_offset_mV']:+.0f}",
                ha='right',va='center',fontsize=8.2,color=INK)
            values=[row['legacy_reported_mesh_pass'],row['candidate_strict_mesh_pass'],row['field_mesh_gate']]
            for j,value in enumerate(values):
                passed = value if j==2 else value==3
                ax.add_patch(Rectangle((j,i),1,1,facecolor=PASS if passed else FAIL,edgecolor='white',lw=1.2))
                text=('PASS' if value else 'FAIL') if j==2 else f'{value}/3'
                ax.text(j+.5,i+.5,text,ha='center',va='center',fontsize=8.6,color=INK,
                    fontweight='bold' if not passed else 'normal')
        for y in (3,6):ax.plot([-1.28,3],[y,y],color=EDGE,lw=.6)
    totals=[sum(r['legacy_reported_mesh_pass'] for r in rows),sum(r['candidate_strict_mesh_pass'] for r in rows),sum(r['field_mesh_gate'] for r in rows)]
    fig.text(.5,.073,f'Legacy reported: {totals[0]}/108 mesh solves   ·   Candidate strict: {totals[1]}/108   ·   Field criterion: {totals[2]}/36 scenarios',
        ha='center',fontsize=8.5,color=INK)
    fig.text(.5,.039,'Counts use 16×8, 24×12 and 36×18 meshes. Field criterion compares 24×12 with 36×18.\nPASS means the named criterion only; it does not release the charged-wall domain.',
        ha='center',va='center',fontsize=8,color=INK,linespacing=1.4)
    save_figure(fig,out,'figure2')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    if not (root/'config/project_config.json').is_file():raise ValueError('NanoporeOS project root not found')
    out=(args.output or root/'manuscript/figures').resolve()
    if not out.is_relative_to(root/'manuscript'):
        raise ValueError('Figure outputs must remain inside manuscript')
    out.mkdir(parents=True,exist_ok=True)
    manifest, scenarios, policies, performance, statistics=build_sources(root)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.pad_inches':.03,'axes.unicode_minus':True})
    neutral=sum(r['analytically_qualified_mesh_records'] for r in scenarios)
    architecture=dict(neutral_records=neutral,neutral_scenarios=sum(r['analytically_qualified_mesh_records']>0 for r in scenarios),
        exploratory_records=108-neutral,experimental_validation='NOT_PERFORMED',figure_type='workflow schematic, not a simulated physical trajectory')
    workflow(out,architecture)
    scenario_map(out,scenarios)
    write_csv(out/'figure2_source_scenarios.csv',scenarios,list(scenarios[0]))
    write_csv(out/'table1_acceptance_policies.csv',policies,[
        'policy','selected_mesh_records','selected_from_field_gate_failed_scenarios','selected_non_neutral_records'])
    method_rows=[]
    for method,label,criterion in [('legacy','Legacy fixed relaxation','Legacy update-success flag'),
                                   ('candidate','Coupled log-density Newton','Coupled residual, positivity and current consistency')]:
        values=performance['methods'][method]
        method_rows.append(dict(method=label,attempts=values['attempts'],method_successes=values['successes'],
            success_criterion=criterion,repetitions=performance['repetitions'],physical_scenarios=performance['physical_scenario_groups'],
            median_attempt_time_s=values['median_seconds'],total_attempt_time_s=values['total_seconds']))
    write_csv(out/'table2_attempted_compute.csv',method_rows,[
        'method','attempts','method_successes','median_attempt_time_s','total_attempt_time_s'])
    difference=dict(comparison='Legacy minus candidate attempted time',
        mean_paired_difference_s=performance['mean_paired_attempt_time_difference_seconds'],
        scenario_bootstrap_95CI_low_s=performance['scenario_bootstrap_95percent_CI_seconds'][0],
        scenario_bootstrap_95CI_high_s=performance['scenario_bootstrap_95percent_CI_seconds'][1],
        bootstrap_seed=performance['bootstrap_seed'],bootstrap_draws=performance['bootstrap_draws'],
        interpretation=performance['interpretation'])
    write_csv(out/'table2_paired_difference.csv',[difference],list(difference))
    write_json(out/'source_data.json',dict(figure1=architecture,figure2=scenarios,table1=policies,
        table2=method_rows,table2_difference=difference,sources=manifest,script_sha256=digest(Path(__file__)),
        environment=dict(python=sys.version,matplotlib=matplotlib.__version__,platform=platform.platform()),
        checks=dict(scenarios=len(scenarios),mesh_records=108,policy_counts_checked_against_audit=True,
            matrix_counts_checked_against_statistics=True,performance_source_hash_matches=True)))
    write_json(out/'manifest.json',dict(schema_version='visual-proposal-1.0',
        inputs=manifest,script_sha256=digest(Path(__file__)),
        outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(status='PASS',output=str(out),scenarios=len(scenarios),
        legacy_mesh_pass=statistics['legacy_reported_converged'],candidate_mesh_pass=statistics['candidate_discrete_pass'],
        field_scenario_pass=statistics['mesh_field_pass'],policy_retained=[p['selected_mesh_records'] for p in policies]),indent=2))


if __name__=='__main__':main()
