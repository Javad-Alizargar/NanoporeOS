from pathlib import Path
import collections
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from nanoporeos.core import Scenario,PNP,VT,ohmic_current,write_json,sha256,source_identity
from nanoporeos.datasets import load_result
from verify_references import compare_fields


def main():
    path=ROOT/'results/matrix/campaign.json';campaign=json.loads(path.read_text());rows=campaign['records']
    if campaign['status']!='FINISHED' or len(rows)!=108:raise RuntimeError('Incomplete matrix')
    groups=collections.defaultdict(list)
    for r in rows:
        s=r['scenario'];groups[tuple(s[k] for k in ['radius_nm','salt_mM','voltage_mV','wall_offset_mV'])].append(r)
    mesh=[]
    for key,records in groups.items():
        records=sorted(records,key=lambda r:r['scenario']['nx']);models=[]
        for rec in records:
            summary,arrays=load_result(ROOT/'results/matrix'/rec['case_id'])
            p=PNP(Scenario(**rec['scenario']))
            y=np.r_[arrays['potential_V'].ravel()/VT,np.log(arrays['cplus_mol_m3'].ravel()/p.s.salt_mM),
                np.log(arrays['cminus_mol_m3'].ravel()/p.s.salt_mM)]
            models.append((p,y))
        currents=[r['diagnostics']['current_A'] for r in records]
        rel=float(np.ptp(currents)/max(abs(currents[-1]),1e-18))
        fields=compare_fields(*models[1],*models[2])
        field_pass=all(f['relative_volume_L2']<=.05 for f in fields) and fields[0]['max_absolute']<=.02
        mesh.append(dict(parameters=key,grids=[[r['scenario']['nx'],r['scenario']['nr']] for r in records],
            current_relative_spread=rel,current_gate='PASS' if rel<=.05 else 'FAIL',
            fields=fields,field_gate='PASS' if field_pass else 'FAIL',
            note='Three original meshes; stronger controlled refinements assessed separately'))
    ohmic=[abs(r['diagnostics']['current_A']-ohmic_current(Scenario(**r['scenario'])))/max(1e-18,abs(ohmic_current(Scenario(**r['scenario']))))
        for r in rows if r['scenario']['wall_offset_mV']==0]
    fail=[r for r in rows if not r['legacy_reported_converged']]
    legacy_strict=sum(r['legacy_diagnostics'] is not None and r['legacy_diagnostics']['coupled_residual_max']<=2e-11
        and r['legacy_diagnostics']['current_consistent'] for r in rows)
    stats=dict(schema_version='1.0',source_sha256=source_identity(),matrix_artifact=str(path.relative_to(ROOT)),
        matrix_sha256=sha256(path),scenario_count=len(groups),mesh_evaluations=len(rows),
        legacy_reported_converged=sum(r['legacy_reported_converged'] for r in rows),
        legacy_strict_coupled_gate_pass=legacy_strict,
        candidate_discrete_pass=sum(r['status']=='PASS' for r in rows),
        candidate_max_coupled_residual=max(r['diagnostics']['coupled_residual_max'] for r in rows),
        candidate_min_concentration_mol_m3=min(r['diagnostics']['min_concentration_mol_m3'] for r in rows),
        candidate_max_current_spread_A=max(r['diagnostics']['current_spread_A'] for r in rows),
        ohmic_max_relative_error=max(ohmic),
        mesh_current_pass=sum(r['current_gate']=='PASS' for r in mesh),mesh_field_pass=sum(r['field_gate']=='PASS' for r in mesh),
        candidate_total_solve_seconds=sum(r['seconds'] for r in rows),
        legacy_total_solve_seconds=sum(r['legacy_seconds'] for r in rows),
        experimental_validation='NOT_PERFORMED',scientific_gate='NUMERICAL_HOLD_PENDING_REFERENCE_AND_REFINEMENT_DOMAIN',
        training_qualification='No charged-wall production targets qualified by this summary alone',
        legacy_failure_cases=[dict(scenario=r['scenario'],case_id=r['case_id']) for r in fail],mesh=mesh,
        timing_note='Initial asymmetric instrumentation: candidate timer includes solve/audit and excludes constructor; legacy timer also includes new-audit PNP construction. Retain as run costs only; not comparable method speedup. Repeated matched construction-plus-method timing is in performance.json.')
    for label,relative in [('equilibrium_reference','results/references/independent_pb.json'),
                           ('learning','results/learning.json'),('performance','results/performance_summary.json'),
                           ('stability','results/gummel_stability.json'),
                           ('nonequilibrium_reference','results/noneq_reference/comparison.json'),
                           ('nonequilibrium_reference_audit','results/noneq_reference/array_audit.json')]:
        artifact=ROOT/relative
        if artifact.exists():stats[label]=dict(artifact=relative,sha256=sha256(artifact),values=json.loads(artifact.read_text()))
    stress=ROOT/'results/stress/campaign.json'
    if stress.exists():
        data=json.loads(stress.read_text());stats['stress']=dict(artifact=str(stress.relative_to(ROOT)),sha256=sha256(stress),
            evaluations=len(data['records']),passes=sum(r['status']=='PASS' for r in data['records']))
    write_json(ROOT/'results/statistics.json',stats)
    print(json.dumps({k:v for k,v in stats.items() if k not in ['mesh','legacy_failure_cases','equilibrium_reference','learning','performance','stability']},indent=2))


if __name__=='__main__':main()
