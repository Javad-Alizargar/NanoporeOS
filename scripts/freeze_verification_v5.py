"""Write once: prespecified distinct-family evaluation; never tune on outputs."""
from pathlib import Path
from datetime import datetime,timezone
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT/'config/verification_v5_protocol.json'
    if path.exists():raise RuntimeError('Frozen protocol exists; never overwrite it')
    families=['polynomial','oscillatory','localized']
    keys=['length_nm','radius_nm','salt_mM','drive_VT','amplitude_psi','bplus','bminus','wall_amplitude']
    tuples=[(16,5,15,1.8,.35,.25,-.15,.2),(22,9,55,-1.7,.75,-.35,.4,-.3),
        (34,12,110,2.4,-.8,.55,-.45,.6),(42,7,80,-2.6,1.,.7,.3,-.65),
        (19,11,35,.8,-.5,-.65,.5,.35),(28,4,160,-.9,.9,.4,-.6,-.45),
        (46,15,25,1.2,1.2,-.75,.65,.75),(36,8,190,-1.4,-1.1,.6,-.7,-.8)]
    cases=[dict(name=f'v5_{family}_{i+1:02}',family=family,**dict(zip(keys,t)))
        for family in families for i,t in enumerate(tuples)]
    development=[dict(name=f'v5_development_{family}',family=family,
        **dict(zip(keys,(20,7,30,2,.4,.25,-.2,.2)))) for family in families]
    payload=dict(version='verification-v5-distinct-families-matched-estimators',output_family='verification_v5',
        declared_utc=datetime.now(timezone.utc).isoformat(),status='FROZEN_BEFORE_ANY_V5_PDE_OUTPUT',
        purpose='Test whether adding field criteria changes acceptance when current estimators are matched',
        design_history='v3/v4 outcomes and matched ablation are exposed prior evidence; no v5 functions/tuples have been solved before freeze',
        model='axisym-forced-pnp-distinct-families-v1',legacy_core_changes=False,
        family_definitions=dict(polynomial='P=16u²(1-u)² H2; M=16u²(1-u)²(2u-1)H3',
            oscillatory='P=sin(3piu)H2; M=sin(4piu)H3',
            localized='P=4u(1-u)exp(-36(u-.35)²)H2; M=4u(1-u)(2u-1)exp(-25(u-.65)²)H3'),
        common_definition='u=x/L,Hm=(1-(r/R)²)^m; W=w sin²(piu); psi=vu+aP+W; logq_z=b_z M-zaP',
        development_cases=development,held_out_cases=cases,mesh_ladder=[[12,6],[24,12],[48,24],[96,48],[192,96]],
        exact_evaluation_tolerances=dict(max_phi_error_VT=.005,max_plus_relative_error=.01,
            max_minus_relative_error=.01,mid_current_relative_error=.01),current_relative_denominator_floor_A=1e-15,
        order_verification=dict(cases='one prespecified development case per new family',
            observables=['max_phi_error_VT','max_plus_relative_error','max_minus_relative_error'],last_pair_order_interval=[1.7,2.3]),
        policies=dict(residual_only='First strict discrete PASS',
            current_matched='First pair with 2/3 successive current difference <=1%',
            field_matched='Same current rule plus potential and both concentration criteria at same 2/3 scaling',
            uniform_finest='Direct192x96 discrete PASS'),
        controller_assumed_order=2,controller_safety_factor=2,controller_consecutive_pairs=1,
        primary_outcomes=['accurate/inaccurate/unresolved counts overall and per family',
            'paired identities of inaccurate acceptance prevented by fields and accurate cases lost/deferred',
            'cumulative visited cells including failures and deferred cases'],
        secondary_outcomes=['cached solve-plus-own-selection replay seconds; no general runtime advantage claim'],
        cost_estimand='Cached-grid replay; construction+solve plus own selection; I/O/scoring/plotting excluded',
        evaluation_separation='Freeze numerical decisions to file before exact scoring; no exact fields in policy inputs',
        failure_handling='Retain all failures; failed development order gate stops dependent evaluation; no replacement, budget changes or controller retuning',
        statistical_unit='24 fixed family/tuple combinations; same8 physical/amplitude tuples across3 families, not24 independent population samples',
        sample_size_reason='Bounded deterministic coverage8tuples per3structurally different smooth profiles; no statistical power/population reliability claim',
        timing_followup=dict(status='PRESPECIFIED_SECONDARY_IF_TIME_PERMITS',repetitions=3,seed=20261121,
            cases=[f'v5_{f}_{i:02}' for f in families for i in [1,8]],
            method='randomize case-policy order per repeat; each policy performs its own new solves and stops numerically; no cached solve reuse',
            warmup='one excluded12x6 polynomial solve; report attempted wall time and coverage, no equal-accuracy speedup'),
        scope_limits=['smooth forced verification only','not physical validation or charged-wall HOLD release',
            'all families share one field parameterization and canonical discretization','heuristic order/safety; shared bias can evade comparison',
            'not external preregistration or human blinding'],maximum_campaign_minutes=25,threads=2)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(path)

if __name__=='__main__':main()
