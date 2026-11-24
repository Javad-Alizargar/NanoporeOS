from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
import numpy as np
from independent_reference import Parameters,values,fields,numerical_forcing,partial_integral,numerical_axis_forcing

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=dict(status='PASS',checked_utc=datetime.now(timezone.utc).isoformat(),families=[],
        imports_canonical_assembly=False,paid_calls=0,physical_validation=False)
    for family in ['polynomial','oscillatory','localized']:
        p=Parameters(family)
        x=p.length_nm*np.array([.13,.29,.47,.71,.86])
        r=p.radius_nm*np.array([.15,.38,.57,.76,.89])
        exact=fields(p,x,r);simple=values(p,x,r)
        np.testing.assert_allclose(exact['psi'],simple['psi'],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(exact['q'],simple['q'],rtol=1e-13,atol=1e-13)
        errors=[]
        for denominator in [80,160,320,640]:
            approximate=numerical_forcing(p,x,r,1/denominator)
            errors.append((np.max(np.abs(approximate-exact['forcing']),axis=1)/
                np.maximum(np.max(np.abs(exact['forcing']),axis=1),1e-6)).tolist())
        errors=np.array(errors)
        orders=np.log2(errors[:-1]/errors[1:])
        assert np.all(errors[-1]<3e-3),(family,errors)
        assert np.all(orders[-1]>1.8),(family,orders)
        integral=partial_integral(p)
        assert max(integral['absolute_imbalance_nm'])<1e-9
        assert max(abs(np.array(integral['integrated_source_nm'])))>1e-3
        ends=fields(p,np.array([0.,p.length_nm]),np.array([.4,.7])*p.radius_nm)
        np.testing.assert_allclose(ends['q'],1,atol=1e-13)
        np.testing.assert_allclose(ends['psi'],[0,p.drive_VT],atol=1e-13)
        axis=fields(p,x,np.zeros_like(x));wall=fields(p,x,np.full_like(x,p.radius_nm))
        assert np.all(np.isfinite(axis['forcing']))
        axis_errors=[]
        for denominator in [160,320,640]:
            numerical=numerical_axis_forcing(p,x,1/denominator)
            axis_errors.append((np.max(abs(numerical-axis['forcing']),axis=1)/
                np.maximum(np.max(abs(axis['forcing']),axis=1),1e-6)).tolist())
        assert np.max(axis_errors[-1])<3e-3
        assert np.all(np.log2(np.array(axis_errors[-2])/axis_errors[-1])>1.8)
        np.testing.assert_allclose(axis['jr'],0,atol=1e-13)
        np.testing.assert_allclose(wall['jr'],0,atol=1e-13)
        np.testing.assert_allclose(wall['psi'],p.drive_VT*x/p.length_nm+
            p.wall_amplitude*np.sin(np.pi*x/p.length_nm)**2,atol=1e-13)
        report['families'].append(dict(family=family,relative_source_errors=errors.tolist(),
            source_halving_orders=orders.tolist(),partial_cylinder_integral=integral,
            axis_relative_source_errors=axis_errors,finite_axis_limit=True,compatible_boundaries=True))
    report['source_hashes']={p.name:digest(p) for p in [HERE/'independent_reference.py',Path(__file__)]}
    report['context_protocol_sha256']=digest(ROOT/'config/verification_v5a_protocol.json')
    (HERE/'reference_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
