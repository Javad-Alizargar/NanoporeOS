from pathlib import Path
import dataclasses
import json
import sys
import numpy as np
from nodal_pnp import NodalPNP,Problem,THERMAL_VOLTAGE,FARADAY,DIFFUSIVITIES


def checks():
    m=NodalPNP(Problem(nx=6,nr=4));rng=np.random.default_rng(20261109)
    y=rng.normal(0,.1,m.n);v=rng.normal(size=m.n)
    _,jac=m.equation(y,.6,True)
    fd=(m.equation(y+1e-6*v,.6)-m.equation(y-1e-6*v,.6))/(2e-6)
    jac_error=float(np.max(np.abs(jac@v-fd)))
    if jac_error>1e-7:raise AssertionError(('Jacobian error',jac_error))
    records=[]
    for voltage,offset in [(100,0),(0,-15),(-100,0)]:
        s=Problem(nx=12,nr=8,voltage_mV=voltage,wall_offset_mV=offset)
        p=NodalPNP(s);o=p.solve()
        if o['status']!='PASS':raise AssertionError(o)
        record=dict(voltage_mV=voltage,wall_offset_mV=offset,audit=o['audit'])
        if offset==0:
            conductivity=FARADAY*sum(DIFFUSIVITIES)*s.salt_mM/THERMAL_VOLTAGE
            reference=-conductivity*np.pi*(s.radius_nm*1e-9)**2/(s.length_nm*1e-9)*s.voltage_mV*.001
            error=float(abs(o['audit']['current_A']-reference)/abs(reference))
            if error>1e-9:raise AssertionError(error)
            record['ohmic_relative_error']=error
            if max(np.max(abs(o['cplus_mol_m3']/s.salt_mM-1)),np.max(abs(o['cminus_mol_m3']/s.salt_mM-1)))>1e-8:raise AssertionError('Neutral density not uniform')
        else:
            phi=o['potential_V']/THERMAL_VOLTAGE
            error=float(max(np.max(np.abs(np.log(o['cplus_mol_m3']/s.salt_mM)+phi)),np.max(np.abs(np.log(o['cminus_mol_m3']/s.salt_mM)-phi))))
            if error>1e-8:raise AssertionError(error)
            record['boltzmann_max_log_error']=error
        records.append(record)
    output=dict(status='PASS',jacobian_max_absolute_error=jac_error,records=records)
    Path(__file__).with_name('checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',jacobian_max_absolute_error=jac_error),indent=2))


if __name__=='__main__':checks()
