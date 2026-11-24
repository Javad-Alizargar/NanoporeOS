"""Conservative analytical qualification for the finite neutral cylinder only."""
import numpy as np
from .core import VT, ohmic_current

def neutral_qualification(s,status,current_A,fields):
    if s.wall_offset_mV!=0 or status!='PASS':return False
    reference=ohmic_current(s)
    if not np.isfinite(current_A) or abs(current_A-reference)>1e-18+abs(reference)*1e-8:return False
    potential=np.asarray(fields['potential_V'])
    expected=np.repeat((np.asarray(fields['x_nm'])/s.length_nm*s.voltage_mV*.001)[:,None],s.nr,axis=1)
    cp=np.asarray(fields['cplus_mol_m3']);cm=np.asarray(fields['cminus_mol_m3'])
    if any(a.shape!=(s.nx,s.nr) or not np.all(np.isfinite(a)) for a in [potential,cp,cm]):return False
    return bool(np.max(np.abs(potential-expected))/VT<=1e-8 and
                max(np.max(np.abs(cp/s.salt_mM-1)),np.max(np.abs(cm/s.salt_mM-1)))<=1e-8)

def qualified_finest(records,qualified_ids):
    """Never replace a declared finest target by a failed or unqualified record."""
    finest=max(records,key=lambda record:record['scenario']['nx'])
    if finest['case_id'] not in qualified_ids:
        raise ValueError('Frozen finest-mesh baseline target is not analytically qualified')
    return finest
