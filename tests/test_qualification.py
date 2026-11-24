import unittest
import numpy as np
from nanoporeos.core import Scenario,ohmic_current,VT
from nanoporeos.qualification import neutral_qualification,qualified_finest

class QualificationTests(unittest.TestCase):
    def test_correct_current_cannot_hide_wrong_local_field(self):
        s=Scenario(nx=16,nr=8,wall_offset_mV=0)
        x=(np.arange(s.nx)+.5)*s.length_nm/s.nx
        arrays=dict(x_nm=x,potential_V=np.repeat((x/s.length_nm*s.voltage_mV*.001)[:,None],s.nr,axis=1),
            cplus_mol_m3=np.full((s.nx,s.nr),s.salt_mM),cminus_mol_m3=np.full((s.nx,s.nr),s.salt_mM))
        self.assertTrue(neutral_qualification(s,'PASS',ohmic_current(s),arrays))
        arrays['potential_V'][5,3]+=.01*VT
        self.assertFalse(neutral_qualification(s,'PASS',ohmic_current(s),arrays))
        arrays['potential_V'][5,3]-=.01*VT;arrays['cplus_mol_m3'][6,4]*=1.01
        self.assertFalse(neutral_qualification(s,'PASS',ohmic_current(s),arrays))

    def test_failed_finest_target_is_never_silently_selected(self):
        records=[dict(case_id='coarse',scenario=dict(nx=16)),dict(case_id='fine',scenario=dict(nx=36))]
        with self.assertRaisesRegex(ValueError,'not analytically qualified'):qualified_finest(records,{'coarse'})
        self.assertEqual(qualified_finest(records,{'coarse','fine'})['case_id'],'fine')

    def test_failure_and_charged_domain_abstain_before_array_use(self):
        self.assertFalse(neutral_qualification(Scenario(wall_offset_mV=0),'FAIL',0,{}))
        self.assertFalse(neutral_qualification(Scenario(wall_offset_mV=-15),'PASS',0,{}))

if __name__=='__main__':unittest.main()
