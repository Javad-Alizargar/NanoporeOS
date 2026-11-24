from dataclasses import replace
from pathlib import Path
import sys
import unittest
import tempfile
import time
import json
import numpy as np
from nanoporeos.manufactured import ManufacturedCase,ManufacturedPNP,continuous
from nanoporeos.accuracy import Snapshot,select

class ManufacturedVerificationTests(unittest.TestCase):
    def test_forcing_mapping_matches_disjoint_continuous_reference(self):
        independent=Path(__file__).resolve().parents[1]/'benchmarks/manufactured'
        sys.path.insert(0,str(independent))
        try:
            from manufactured_math import Parameters,fields
            from check_derivatives import integral_checks
            c=ManufacturedCase('independent_mapping');p=Parameters(length_nm=c.length_nm,radius_nm=c.radius_nm,
                bulk_mol_m3=c.salt_mM,applied_thermal_voltages=c.drive_VT,potential_amplitude=c.amplitude_psi,
                plus_mu_amplitude=c.bplus,minus_mu_amplitude=c.bminus,wall_amplitude=c.wall_amplitude)
            x=c.length_nm*np.array([.13,.37,.71]);r=c.radius_nm*np.array([0.,.4,.8])
            ours=continuous(c,x,r);reference=fields(x,r,p)
            expected=np.array([reference['poisson_forcing_nm2'],reference['plus']['forcing_nm2'],reference['minus']['forcing_nm2']])
            np.testing.assert_allclose(ours['forcing'],expected,rtol=1e-12,atol=1e-13)
            # A nonzero source integral prevents a zero-sum sign error escaping.
            balance=integral_checks(p,48,.37)
            self.assertGreater(abs(balance['plus']['integrated_source_nm']),1e-3)
            self.assertLess(balance['plus']['absolute_imbalance_nm'],1e-10)
        finally:sys.path.remove(str(independent))

    def test_source_aware_balance_and_refinement(self):
        case=ManufacturedCase('refinement')
        errors=[]
        for nx,nr in [(12,6),(24,12),(48,24)]:
            model=ManufacturedPNP(case,nx,nr);solved=model.solve()
            self.assertEqual(solved['status'],'PASS');self.assertFalse(solved['training_qualified'])
            self.assertEqual(solved['model'],'axisym-forced-pnp-manufactured-v1')
            audit=solved['diagnostics']
            self.assertGreater(audit['current_spread_A'],1e-14)
            self.assertLess(audit['source_corrected_current_spread_A'],1e-18)
            self.assertLess(max(audit['species_max_local_imbalance_mol_s']),1e-20)
            errors.append(model.exact_errors(solved['y'],model.mid_current(solved))['max_phi_error_VT'])
        self.assertTrue(all(b<a/2 for a,b in zip(errors,errors[1:])))

    def test_forced_identity_includes_amplitudes(self):
        c=ManufacturedCase('identity')
        a=ManufacturedPNP(c,12,6);b=ManufacturedPNP(replace(c,wall_amplitude=-.3),12,6)
        self.assertNotEqual(a.identity(),b.identity())
        self.assertNotEqual(a.identity(),a.s.identity())

    def test_hash_valid_but_semantically_wrong_cache_is_rejected(self):
        scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
        try:
            from verification_v3 import one
            from nanoporeos.core import write_json,sha256
            case=ManufacturedCase('cache_guard')
            with tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                snapshot,saved,model=one(case,[12,6],directory,time.monotonic()+30,'test_protocol')
                parent=directory/'cases'/saved['identity'];saved['scenario']['model']='unforced_mislabeled'
                write_json(parent/'result.json',saved)
                manifest=json.loads((parent/'manifest.json').read_text());manifest['files']['result.json']=sha256(parent/'result.json')
                write_json(parent/'manifest.json',manifest)
                with self.assertRaisesRegex(ValueError,'identity/schema mismatch'):
                    one(case,[12,6],directory,time.monotonic()+30,'test_protocol')
        finally:sys.path.remove(str(scripts))

class AccuracyEvidenceTests(unittest.TestCase):
    budgets=dict(max_phi_error_VT=.005,max_plus_relative_error=.01,max_minus_relative_error=.01,mid_current_relative_error=.01)

    def snapshots(self):
        return [Snapshot('same_problem','verification_model',(np.arange(nx)+.5)/nx,
            (np.arange(nr)+.5)/nr,np.array([np.full((nx,nr),.1),np.zeros((nx,nr)),np.zeros((nx,nr))]),1e-10,'PASS',.1)
            for nx,nr in [(12,6),(24,12),(48,24),(96,48)]]

    def test_mesh_invariant_bias_is_not_a_certified_bound(self):
        decision=select('field_controller',self.snapshots(),self.budgets)
        self.assertEqual(decision['status'],'ESTIMATED_ACCEPT')
        self.assertFalse(decision['certified_error_bound'])
        self.assertFalse(decision['training_qualified'])

    def test_failed_mesh_breaks_consecutive_evidence(self):
        snapshots=self.snapshots();snapshots[1].status='FAIL'
        self.assertEqual(select('field_controller',snapshots,self.budgets)['status'],'UNRESOLVED')

    def test_cross_problem_comparison_is_rejected(self):
        snapshots=self.snapshots();snapshots[-1].problem_id='different_boundary_problem'
        with self.assertRaises(ValueError):select('field_controller',snapshots,self.budgets)

if __name__=='__main__':unittest.main()
