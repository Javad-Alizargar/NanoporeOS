from dataclasses import replace
import unittest
from pathlib import Path
import sys
import numpy as np
from nanoporeos.manufactured_families import FamilyCase,FamilyPNP,continuous,FAMILIES
from nanoporeos.acceptance_matched import select
from nanoporeos.accuracy import Snapshot

class DistinctFamilyTests(unittest.TestCase):
    def test_forcing_and_flux_against_independent_symbolic_derivatives(self):
        independent=Path(__file__).resolve().parents[1]/'benchmarks/manufactured_families'
        sys.path.insert(0,str(independent))
        try:
            from independent_reference import fields,parameter_mapping
            for family in FAMILIES:
                c=FamilyCase('symbolic_check',family=family)
                x=c.length_nm*np.array([0.,.13,.39,.71,1.]);r=c.radius_nm*np.array([0.,.3,.8,1.,.5])
                ours=continuous(c,x,r);reference=fields(parameter_mapping(c),x,r)
                for key in ['forcing','jx','jr','psi']:np.testing.assert_allclose(ours[key],reference[key],rtol=2e-12,atol=2e-13)
        finally:sys.path.remove(str(independent))

    def test_family_identity_and_compatible_boundaries(self):
        identities=[]
        for family in FAMILIES:
            c=FamilyCase('boundaries',family=family);r=np.linspace(0,c.radius_nm,7)
            for x in [0,c.length_nm]:
                fields=continuous(c,x,r)
                np.testing.assert_allclose(fields['cp'],1,atol=1e-14)
                np.testing.assert_allclose(fields['cm'],1,atol=1e-14)
                np.testing.assert_allclose(fields['psi'],c.drive_VT*x/c.length_nm,atol=1e-14)
            x=np.linspace(0,c.length_nm,9)
            np.testing.assert_allclose(continuous(c,x,0)['jr'],0,atol=1e-14)
            np.testing.assert_allclose(continuous(c,x,c.radius_nm)['jr'],0,atol=1e-14)
            identities.append(FamilyPNP(c,12,6).identity())
        self.assertEqual(len(set(identities)),3)
        with self.assertRaises(ValueError):FamilyCase('invalid',family='not_a_family')

    def test_source_balance_for_distinct_shapes(self):
        for family in FAMILIES:
            model=FamilyPNP(FamilyCase('source_balance',family=family),12,6)
            result=model.solve();self.assertEqual(result['status'],'PASS')
            self.assertFalse(result['training_qualified']);self.assertTrue(result['verification_only'])
            self.assertEqual(result['scenario']['model'],result['model'])
            self.assertLess(result['diagnostics']['source_corrected_current_spread_A'],1e-18)
            self.assertLess(max(result['diagnostics']['species_max_local_imbalance_mol_s']),1e-20)

class MatchedAcceptanceTests(unittest.TestCase):
    budgets=dict(max_phi_error_VT=.005,max_plus_relative_error=.01,max_minus_relative_error=.01,mid_current_relative_error=.01)

    def test_identical_current_scaling_and_extra_field_rejection(self):
        snapshots=[]
        for i,(nx,nr) in enumerate([(12,6),(24,12)]):
            # Current difference1.2%: both shared2/3 estimators pass1%; large
            # field discrepancy must make only the field-aware policy abstain.
            snapshots.append(Snapshot('same','forced',(np.arange(nx)+.5)/nx,(np.arange(nr)+.5)/nr,
                np.array([np.full((nx,nr),.1*(1-i)),np.zeros((nx,nr)),np.zeros((nx,nr))]),
                1.012e-10 if i==0 else 1e-10,'PASS',.1))
        current=select('current_matched',snapshots,self.budgets)
        field=select('field_matched',snapshots,self.budgets)
        self.assertEqual(current['selected_index'],1);self.assertIsNone(field['selected_index'])
        self.assertAlmostEqual(current['indicators'][0]['estimated_errors']['mid_current_relative_error'],
            field['indicators'][0]['estimated_errors']['mid_current_relative_error'])
        self.assertFalse(current['certified_error_bound'])

    def test_failed_mesh_and_inconsistent_refinement_rejected(self):
        snapshots=[]
        for nx,nr,status in [(12,6,'FAIL'),(24,12,'PASS')]:
            snapshots.append(Snapshot('same','forced',(np.arange(nx)+.5)/nx,(np.arange(nr)+.5)/nr,
                np.zeros((3,nx,nr)),1e-10,status,.1))
        for policy in ['current_matched','field_matched']:
            self.assertIsNone(select(policy,snapshots,self.budgets)['selected_index'])
            with self.assertRaises(ValueError):select(policy,snapshots,self.budgets,consecutive=2)

if __name__=='__main__':unittest.main()
