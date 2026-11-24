from pathlib import Path
import sys,unittest
import numpy as np
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'benchmarks/independent'))
from nodal_pnp import NodalPNP,Problem
from nodal_pnp_ordered import NodalPNP as OrderedPNP,Problem as OrderedProblem

class ReferenceOrderingTests(unittest.TestCase):
    def test_factorization_ordering_preserves_equations_and_converged_fields(self):
        for voltage in [-100.,100.]:
            problem=Problem(nx=16,nr=8,voltage_mV=voltage)
            original=NodalPNP(problem);ordered=OrderedPNP(OrderedProblem(**asdict(problem)))
            y=np.random.default_rng(71).normal(0,.02,original.n)
            a,ja=original.equation(y,.25,True);b,jb=ordered.equation(y,.25,True)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ja.toarray(),jb.toarray())
            first=original.solve(seconds=20);second=ordered.solve(seconds=20)
            self.assertEqual(first['status'],'PASS');self.assertEqual(second['status'],'PASS')
            for key in ['potential_V','cplus_mol_m3','cminus_mol_m3']:
                np.testing.assert_allclose(first[key],second[key],rtol=1e-11,atol=1e-11)
            self.assertLess(second['audit']['coupled_scaled_residual_max'],1e-10)

if __name__=='__main__':unittest.main()
