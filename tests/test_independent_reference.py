from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/independent'))
from nodal_pnp import NodalPNP,Problem,THERMAL_VOLTAGE


class IndependentReference(unittest.TestCase):
    def test_reference_jacobian(self):
        model=NodalPNP(Problem(nx=5,nr=4));rng=np.random.default_rng(20261109)
        y=rng.normal(0,.1,model.n);v=rng.normal(size=model.n)
        _,jac=model.equation(y,.5,True)
        np.testing.assert_allclose(jac@v,(model.equation(y+1e-6*v,.5)-model.equation(y-1e-6*v,.5))/2e-6,rtol=1e-6,atol=1e-8)

    def test_reference_neutral_fields(self):
        model=NodalPNP(Problem(nx=8,nr=6,wall_offset_mV=0));out=model.solve()
        self.assertEqual(out['status'],'PASS')
        expected=np.repeat((model.x/20*.1)[:,None],7,axis=1)
        np.testing.assert_allclose(out['potential_V'],expected,atol=1e-10)
        np.testing.assert_allclose(out['cplus_mol_m3'],100,atol=1e-8)
        np.testing.assert_allclose(out['cminus_mol_m3'],100,atol=1e-8)

    def test_reference_equilibrium_fields(self):
        model=NodalPNP(Problem(nx=8,nr=6,voltage_mV=0));out=model.solve()
        self.assertEqual(out['status'],'PASS')
        np.testing.assert_allclose(np.log(out['cplus_mol_m3']/100),-out['potential_V']/THERMAL_VOLTAGE,atol=1e-8)
        np.testing.assert_allclose(np.log(out['cminus_mol_m3']/100),out['potential_V']/THERMAL_VOLTAGE,atol=1e-8)


if __name__=='__main__':unittest.main()
