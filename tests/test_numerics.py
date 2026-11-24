import dataclasses
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nanoporeos.core import PNP,Scenario,bern,bern_prime,ohmic_current,VT,StopRequested,check_stop,write_json,sha256
from nanoporeos.datasets import save_result,load_result


class Numerics(unittest.TestCase):
    def test_bernoulli(self):
        x=np.array([-500.,-10.,-.001,0,.001,10.,500.])
        np.testing.assert_allclose(bern(-x)-bern(x),x,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(bern_prime(np.array([-1.,0.,1.])),
            (bern(np.array([-1.,0.,1.])+1e-6)-bern(np.array([-1.,0.,1.])-1e-6))/2e-6,atol=1e-9)

    def test_coupled_jacobian_independent_difference(self):
        p=PNP(Scenario(nx=6,nr=4))
        rng=np.random.default_rng(20261109);y=rng.normal(0,.1,3*p.n);v=rng.normal(size=3*p.n)
        _,jac=p.residual(y,.7,True)
        np.testing.assert_allclose(jac@v,(p.residual(y+1e-6*v,.7)-p.residual(y-1e-6*v,.7))/2e-6,rtol=1e-7,atol=2e-8)

    def test_analytical_ohmic(self):
        for voltage in [-100,0,100]:
            s=Scenario(nx=12,nr=6,wall_offset_mV=0,voltage_mV=voltage)
            p=PNP(s);out=p.solve()
            self.assertEqual(out['status'],'PASS')
            phi,up,um=out['y'].reshape(3,p.n)
            np.testing.assert_allclose(phi,np.repeat(p.x,s.nr)/s.length_nm*voltage*.001/VT,atol=1e-10)
            np.testing.assert_allclose(np.r_[up,um],0,atol=1e-10)
            self.assertLess(abs(out['diagnostics']['current_A']-ohmic_current(s)),1e-20)

    def test_equilibrium_boltzmann(self):
        p=PNP(Scenario(nx=16,nr=8,voltage_mV=0))
        out=p.solve();self.assertEqual(out['status'],'PASS')
        phi,up,um=out['y'].reshape(3,p.n)
        self.assertLess(np.max(np.abs(up+phi)),1e-8)
        self.assertLess(np.max(np.abs(um-phi)),1e-8)
        self.assertLess(abs(out['diagnostics']['current_A']),1e-18)

    def test_export_integrity(self):
        p=PNP(Scenario(nx=8,nr=4));out=p.solve()
        with tempfile.TemporaryDirectory() as temp:
            save_result(temp,p,out);summary,arrays=load_result(temp)
            np.testing.assert_array_equal(arrays['potential_V'],out['y'][:p.n].reshape(8,4)*VT)
            with (Path(temp)/'sections.csv').open('a') as f:f.write('corrupt')
            with self.assertRaises(ValueError):load_result(temp)

    def test_limits_and_strict_json(self):
        for values in [dict(nx=1000000),dict(nr=True),dict(salt_mM=float('nan')),dict(model='stokes')]:
            with self.assertRaises(ValueError):Scenario(**values)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):write_json(Path(temp)/'bad.json',dict(x=float('nan')))
        with self.assertRaises(StopRequested):check_stop(deadline=0)

    def test_forged_export_schema_rejected(self):
        import json
        p=PNP(Scenario(nx=8,nr=4));out=p.solve()
        for malformed in ['coordinates','concentrations']:
            with tempfile.TemporaryDirectory() as temp:
                save_result(temp,p,out);_,arrays=load_result(temp)
                if malformed=='coordinates':arrays['x_nm']=np.array([float('nan')])
                else:arrays['cplus_mol_m3'][0,0]=-1
                np.savez_compressed(Path(temp)/'fields.npz',**arrays)
                manifest=json.loads((Path(temp)/'manifest.json').read_text())
                manifest['files']['fields.npz']=sha256(Path(temp)/'fields.npz')
                write_json(Path(temp)/'manifest.json',manifest)
                with self.assertRaises(ValueError):load_result(temp)

    def test_stop_marker_and_deterministic_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'STOP_REQUESTED').write_text('owned test fixture')
            with patch('nanoporeos.core.ROOT',Path(temp)):
                with self.assertRaises(StopRequested):PNP(Scenario(nx=8,nr=4)).solve()
        p=PNP(Scenario(nx=8,nr=4));a=p.solve();b=p.solve()
        np.testing.assert_array_equal(a['y'],b['y'])


if __name__=='__main__':unittest.main()
