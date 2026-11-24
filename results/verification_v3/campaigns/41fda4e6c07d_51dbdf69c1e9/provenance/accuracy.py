"""Numerical-only mesh selection. No exact solution or physical qualification.

The field estimator assumes smooth second-order error. It is a heuristic,
not a certified bound, and is deliberately scored against held-out exact tests.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator

@dataclass
class Snapshot:
    problem_id: str
    model: str
    x: np.ndarray
    r: np.ndarray
    y: np.ndarray
    current_A: float
    status: str
    seconds: float

    @property
    def grid(self):return [len(self.x),len(self.r)]

def differences(coarse,fine):
    if coarse.problem_id!=fine.problem_id or coarse.model!=fine.model:
        raise ValueError('Cannot compare different problems or model identities')
    if any(a>=b for a,b in zip(coarse.grid,fine.grid)):raise ValueError('Both axes must refine')
    for s in [coarse,fine]:
        if s.status!='PASS':raise ValueError('A failed solve cannot support refinement acceptance')
        if np.asarray(s.y).shape!=(3,*s.grid) or not np.all(np.isfinite(s.y)):
            raise ValueError('Invalid numerical snapshot')
        if not np.isfinite(s.current_A):raise ValueError('Nonfinite observable')
        if not all(np.all(np.diff(a)>0) for a in [s.x,s.r]):raise ValueError('Coordinates must increase')
    xx,rr=np.meshgrid(coarse.x,coarse.r,indexing='ij');points=np.c_[xx.ravel(),rr.ravel()]
    interpolated=np.array([RegularGridInterpolator((fine.x,fine.r),field)(points).reshape(coarse.grid) for field in fine.y])
    return dict(max_phi_error_VT=float(np.max(np.abs(coarse.y[0]-interpolated[0]))),
        max_plus_relative_error=float(np.max(np.abs(np.expm1(coarse.y[1]-interpolated[1])))),
        max_minus_relative_error=float(np.max(np.abs(np.expm1(coarse.y[2]-interpolated[2])))),
        mid_current_relative_error=float(abs(coarse.current_A-fine.current_A)/max(abs(fine.current_A),1e-15)))

def select(policy,snapshots,budgets,assumed_order=2,safety=2,consecutive=2):
    """Accepts only numerical snapshots and frozen budgets, never analytic truth."""
    if policy not in ['residual_only','current_mesh','field_controller','uniform_finest']:raise ValueError('Unknown policy')
    if not snapshots:raise ValueError('No attempted meshes')
    if len({(s.problem_id,s.model) for s in snapshots})!=1:raise ValueError('Cannot mix problem identities')
    if not all(np.isfinite(v) and v>0 for v in budgets.values()):raise ValueError('Positive finite budgets required')
    if policy=='field_controller':
        if not np.isfinite(assumed_order) or assumed_order<=0 or not np.isfinite(safety) or safety<1:
            raise ValueError('Invalid error-estimator assumptions')
        if type(consecutive) is not int or consecutive<1:raise ValueError('Invalid evidence length')
        if any(f.grid!=[2*n for n in c.grid] for c,f in zip(snapshots,snapshots[1:])):
            raise ValueError('Frozen estimator requires twofold refinement on both axes')
    visited=[];indicators=[];streak=0;selected=None
    if policy=='uniform_finest':
        visited=[len(snapshots)-1]
        if snapshots[-1].status=='PASS':selected=visited[0]
    else:
        for i,snapshot in enumerate(snapshots):
            visited.append(i)
            if policy=='residual_only':
                if snapshot.status=='PASS':selected=i;break
                continue
            if i==0:continue
            if snapshot.status!='PASS' or snapshots[i-1].status!='PASS':streak=0;continue
            if policy=='current_mesh':
                delta={'mid_current_relative_error':float(abs(snapshots[i-1].current_A-snapshot.current_A)/max(abs(snapshot.current_A),1e-15))}
                passed=delta['mid_current_relative_error']<=budgets['mid_current_relative_error']
                estimated={'mid_current_relative_error':delta['mid_current_relative_error']}
            else:
                delta=differences(snapshots[i-1],snapshot)
                factor=safety/(2**assumed_order-1)
                estimated={k:v*factor for k,v in delta.items()}
                passed=all(estimated[k]<=budgets[k] for k in budgets)
            streak=streak+1 if passed else 0
            indicators.append(dict(coarse_grid=snapshots[i-1].grid,fine_grid=snapshot.grid,
                raw_differences=delta,estimated_errors=estimated,pair_pass=bool(passed),consecutive_passes=streak))
            if passed and (policy=='current_mesh' or streak>=consecutive):selected=i;break
    return dict(policy=policy,status='ESTIMATED_ACCEPT' if selected is not None else 'UNRESOLVED',
        selected_index=selected,selected_grid=None if selected is None else snapshots[selected].grid,
        visited_indices=visited,attempted_solve_seconds=float(sum(snapshots[i].seconds for i in visited)),
        attempted_cells=sum(np.prod(snapshots[i].grid).item() for i in visited),indicators=indicators,
        reference_access='NONE_NUMERICAL_INPUTS_ONLY',certified_error_bound=False,training_qualified=False)
