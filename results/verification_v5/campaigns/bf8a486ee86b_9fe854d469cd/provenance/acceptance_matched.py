"""Matched smooth-error assumptions isolate field checks in an offline study."""
import numpy as np
from .accuracy import select as original_select


def select(policy,snapshots,budgets,assumed_order=2,safety=2,consecutive=1):
    if policy in ['residual_only','uniform_finest']:
        return original_select(policy,snapshots,budgets,assumed_order,safety,consecutive)
    if policy not in ['current_matched','field_matched']:raise ValueError('Unknown matched policy')
    if not np.isfinite(assumed_order) or assumed_order<=0 or not np.isfinite(safety) or safety<1:
        raise ValueError('Invalid shared estimator assumptions')
    if consecutive!=1:raise ValueError('Matched study fixes one passing pair')
    if any(f.grid!=[2*n for n in c.grid] for c,f in zip(snapshots,snapshots[1:])):
        raise ValueError('Matched estimator requires twofold refinement on both axes')
    factor=safety/(2**assumed_order-1)
    if policy=='field_matched':
        decision=original_select('field_controller',snapshots,budgets,assumed_order,safety,consecutive)
    else:
        current_budget=dict(budgets);current_budget['mid_current_relative_error']/=factor
        decision=original_select('current_mesh',snapshots,current_budget,assumed_order,safety,consecutive)
        for row in decision['indicators']:
            row['estimated_errors']={k:factor*v for k,v in row['raw_differences'].items()}
    decision.update(policy=policy,shared_estimator_factor=factor,
        comparison='Same grids/order/safety/current budget/one-pair evidence; field criterion is the added component')
    return decision
