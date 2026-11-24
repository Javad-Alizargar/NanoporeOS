"""Narrow analytical baseline with an explicit, tested abstention contract.

There is no qualified general charged-wall learned surrogate. The offline
learning exercise is separately recorded in results/learning.json.
"""
from .core import Scenario,ohmic_current


def predict_neutral(s: Scenario):
    domain=(s.length_nm==20 and s.radius_nm in [5,10] and s.salt_mM in [10,100]
        and s.voltage_mV in [-100,0,100] and s.wall_offset_mV==0)
    if not domain:return dict(status='OUTSIDE_VALIDATED_DOMAIN',prediction_A=None,
        reason='Abstention: only the enumerated neutral-wall analytic reference domain is qualified')
    return dict(status='PASS',prediction_A=ohmic_current(s),dataset_class='analytical_reference',
        experimental_validation='NOT_PERFORMED',learned_model=False)
