import unittest
from nanoporeos.core import Scenario
from nanoporeos.surrogates import predict_neutral


class Domain(unittest.TestCase):
    def test_unsupported_predictions_abstain(self):
        for kwargs in [dict(wall_offset_mV=-15),dict(wall_offset_mV=0,salt_mM=150),
                       dict(wall_offset_mV=0,radius_nm=7),dict(wall_offset_mV=0,length_nm=21)]:
            result=predict_neutral(Scenario(**kwargs))
            self.assertEqual(result['status'],'OUTSIDE_VALIDATED_DOMAIN')
            self.assertIsNone(result['prediction_A'])
        self.assertEqual(predict_neutral(Scenario(wall_offset_mV=0))['dataset_class'],'analytical_reference')


if __name__=='__main__':unittest.main()
