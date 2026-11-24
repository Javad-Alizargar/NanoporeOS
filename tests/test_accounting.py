from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from nanoporeos.accounting import Ledger,Policy,AdmissionDenied

PRICE=dict(verified=True,source='local_mock_tariff_not_a_real_provider',date='2026-11-09',
    bounded_token_only_tariff=True,input_USD_per_million='1',output_USD_per_million='1')


class Accounting(unittest.TestCase):
    def reserve(self,l,provider='mock',task='task',tokens=200000,review=False):
        return l.reserve(provider,'mock-model','run',task,tokens,1,token_count_verified=True,
            output_limit_supported=True,final_review=review)

    def test_disabled_unknown_unpriced(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger=Ledger(Path(temp)/'ledger.sqlite')
            with self.assertRaises(AdmissionDenied):self.reserve(ledger)
            ledger=Ledger(ledger.path,Policy(enabled=True,historical_usage_reconciled=True))
            with self.assertRaises(AdmissionDenied):self.reserve(ledger)
            self.assertEqual(ledger.state()['requests'],0)

    def test_atomic_reservations_restart_timeout_and_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            policy=replace(Policy(),enabled=True,historical_usage_reconciled=True,cap_micro_USD=1_000_000,reserve_micro_USD=500_000)
            path=Path(temp)/'ledger.sqlite';models={('mock','mock-model'):PRICE}
            l=Ledger(path,policy,models)
            first=self.reserve(l,tokens=250000-1);l.unresolved(first)
            restarted=Ledger(path,policy,models)
            self.assertEqual(restarted.state()['accounted_micro_USD_since_ledger_initialization'],250000)
            def attempt(i):
                try:self.reserve(Ledger(path,policy,models),task=str(i),tokens=200000);return True
                except AdmissionDenied:return False
            with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(attempt,[1,2]))
            self.assertEqual(sum(results),1)
            with self.assertRaises(AdmissionDenied):self.reserve(l,tokens=200000)
            self.reserve(l,tokens=200000,review=True)
            self.assertEqual(l.state()['unresolved_or_reserved'],3)

    def test_openai_cap_retries_and_overrun_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'ledger.sqlite'
            p=replace(Policy(),enabled=True,historical_usage_reconciled=True,openai_micro_USD=250000)
            l=Ledger(path,p,{('openai','mock-model'):PRICE})
            rid=self.reserve(l,'openai',tokens=200000);l.settle(rid,200000,1)
            with self.assertRaises(AdmissionDenied):self.reserve(l,'openai',tokens=60000)
            with self.assertRaises(AdmissionDenied):l.settle(rid,0,0)
            # An actual provider overrun is recorded, not discarded/refunded.
            rid=l.reserve('openai','mock-model','run','review',100,1,token_count_verified=True,output_limit_supported=True)
            l.settle(rid,200000,1)
            with self.assertRaises(AdmissionDenied):self.reserve(l,'openai',tokens=1)
            self.assertGreater(l.state()['accounted_micro_USD_since_ledger_initialization'],400000)


if __name__=='__main__':unittest.main()
