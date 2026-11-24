"""Save a test result bound to the engine and test-source bytes."""
from pathlib import Path
import json
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import sha256,write_json


def main():
    files=sorted([*ROOT.glob('src/nanoporeos/*.py'),*ROOT.glob('tests/*.py'),*ROOT.glob('benchmarks/**/*.py'),
        ROOT/'scripts/verification_v3.py',ROOT/'scripts/analyze_verification.py',
        ROOT/'scripts/verification_v5.py',ROOT/'scripts/analyze_verification_v5.py',ROOT/'scripts/time_verification_v5.py',
        ROOT/'scripts/driven_refinement_v2.py',ROOT/'scripts/driven_refinement_ordered.py',ROOT/'scripts/analyze_driven_refinement_v2.py',
        ROOT/'scripts/audit_verification_v5_timing.py'])
    sources={p.relative_to(ROOT).as_posix():sha256(p) for p in files}
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/tests_final.txt').open('w',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_hashes=sources,
        log='logs/tests_final.txt',log_sha256=sha256(ROOT/'logs/tests_final.txt'))
    write_json(ROOT/'reports/tests.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
