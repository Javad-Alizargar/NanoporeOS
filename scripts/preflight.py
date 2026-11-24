"""Read-only project preflight; never outputs helper source or key values."""
from pathlib import Path
import ast
import importlib.metadata
import json
import platform
import re
import sys
import zipfile
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nanoporeos.core import write_json,sha256,environment


def main():
    helpers=[]
    for p in (ROOT/'apis').glob('*.py'):
        raw=p.read_text(encoding='utf-8-sig',errors='replace')
        try:
            tree=ast.parse(raw)
            calls=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)})
            imports=sorted({n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)} |
                           {a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names})
            hosts=sorted({urlsplit(n.value).hostname for n in ast.walk(tree) if isinstance(n,ast.Constant)
                          and isinstance(n.value,str) and n.value.startswith(('https://','http://'))
                          and urlsplit(n.value).hostname})
            helpers.append(dict(file=p.name,sha256=sha256(p),imports=imports,calls=calls,hosts=hosts,
                execution='DISABLED_UNAUDITED',note='AST inventory only; no permission or security certification'))
        except (SyntaxError,ValueError):
            helpers.append(dict(file=p.name,status='STATIC_PARSE_FAILED',execution='DISABLED_UNAUDITED'))
    archives=[]
    for p in (ROOT/'handoff/legacy/review_packages').glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            records=[]
            for name in z.namelist():
                if re.search(r'(budget|spend|ledger|cost|account)',name,re.I):
                    records.append(name)
            archives.append(dict(file=p.name,members=len(z.namelist()),possible_accounting_members=records))
    payload=dict(environment=environment(),helpers=helpers,archives=archives,
        credential_presence={name:(ROOT/'apis'/name).is_file() for name in ['chatgpt.txt','deepseek.txt','NCBI.txt',
            'OPENALEX.txt','GITHUB_fine_grained 2.txt','Open Access API.txt','Meta API.txt']},
        accounting=dict(historical_paid_usage='UNKNOWN',additional_metered_API_calls_this_run=0,
            additional_metered_API_cost_USD=0,paid_calls_enabled=False,
            reason='Historical accounting unreconciled; no verified model pricing or full admission gateway',
            codex_subscription_usage='not observable or controlled by local project ledger'),
        dependency_versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','matplotlib','python-docx']})
    write_json(ROOT/'reports/preflight.json',payload)
    ledger=ROOT/'orchestration/spending_state.json'
    if not ledger.exists():write_json(ledger,payload['accounting'])
    print(json.dumps(dict(status='SAVED',helpers_kept_disabled=len(helpers),archives=len(archives),
        existing_queue_files=[str(p.relative_to(ROOT)) for p in (ROOT/'orchestration').glob('*')]),indent=2))


if __name__=='__main__':main()
