"""Transactional admission accounting for future explicitly priced API clients.

No provider client or gateway is implemented. Default production admission is
disabled and historical spending is unknown. This cannot meter Codex sessions,
hidden billing or adversarial same-user code that bypasses the module.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal,ROUND_CEILING
from pathlib import Path
import json
import sqlite3
import time
import uuid


class AdmissionDenied(RuntimeError):pass


@dataclass(frozen=True)
class Policy:
    enabled: bool = False
    historical_usage_reconciled: bool = False
    cap_micro_USD: int = 5_000_000
    openai_micro_USD: int = 1_000_000
    reserve_micro_USD: int = 500_000
    per_request_micro_USD: int = 250_000
    max_concurrent: int = 3
    max_run_attempts: int = 80
    max_task_attempts: int = 12


class Ledger:
    def __init__(self,path,policy=Policy(),models=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.policy=policy;self.models=models or {}
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, provider TEXT,model TEXT,run_id TEXT,task_id TEXT,status TEXT,reserved INTEGER,cost INTEGER,price TEXT,usage TEXT,created REAL)')
            c.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
            c.execute("INSERT OR IGNORE INTO metadata VALUES('historical_usage','UNKNOWN')")
            c.execute("INSERT OR IGNORE INTO metadata VALUES('disabled_due_to_overrun','false')")

    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.path,timeout=10)
        try:
            with c:yield c
        finally:c.close()

    @staticmethod
    def cost(price,inputs,outputs):
        if type(inputs) is not int or type(outputs) is not int or min(inputs,outputs)<0:
            raise AdmissionDenied('Verified nonnegative integer token bounds required')
        # USD/million tokens numerically equals micro-USD/token.
        value=Decimal(str(price['input_USD_per_million']))*inputs+Decimal(str(price['output_USD_per_million']))*outputs
        if not value.is_finite() or value<0:raise AdmissionDenied('Invalid tariff')
        return int(value.to_integral_value(rounding=ROUND_CEILING))

    def reserve(self,provider,model,run_id,task_id,input_tokens,output_bound,*,
                token_count_verified=False,output_limit_supported=False,final_review=False,critical=False):
        p=self.policy
        if not p.enabled or not p.historical_usage_reconciled:
            raise AdmissionDenied('Paid calls disabled: historical accounting/pricing/admission prerequisites unresolved')
        price=self.models.get((provider,model))
        if not price or not all(price.get(k) for k in ['verified','source','date','bounded_token_only_tariff']):
            raise AdmissionDenied('Unpriced/unverified model or unbounded tool charges')
        if not token_count_verified or not output_limit_supported:raise AdmissionDenied('Unsupported input/output bound')
        reservation=self.cost(price,input_tokens,output_bound)
        if not 0<reservation<=p.per_request_micro_USD:raise AdmissionDenied('Per-request cap')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute("SELECT value FROM metadata WHERE key='disabled_due_to_overrun'").fetchone()[0]=='true':
                raise AdmissionDenied('Accounting overrun requires reconciliation')
            rows=c.execute('SELECT provider,status,reserved,cost FROM requests').fetchall()
            total=sum(cost if status=='SETTLED' else max(reserved,cost or 0) for _,status,reserved,cost in rows)
            provider_total=sum(cost if status=='SETTLED' else max(reserved,cost or 0)
                for prov,status,reserved,cost in rows if prov==provider)
            if total>=int(.85*p.cap_micro_USD) and not (critical or final_review):raise AdmissionDenied('85-percent admission restriction')
            ceiling=p.cap_micro_USD-(0 if final_review else p.reserve_micro_USD)
            if total+reservation>ceiling:raise AdmissionDenied('Project cap or final-review reserve')
            if provider=='openai' and provider_total+reservation>p.openai_micro_USD:raise AdmissionDenied('OpenAI subbudget')
            if sum(status!='SETTLED' for _,status,_,_ in rows)>=p.max_concurrent:raise AdmissionDenied('Concurrent/unresolved request limit')
            if c.execute('SELECT COUNT(*) FROM requests WHERE run_id=?',(run_id,)).fetchone()[0]>=p.max_run_attempts:raise AdmissionDenied('Run attempt cap')
            if c.execute('SELECT COUNT(*) FROM requests WHERE run_id=? AND task_id=?',(run_id,task_id)).fetchone()[0]>=p.max_task_attempts:raise AdmissionDenied('Task attempt cap')
            rid=uuid.uuid4().hex
            c.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (rid,provider,model,run_id,task_id,'RESERVED',reservation,None,json.dumps(price),None,time.time()))
        return rid

    def unresolved(self,request_id):
        with self.connect() as c:
            c.execute("UPDATE requests SET status='UNRESOLVED' WHERE id=? AND status='RESERVED'",(request_id,))

    def settle(self,request_id,input_tokens,output_tokens,provider_request_id=None):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT status,reserved,price FROM requests WHERE id=?',(request_id,)).fetchone()
            if not row or row[0]=='SETTLED':raise AdmissionDenied('Unknown/already settled request')
            cost=self.cost(json.loads(row[2]),input_tokens,output_tokens)
            if cost>row[1]:c.execute("UPDATE metadata SET value='true' WHERE key='disabled_due_to_overrun'")
            c.execute("UPDATE requests SET status='SETTLED',cost=?,usage=? WHERE id=?",(cost,json.dumps(dict(
                input_tokens=input_tokens,output_tokens=output_tokens,provider_request_id=provider_request_id,
                cost_class='calculated_from_reported_usage_not_invoice')),request_id))

    def state(self):
        with self.connect() as c:
            rows=c.execute('SELECT status,reserved,cost FROM requests').fetchall()
        known=sum(cost if status=='SETTLED' else max(reserved,cost or 0) for status,reserved,cost in rows)
        return dict(accounted_micro_USD_since_ledger_initialization=known,requests=len(rows),
            unresolved_or_reserved=sum(status!='SETTLED' for status,_,_ in rows),
            historical_usage='RECONCILED_EXTERNALLY' if self.policy.historical_usage_reconciled else 'UNKNOWN',
            paid_admission_enabled=self.policy.enabled and self.policy.historical_usage_reconciled,
            warning_70_percent=known>=int(.7*self.policy.cap_micro_USD),
            Codex_subscription_metering='NOT_OBSERVABLE',gateway_or_provider_client='NOT_IMPLEMENTED')
