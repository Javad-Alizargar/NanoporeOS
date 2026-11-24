"""Loopback-only bounded demo API with persistent SQLite queue.

One small solve at a time; no arbitrary paths, URLs, code, archive imports or
provider credentials. Deployment requires a separately configured auth proxy.
"""
from dataclasses import asdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from urllib.parse import urlsplit
from .core import ROOT,PNP,Scenario,StopRequested,check_stop,write_json
from .datasets import save_result,load_result,export_zip


class Queue:
    def __init__(self,workspace):
        self.root=Path(workspace).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.owner=(self.root/'owner.lock').open('a+b')
        self.owner.seek(0);self.owner.write(b'0');self.owner.flush();self.owner.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.owner.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.owner.close();raise RuntimeError('Queue workspace already owned by another server')
        self.db=self.root/'queue.sqlite';self.halt=threading.Event();self.events={};self.lock=threading.Lock()
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, identity TEXT UNIQUE, config TEXT, status TEXT, attempt INTEGER, error TEXT, created REAL)')
            c.execute("UPDATE jobs SET status='INTERRUPTED',error='Server interrupted; explicit resume available' WHERE status='RUNNING'")
        self.thread=threading.Thread(target=self.work,name='nanoporeos-single-worker',daemon=True);self.thread.start()

    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.db,timeout=10)
        try:
            with connection:yield connection
        finally:connection.close()

    def list(self):
        with self.connect() as c:
            c.row_factory=sqlite3.Row
            return [dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT 100')]

    def get(self,jid):
        if not re.fullmatch(r'[a-f0-9]{32}',jid):raise ValueError('Invalid job id')
        with self.connect() as c:
            c.row_factory=sqlite3.Row
            r=c.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
        if r is None:raise KeyError('Unknown job')
        result=dict(r);result['config']=json.loads(result['config']);return result

    def submit(self,config):
        check_stop()
        s=Scenario(**config)
        if s.nx>36 or s.nr>18 or s.nx*s.nr>648:raise ValueError('Demo limit 36 x 18 cells')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            found=c.execute('SELECT id FROM jobs WHERE identity=?',(s.identity(),)).fetchone()
            if found:return found[0]
            if c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]>=100:raise ValueError('Demo storage quota (100 jobs) reached')
            if c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone()[0]>=10:raise ValueError('Demo queue quota (10) reached')
            jid=uuid.uuid4().hex
            c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)',(jid,s.identity(),json.dumps(asdict(s)),'QUEUED',0,None,time.time()))
        return jid

    def cancel(self,jid):
        self.get(jid)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            with self.lock:
                if jid in self.events:self.events[jid].set()
            c.execute("UPDATE jobs SET status='CANCELLED' WHERE id=? AND status='QUEUED'",(jid,))

    def resume(self,jid):
        check_stop();self.get(jid)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone()[0]>=10:
                raise ValueError('Demo queue quota (10) reached')
            c.execute("UPDATE jobs SET status='QUEUED',error=NULL WHERE id=? AND status IN ('CANCELLED','INTERRUPTED','FAIL')",(jid,))

    def directory(self,job):return self.root/job['id']/f"attempt-{job['attempt']}"

    def work(self):
        while not self.halt.wait(.15):
            try:
                check_stop()
                with self.connect() as c:
                    c.execute('BEGIN IMMEDIATE')
                    row=c.execute("SELECT id,config,attempt FROM jobs WHERE status='QUEUED' ORDER BY created LIMIT 1").fetchone()
                    if row is None:continue
                    jid,config,attempt=row
                    event=threading.Event()
                    with self.lock:self.events[jid]=event
                    c.execute("UPDATE jobs SET status='RUNNING',attempt=attempt+1 WHERE id=?",(jid,))
                model=PNP(Scenario(**json.loads(config)))
                out=model.solve(deadline=time.monotonic()+60,cancel=event)
                check_stop()
                if event.is_set():raise StopRequested('Job cancellation requested')
                save_result(self.root/jid/f'attempt-{attempt+1}',model,out)
                status,error=out['status'],None
            except StopRequested as exc:
                if 'jid' not in locals():continue
                status,error='CANCELLED',str(exc)
            except Exception as exc:
                if 'jid' not in locals():continue
                status,error='FAIL',type(exc).__name__
            else:pass
            if 'jid' in locals():
                with self.connect() as c:c.execute('UPDATE jobs SET status=?,error=? WHERE id=?',(status,error,jid))
                with self.lock:self.events.pop(jid,None)
                del jid

    def close(self):
        self.halt.set()
        with self.lock:
            for e in self.events.values():e.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():raise RuntimeError('Worker still stopping; queue ownership retained')
        if not self.owner.closed:self.owner.close()


class Handler(BaseHTTPRequestHandler):
    server_version='NanoporeOS/0.1'
    def log_message(self,*args):pass  # no arbitrary request/user payloads in logs

    def respond(self,status,data,content_type='application/json'):
        raw=data if isinstance(data,bytes) else json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(raw)

    def trusted(self):
        expected={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in expected:raise PermissionError('Unexpected Host')
        origin=self.headers.get('Origin')
        if origin and origin not in {'http://'+h for h in expected}:raise PermissionError('Cross-origin request denied')

    def do_GET(self):
        try:
            self.trusted();path=urlsplit(self.path).path
            if path=='/health':return self.respond(200,dict(status='ok',mode='loopback_bounded_demo'))
            if path=='/api/models':return self.respond(200,dict(model='axisym-pnp-ramp-offset-v1',
                scientific_gate='NUMERICAL_HOLD',training_qualified=False,
                unsupported=['Stokes','molecule feedback','transient PNP','experimental validation'],
                max_mesh=[36,18],max_job_seconds=60,queue_quota=10))
            if path=='/api/jobs':return self.respond(200,self.server.queue.list())
            match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(fields|export))?',path)
            if match:
                job=self.server.queue.get(match[1]);directory=self.server.queue.directory(job)
                if not match[2]:return self.respond(200,job)
                if job['status'] not in ['PASS','FAIL'] or not (directory/'manifest.json').exists():raise ValueError('Result unavailable')
                if match[2]=='export':
                    return self.respond(200,export_zip(directory),'application/zip')
                summary,arrays=load_result(directory)
                return self.respond(200,dict(summary=summary,fields={k:v.tolist() for k,v in arrays.items()}))
            assets={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in assets:
                p=ROOT/'web'/assets[path]
                return self.respond(200,p.read_bytes(),{'html':'text/html; charset=utf-8','js':'text/javascript','css':'text/css'}[p.suffix[1:]])
            self.respond(404,dict(error='Not found'))
        except PermissionError as e:self.respond(403,dict(error=str(e)))
        except KeyError:self.respond(404,dict(error='Unknown job'))
        except (ValueError,TypeError) as e:self.respond(400,dict(error=str(e)))
        except Exception:self.respond(500,dict(error='Internal error; inspect local project state'))

    def do_POST(self):
        try:
            path=urlsplit(self.path).path
            if self.headers.get('Content-Type')!='application/json':raise ValueError('JSON required')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=8192:raise ValueError('Body size limit')
            raw=self.rfile.read(size)
            # Consume the bounded request before rejecting its Origin/Host.
            # Closing with unread data can reset Windows TCP before 403 arrives.
            self.trusted()
            body=json.loads(raw)
            if not isinstance(body,dict):raise ValueError('Object required')
            if path=='/api/jobs':return self.respond(202,dict(id=self.server.queue.submit(body)))
            match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})/(cancel|resume)',path)
            if match:
                getattr(self.server.queue,match[2])(match[1]);return self.respond(200,dict(status='accepted'))
            self.respond(404,dict(error='Not found'))
        except PermissionError as e:self.respond(403,dict(error=str(e)))
        except KeyError:self.respond(404,dict(error='Unknown job'))
        except (ValueError,TypeError,StopRequested) as e:self.respond(400,dict(error=str(e)))
        except Exception:self.respond(500,dict(error='Internal error'))


def make_server(port,workspace):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    try:server.queue=Queue(workspace)
    except Exception:
        server.server_close();raise
    server.daemon_threads=True
    return server


def serve(port=8765,workspace='runs/local_api'):
    server=make_server(port,workspace)
    print(f'NanoporeOS local bounded demo: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    finally:server.queue.close();server.server_close()
