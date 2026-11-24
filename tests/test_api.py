from pathlib import Path
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from nanoporeos.api import Queue,make_server
from nanoporeos.core import PNP,Scenario,StopRequested


def wait_job(q,jid,states=('PASS','FAIL','CANCELLED'),seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        job=q.get(jid)
        if job['status'] in states:return job
        time.sleep(.03)
    raise AssertionError('Timed out waiting for job')


class APITests(unittest.TestCase):
    def test_workspace_lock_and_resume_quota(self):
        with tempfile.TemporaryDirectory() as temp:
            q=Queue(temp)
            try:
                with self.assertRaises(RuntimeError):Queue(temp)
                q.halt.set();q.thread.join()
                cancelled=q.submit(dict(nx=8,nr=4,voltage_mV=10));q.cancel(cancelled)
                for i in range(10):q.submit(dict(nx=8,nr=4,voltage_mV=20+i))
                with self.assertRaises(ValueError):q.resume(cancelled)
                self.assertEqual(sum(r['status']=='QUEUED' for r in q.list()),10)
            finally:q.close()

    def test_queue_duplicate_export_and_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            q=Queue(temp)
            try:
                config=dict(nx=8,nr=4)
                a=q.submit(config);self.assertEqual(a,q.submit(config))
                out=wait_job(q,a);self.assertEqual(out['status'],'PASS')
                self.assertTrue((q.directory(out)/'manifest.json').exists())
                with self.assertRaises(ValueError):q.submit(dict(nx=160,nr=80))
            finally:q.close()
            with closing(sqlite3.connect(Path(temp)/'queue.sqlite')) as c:
                c.execute("UPDATE jobs SET status='RUNNING'");c.commit()
            q=Queue(temp)
            try:
                self.assertEqual(q.get(a)['status'],'INTERRUPTED')
                q.resume(a);out=wait_job(q,a);self.assertEqual(out['status'],'PASS')
                self.assertEqual(out['attempt'],2)
                self.assertTrue((Path(temp)/a/'attempt-1'/'manifest.json').exists())
            finally:q.close()

    def test_cancellation_and_resume(self):
        event=threading.Event();event.set()
        with self.assertRaises(StopRequested):PNP(Scenario(nx=8,nr=4)).solve(cancel=event)
        with tempfile.TemporaryDirectory() as temp:
            q=Queue(temp)
            try:
                jid=q.submit(dict(nx=36,nr=18));q.cancel(jid)
                self.assertEqual(wait_job(q,jid)['status'],'CANCELLED')
                q.resume(jid);self.assertEqual(wait_job(q,jid)['status'],'PASS')
            finally:q.close()

    def test_http_flow_and_hostile_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            server=make_server(0,temp);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            def request(path,body=None,headers=None):
                data=None if body is None else json.dumps(body).encode()
                r=Request(base+path,data=data,headers=headers or ({'Content-Type':'application/json'} if data else {}))
                return urlopen(r,timeout=10)
            try:
                self.assertEqual(json.load(request('/health'))['status'],'ok')
                self.assertIn(b'NanoporeOS',request('/').read())
                jid=json.load(request('/api/jobs',dict(nx=8,nr=4)))['id']
                self.assertEqual(wait_job(server.queue,jid)['status'],'PASS')
                self.assertEqual(json.load(request('/api/jobs/'+jid+'/fields'))['summary']['status'],'PASS')
                self.assertTrue(request('/api/jobs/'+jid+'/export').read().startswith(b'PK'))
                for path,body,headers,code in [
                    ('/api/jobs',{'nx':999999},None,400),
                    ('/api/jobs',{'salt_mM':-1},None,400),
                    ('/api/jobs',{'model':'stokes'},None,400),
                    ('/api/jobs',{'url':'http://internal.local'},None,400),
                    ('/health',None,{'Host':'evil.test'},403),
                    ('/api/jobs',{}, {'Origin':'https://evil.test','Content-Type':'application/json'},403),
                    ('/api/jobs/../../apis/chatgpt.txt',None,None,404)]:
                    with self.subTest(path=path,headers=headers,body=body):
                        with self.assertRaises(HTTPError) as caught:request(path,body,headers)
                        self.assertEqual(caught.exception.code,code)
            finally:
                server.shutdown();server.queue.close();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
