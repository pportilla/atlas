import json
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.app import AtlasServer, Handler
from engine.search_jobs import SearchJobs


class Quiet(Handler):
    def log_message(self, *args):
        pass


class SearchEndpoints(unittest.TestCase):
    def setUp(self):
        self.server = AtlasServer(("127.0.0.1",0),Quiet)
        # One polynomial completes immediately; the other stays running until
        # the server kills its process group. The stub verifies the effective
        # worker request and avoids depending on a specific algebraic example.
        command = """import json,sys,signal
d=json.load(sys.stdin)
if d['polynomial']=='x^2': signal.pause()
print(json.dumps([200,{'status':'resolved','search':d['search']}]))
"""
        self.server.search_jobs = SearchJobs(command=[sys.executable,"-c",command],retention=60)
        self.thread = threading.Thread(target=self.server.serve_forever,kwargs={"poll_interval":0.015},daemon=True)
        self.thread.start()
        self.base=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        with urlopen(Request(self.base+path,method=method,data=body,headers={"Content-Type":"application/json"}),
                     timeout=3) as response:
            return response.status,json.load(response)

    def post(self, polynomial):
        return self.request("POST","/api/searches",{"variables":["x","y"],"polynomial":polynomial})

    def wait(self, ident):
        for _ in range(150):
            _, job=self.request("GET","/api/searches/"+ident)
            if job["state"] not in ("running","cancelling"):
                return job
            time.sleep(0.01)
        self.fail("Worker did not terminate")

    def test_completed_result_and_reattachment(self):
        code, job=self.post("x+y")
        self.assertEqual(code,202)
        done=self.wait(job["id"])
        self.assertEqual(done["result"],{"status":"resolved","search":"exhaustive"})
        _,current=self.request("GET","/api/searches/current")
        self.assertEqual(current["job"]["id"],job["id"])

    def test_cancel_conflict_and_replacement(self):
        _,job=self.post("x^2")
        with self.assertRaises(HTTPError) as err: self.post("x+y")
        self.assertEqual(err.exception.code,409); err.exception.close()
        self.request("DELETE","/api/searches/"+job["id"])
        self.assertEqual(self.wait(job["id"])["state"],"cancelled")
        _,next_job=self.post("x+y")
        self.assertEqual(self.wait(next_job["id"])["state"],"completed")
        with self.assertRaises(HTTPError) as err: self.request("GET","/api/searches/"+job["id"])
        self.assertEqual(err.exception.code,404); err.exception.close()

    def test_shutdown_reaps_active_worker(self):
        self.post("x^2")
        worker=self.server.search_jobs.job.child
        self.server.search_jobs.close()
        self.assertIsNotNone(worker.poll())
        self.assertIsNone(self.server.search_jobs.get())

    def test_invalid_request(self):
        for path,data in [("/api/searches",{"variables":[],"polynomial":"x"}),
                          ("/api/resolve",{"variables":[],"polynomial":"x^2","search":"exhaustive"})]:
            with self.assertRaises(HTTPError) as err: self.request("POST",path,data)
            self.assertEqual(err.exception.code,400); err.exception.close()

    def test_default_solve_continues_after_a_quick_limit(self):
        code, job=self.request("POST","/api/resolve",{"variables":["x","y"],"polynomial":"x+y^150"})
        self.assertEqual(code,202)
        self.assertEqual(self.wait(job["id"])["result"]["search"],"exhaustive")
