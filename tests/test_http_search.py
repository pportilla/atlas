import json
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine.app import AtlasServer
from engine.search_jobs import SearchJobs
from tests.http_support import QuietHandler


class SearchEndpoints(unittest.TestCase):
    def setUp(self):
        self.server = AtlasServer(("127.0.0.1",0),QuietHandler)
        # One polynomial completes immediately; the other stays running until
        # the server kills its process group. The stub verifies the effective
        # worker request and avoids depending on a specific algebraic example.
        command = """import json,os,sys,signal
from pathlib import Path
d=json.load(sys.stdin)
if 'ATLAS_PROGRESS_FILE' in os.environ:
    Path(os.environ['ATLAS_PROGRESS_FILE']).write_text(json.dumps({'stage': d['polynomial'], 'blowups': 2}))
if d['polynomial']=='x^2': signal.pause()
if d['polynomial']=='x^3' and d['search']=='quick':
    print(json.dumps([200,{'status':'inconclusive'}])); sys.exit(0)
print(json.dumps([200,{'status':'resolved','search':d['search'],'dual_complex':{'status':'complete'}}]))
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
            if job["state"] not in ("queued","running","cancelling"):
                return job
            time.sleep(0.01)
        self.fail("Worker did not terminate")

    def test_completed_result_and_reattachment(self):
        code, job=self.post("x+y")
        self.assertEqual(code,202)
        done=self.wait(job["id"])
        self.assertEqual(done["result"],{"status":"resolved","search":"exhaustive","dual_complex":{"status":"complete"}})
        _,current=self.request("GET","/api/searches/current")
        self.assertEqual(current["job"]["id"],job["id"])

    def test_parallel_and_queued_jobs_cancel_independently(self):
        _, first = self.post("x^2")
        _, second = self.post("x^2")
        _, queued = self.post("x+y")
        _, dropped = self.post("x^2")
        self.assertEqual([first["state"], second["state"], queued["state"], dropped["state"]],
                         ["running", "running", "queued", "queued"])
        self.request("DELETE", "/api/searches/" + dropped["id"])
        self.request("DELETE", "/api/searches/" + first["id"])
        self.assertEqual(self.wait(first["id"])["state"], "cancelled")
        self.assertEqual(self.wait(queued["id"])["state"], "completed")
        jobs = self.request("GET", "/api/searches")[1]["jobs"]
        self.assertEqual([(j["id"], j["state"]) for j in jobs],
                         [(dropped["id"], "cancelled"), (queued["id"], "completed"),
                          (second["id"], "running"), (first["id"], "cancelled")])

    def test_automatic_job_retries_an_inconclusive_quick_attempt(self):
        for polynomial, expected in (("x+y", "quick"), ("x^3", "exhaustive")):
            _, job = self.request("POST", "/api/searches", {"variables": ["x", "y"], "polynomial": polynomial,
                                                             "search": "auto"})
            self.assertEqual(self.wait(job["id"])["result"]["search"], expected)

    def test_live_progress_is_scoped_to_one_worker(self):
        _, running = self.post("x^2")
        other = self.wait(self.post("x+y")[1]["id"])
        self.assertEqual(other["progress"], {"stage": "x+y", "blowups": 2})
        for _ in range(150):
            current = self.request("GET", "/api/searches/" + running["id"])[1]
            if current.get("progress"):
                break
            time.sleep(0.01)
        self.assertEqual(current.get("progress"), {"stage": "x^2", "blowups": 2})
        self.assertEqual(current["state"], "running")

    def test_independent_zetas_and_replay_on_completed_jobs(self):
        self.server.search_jobs = SearchJobs()
        request = {"variables": ["x", "y"], "polynomial": "y^2-x^3", "mode": "embedded", "search": "quick"}
        _, plain = self.request("POST", "/api/searches", dict(request, zetas=[]))
        plain_record = self.wait(plain["id"])["result"]
        _, initial = self.request("POST", "/api/searches", dict(request, zetas=["topological"]))
        job = self.wait(initial["id"])
        record = job["result"]
        self.assertEqual(record["topological_zeta"]["expression"], "(4*s + 5)/((s + 1)*(6*s + 5))")
        self.assertNotIn("monodromy_zeta", record)
        self.assertNotIn("monodromy_check", record)
        _, job = self.request("POST", "/api/searches/" + job["id"] + "/zetas", {"zetas": ["monodromy"]})
        self.assertEqual(job["result"]["monodromy_zeta"]["expression"], "(1-T^6)/((1-T^2)*(1-T^3))")
        self.assertEqual(job["result"]["topological_zeta"]["expression"], record["topological_zeta"]["expression"])
        check = job["result"]["monodromy_check"]
        self.assertEqual(check["status"], "verified")
        self.assertEqual([(p["value"], p["eigenvalue_order"], p["witness"]) for p in check["poles"]],
                         [("-1", 1, "origin zeta"), ("-5/6", 6, "origin zeta")])
        self.assertEqual(job["result"]["charts"], record["charts"])

        _, intrinsic = self.request("POST", "/api/searches", dict(request, mode="intrinsic", zetas=[]))
        saved = self.wait(intrinsic["id"])["result"]
        self.assertNotIn("embedded_companion", saved)
        _, updated = self.request("POST", "/api/searches/" + intrinsic["id"] + "/zetas", {"zetas": ["topological"]})
        self.assertEqual(updated["result"]["topological_zeta"]["expression"], record["topological_zeta"]["expression"])
        self.assertEqual(updated["result"]["topological_zeta"]["source"], "companion embedded resolution")
        self.assertEqual(saved["charts"], updated["result"]["charts"])

        # A stale pole imported with a saved record provides no evidence.
        record["topological_zeta"]["poles"].append({"value": "-2/7", "order": 1, "components": []})
        _, outputs = self.request("POST", "/api/zetas", {"record": record, "zetas": ["monodromy"]})
        self.assertEqual(outputs["monodromy_check"], check)
        self.assertEqual(outputs["topological_zeta"], job["result"]["topological_zeta"])
        self.assertEqual(outputs["monodromy_zeta"], job["result"]["monodromy_zeta"])
        from engine.resolution_zetas import replay_zetas
        with patch("engine.resolve._resolve_polynomial", side_effect=AssertionError("started a new resolution")):
            both = replay_zetas({"record": plain_record, "zetas": ["monodromy", "topological"]})
        self.assertEqual(both["monodromy_zeta"]["expression"], job["result"]["monodromy_zeta"]["expression"])
        self.assertEqual(both["topological_zeta"]["expression"], record["topological_zeta"]["expression"])
        self.assertEqual(both["monodromy_check"], check)

    def test_replay_completes_a_bounded_record_from_its_original_polynomial(self):
        self.server.search_jobs = SearchJobs()
        request = {"variables": ["x"], "polynomial": "x^200", "search": "quick", "zetas": ["monodromy"]}
        _, job = self.request("POST", "/api/searches", request)
        record = self.wait(job["id"])["result"]
        self.assertEqual(record["status"], "inconclusive")
        for path, obj in (("/api/zetas", {"record": record}),
                          ("/api/searches/" + job["id"] + "/zetas", {})):
            with self.subTest(path=path):
                code, replay = self.request("POST", path, dict(obj, zetas=["topological", "monodromy"]))
                output = replay.get("result", replay)
                self.assertEqual(code, 200)
                self.assertEqual(output["monodromy_check"]["status"], "verified")
                self.assertEqual(output["monodromy_zeta"]["expression"], "1/(1-T^200)")

    def test_three_graph_zetas_only_and_saved_certificate(self):
        from tests.test_three_graph_monodromy import POLYNOMIAL
        request = {"variables": ["x", "y", "z"], "polynomial": POLYNOMIAL, "search": "quick",
                   "zetas_only": True, "zetas": ["topological"]}
        _, record = self.request("POST", "/api/resolve", request)
        self.assertEqual(record["topological_zeta"]["status"], "computed")
        self.assertNotIn("monodromy_zeta", record)
        _, pair = self.request("POST", "/api/zetas", {"record": record, "zetas": ["monodromy"]})
        self.assertEqual(pair["monodromy_check"]["status"], "verified")
        self.assertEqual(pair["topological_zeta"]["expression"], record["topological_zeta"]["expression"])
        certificate = record["contact_certificate"]
        certificate["divisors"][0]["nu"] += 1
        with self.assertRaises(HTTPError) as failure:
            self.request("POST", "/api/zetas", {"record": record, "zetas": ["monodromy"]})
        with failure.exception as response:
            self.assertEqual(response.code, 400)

    def test_quadratic_translate_saved_http_certificate(self):
        from tests.test_quadratic_translate_monodromy import POLYNOMIAL, REFERENCE
        request = {"variables": ["x", "y", "z"], "polynomial": POLYNOMIAL, "search": "quick",
                   "zetas_only": True, "zetas": ["monodromy"], "compute": "cpu"}
        _, record = self.request("POST", "/api/resolve", request)
        self.assertEqual(record["engine"], "quadratic-translate-contact")
        self.assertEqual(record["monodromy_zeta"]["factors"], REFERENCE["monodromy_zeta"]["factors"])
        self.assertNotIn("topological_zeta", record)
        _, both = self.request("POST", "/api/zetas", {"record": record, "zetas": ["topological"]})
        self.assertEqual(both["monodromy_check"]["status"], "verified")
        record["contact_certificate"]["divisors"][0]["orders"][0] += 1
        with self.assertRaises(HTTPError) as failure:
            self.request("POST", "/api/zetas", {"record": record, "zetas": ["monodromy"]})
        with failure.exception as response:
            self.assertEqual(response.code, 400)

    def test_concurrent_zeta_requests_attach_the_same_check_as_the_initial_pair(self):
        self.server.search_jobs = SearchJobs()
        request = {"variables": ["x", "y"], "polynomial": "x*y", "mode": "embedded", "search": "quick"}
        _, both = self.request("POST", "/api/resolve", dict(request, zetas=["monodromy", "topological"]))
        self.assertEqual(both["monodromy_check"]["poles"][0]["witness"], "origin H^0")
        _, job = self.request("POST", "/api/searches", dict(request, zetas=[]))
        self.wait(job["id"])
        path = "/api/searches/" + job["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = [pool.submit(self.request, "POST", path + "/zetas", {"zetas": [kind]})
                         for kind in ("monodromy", "topological")]
            for response in responses:
                self.assertEqual(response.result()[0], 200)
        _, job = self.request("GET", path)
        self.assertEqual(job["result"]["monodromy_check"], both["monodromy_check"])

    def test_automatic_check_preserves_transverse_rationals_and_wide_orders(self):
        self.server.search_jobs = SearchJobs()
        for equation, variables, pole, order, witness in (
                ("3/7*x^2+11/13*y^2+t*z^2+z^3", list("xyzt"), "-3/2", 2, "nearby transverse Morse"),
                ("(123456789012345678901234567890123456789/7)*x^4294967299", ["x"],
                 "-1/4294967299", 4294967299, "origin zeta"),
                ("(123456789012345678901234567890123456789/7)*x^9007199254740993*y^4", list("xyz"),
                 "-1/9007199254740993", 9007199254740993, "nearby smooth component")):
            with self.subTest(equation=equation):
                _, job = self.request("POST", "/api/searches", {"polynomial": equation, "variables": variables,
                                      "search": "exhaustive", "zetas": ["monodromy", "topological"]})
                result = self.wait(job["id"])["result"]
                check = result["monodromy_check"]
                evidence = next(p for p in check["poles"] if p["value"] == pole)
                self.assertEqual((check["status"], evidence["eigenvalue_order"], evidence["witness"]),
                                 ("verified", order, witness))
                nearby = result["monodromy_zeta"]["nearby_strata"]
                if witness.startswith("nearby"):
                    match, = evidence["nearby_matches"]
                    self.assertEqual(match["cyclotomic_exponent"], -1)
                    self.assertEqual(nearby[match["stratum_index"]]["expression"], f"1/(1-T^{order})")
                else:
                    self.assertEqual((nearby, evidence["nearby_matches"]), ([], []))
                if evidence["transverse"]:
                    self.assertEqual(evidence["transverse"]["transverse_hessian_determinant"], "264/91")

    def test_shutdown_reaps_active_workers(self):
        first = self.post("x^2")[1]
        second = self.post("x^2")[1]
        self.post("x+y")
        workers = [self.server.search_jobs.jobs[j["id"]].child for j in (first, second)]
        self.server.search_jobs.close()
        self.assertTrue(all(worker.poll() is not None for worker in workers))
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

    def test_default_retries_limits_and_incomplete_requested_outputs(self):
        body = {"variables":["x","y"], "polynomial":"x+y", "monodromy":True}
        for reply in ((422, {"code":"computation_limit", "error":"Euler budget"}),
                      (200, {"status":"resolved", "monodromy_zeta":{"status":"computed"},
                             "topological_zeta":{"status":"unavailable"}}),
                      (200, {"status":"resolved", "monodromy_zeta":{"status":"computed"},
                             "topological_zeta":{"status":"computed"}, "dual_complex":{"status":"partial"}}),
                      (200, {"status":"resolved", "monodromy_zeta":{"status":"computed"},
                             "topological_zeta":{"status":"computed"}})):
            with self.subTest(reply=reply), patch("engine.app.run_job", return_value=reply):
                code, job = self.request("POST", "/api/resolve", body)
                self.assertEqual(code, 202)
                self.assertEqual(self.wait(job["id"])["state"], "completed")
