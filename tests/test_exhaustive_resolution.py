import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from engine import exhaustive_resolution as exhaustive
from engine.polynomial import LimitError, parse
from engine.resolve import resolve


class ExhaustiveFallback(unittest.TestCase):
    def test_positive_dimensional_newton_failure_reaches_constructive_blowups(self):
        request = dict(polynomial="(x*y-z*t+u^2)^2 + x^5+2*y^5+3*z^5+4*t^5+5*u^5",
                       variables=list("xyztu"), mode="embedded", search="exhaustive")
        result = resolve(request)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual([step["rule"] for step in result["steps"]], ["repeated tangent"] * 3)
        first, second, third = [step["center_charts"] for step in result["steps"]]
        self.assertEqual(first, [dict(history=[], coordinates=list("xyztu"), ideal=list("xyztu"))])
        self.assertEqual(sorted(piece["history"] for piece in second), [[[1, pivot]] for pivot in range(5)])
        for piece in third:
            self.assertEqual([stage for stage, pivot in piece["history"]], [1, 2])
            self.assertTrue(piece["ideal"])
        self.assertEqual(len(result["atlas"]), len(result["charts"]))
        self.assertEqual(result["certificate"]["atlas_verification"], "Embedded normal crossings on the owned origin fibre")

    def test_certificate_resolution_when_geometric_rule_does_not_apply(self):
        from engine.exhaustive_resolution import search
        names = list("wxyz")
        # The singular locus is a plane. Blowing up (x,y) gives a certified
        # resolution, including away from the centre of the origin blowup.
        result = search(parse("x^2-y^2*z", names), names, "embedded")
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["charts"])

    def test_quick_keeps_a_bounded_search(self):
        request = dict(polynomial="(x*y-z*t+u^2)^2 + x^5+2*y^5+3*z^5+4*t^5+5*u^5",
                       variables=list("xyztu"), mode="embedded", search="quick", monodromy=True)
        with patch("engine.newton_backend.resolve_newton", side_effect=LimitError("fan limit")), \
             patch("engine.origin_backend.resolve_origin", return_value=None) as origin, \
             patch("engine.general_resolution.search", return_value=None) as quick, \
             patch("engine.exhaustive_resolution.search") as exhaustive:
            result = resolve(request)
        self.assertFalse(quick.call_args.kwargs["exhaustive"])
        # The repeated-tangent rule, then the smooth-centre rules, each with a deadline.
        calls = [call.kwargs for call in origin.call_args_list]
        self.assertTrue(calls[0]["tangent_only"])
        self.assertTrue(all(call["seconds"] > 0 for call in calls))
        exhaustive.assert_not_called()
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["monodromy_zeta"]["status"], "inconclusive")
        reason = result["diagnostic"]["description"]
        self.assertIn("Newton search reached a limit (fan limit)", reason)
        self.assertIn("repeated-tangent search", reason)
        self.assertIn("general search checked 0 candidates", reason)

    def test_newton_limit_leaves_fallbacks_eligible(self):
        payload = dict(terms=[[[2, 0, 0], "1"]], variables=list("xyz"), mode="embedded",
                       compute="cpu", replay={"compute": "cpu"})
        with patch("engine.newton_backend.resolve_newton", side_effect=LimitError("fan limit")) as newton:
            self.assertIsNone(exhaustive._run("newton", payload)["record"])
        # The lane keeps the fan path of the race and collects a certificate
        # its coordinator can attach. Zeta requests verify again with integrals.
        self.assertEqual(newton.call_args.kwargs, {"compute": "cpu", "seconds": 1200, "defer_outputs": True,
                                                   "verified": [], "integrals": False, "construct": False})

    def test_newton_worker_verifies_geometry_before_reporting_a_proposal(self):
        payload = dict(terms=[[[2, 0, 0], "1"]], variables=list("xyz"), mode="intrinsic",
                       compute="cpu", replay={"compute": "cpu"})
        record = {"status": "resolved"}
        bundle = [], {"dual_complex": {"status": "complete"}}
        with patch("engine.newton_backend.resolve_newton", return_value=record), \
             patch("engine.strata.proposal_chart_outputs", return_value=bundle) as verify:
            reply = exhaustive._run("newton", payload)
        self.assertEqual(reply["certified"], list(bundle))
        verify.assert_called_once_with(record, parse("x^2", list("xyz")), integrals=False)

    def test_canonical_head_start_covers_only_certified_newton_failures(self):
        from engine.newton_backend import coordinate_choices, resolve_newton
        from engine.newton_faces import face_search
        from engine.newton_fan import build_fan
        from engine.native_library import _kernel
        if not hasattr(_kernel(), "atlas_newton_tangent_obstruction_abi"):
            self.skipTest("Packed tangent obstruction unavailable.")
        names = list("xyz")
        for source in ("(x+y+z)^2-z*(x-y)^2", "(x/2+2*y/3-z/5)^2-z*(3*x/4-y/5)^2",
                       "(x+y+z)^3/7+x*y*z^2", "(2*x+4*y-3*z)^2+z^5", "(1+x)*((x+y+z)^2-x*y*z)",
                       "(x+y+z)^2*(1+x)", "(x+y+z)^2+z^3/65521"):
            with self.subTest(source=source), face_search(True):
                poly = parse(source,names)
                self.assertTrue(exhaustive._canonical_head_start(poly,names,"embedded"))
                self.assertEqual([(f, changes) for f,_,changes in coordinate_choices(poly,3)], [(poly,[])])
                fan = build_fan(poly, exhaustive=True)
                self.assertEqual(fan["rays"][3:], [[1,1,1]])
                self.assertIsNone(resolve_newton(poly,names,"embedded",compute="cpu",defer_outputs=True))
                self.assertFalse(exhaustive._canonical_head_start(poly,names,"intrinsic"))
        for source in ("(x+y)^2-z*(x-y)^2", "(x+y+z)^2+x*y+z^3", "(x+y+z)^2+x^17", "2^70*(x+y+z)^2+z^3"):
            with self.subTest(source=source):
                self.assertFalse(exhaustive._canonical_head_start(parse(source,names),names,"embedded"))
        from engine.native_library import Unavailable
        for exc in (Unavailable("unavailable"), OSError("optional kernel missing")):
            with patch("engine.kernel_json.kernel_json", side_effect=exc):
                self.assertFalse(exhaustive._canonical_head_start(parse("(x+y+z)^2-z*(x-y)^2",names),names,"embedded"))
        with patch("engine.native_library._kernel", return_value=None):
            self.assertFalse(exhaustive._canonical_head_start(parse("(x+y+z)^2-z*(x-y)^2",names),names,"embedded"))

    def test_obstructed_newton_releases_certificates_on_origin_decline_failure_and_timeout(self):
        from subprocess import Popen
        from pathlib import Path
        import time
        program = """
import sys,json,time
from pathlib import Path
method, outcome, mode, path = sys.argv[1:]
args=json.load(sys.stdin)
assert args['mode']==mode and args['replay']['compute']=='cpu'
p=Path(path)
if method=='origin':
    (p/'started').touch()
    if outcome=='slow':
        for _ in range(200):
            if (p/'racing').exists(): break
            time.sleep(.01)
        else: raise AssertionError('certificate lane was never started')
        time.sleep(1)
    if outcome=='error': sys.exit(3)
    if outcome=='nonreduced': print(json.dumps({'nonreduced':'repeated factor'}));sys.exit(0)
elif method=='certificates':
    assert bool(args.get('defer_spare')) == (outcome=='slow')
    (p/'racing').touch()
else:
    raise AssertionError('proven Newton failure should never start')
record = None if outcome=='decline' and method=='origin' else {'engine':method}
bundle = [[], {'dual_complex': {'status': 'complete'}}]
if method == 'origin' and outcome in ('partial', 'zeta_only_partial'):
    bundle[1]['dual_complex']['status'] = 'partial'
if method == 'origin' and outcome == 'missing':
    bundle = None
print(json.dumps({'record':record, 'certified':bundle, 'replay':{'winner':method}}))
"""
        for outcome in ("win","decline","missing","partial","zeta_only_partial","error","slow","failed_start","nonreduced"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as path:
                processes={}; elapsed={}
                def launch(argv, **kwargs):
                    method = argv[-2]
                    elapsed[method] = time.monotonic()
                    if outcome=="failed_start" and method=="origin":
                        raise OSError("cannot start canonical worker")
                    worker=Popen([argv[0],"-c",program,method,outcome,"embedded",path], **kwargs)
                    processes[method]=worker
                    return worker
                replay=Mock()
                replay.settings.return_value={"compute":"cpu"}
                # A long grace makes early release distinguishable from expiration.
                grace=.1 if outcome=="slow" else 5
                with patch.object(exhaustive,"_canonical_head_start",return_value=True), \
                     patch.object(exhaustive,"_forkable",return_value=False), \
                     patch.object(exhaustive,"ORIGIN_GRACE_SECONDS",grace), \
                     patch.object(exhaustive.subprocess,"Popen",side_effect=launch), \
                     patch.object(exhaustive,"current_replay_scope",return_value=replay):
                    if outcome=="nonreduced":
                        from engine.polynomial import NonReducedError
                        with self.assertRaisesRegex(NonReducedError,"repeated factor"):
                            exhaustive.search({(2,0,0):1},list("xyz"),"embedded",compute="cpu")
                        replay.merge.assert_not_called()
                    else:
                        expected = "origin" if outcome in ("win", "zeta_only_partial") else "certificates"
                        result=exhaustive.search({(2,0,0):1},list("xyz"),"embedded",compute="cpu",
                                                 zetas_only=outcome=="zeta_only_partial")
                        self.assertEqual(result,{"engine":expected})
                        replay.merge.assert_called_once_with({"winner":expected})
                self.assertTrue(all(p.poll() is not None for p in processes.values()))
                self.assertEqual(set(elapsed), {"origin"} if outcome in ("win","zeta_only_partial","nonreduced")
                                 else {"origin","certificates"})
                if "certificates" in elapsed:
                    wait=elapsed["certificates"]-elapsed["origin"]
                    self.assertLess(wait,3)
                    if outcome=="slow":
                        self.assertGreaterEqual(wait,grace)
                        self.assertTrue((Path(path)/"racing").exists())

    def test_newton_and_fallbacks_complete_independently(self):
        # Real workers use marker files to synchronize, so a slow CI machine
        # cannot change which method wins a given case.
        import sys
        from subprocess import Popen
        host_limit = sys.get_int_max_str_digits()
        exponent = 10 ** max(5000, host_limit + 1)
        script = """
import json, os, sys, time
from pathlib import Path
from engine.exact_json import result_json
from engine.polynomial import integer_from_decimal
method, winner, root = sys.argv[1:]
root = Path(root)
args = json.load(sys.stdin, parse_int=integer_from_decimal)
exponent, = [e[0] for e, coefficient in args['terms']]
assert args['replay'] == {'compute': 'gpu', 'backend': 'opencl', 'index': 1}
if method=='certificates':
    assert args['defer_spare']
(root / method).write_text(str(os.getpid()))
if method == 'newton' and winner == 'fast':
    pass
elif method == winner:
    deadline = time.monotonic() + 5
    for sibling in ('newton', 'origin', 'certificates'):
        while not (root / sibling).exists() and time.monotonic() < deadline:
            time.sleep(.01)
    if not all((root / s).exists() for s in ('newton', 'origin', 'certificates')):
        sys.exit(3)
else:
    time.sleep(6)
    sys.exit(4)
print(result_json({'record': {'engine': method, 'exponent': exponent},
                   'certified': [[], {'dual_complex': {'status':'complete'}}], 'replay': {'winner': method}}))
"""
        for winner in ("fast", "origin", "newton", "certificates"):
            with self.subTest(winner=winner), tempfile.TemporaryDirectory() as root:
                workers = {}
                def spawn(argv, **kwargs):
                    method = argv[-2]
                    self.assertNotIn(method, workers)
                    workers[method] = Popen([argv[0], "-c", script, method, winner, root], **kwargs)
                    return workers[method]
                scope = Mock()
                scope.settings.return_value = {"compute": "gpu", "backend": "opencl", "index": 1}
                grace = 2 if winner == "fast" else .02
                with patch.object(exhaustive.subprocess, "Popen", side_effect=spawn), \
                     patch.object(exhaustive, "_forkable", return_value=False), \
                     patch.object(exhaustive, "NEWTON_GRACE_SECONDS", grace), \
                     patch.object(exhaustive, "current_replay_scope", return_value=scope):
                    result = exhaustive.search({(exponent, 0, 0): 1}, list("xyz"), "embedded", compute="gpu")
                expected = "newton" if winner == "fast" else winner
                self.assertEqual(result, {"engine": expected, "exponent": exponent})
                self.assertEqual(sys.get_int_max_str_digits(), host_limit)
                scope.merge.assert_called_once_with({"winner": expected})
                self.assertEqual(set(workers), {"newton"} if winner == "fast" else {"newton", "origin", "certificates"})
                self.assertTrue(all(child.poll() is not None for child in workers.values()))

    def test_lane_reaps_native_descendant_on_success_and_outer_cancellation(self):
        import os
        import select
        import subprocess
        import sys

        # A pipe borrowed by the native grandchild gives observable EOF
        # when it exits, including on systems where an orphan remains a
        # zombie until the OS reaps it. A separate pipe signals readiness.
        lane_script = """
import json, os, sys, subprocess, threading
from engine.exhaustive_resolution import _watch_coordinator
method, guard, live, ready, let_win = sys.argv[1:]
threading.Thread(target=_watch_coordinator, args=(int(guard),), daemon=True).start()
json.load(sys.stdin)
if method == 'newton':
    fds = (int(live), int(ready))
    native = subprocess.Popen([sys.executable, '-c',
        'import os,sys,time; os.write(int(sys.argv[1]), b"r"); time.sleep(15)', ready],
        pass_fds=fds, stdout=subprocess.DEVNULL)
    for fd in fds: os.close(fd)
    native.wait()
else:
    if os.read(int(let_win), 1):
        print(json.dumps({'record': {'engine': method}, 'certified': [[], {'dual_complex': {'status':'complete'}}]}))
"""
        coordinator_script = """
from engine import exhaustive_resolution as e
import os, subprocess, sys
worker, live, ready, let_win = sys.argv[1:]
spawn = subprocess.Popen
def start(argv, **kw):
    method, guard = argv[-2:]
    extra = (int(live), int(ready)) if method == 'newton' else (int(let_win),)
    kw['pass_fds'] += extra
    process = spawn([sys.executable, '-c', worker, method, guard, live, ready, let_win], **kw)
    if method == 'newton':
        os.close(int(live)); os.close(int(ready))
    return process
e.subprocess.Popen = start
e._forkable = lambda *args: False
e.NEWTON_GRACE_SECONDS = .02
print(e.search({(2,0,0): 1}, list('xyz'), 'embedded'))
"""
        for killed in (False, True):
            with self.subTest(killed=killed):
                live, live_w = os.pipe()
                ready, ready_w = os.pipe()
                go, go_w = os.pipe()
                process = subprocess.Popen([sys.executable, "-c", coordinator_script,
                                            lane_script, str(live_w), str(ready_w), str(go)],
                                           pass_fds=(live_w, ready_w, go), stdout=subprocess.PIPE)
                os.close(live_w)
                os.close(ready_w)
                os.close(go)
                try:
                    self.assertTrue(select.select([ready], [], [], 5)[0])
                    self.assertEqual(os.read(ready, 1), b"r")
                    if killed:
                        process.kill()
                    else:
                        os.write(go_w, b"g")
                    process.communicate(timeout=6)
                    self.assertEqual(process.returncode, -9 if killed else 0)
                    self.assertTrue(select.select([live], [], [], 5)[0])
                    self.assertEqual(os.read(live, 1), b"")
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    for fd in (ready, live, go_w):
                        os.close(fd)


class SpeculativeLanes(unittest.TestCase):
    """Forked lanes start together; the first lane keeps its priority."""

    def test_first_lane_keeps_priority_over_earlier_answers(self):
        from subprocess import Popen
        script = """
import json, sys, time
from pathlib import Path
method, plan, root = sys.argv[1:]
args = json.load(sys.stdin)
(Path(root) / method).write_text("started")
delay, wins = json.loads(plan)[method]
time.sleep(delay)
print(json.dumps({'record': {'engine': method} if wins else None,
                  'certified': [[], {'dual_complex': {'status':'complete'}}]}))
"""
        # Each lane's (delay in seconds, whether it finds a record).
        plans = {
            "newton wins after origin": ({"newton": (.6, True), "origin": (.05, True), "certificates": (5, False)},
                                         "newton"),
            "origin waits for newton": ({"newton": (.6, False), "origin": (.05, True), "certificates": (5, False)},
                                        "origin"),
        }
        for name, (plan, expected) in plans.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as root:
                started = {}

                def spawn(argv, **kwargs):
                    method = argv[-2]
                    started[method] = time.monotonic()
                    return Popen([argv[0], "-c", script, method, json.dumps(plan), root], **kwargs)

                with patch.object(exhaustive.subprocess, "Popen", side_effect=spawn), \
                     patch.object(exhaustive, "_forkable", return_value=False), \
                     patch.object(exhaustive.fork_server, "available", return_value=True), \
                     patch.object(exhaustive, "NEWTON_GRACE_SECONDS", 3), \
                     patch.object(exhaustive, "_canonical_head_start", return_value=False), \
                     patch.object(exhaustive, "current_replay_scope", return_value=None):
                    begin = time.monotonic()
                    result = exhaustive.search({(2, 0, 0): 1}, list("xyz"), "embedded", compute="cpu")
                    elapsed = time.monotonic() - begin
                self.assertEqual(result, {"engine": expected})
                self.assertEqual(set(started), {"newton", "origin", "certificates"})
                self.assertLess(max(started.values()) - min(started.values()), .5)
                # No answer is taken before the first lane's own answer.
                self.assertGreaterEqual(elapsed, .6)
                self.assertLess(elapsed, 3)


ROOT = Path(__file__).resolve().parents[1]

# A coordinator in a fresh interpreter, which has a single thread, so it
# forks its lanes. Each lane runs a scripted method and reports its state.
FORKED_COORDINATOR = r"""
import json, os, sys, time
from pathlib import Path
from engine import exhaustive_resolution as e
from engine.native_library import replay_scope
plan, root, cpu, grace = json.loads(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
if cpu != "all":
    os.sched_setaffinity(0, {int(cpu)})

def interpreter(*args, **kwargs):
    raise AssertionError("a lane started an interpreter")

def scripted(method, payload):
    from engine.native_library import current_replay_scope
    from engine.newton_faces import _WORKER
    begin = time.time()
    fds = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            fds[name] = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            pass
    with open("/proc/self/stat") as stat:
        started = int(stat.read().rpartition(")")[2].split()[19]) / os.sysconf("SC_CLK_TCK")
    (root / method).write_text(json.dumps(dict(
        begin=begin, started=started, pid=os.getpid(), parent=os.getppid(), session=os.getsid(0),
        fds=fds, cwd=os.getcwd(), digits=sys.get_int_max_str_digits(), scope=current_replay_scope() is None,
        worker=_WORKER.get() is None, defer=payload.get("defer_spare", False),
        progress=os.environ.get("ATLAS_PROGRESS_METHOD"))))
    delay, outcome = plan[method]
    time.sleep(delay)
    if outcome == "crash":
        raise RuntimeError("scripted failure")
    if outcome == "nonreduced":
        return {"nonreduced": "repeated factor"}
    return {"record": {"engine": method} if outcome == "record" else None,
            "certified": [[], {"dual_complex": {"status":"complete"}}], "replay": {}}

e.fork_server.popen = interpreter
e._run = scripted
e._canonical_head_start = lambda *args: False
e.NEWTON_GRACE_SECONDS = grace
# An inheritable descriptor and a working directory that lanes must not keep.
os.set_inheritable(os.open(os.devnull, os.O_RDONLY), True)
os.chdir("/")
with open("/proc/self/stat") as stat:
    started = int(stat.read().rpartition(")")[2].split()[19]) / os.sysconf("SC_CLK_TCK")
begin = time.time()
try:
    with replay_scope("cpu"):
        outcome = {"record": e.search({(2, 0, 0): 1}, list("xyz"), "embedded", compute="cpu")}
except Exception as error:
    outcome = {"error": type(error).__name__, "message": str(error)}
print(json.dumps(dict(outcome, begin=begin, end=time.time(), pid=os.getpid())))
"""


class ForkedLanes(unittest.TestCase):
    """Without a fork server, a single-threaded solver forks its lanes and keeps the race semantics."""

    def coordinate(self, plan, *, cpu="all", grace=3):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as root:
            env = {**os.environ, "ATLAS_PROGRESS_FILE": str(Path(root) / "progress")}
            env.pop("ATLAS_PROGRESS_METHOD", None)
            out = subprocess.run([sys.executable, "-c", FORKED_COORDINATOR, json.dumps(plan), root, str(cpu),
                                  str(grace)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 0, out.stderr)
            result = json.loads(out.stdout)
            lanes = {method: json.loads((Path(root) / method).read_text())
                     for method in plan if (Path(root) / method).exists()}
        for method, lane in lanes.items():
            self.assertEqual(lane["parent"], result["pid"])
            self.assertEqual(lane["session"], lane["pid"])
            self.assertEqual(Path(lane["cwd"]), ROOT)
            self.assertEqual(lane["digits"], 0)
            self.assertTrue(lane["scope"] and lane["worker"])
            self.assertEqual(lane["progress"], method)
            # Standard streams, the coordinator pipe and the listing itself.
            self.assertNotIn(os.devnull, lane["fds"].values())
            self.assertLessEqual(len(lane["fds"]), 5)
            self.assertTrue(lane["fds"]["0"].startswith("pipe:") and lane["fds"]["1"].startswith("pipe:"))
        return result, lanes

    def test_lanes_start_together_and_the_first_keeps_priority(self):
        import os
        if len(os.sched_getaffinity(0)) < 3:
            self.skipTest("Speculative lanes need a processor for each lane.")
        plans = {
            "newton wins after origin": ({"newton": (.6, "record"), "origin": (.05, "record"),
                                          "certificates": (5, "none")}, {"record": {"engine": "newton"}}),
            "origin waits for newton": ({"newton": (.6, "none"), "origin": (.05, "record"),
                                         "certificates": (5, "none")}, {"record": {"engine": "origin"}}),
            "newton crashes": ({"newton": (.3, "crash"), "origin": (.05, "none"),
                                "certificates": (.1, "record")}, {"record": {"engine": "certificates"}}),
            "nonreduced": ({"newton": (.3, "none"), "origin": (.05, "nonreduced"),
                            "certificates": (5, "none")}, {"error": "NonReducedError"}),
            "all fail": ({"newton": (.05, "crash"), "origin": (.05, "none"), "certificates": (.05, "none")},
                         {"error": "RuntimeError"}),
        }
        for name, (plan, expected) in plans.items():
            with self.subTest(name):
                result, lanes = self.coordinate(plan)
                self.assertEqual({k: result[k] for k in expected}, expected)
                self.assertEqual(set(lanes), set(plan))
                begins = [lane["begin"] for lane in lanes.values()]
                self.assertLess(max(begins) - min(begins), .5)
                self.assertTrue(lanes["certificates"]["defer"])
                elapsed = result["end"] - result["begin"]
                # No answer is taken before the first lane's own answer.
                self.assertGreaterEqual(elapsed, plan["newton"][0])
                self.assertLess(elapsed, 3)
        self.assertIn("newton worker exited with status 1", result["message"])

    def test_one_processor_sends_later_lanes_their_request_after_the_head_start(self):
        import os
        cpu = min(os.sched_getaffinity(0))
        for name, plan, grace, winner in (
                ("newton declines", {"newton": (.4, "none"), "origin": (.05, "record"),
                                     "certificates": (5, "none")}, 3, "origin"),
                ("head start expires", {"newton": (10, "none"), "origin": (5, "none"),
                                        "certificates": (.05, "record")}, .3, "certificates")):
            with self.subTest(name):
                result, lanes = self.coordinate(plan, cpu=cpu, grace=grace)
                self.assertEqual(result["record"], {"engine": winner})
                newton = lanes["newton"]
                for method in ("origin", "certificates"):
                    # Forked with Newton, they wait for their request.
                    self.assertLess(abs(lanes[method]["started"] - newton["started"]), .1)
                    self.assertGreaterEqual(lanes[method]["begin"] - newton["begin"], min(plan["newton"][0], grace) - .05)
                self.assertTrue(lanes["certificates"]["defer"])
                self.assertLess(result["end"] - result["begin"], 4)

    def test_forked_lanes_stop_with_their_coordinator(self):
        import os
        import select
        import signal
        import subprocess
        import sys
        # The Newton lane starts a native-like grandchild that holds a FIFO.
        # Its EOF shows the grandchild exited, even before it is reaped.
        coordinator = r"""
import os, sys, time, subprocess
from pathlib import Path
from engine import exhaustive_resolution as e
fifo, go = sys.argv[1:]
def scripted(method, payload):
    if method == "newton":
        subprocess.Popen([sys.executable, "-c",
                          "import os,sys,time; f=os.open(sys.argv[1], os.O_WRONLY); os.write(f, b'r'); time.sleep(30)",
                          fifo])
    if method != "origin":
        time.sleep(30)
    while not Path(go).exists():
        time.sleep(.01)
    return {"record": {"engine": method}, "certified": [[], {"dual_complex": {"status":"complete"}}], "replay": {}}
e._run = scripted
e._canonical_head_start = lambda *args: False
e.NEWTON_GRACE_SECONDS = .05
print(e.search({(2, 0, 0): 1}, list("xyz"), "embedded", compute="cpu"), flush=True)
"""
        for killed in (False, True):
            with self.subTest(killed=killed), tempfile.TemporaryDirectory() as root:
                fifo, go = Path(root) / "fifo", Path(root) / "go"
                os.mkfifo(fifo)
                reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
                process = subprocess.Popen([sys.executable, "-c", coordinator, str(fifo), str(go)], cwd=ROOT,
                                           stdout=subprocess.PIPE, text=True)
                try:
                    self.assertTrue(select.select([reader], [], [], 10)[0])
                    self.assertEqual(os.read(reader, 1), b"r")
                    if killed:
                        process.send_signal(signal.SIGKILL)
                    else:
                        go.touch()
                    stdout, _ = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, -signal.SIGKILL if killed else 0)
                    if not killed:
                        self.assertIn("'engine': 'origin'", stdout)
                    self.assertTrue(select.select([reader], [], [], 5)[0])
                    self.assertEqual(os.read(reader, 1), b"")
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    os.close(reader)

    def test_interpreters_remain_for_threads_gpus_and_fork_servers(self):
        import subprocess
        import sys
        script = """
import os, threading, time
from types import SimpleNamespace
from engine import exhaustive_resolution as e, fork_server
checks = [e._forkable(None, "cpu"), e._forkable(SimpleNamespace(compute="cpu"), None),
          e._forkable(None, None), e._forkable(None, "gpu"), e._forkable(SimpleNamespace(compute="auto"), "cpu")]
stop = threading.Event()
thread = threading.Thread(target=stop.wait, daemon=True)
thread.start()
checks.append(e._forkable(None, "cpu"))
stop.set(); thread.join()
# The joined thread's OS task may take a moment to disappear.
for _ in range(100):
    if len(os.listdir("/proc/self/task")) == 1:
        break
    time.sleep(.01)
checks.append(e._forkable(None, "cpu"))
fork_server.enable()
checks.append(e._forkable(None, "cpu") if fork_server.supported() else False)
print(checks)
"""
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), str([True, True, False, False, False, False, True, False]))

    def test_forked_and_interpreter_lanes_give_the_same_record(self):
        import subprocess
        import sys
        script = """
import json
from engine import exhaustive_resolution as e
from engine.worker import evaluate
forks = []
class Counted(e._ForkedLane):
    def __init__(self, method, coordinator):
        forks.append(method)
        super().__init__(method, coordinator)
e._ForkedLane = Counted
def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in ("elapsed_ms", "native_elapsed_ms")}
    return [strip(v) for v in value] if isinstance(value, list) else value
request = dict(polynomial="x*y*z+x^4+y^4+z^4", variables=list("xyz"), mode="embedded", search="exhaustive")
forked = evaluate(request)
lanes = list(forks)
e._forkable = lambda *args: False
started = evaluate(request)
print(json.dumps(dict(lanes=lanes, forks=len(forks), same=list(strip(list(forked))) == strip(list(started)),
                      status=forked[0], engine=forked[1].get("engine"))))
"""
        out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        result = json.loads(out.stdout)
        self.assertEqual(result["lanes"], ["newton", "origin", "certificates"])
        self.assertEqual(result["forks"], 3)
        self.assertEqual((result["status"], result["engine"]), (200, "newton-polyhedron"))
        self.assertTrue(result["same"])
