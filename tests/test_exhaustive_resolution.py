import tempfile
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
        self.assertTrue(origin.call_args.kwargs["tangent_only"])
        self.assertGreater(origin.call_args.kwargs["seconds"], 0)
        exhaustive.assert_not_called()
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["monodromy_zeta"]["status"], "unavailable")

    def test_newton_limit_leaves_fallbacks_eligible(self):
        payload = dict(terms=[[[2, 0, 0], "1"]], variables=list("xyz"), mode="embedded",
                       compute="cpu", replay={"compute": "cpu"})
        with patch("engine.newton_backend.resolve_newton", side_effect=LimitError("fan limit")) as newton:
            self.assertIsNone(exhaustive._run("newton", payload)["record"])
        self.assertEqual(newton.call_args.kwargs, {"compute": "cpu", "seconds": 1200, "defer_outputs": True})

    def test_canonical_head_start_covers_only_certified_newton_failures(self):
        from engine.newton_backend import coordinate_choices, resolve_newton
        from engine.newton_faces import face_search
        from engine.newton_fan import build_fan
        from engine.native_library import _MODE, _kernel
        if _MODE != "native" or not hasattr(_kernel(), "atlas_newton_tangent_obstruction_abi"):
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
    (p/'racing').touch()
else:
    raise AssertionError('proven Newton failure should never start')
record = None if outcome=='decline' and method=='origin' else {'engine':method}
print(json.dumps({'record':record, 'replay':{'winner':method}}))
"""
        for outcome in ("win","decline","error","slow","failed_start","nonreduced"):
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
                     patch.object(exhaustive,"ORIGIN_GRACE_SECONDS",grace), \
                     patch.object(exhaustive.subprocess,"Popen",side_effect=launch), \
                     patch.object(exhaustive,"current_replay_scope",return_value=replay):
                    if outcome=="nonreduced":
                        from engine.polynomial import NonReducedError
                        with self.assertRaisesRegex(NonReducedError,"repeated factor"):
                            exhaustive.search({(2,0,0):1},list("xyz"),"embedded",compute="cpu")
                        replay.merge.assert_not_called()
                    else:
                        expected = "origin" if outcome=="win" else "certificates"
                        result=exhaustive.search({(2,0,0):1},list("xyz"),"embedded",compute="cpu")
                        self.assertEqual(result,{"engine":expected})
                        replay.merge.assert_called_once_with({"winner":expected})
                self.assertTrue(all(p.poll() is not None for p in processes.values()))
                self.assertEqual(set(elapsed), {"origin"} if outcome in ("win","nonreduced") else {"origin","certificates"})
                if "certificates" in elapsed:
                    wait=elapsed["certificates"]-elapsed["origin"]
                    self.assertLess(wait,3)
                    if outcome=="slow":
                        self.assertGreaterEqual(wait,grace)
                        self.assertTrue((Path(path)/"racing").exists())

    def test_newton_and_fallbacks_complete_independently(self):
        # Real workers use marker files to synchronize, so a slow CI machine
        # cannot change which method wins a given case.
        from subprocess import Popen
        script = """
import json, os, sys, time
from pathlib import Path
method, winner, root = sys.argv[1:]
root = Path(root)
args = json.load(sys.stdin)
assert args['replay'] == {'compute': 'gpu', 'backend': 'opencl', 'index': 1}
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
print(json.dumps({'record': {'engine': method, 'compute': args['compute']}, 'replay': {'winner': method}}))
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
                     patch.object(exhaustive, "NEWTON_GRACE_SECONDS", grace), \
                     patch.object(exhaustive, "current_replay_scope", return_value=scope):
                    result = exhaustive.search({(2, 0, 0): 1}, list("xyz"), "embedded", compute="gpu")
                expected = "newton" if winner == "fast" else winner
                self.assertEqual(result, {"engine": expected, "compute": "gpu"})
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
        print(json.dumps({'record': {'engine': method}}))
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
