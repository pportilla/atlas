"""Coordinate independent exhaustive resolution methods for the same germ."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from . import fork_server
from .exact_json import result_json
from .native_library import current_replay_scope
from .polynomial import LimitError, NonReducedError, integer_from_decimal, rational_text
from .resolution_progress import report


_ROOT = Path(__file__).resolve().parents[1]
NEWTON_GRACE_SECONDS = 2
# With forked lanes, from a preloaded engine or from this solver, the later
# lanes start with the first one. Their answers wait until the first lane
# answers or its grace ends, so the first lane keeps its priority.
SPECULATIVE_LANES = True
# A short wall-clock prefix for the canonical rule only when the native
# proof excludes Newton and the early identity/origin/coordinate graphs.
ORIGIN_GRACE_SECONDS = 0.25


def _canonical_head_start(poly, variables, mode):
    from .native_library import Unavailable, _kernel
    if (mode != "embedded" or len(variables) != 3 or not 6 <= len(poly) <= 128
            or not hasattr(_kernel(), "atlas_newton_tangent_obstruction_abi")):
        return False
    # Bound decimal transport even for unbounded exhaustive caller inputs.
    if any(c.numerator.bit_length() > 64 or c.denominator.bit_length() > 64 or
           any(k > 16 for k in e) for e, c in poly.items()):
        return False
    from .kernel_json import kernel_json
    from .point_geometry import encode_terms
    try:
        return kernel_json("atlas_resolution_candidate", "repeated_tangent_obstruction",
                           width=len(variables), terms=encode_terms(poly))
    except (Unavailable, OSError):
        return False


def _restored(certified):
    """A Newton lane's (atlas, outputs) as the solver builds them in place.

    JSON transport turns each chart's torus rays and linear map into lists.
    """
    atlas, outputs = certified
    for chart in atlas:
        torus = chart.get("torus")
        if isinstance(torus, dict):
            chart["torus"] = {**torus, "rays": tuple(map(tuple, torus["rays"])),
                              "linear_map": tuple(map(tuple, torus["linear_map"]))}
    return atlas, outputs


def _processors():
    return os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)


def _forkable(scope, compute):
    """Whether this process can fork its lanes instead of starting an interpreter for each.

    A fork copies only the calling thread, so no other thread may exist,
    Python or native. A GPU runtime does not survive a fork, so the request
    must use the CPU. A process with a fork server starts its lanes there.
    """
    if not sys.platform.startswith("linux") or fork_server.attached():
        return False
    if (scope.compute if scope is not None else compute) != "cpu":
        return False
    try:
        return threading.active_count() == 1 and len(os.listdir("/proc/self/task")) == 1
    except OSError:
        return False


class _ForkedLane:
    """A lane forked from this process, with the part of Popen that `search` uses.

    The lane leads its own session and waits for its request on standard
    input. It reads the whole request before it answers on standard output.
    """

    def __init__(self, method, coordinator):
        from .native_library import fork_reset
        from .native_process import c_stream_reset
        reset, streams = fork_reset(), c_stream_reset()
        for stream in {sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__}:
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        fds = [*os.pipe(), *os.pipe(), *os.pipe()]
        request_r, request_w, reply_r, reply_w, ready_r, ready_w = fds
        try:
            pid = os.fork()
        except BaseException:
            for fd in fds:
                os.close(fd)
            raise
        if pid == 0:
            _lane(method, request_r, reply_w, coordinator, ready_w, reset, streams)
        try:
            for fd in (request_r, reply_w, ready_w):
                os.close(fd)
            # Like Popen(start_new_session=True), return once the child leads
            # its session, so its process group can be signalled.
            os.read(ready_r, 1)
            os.close(ready_r)
        except BaseException:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            for fd in (request_w, reply_r):
                os.close(fd)
            raise
        self.pid, self.returncode = pid, None
        self.stdin, self.stdout = open(request_w, "wb", buffering=0), open(reply_r, "rb", buffering=0)
        self._lock = threading.Lock()

    def _reap(self, options):
        # Like Popen, a child reaped elsewhere (SIGCHLD ignored) exited with 0.
        try:
            pid, status = os.waitpid(self.pid, options)
        except ChildProcessError:
            pid, status = self.pid, 0
        if pid:
            self.returncode = os.waitstatus_to_exitcode(status)

    def poll(self):
        # Another thread waiting for the lane holds the lock: still running.
        if self.returncode is None and self._lock.acquire(False):
            try:
                if self.returncode is None:
                    self._reap(os.WNOHANG)
            finally:
                self._lock.release()
        return self.returncode

    def wait(self):
        with self._lock:
            if self.returncode is None:
                self._reap(0)
        return self.returncode

    def communicate(self, message):
        data = memoryview(message.encode())
        try:
            while data:
                data = data[self.stdin.write(data):]
        except BrokenPipeError:
            pass  # The lane stopped early; its exit status reports why.
        self.stdin.close()
        output = self.stdout.read()
        self.stdout.close()
        self.wait()
        return output.decode(), None


def _lane(method, request, reply, coordinator, ready, reset, streams):
    """Run one lane in a fork of the coordinator, as an interpreter started for it would. Never returns."""
    code = 1
    try:
        os.setsid()
        os.write(ready, b"1")
        # Like Popen(close_fds=True) and exec: keep the request and reply as
        # standard input and output, an inheritable standard error and the
        # coordinator pipe. Close every other inherited descriptor.
        import fcntl
        try:
            stderr = os.get_inheritable(2)
        except OSError:
            stderr = False
        parked = [fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3) for fd in (request, reply, coordinator)]
        keep = {*parked, *((2,) if stderr else ())}
        for name in os.listdir("/proc/self/fd"):
            if int(name) not in keep:
                try:
                    os.close(int(name))
                except OSError:
                    pass
        for fd, target in zip(parked[:2], (0, 1)):
            os.dup2(fd, target)
            os.close(fd)
        threading.Thread(target=_watch_coordinator, args=(parked[2],), daemon=True).start()
        # The state of a fresh interpreter: inherited objects whose descriptors
        # are closed are never collected here, no handler of the coordinator
        # runs, the native replay hooks are cleared and C buffers are empty.
        import gc
        gc.freeze()
        for sig in signal.valid_signals():
            handler = signal.getsignal(sig)
            if callable(handler) and handler is not signal.default_int_handler:
                signal.signal(sig, signal.default_int_handler if sig == signal.SIGINT else signal.SIG_DFL)
        reset()
        streams()
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(0)
        os.chdir(_ROOT)
        if os.getenv("ATLAS_PROGRESS_FILE"):
            os.environ["ATLAS_PROGRESS_METHOD"] = method
        sys.stdin, sys.stdout = open(0, closefd=False), open(1, "w", closefd=False)
        sys.stderr = open(2, "w", buffering=1, errors="backslashreplace", closefd=False) if stderr else None
        # Fresh face_search and replay_scope contexts, as in a new process.
        import contextvars
        answer = contextvars.Context().run(_run, method, json.load(sys.stdin))
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else 0 if stop.code is None else 1
    except BaseException:
        try:
            import traceback
            traceback.print_exc()
        except BaseException:
            pass
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except BaseException:
                pass
        os._exit(code & 0xff)


def search(poly, variables, mode, *, compute=None, integrals=False, verified=None):
    """Give the first eligible method a short head start, then race the remaining workers.

    Each method has its own process group. Its inherited pipe closes if
    the outer job is killed, so cancellation also reaches native descendants.
    Without a fork server, a single-threaded CPU solver forks every lane
    before any reader thread exists. A lane then waits for its request until
    its turn to race. Otherwise each lane starts an interpreter when it races.
    """
    scope = current_replay_scope()
    request = {
        "terms": [[list(e), rational_text(c)] for e, c in poly.items()],
        "variables": variables, "mode": mode, "compute": compute,
        "replay": scope.settings() if scope else {"compute": compute},
        "integrals": integrals,
    }
    payload = result_json(request)
    replies = queue.SimpleQueue()
    # Every lane process with its coordinator pipe, the lanes that received
    # their request, and forked lanes still waiting for it.
    children, readers, started, waiting = [], [], [], {}

    def collect(method, child, message):
        try:
            output, _ = child.communicate(message)
            if child.returncode:
                raise RuntimeError(f"{method} worker exited with status {child.returncode}")
            reply = json.loads(output, parse_int=integer_from_decimal)
            if not isinstance(reply, dict) or not ("record" in reply or "nonreduced" in reply):
                raise ValueError("incomplete resolution worker reply")
        except (OSError, ValueError, RuntimeError) as error:
            reply = {"error": str(error)}
        replies.put((method, reply))

    errors = []

    def spawn(method, fork):
        """A lane process waiting for its request, or None after recording why it could not start."""
        read_fd, write_fd = os.pipe()
        try:
            child = None
            if fork:
                try:
                    child = _ForkedLane(method, read_fd)
                except OSError:
                    pass  # Start an interpreter instead.
            if child is None:
                environment = {**os.environ, "ATLAS_PROGRESS_METHOD": method} if os.getenv("ATLAS_PROGRESS_FILE") else None
                child = fork_server.popen([sys.executable, "-m", "engine.exhaustive_resolution", method, str(read_fd)],
                                         cwd=_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         env=environment, start_new_session=True, pass_fds=(read_fd,))
        except OSError as error:
            os.close(write_fd)
            errors.append(f"{method}: {error}")
            return None
        finally:
            os.close(read_fd)
        children.append((child, write_fd))
        return child

    def start(method):
        """Send a lane its request, starting it unless it is waiting. Return whether it races."""
        child = waiting.pop(method) if method in waiting else spawn(method, False)
        if child is None:
            return False
        message = payload
        if method == "certificates" and any(peer.poll() is None for peer in started):
            message = result_json(dict(request, defer_spare=True))
        started.append(child)
        reader = threading.Thread(target=collect, args=(method, child, message), daemon=True)
        reader.start()
        readers.append(reader)
        return True

    try:
        if _canonical_head_start(poly, variables, mode):
            first, others, grace = "origin", ("certificates",), ORIGIN_GRACE_SECONDS
        else:
            first, others, grace = "newton", ("origin", "certificates"), NEWTON_GRACE_SECONDS
        def answer(method, reply):
            """The winning record, or None to keep waiting."""
            if "nonreduced" in reply:
                raise NonReducedError(reply["nonreduced"])
            if reply.get("record") is not None:
                bundle = reply.get("certified")
                if integrals and mode == "embedded" and not (isinstance(bundle, list) and len(bundle) == 2
                        and isinstance(bundle[1], dict) and "strata" in bundle[1] and "numerical_data" in bundle[1]):
                    return None
                if scope is not None:
                    scope.merge(reply.get("replay"))
                if verified is not None and isinstance(bundle, list) and len(bundle) == 2:
                    verified.append((reply["record"], _restored(bundle)))
                return reply["record"]
            if "error" in reply:
                errors.append(f"{method}: {reply['error']}")
            return None

        forked = _forkable(scope, compute)
        if forked:
            # Fork every lane while this is the process's only thread.
            for method in (first, *others):
                waiting[method] = spawn(method, True)
        head_start = start(first)
        speculative = (head_start and SPECULATIVE_LANES and len(_processors()) > len(others)
                       and (forked or fork_server.available()))
        if not head_start or speculative:
            for method in others:
                start(method)
        held = []
        received = 0
        end = time.monotonic() + grace
        while received < len(started):
            try:
                method, reply = replies.get(timeout=max(0, end - time.monotonic()) if head_start else None)
            except queue.Empty:
                pass
            else:
                received += 1
                if head_start and speculative and method != first:
                    held.append((method, reply))
                    continue
                record = answer(method, reply)
                if record is not None:
                    return record
            if head_start:
                head_start = False
                if not speculative:
                    for method in others:
                        start(method)
                # The later lanes now race as if they had started here.
                for method, reply in held:
                    record = answer(method, reply)
                    if record is not None:
                        return record
                held.clear()
        if errors:
            raise RuntimeError("Exhaustive resolution could not finish. " + " / ".join(errors))
        return None
    finally:
        for child, write_fd in children:
            os.close(write_fd)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child, _ in children:
            child.wait()
        for reader in readers:
            reader.join()
        # Lanes that never raced still hold their pipes.
        for child, _ in children:
            for stream in (child.stdin, child.stdout):
                if stream is not None:
                    stream.close()


def _run(method, payload):
    from .polynomial import format_poly, kernel_fraction
    from .native_library import replay_scope
    from .newton_faces import face_search

    terms = {tuple(e): kernel_fraction(c) for e, c in payload["terms"]}
    need_strata = payload.get("integrals", False) and payload["mode"] == "embedded"
    with replay_scope(**payload["replay"]) as replay, face_search(True):
        certified = None
        try:
            if method == "newton":
                from .newton_backend import resolve_newton
                report("Testing a Newton polyhedron resolution")
                verified = []
                try:
                    record = resolve_newton(terms, payload["variables"], payload["mode"],
                                            compute=payload["compute"], seconds=1200, defer_outputs=True,
                                            verified=verified, integrals=payload.get("integrals", False),
                                            construct=False)
                except (LimitError, subprocess.TimeoutExpired):
                    record = None
                certified = next((value for producer, value in verified if producer is record), None)
            elif method == "origin":
                from .origin_backend import _resolve, certified_outputs
                report("Searching for smooth blow-up centres over the origin")
                record = _resolve(terms, payload["variables"], payload["mode"], None, defer_outputs=True)
                if record is not None:
                    certified = certified_outputs(record, payload["variables"], payload["mode"],
                                                  payload.get("integrals", False))
            elif method == "certificates":
                from .general_resolution import search as certificates
                report("Searching for a certified blow-up centre")
                verified = []
                record = certificates(terms, payload["variables"], payload["mode"], exhaustive=True,
                                      prewarm_spare=not payload.get("defer_spare", False),
                                      integrals=payload.get("integrals", False), verified=verified,
                                      resolve_reduced_first=True)
                certified = next((value for producer, value in verified if producer is record), None)
            else:
                raise ValueError("Unknown resolution method.")
        except NonReducedError as error:
            return {"nonreduced": str(error)}
        if need_strata and record is not None and method != "certificates":
            from .general_resolution import integral_certificate
            names = payload["variables"]
            record.update(variables=names, mode="embedded", polynomial=format_poly(terms,names), origin=[0]*len(names))
            certified = integral_certificate(record, terms, certified)
            if certified is None:
                record = None
        if record is not None and certified is not None:
            return {"record":record, "certified":list(certified), "replay":replay.report()}
        return {"record": record, "replay": replay.report()}


def _watch_coordinator(read_fd):
    # The coordinator exclusively owns the write end. Its exit (including
    # SIGKILL during UI cancellation) closes this pipe, even if the worker
    # is inside a synchronous native calculation or waiting on a compiler.
    with os.fdopen(read_fd, "rb") as parent:
        parent.read()
    os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        threading.Thread(target=_watch_coordinator, args=(int(sys.argv[2]),), daemon=True).start()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    print(json.dumps(_run(sys.argv[1], json.load(sys.stdin)), separators=(",", ":")))
