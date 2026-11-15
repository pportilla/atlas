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
from .native_library import current_replay_scope
from .polynomial import LimitError, NonReducedError, integer_from_decimal, rational_text
from .resolution_progress import report


NEWTON_GRACE_SECONDS = 2
# With lanes forked from a preloaded engine, the later lanes start with the
# first one. Their answers wait until the first lane answers or its grace
# ends, so the first lane keeps its priority.
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


def search(poly, variables, mode, *, compute=None, integrals=False, verified=None):
    """Give the first eligible method a short head start, then race the remaining workers.

    Each method has its own process group. Its inherited pipe closes if
    the outer job is killed, so cancellation also reaches native descendants.
    """
    scope = current_replay_scope()
    request = {
        "terms": [[list(e), rational_text(c)] for e, c in poly.items()],
        "variables": variables, "mode": mode, "compute": compute,
        "replay": scope.settings() if scope else {"compute": compute},
        "integrals": integrals,
    }
    payload = json.dumps(request)
    replies = queue.SimpleQueue()
    children, readers = [], []

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

    def start(method):
        message = payload
        if method == "certificates" and any(peer.poll() is None for peer, _ in children):
            message = json.dumps(dict(request, defer_spare=True))
        read_fd, write_fd = os.pipe()
        try:
            environment = {**os.environ, "ATLAS_PROGRESS_METHOD": method} if os.getenv("ATLAS_PROGRESS_FILE") else None
            child = fork_server.popen([sys.executable, "-m", "engine.exhaustive_resolution", method, str(read_fd)],
                                     cwd=Path(__file__).resolve().parents[1],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=environment,
                                     start_new_session=True, pass_fds=(read_fd,))
        except OSError as error:
            os.close(write_fd)
            errors.append(f"{method}: {error}")
            return
        finally:
            os.close(read_fd)
        children.append((child, write_fd))
        reader = threading.Thread(target=collect, args=(method, child, message), daemon=True)
        reader.start()
        readers.append(reader)

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
                if scope is not None:
                    scope.merge(reply.get("replay"))
                if verified is not None and isinstance(reply.get("certified"), list) and len(reply["certified"]) == 2:
                    verified.append((reply["record"], _restored(reply["certified"])))
                return reply["record"]
            if "error" in reply:
                errors.append(f"{method}: {reply['error']}")
            return None

        start(first)
        head_start = bool(children)
        speculative = (head_start and SPECULATIVE_LANES and len(_processors()) > len(others)
                       and fork_server.available())
        if not head_start or speculative:
            for method in others:
                start(method)
        held = []
        received = 0
        end = time.monotonic() + grace
        while received < len(children):
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


def _run(method, payload):
    from .polynomial import kernel_fraction
    from .native_library import replay_scope
    from .newton_faces import face_search

    terms = {tuple(e): kernel_fraction(c) for e, c in payload["terms"]}
    with replay_scope(**payload["replay"]) as replay, face_search(True):
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
                if record is not None and certified is not None:
                    return {"record": record, "certified": list(certified), "replay": replay.report()}
            elif method == "origin":
                from .origin_backend import _resolve
                report("Searching for smooth blow-up centres over the origin")
                record = _resolve(terms, payload["variables"], payload["mode"], None, defer_outputs=True)
            elif method == "certificates":
                from .general_resolution import search as certificates
                report("Searching for a certified blow-up centre")
                record = certificates(terms, payload["variables"], payload["mode"], exhaustive=True,
                                      prewarm_spare=not payload.get("defer_spare", False))
            else:
                raise ValueError("Unknown resolution method.")
        except NonReducedError as error:
            return {"nonreduced": str(error)}
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
