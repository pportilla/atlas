"""Run constructive blowups and the general certificate search for the same germ."""
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .native_library import current_replay_scope
from .polynomial import NonReducedError, integer_from_decimal, rational_text


def search(poly, variables, mode):
    """A slow centre calculation can continue while ideal certificates are tried.

    Both children stay in the outer job's process group for UI cancellation.
    The verifier's own workers monitor their parent and exit when it is reaped.
    """
    scope = current_replay_scope()
    payload = json.dumps({
        "terms": [[list(e), rational_text(c)] for e, c in poly.items()],
        "variables": variables, "mode": mode, "replay": scope.settings() if scope else {},
    })
    replies = queue.SimpleQueue()
    children, readers = [], []

    def collect(method, child):
        try:
            output, _ = child.communicate(payload)
            if child.returncode:
                raise RuntimeError(f"{method} worker exited with status {child.returncode}")
            reply = json.loads(output, parse_int=integer_from_decimal)
            if not isinstance(reply, dict) or not ("record" in reply or "nonreduced" in reply):
                raise ValueError("incomplete resolution worker reply")
        except (OSError, ValueError, RuntimeError) as error:
            reply = {"error": str(error)}
        replies.put((method, reply))

    errors = []
    try:
        for method in ("origin", "certificates"):
            try:
                child = subprocess.Popen([sys.executable, "-m", "engine.exhaustive_resolution", method],
                                         cwd=Path(__file__).resolve().parents[1],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            except OSError as error:
                errors.append(f"{method}: {error}")
                continue
            children.append(child)
            reader = threading.Thread(target=collect, args=(method, child), daemon=True)
            reader.start()
            readers.append(reader)
        for _ in children:
            method, reply = replies.get()
            if "nonreduced" in reply:
                raise NonReducedError(reply["nonreduced"])
            if reply.get("record") is not None:
                if scope is not None:
                    scope.merge(reply.get("replay"))
                return reply["record"]
            if "error" in reply:
                errors.append(f"{method}: {reply['error']}")
        if errors:
            raise RuntimeError("Exhaustive resolution could not finish. " + " / ".join(errors))
        return None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait()
        for reader in readers:
            reader.join()


def _run(method, payload):
    from fractions import Fraction
    from .native_library import replay_scope
    from .newton_faces import face_search

    terms = {tuple(e): Fraction(c) for e, c in payload["terms"]}
    with replay_scope(**payload["replay"]) as replay, face_search(True):
        try:
            if method == "origin":
                from .origin_backend import _resolve
                record = _resolve(terms, payload["variables"], payload["mode"], None, defer_outputs=True)
            elif method == "certificates":
                from .general_resolution import search as certificates
                record = certificates(terms, payload["variables"], payload["mode"], exhaustive=True)
            else:
                raise ValueError("Unknown resolution method.")
        except NonReducedError as error:
            return {"nonreduced": str(error)}
        return {"record": record, "replay": replay.report()}


if __name__ == "__main__":
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    print(json.dumps(_run(sys.argv[1], json.load(sys.stdin)), separators=(",", ":")))
