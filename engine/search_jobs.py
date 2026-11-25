"""A bounded queue of cancellable resolutions with retained result records."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time

from . import fork_server
from .polynomial import integer_from_decimal
from .resolution_zetas import selected_zetas

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    id: str
    request: dict
    started: float
    state: str = "queued"
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    child: subprocess.Popen | None = None
    watcher: threading.Thread | None = None
    progress: dict | None = None
    zeta_lock: threading.Lock = field(default_factory=threading.Lock)


class SearchJobs:
    def __init__(self, *, command=None, retention=1800, slots=2):
        self.command = command or [sys.executable, "-m", "engine.worker"]
        self.retention = retention
        self.slots = slots
        self.lock = threading.Lock()
        self.jobs = {}
        self.closed = False
        self.progress_dir = None

    def _read_progress(self, job):
        if self.progress_dir is None:
            return
        path = Path(self.progress_dir.name) / job.id
        try:
            snapshot = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return
        if isinstance(snapshot, dict) and isinstance(snapshot.get("stage"), str):
            progress = {"stage": snapshot["stage"]}
            if isinstance(snapshot.get("method"), str):
                progress["method"] = snapshot["method"]
            if type(snapshot.get("blowups")) is int and snapshot["blowups"] >= 0:
                progress["blowups"] = snapshot["blowups"]
            job.progress = progress

    def _view(self, job):
        if job.state in ("running", "cancelling"):
            self._read_progress(job)
        end = job.finished if job.finished is not None else time.monotonic()
        view = {"id":job.id, "state":job.state, "request":job.request,
                "elapsed_ms":round(1000*(end-job.started))}
        if job.result is not None:
            view["result"] = job.result
        if job.progress is not None:
            view["progress"] = job.progress
        if job.error:
            view["error"] = job.error
        return view

    def _prune(self):
        cutoff = time.monotonic() - self.retention
        for ident in [job.id for job in self.jobs.values() if job.finished is not None and job.finished < cutoff]:
            del self.jobs[ident]

    def start(self, data):
        if not isinstance(data, dict):
            raise ValueError("Send a polynomial, its variables, and a resolution mode.")
        fields = ("polynomial", "variables", "mode", "compute", "monodromy", "zetas", "zetas_only", "search", "method",
                  "seconds")
        request = {key: data[key] for key in fields if key in data}
        # A budget in seconds asks for a quick search.
        request.setdefault("search", "exhaustive" if request.get("seconds") is None else "quick")
        if request["search"] not in ("auto", "quick", "exhaustive"):
            raise ValueError("Search must be auto, quick or exhaustive.")
        from .quick_budget import request_seconds
        request_seconds(request)
        if not isinstance(request.get("polynomial"), str) or not request["polynomial"].strip():
            raise ValueError("Enter a nonempty polynomial.")
        names = request.get("variables")
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) or
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", name) for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Provide distinct variable names starting with a letter.")
        if request.get("method", "auto") not in ("auto", "weighted", "jung"):
            raise ValueError("Method must be auto, weighted or jung.")
        if request.get("method") == "jung":
            request.setdefault("mode", "intrinsic")
        if request.get("mode", "embedded") not in ("embedded", "intrinsic"):
            raise ValueError("Mode must be embedded or intrinsic.")
        if request.get("compute") not in (None, "auto", "cpu", "gpu"):
            raise ValueError("Compute must be cpu, auto, or gpu.")
        selected_zetas(request)
        with self.lock:
            if self.closed:
                raise RuntimeError("The search server is shutting down.")
            self._prune()
            job = Job(secrets.token_hex(16), request, time.monotonic())
            self.jobs[job.id] = job
            self._dispatch()
            return self._view(job)

    def _launch(self, job):
        if self.progress_dir is None:
            self.progress_dir = tempfile.TemporaryDirectory(prefix="atlas-search-")
        path = Path(self.progress_dir.name) / job.id
        path.unlink(missing_ok=True)
        job.progress = None
        return fork_server.popen(self.command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                env={**os.environ, "ATLAS_PROGRESS_FILE": str(path)},
                                text=True, start_new_session=True)

    def _dispatch(self):
        """Start waiting jobs in submission order. Caller holds the lock."""
        if self.closed:
            return
        active = sum(job.state in ("running", "cancelling") for job in self.jobs.values())
        for job in self.jobs.values():
            if active >= self.slots:
                break
            if job.state != "queued":
                continue
            try:
                job.child = self._launch(job)
                job.watcher = threading.Thread(target=self._collect, args=(job,), daemon=True)
                job.watcher.start()
            except (OSError, RuntimeError) as exc:
                if job.child is not None:
                    self._stop(job)
                    job.child.communicate()
                job.state, job.error, job.finished = "failed", str(exc), time.monotonic()
                continue
            job.state = "running"
            active += 1

    @staticmethod
    def _stop(job):
        try:
            os.killpg(job.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _run_attempt(self, job, policy):
        from .quick_budget import expired, read_progress, request_seconds
        # The quick attempt has the budget of the request. An exhaustive one has none.
        request = {key: value for key, value in job.request.items() if policy == "quick" or key != "seconds"}
        request["search"] = policy
        timeout = request_seconds(request)
        started = time.monotonic()
        try:
            output, _ = job.child.communicate(json.dumps(request), timeout=timeout)
        except subprocess.TimeoutExpired:
            self._stop(job)
            job.child.communicate()
            progress = read_progress(Path(self.progress_dir.name) / job.id) if self.progress_dir else None
            return expired(request, timeout, time.monotonic() - started, progress)
        if job.child.returncode:
            raise RuntimeError("The search process exited before returning a result.")
        return json.loads(output, parse_int=integer_from_decimal)

    def _collect(self, job):
        from .resolve_cli import needs_exhaustive_search
        result, error = None, None
        policy = job.request["search"]
        try:
            code, payload = self._run_attempt(job, "quick" if policy == "auto" else policy)
            if policy == "auto" and needs_exhaustive_search(job.request, code, payload):
                with self.lock:
                    if job.state != "running":
                        raise RuntimeError("Search cancelled.")
                    job.child = self._launch(job)
                code, payload = self._run_attempt(job, "exhaustive")
            if code == 200 and isinstance(payload, dict):
                result = payload
            else:
                error = payload.get("error", "The search did not finish.")
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        except (ValueError, TypeError, AttributeError):
            error = "The search process did not return a readable result."
        with self.lock:
            if job.state == "cancelling":
                job.state = "cancelled"
            elif error:
                job.state, job.error = "failed", error
            else:
                job.state, job.result = "completed", result
            self._read_progress(job)
            (Path(self.progress_dir.name) / job.id).unlink(missing_ok=True)
            job.finished = time.monotonic()
            self._dispatch()

    def get(self, ident=None):
        with self.lock:
            self._prune()
            job = next(reversed(self.jobs.values()), None) if ident is None else self.jobs.get(ident)
            return self._view(job) if job else None

    def list(self):
        with self.lock:
            self._prune()
            return [self._view(job) for job in reversed(self.jobs.values())]

    def embedded_companion(self, record):
        """Locate a finished embedded atlas for exactly the same canonical input."""
        if record.get("mode") != "intrinsic" or record.get("embedded_companion"):
            return None
        with self.lock:
            for job in reversed(self.jobs.values()):
                other = job.result
                if (other and other.get("mode") == "embedded" and other.get("status") == "resolved"
                        and other.get("variables") == record.get("variables")
                        and other.get("origin") == record.get("origin")
                        and other.get("polynomial") == record.get("polynomial") and other.get("atlas")):
                    return {"atlas": other["atlas"]}
        return None

    def zeta_lock(self, ident):
        """Serialize follow-ups for this result so each worker sees the previous zeta."""
        with self.lock:
            self._prune()
            job = self.jobs.get(ident)
            return job.zeta_lock if job else None

    def attach_zetas(self, ident, outputs, companion=None):
        with self.lock:
            job = self.jobs.get(ident)
            if job is None:
                return None
            # Replace the mapping: earlier snapshots can be serialized on another thread.
            job.result = {**job.result, **outputs}
            if companion is not None:
                job.result["embedded_companion"] = companion
            return self._view(job)

    def cancel(self, ident):
        with self.lock:
            self._prune()
            job = self.jobs.get(ident)
            if job is None:
                return None
            if job.state == "running":
                job.state = "cancelling"
                self._stop(job)
            elif job.state == "queued":
                job.state, job.finished = "cancelled", time.monotonic()
            return self._view(job)

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                if job.state in ("running", "cancelling"):
                    job.state = "cancelling"
                    self._stop(job)
            watchers = [job.watcher for job in self.jobs.values() if job.watcher is not None and job.watcher.ident]
        for watcher in watchers:
            watcher.join()
        with self.lock:
            self.jobs.clear()
            if self.progress_dir is not None:
                self.progress_dir.cleanup()
                self.progress_dir = None
