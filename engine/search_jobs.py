"""A server-owned, cancellable exhaustive search, isolated in a process group."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    id: str
    request: dict
    child: subprocess.Popen
    started: float
    state: str = "running"
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    watcher: threading.Thread | None = None


class SearchJobs:
    """Keep one current job. Only completed records expire, or are replaced."""
    def __init__(self, *, command=None, retention=1800):
        self.command = command or [sys.executable, "-m", "engine.worker"]
        self.retention = retention
        self.lock = threading.Lock()
        self.job = None
        self.expiry = None
        self.closed = False

    @staticmethod
    def _view(job):
        end = job.finished if job.finished is not None else time.monotonic()
        view = {"id":job.id, "state":job.state, "request":job.request,
                "elapsed_ms":round(1000*(end-job.started))}
        if job.result is not None:
            view["result"] = job.result
        if job.error:
            view["error"] = job.error
        return view

    def start(self, data):
        if not isinstance(data,dict):
            raise ValueError("Send a polynomial, its variables, and a resolution mode.")
        request = {key:data[key] for key in ("polynomial","variables","mode","compute","monodromy") if key in data}
        # All polynomial parsing/expansion belongs in the isolated worker:
        # a large exhaustive input is permitted and may itself take time.
        if not isinstance(request.get("polynomial"),str) or not request["polynomial"].strip():
            raise ValueError("Enter a nonempty polynomial.")
        names=request.get("variables")
        if (not isinstance(names,list) or not names or any(not isinstance(name,str) or
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}",name) for name in names)
            or len(set(names))!=len(names)):
            raise ValueError("Provide distinct variable names starting with a letter.")
        if request.get("mode","embedded") not in ("embedded","intrinsic"):
            raise ValueError("Mode must be embedded or intrinsic.")
        if request.get("compute") not in (None,"auto","cpu","gpu"):
            raise ValueError("Compute must be cpu, auto, or gpu.")
        if not isinstance(request.get("monodromy", False), bool):
            raise ValueError("monodromy must be true or false.")
        with self.lock:
            if self.closed:
                raise RuntimeError("The search server is shutting down.")
            if self.job and self.job.finished is None:
                raise RuntimeError("An exhaustive search is already running on this server.")
            child = subprocess.Popen(self.command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, start_new_session=True)
            if self.expiry:
                self.expiry.cancel()
                self.expiry = None
            job = Job(secrets.token_hex(16), request, child, time.monotonic())
            self.job = job
            job.watcher = threading.Thread(target=self._collect, args=(job,), daemon=True)
            try:
                job.watcher.start()
            except RuntimeError:
                self._stop(job)
                child.communicate()
                self.job = None
                raise
            return self._view(job)

    @staticmethod
    def _stop(job):
        try:
            os.killpg(job.child.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _collect(self, job):
        result, error = None, None
        try:
            output, _ = job.child.communicate(json.dumps({**job.request, "search":"exhaustive"}))
            if job.child.returncode == 0:
                code, payload = json.loads(output)
                if code==200 and isinstance(payload,dict):
                    result = payload
                else:
                    error = payload.get("error","The exhaustive search did not finish.")
            else:
                error = "The search process exited before returning a result."
        except (OSError, ValueError, TypeError, AttributeError):
            error = "The search process did not return a readable result."
        with self.lock:
            if job.state == "cancelling":
                job.state = "cancelled"
            elif error:
                job.state, job.error = "failed", error
            else:
                job.state, job.result = "completed", result
            job.finished = time.monotonic()
            if not self.closed and self.job is job:
                self.expiry = threading.Timer(self.retention,self._expire,args=(job.id,))
                self.expiry.daemon = True
                self.expiry.start()

    def _expire(self, ident):
        with self.lock:
            if self.job and self.job.id==ident and self.job.finished is not None:
                self.job = self.expiry = None

    def get(self, ident=None):
        with self.lock:
            if self.job is None or (ident is not None and ident!=self.job.id):
                return None
            return self._view(self.job)

    def cancel(self, ident):
        with self.lock:
            if self.job is None or self.job.id!=ident:
                return None
            if self.job.state=="running":
                self.job.state="cancelling"
                self._stop(self.job)
            return self._view(self.job)

    def close(self):
        with self.lock:
            self.closed = True
            job = self.job
            if self.expiry:
                self.expiry.cancel()
            if job and job.finished is None:
                job.state="cancelling"
                self._stop(job)
        if job and job.watcher:
            job.watcher.join()
        with self.lock:
            self.job = self.expiry = None
