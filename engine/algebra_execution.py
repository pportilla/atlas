"""Device selection and observed execution stages for one rational algebra request."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter, perf_counter_ns

from .native_library import current_replay_scope, replay_scope


_current = ContextVar("atlas_algebra_execution", default=None)


def current_execution():
    run = _current.get()
    return run if run is not None and current_replay_scope() is run.replay else None


class _Execution:
    def __init__(self, replay, started):
        self.replay = replay
        self.started = started
        self.stages = []
        self.reasons = []
        self.label = None
        self.previous = {}

    def append(self, operation, backend, details):
        stage = dict(operation=operation, backend=backend, **details)
        if self.label is not None:
            stage["label"] = self.label
        self.stages.append(stage)

    def collect(self):
        # Snapshot monotonically increasing, request-local counters at stage
        # boundaries. One quotient can compute several different bases.
        report = self.replay.report()
        if not report:
            return
        previous = self.previous
        for op, now, before in (
            ("groebner_trace", report.get("trace", {}).get("backends", {}),
             previous.get("trace", {}).get("backends", {})),
            ("groebner_replay", report.get("backends", {}), previous.get("backends", {})),
        ):
            for backend, counts in now.items():
                delta = {key: n - before.get(backend, {}).get(key, 0) for key, n in counts.items()}
                if delta["batches"]:
                    device = report.get("devices", {}).get(backend)
                    if device:
                        delta["device"] = device
                    self.append(op, backend, delta)
        cert = report.get("certification", {})
        delta = {key: n - previous.get("certification", {}).get(key, 0) for key, n in cert.items()}
        if any(delta.values()):
            self.append("groebner_certification", "cpu", delta)
        for reason in report.get("fallback_reasons", ()):
            if reason not in self.reasons:
                self.reasons.append(reason)
        self.previous = report

    def report(self):
        self.collect()
        devices = {stage["backend"]: stage["device"] for stage in self.stages
                   if stage["backend"] in ("opencl", "triton") and "device" in stage}
        names = [f"{device.get('name', 'GPU')} ({backend}, index {device['index']})"
                 for backend, device in devices.items()]
        backends = list(devices)
        if any(s["backend"] == "reference" for s in self.stages):
            backends.append("reference")
        backend = "cpu"
        if backends:
            backend = backends[0] if len(backends) == 1 else "mixed"
        result = dict(requested=self.replay.compute, backend=backend,
                      device=" and ".join([*names, "CPU"]), stages=list(self.stages),
                      fallback_reasons=list(self.reasons),
                      elapsed_ms=round(1000 * (perf_counter() - self.started), 3))
        if self.replay.compute == "gpu" and not devices:
            result["note"] = (" ".join(self.reasons) if self.reasons else
                              "The request ran on the CPU. No eligible device batch was executed.")
        return result


@contextmanager
def algebra_execution(compute="auto", *, backend=None, index=None):
    """One scope across parsing, every computational stage and host checks."""
    started = perf_counter()
    with replay_scope(compute, backend=backend, index=index) as replay:
        run = _Execution(replay, started)
        token = _current.set(run)
        try:
            yield run
        finally:
            _current.reset(token)


@contextmanager
def stage_label(name):
    """Attribute nested basis work to its place in a derived ideal operation."""
    run = current_execution()
    if run is None:
        yield
        return
    run.collect()
    previous = run.label
    run.label = name
    try:
        yield
    finally:
        try:
            run.collect()
        finally:
            run.label = previous


def record_stage(operation, backend, **details):
    run = current_execution()
    if run is not None:
        run.collect()
        run.append(operation, backend, details)


@contextmanager
def cpu_stage(operation, **details):
    """Record the elapsed time and work counts of a completed CPU stage."""
    started = perf_counter_ns()
    yield
    record_stage(operation, "cpu", elapsed_ns=perf_counter_ns() - started, **details)


def record_fallback(reason):
    run = current_execution()
    if run is not None and run.replay.compute != "cpu" and reason not in run.reasons:
        run.reasons.append(reason)


def packed_basis(library, payload, lines, can_homogenize, known, order):
    """Offer eligible explicit GPU requests to the certified modular solver."""
    run = current_execution()
    if run is None:
        return library.atlas_groebner(payload, len(payload))
    retry_on_cpu = False
    if run.replay.compute == "gpu" and known < len(lines):
        # The trace homogenizes with one additional packed variable. One
        # generator, a monomial ideal or a supplied constant needs no pairs.
        nonzero = [line for line in lines if line]
        needs_pairs = (len(nonzero) > 1 and any(";" in line or "*" in line for line in nonzero)
                       and not any(_constant_row(line) for line in nonzero))
        if needs_pairs and can_homogenize:
            modular = getattr(library, "atlas_groebner_modular", None)
            if modular is not None:
                handle = modular(payload, len(payload), None, None)
                if handle:
                    return handle
                reason = library.atlas_groebner_error().decode("utf-8", errors="replace")
                record_fallback(f"Modular basis declined: {reason}")
            else:
                record_fallback("The installed kernel has no certified modular entry point.")
            retry_on_cpu = True
        elif needs_pairs:
            record_fallback("The modular basis needs one more packed variable for homogenization.")
    if retry_on_cpu:
        # The general solver can enter modular_basis again on coefficient swell.
        # Retry with host hooks so a declined device route cannot be repeated.
        with replay_scope("cpu") as host:
            handle = library.atlas_groebner(payload, len(payload))
        run.replay.merge(host.report())
    else:
        handle = library.atlas_groebner(payload, len(payload))
    if handle:
        algorithm = library.atlas_groebner_route(handle)
        if algorithm != 1:
            try:
                record_stage("groebner_basis", "cpu", algorithm="hilbert" if algorithm == 2 else "direct",
                             order=order, generators=len(lines), known=known)
            except BaseException:
                library.atlas_groebner_free(handle)
                raise
    return handle


def _constant_row(row):
    return ";" not in row and "*" not in row and all(power == "0" for power in row.split()[1:])
