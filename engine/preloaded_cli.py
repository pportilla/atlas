"""Advertise a ready Linux engine to standalone resolve supervisors in this checkout.

The caller continues to enforce its CLI budget. A guardian beside the warm
worker reaps its whole subtree before acknowledging completion or cancellation.
"""
import ctypes
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import struct
import sys


def listener(root):
    """One local listener per user and checkout, released by the kernel on exit."""
    try:
        info = (root / "atlas").stat()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            sock.bind(f"\0atlas.cli.1.{os.geteuid()}.{info.st_dev}.{info.st_ino}")
            sock.listen(8)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
    except OSError:
        return None
    return sock


def _context(proc):
    # A named engine is shared only across equivalent execution contexts.
    # This also excludes seccomp filters, whose programs cannot be compared.
    status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines())
    if status.get("Seccomp", "").strip() != "0":
        raise ValueError("filtered process")
    credentials = tuple(status.get(k) for k in ("Uid", "Gid", "Groups", "Umask", "NoNewPrivs",
                        "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))
    return (credentials, tuple(sorted((n.name, n.stat().st_ino) for n in (proc / "ns").iterdir())),
            (proc / "root").stat().st_ino, (proc / "root").stat().st_dev,
            (proc / "cgroup").read_bytes(), (proc / "limits").read_bytes(),
            (proc / "attr/current").read_bytes())


def _compatible(env):
    # CPython, the dynamic loader and native thread runtimes consult these
    # before the worker can install its request environment.
    prefixes = ("PYTHON", "LD_", "LC_", "ATLAS_GROEBNER", "OMP_", "OPENBLAS_", "FLINT_")
    def startup(items):
        return {k: v for k, v in items.items() if k.startswith(prefixes) or k in ("LANG", "TZ", "VIRTUAL_ENV", "TMPDIR")}
    if any(stream is None for stream in (sys.stdin, sys.stdout, sys.stderr)):
        return False
    if any(env.get(k) for k in ("PYTHONPROFILEIMPORTTIME", "PYTHONINSPECT", "PYTHONTRACEMALLOC", "LD_DEBUG")):
        return False
    python = shutil.which("python3", path=env.get("PATH"))
    return (python and Path(python).parent.resolve() == Path(sys.executable).parent.resolve()
            and os.path.samefile(python, sys.executable) and startup(env) == startup(os.environ))


def accept(sock, root, data, descriptors, flags, fresh):
    """Fork a guardian only for the current supervisor, code image and credentials."""
    try:
        if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(descriptors) != 5 or not fresh():
            return None
        pid, uid, gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        proc = Path("/proc") / str(pid)
        if (uid != os.geteuid() or gid != os.getegid()
                or not (proc / "exe").samefile(root / "native/build/resolve_supervisor")
                or _context(proc) != _context(Path("/proc/self"))
                or os.getpriority(os.PRIO_PROCESS, pid) != os.getpriority(os.PRIO_PROCESS, 0)
                or os.sched_getscheduler(pid) != os.sched_getscheduler(0)
                or os.sched_getparam(pid) != os.sched_getparam(0)):
            return None
        affinity = os.sched_getaffinity(pid)
        argc, = struct.unpack_from("=I", data)
        fields = [os.fsdecode(s) for s in data[4:].split(b"\0")]
        if fields.pop() or argc < 1 or len(fields) != argc + 2 or fields[0] != "resolve":
            return None
        args = fields[:argc]
        environ = (proc / "environ").read_bytes().split(b"\0")
        env = dict(os.fsdecode(s).split("=", 1) for s in environ if s)
        for key, value in zip(("ATLAS_PROGRESS_FILE", "ATLAS_PROGRESS_OWNED"), fields[argc:]):
            if value or key in env:
                env[key] = value
        if not _compatible(env):
            return None
        guardian = os.fork()
        if guardian == 0:
            try:
                _guard(sock, descriptors, affinity, args, env)
            finally:
                os._exit(1)
        return guardian
    except (OSError, ValueError, IndexError, struct.error):
        return None


def _guard(sock, descriptors, affinity, args, env):
    import fcntl
    os.setsid()
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        return
    # Close every descriptor of the engine, including its session control
    # socket. The only objects inherited by the calculation belong to the CLI.
    kept = [fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 16) for fd in [*descriptors, sock.detach()]]
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd not in kept:
            try:
                os.close(fd)
            except OSError:
                pass
    channel = socket.socket(fileno=kept.pop())
    channel.setblocking(True)
    cwd = kept.pop()
    os.fchdir(cwd)
    os.close(cwd)
    os.sched_setaffinity(0, affinity)
    os.environ.clear()
    os.environ.update(env)
    # Imported modules may have cached a temporary directory while preloading.
    if "tempfile" in sys.modules:
        sys.modules["tempfile"].tempdir = None
    channel.sendall(b"R")
    if channel.recv(1) != b"G":
        return
    worker = os.fork()
    if worker == 0:
        channel.close()
        os.setsid()
        for target, fd in enumerate(kept):
            os.dup2(fd, target)
            os.close(fd)
        os.set_inheritable(3, False)
        # Match Python's standard streams after replacing the server's files.
        for name, fd in (("stdin", 0), ("stdout", 1), ("stderr", 2)):
            old = getattr(sys, name)
            stream = open(fd, "r" if fd == 0 else "w", encoding=old.encoding, errors=old.errors,
                          buffering=1 if fd == 2 or os.isatty(fd) else -1, closefd=False)
            setattr(sys, name, stream)
            setattr(sys, "__" + name + "__", stream)
        from .fork_server import _run_main
        os._exit(_run_main("engine.supervised_cli", ["3", *args]))

    for fd in kept:
        os.close(fd)
    pidfd = os.pidfd_open(worker)
    poller = select.poll()
    poller.register(channel, select.POLLIN)
    poller.register(pidfd, select.POLLIN)
    try:
        poller.poll()
    finally:
        # Either the worker finished or the client cancelled/disconnected.
        # A waitable zombie still reserves its PID and process-group ID.
        for kill in (os.killpg, os.kill):
            try:
                kill(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status = os.waitpid(worker, 0)
        while True:
            try:
                for child in Path("/proc/thread-self/children").read_bytes().split():
                    try:
                        os.kill(int(child), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                os.waitpid(-1, 0)
            except ChildProcessError:
                break
    channel.sendall(struct.pack("=i", status))
