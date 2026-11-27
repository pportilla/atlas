"""Command-line parsing and output for an isolated resolution at the origin."""
import argparse
from contextlib import contextmanager
import json
import os
import sys
from time import monotonic

from .quick_budget import QUICK_SECONDS, checked_seconds, expired, expired_line, read_progress, request_seconds

# Seconds that a standard algebra request has, and the longest budget that a
# quick resolution may ask for where a request holds a computation slot.
MAX_SECONDS = 55
# The file in which a search publishes its progress (engine/resolution_progress.py).
PROGRESS_FILE = "ATLAS_PROGRESS_FILE"
# Set by the command-line supervisor when it made that file for this run.
PROGRESS_OWNED = "ATLAS_PROGRESS_OWNED"


def _timed_out(data, timeout, elapsed=None, progress=None):
    """The answer of a quick search stopped at its budget of `timeout` seconds."""
    return expired(data, timeout, elapsed, progress)


@contextmanager
def _progress_file():
    """A file for the progress of one search, removed afterwards: its path, and the worker's environment.

    A file that the environment already names is used as it is and left in place.
    """
    if os.environ.get(PROGRESS_FILE):
        yield os.environ[PROGRESS_FILE], None
        return
    import tempfile
    handle, path = tempfile.mkstemp(prefix="atlas-progress-", suffix=".json")
    os.close(handle)
    try:
        yield path, {**os.environ, PROGRESS_FILE: path}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def needs_exhaustive_search(request, status, result):
    from .resolution_zetas import selected_zetas
    if status == 422:
        return result.get("code") == "computation_limit"
    if status != 200:
        return False
    if not request.get("zetas_only"):
        if result.get("status") == "inconclusive":
            return True
        complex_status = result.get("dual_complex", {}).get("status")
        if result.get("engine") != "weighted" and complex_status != "complete":
            return True
    if result.get("monodromy_check", {}).get("status") == "unresolved":
        return True
    return any(result.get(kind + "_zeta", {}).get("status") != "computed" for kind in selected_zetas(request))


def _stopped():
    return 503, {'error':'Worker stopped before it produced a certified result.', 'code':'worker_failure'}


_OF_REQUEST = object()


def run_job(data, timeout=_OF_REQUEST, *, owner=None):
    """Run one resolution request in a worker of its own: (status, record).

    A quick request has a time budget: `seconds` in the request, or
    QUICK_SECONDS. When it ends the worker is stopped and the answer is the
    record of `quick_budget.expired`, with the progress that the search had
    published. `timeout` replaces the budget of the request when given.
    """
    from .worker_process import run_worker
    from subprocess import TimeoutExpired
    if timeout is _OF_REQUEST:
        try:
            timeout = request_seconds(data) if isinstance(data, dict) else None
        except ValueError as error:
            return 400, {'error': str(error)}
    if timeout is None:
        try:
            return run_worker(['engine.worker'], data, None, owner=owner)
        except RuntimeError:
            return _stopped()
    with _progress_file() as (path, env):
        started = monotonic()
        try:
            return run_worker(['engine.worker'], data, timeout, owner=owner, env=env)
        except TimeoutExpired:
            return _timed_out(data, timeout, monotonic() - started, read_progress(path))
        except RuntimeError:
            return _stopped()


def show_zeta(zeta):
    if zeta['status'] != 'computed':
        print(f"Monodromy zeta ({zeta['status']}):", zeta['reason'])
        return
    print("Local monodromy zeta(T) =", zeta['expression'])
    print("Euler characteristic of the complex Milnor fiber:", zeta['milnor_fiber_euler_characteristic'])


def show(r, *, zetas_only=False):
    if r.get('budget', {}).get('expired'):
        # The budget ended: one line, and how to go on.
        print(expired_line(r))
        return
    if (r.get('diagnostic') or {}).get('code') == 'input_limit':
        # The search did not start: the input is beyond what a quick search reads.
        print(f"{r['summary']} {r['diagnostic']['description']} Run without --quick to resolve it.")
        return
    print(f"{r['status'].upper()}  {r['summary']}\n{r['engine']} · {r['mode']} · {r['elapsed_ms']:g} ms")
    print(f"{r['polynomial']} = 0 at the origin in ({', '.join(r['variables'])})")
    if 'monodromy_zeta' in r:
        show_zeta(r['monodromy_zeta'])
    if zetas_only:
        top = r.get("topological_zeta", {})
        if top.get("status") == "computed":
            print("Local topological zeta(s) =", top["expression"])
        else:
            print(f"Topological zeta ({top.get('status', 'inconclusive')}):", top.get("reason", "Computation did not finish."))
        check = r.get("monodromy_check")
        if check:
            print("Local monodromy conjecture:", check["status"])
            for pole in check["poles"]:
                witness = pole["witness"] or ("absent near the origin" if check["status"] == "refuted"
                                               else "witness undetermined")
                print(f"  {pole['value']} (pole order {pole['order']}): eigenvalue order {pole['eigenvalue_order']}, "
                      f"{witness}")
        return
    dual=r['dual_complex'];print(f"\nDual complex ({dual['status']}): {dual['convention']}")
    weighted=bool(dual['vertices']) and all('genus' in v and 'self_intersection' in v for v in dual['vertices'])
    print("  Vertices:", ', '.join(v['id']+(f" (genus {v['genus']}, self-intersection {v['self_intersection']})"
                                            if weighted else '') for v in dual['vertices']) or 'none')
    if r.get('invariants'):
        inv=r['invariants']
        print(f"  K^2 = {inv['K2']}, Euler characteristic {inv['euler_characteristic']}, "
              f"geometric genus {inv['geometric_genus']}, 12 p_g + K^2 + chi - 1 = {inv['milnor_number_from_laufer']}")
    for check in r.get('certificate',{}).get('hypotheses',()):
        print('  Verified:',check['statement'])
    if r.get('coefficient_field',{}).get('description'): print('  Field:', r['coefficient_field']['description'])
    if r.get('algebraic_numbers'):
        for symbol,defn in r['algebraic_numbers'].items(): print(f"  {symbol} = {defn['exact']}")
    if r.get('execution'):
        print('  Device:', r['execution']['device'])
        if r['execution'].get('note'): print(' ',r['execution']['note'])
    if dual['cells']: print("  Cells:",'; '.join('–'.join(c['vertices']) for c in dual['cells']))
    for c in r['charts']:
        strict=c['strict_transform']
        equations=strict if isinstance(strict,list) else [strict]
        print(f"\n{c['id']} ({', '.join(c['coordinates'])})")
        if c.get('ambient_ideal'): print('  Ambient chart:',', '.join(f'{eq} = 0' for eq in c['ambient_ideal']))
        print('  Strict transform:',', '.join(f'{eq} = 0' for eq in equations) or 'whole chart')
        print('  Map:',', '.join(f'{v} = {e}' for v,e in c['map'].items()))
        print('  Exceptional:',', '.join(f"{e['id']}: {e['equation']} = 0" for e in c['exceptional']) or 'none')
        if c.get('exclude_points'): print('  Exclude points:',c['exclude_points'])
    if r['charts']: print('\nUse --json for full chart domains, certificates, and incidence data.')


def add_resolution_options(calc):
    calc.add_argument('polynomial')
    calc.add_argument('--vars',required=True,metavar='x,y[,z,...]')
    calc.add_argument('--mode',choices=['embedded','intrinsic'],default=None,
                      help='embedded (default) or intrinsic; method jung resolves the surface itself (intrinsic)')
    calc.add_argument('--method',choices=['auto','weighted','jung'],default='auto',
                      help="weighted tries weighted blowups in embedded mode. An exhaustive search continues "
                      "with ordinary resolution when the weighted method declines. jung checks that Jung's "
                      "method applies to this surface and then uses it")
    policies=calc.add_mutually_exclusive_group()
    policies.add_argument('--quick',dest='search',action='store_const',const='quick',
                          help=f'give the same search a time budget of {QUICK_SECONDS} seconds. When the budget ends '
                               'the answer says so and the exit code is 3: not finished')
    policies.add_argument('--exhaustive',dest='search',action='store_const',const='exhaustive',
                          help='continue until certified or interrupted (default)')
    calc.set_defaults(search=None)
    calc.add_argument('--seconds',type=_seconds_argument,metavar='N',
                      help='the time budget of a quick search in seconds; implies --quick')
    calc.add_argument('--json',action='store_true',help='print the full machine-readable atlas')
    calc.add_argument('--monodromy',action='store_true',help="compute the local monodromy zeta function of the complex Milnor fiber")
    calc.add_argument('--zetas-only', action='store_true',
                      help="compute both local zetas, allowing a plane contact certificate in place of a chart atlas")
    compute=calc.add_mutually_exclusive_group()
    compute.add_argument('--compute',choices=['cpu','auto','gpu'],help='select the CPU, automatic dispatch, or GPU acceleration')
    compute.add_argument('--gpu',dest='compute',action='store_const',const='gpu',
                         help='use available GPU kernels (equivalent to --compute gpu)')


def _seconds_argument(text):
    try:
        value = float(text)
        return checked_seconds(int(value) if value.is_integer() else value)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("give a positive number of seconds") from None


def calculate_resolution(polynomial, variables, *, mode, search, compute=None, monodromy=False, method="auto",
                         zetas_only=False, seconds=None, job=run_job):
    """Resolve in a worker: (status, record). `seconds` is the budget of a quick search and implies one."""
    if mode is None:
        mode = 'intrinsic' if method == 'jung' else 'embedded'
    if seconds is not None:
        try:
            checked_seconds(seconds)
        except ValueError as error:
            return 400, {'error': str(error)}
        search = 'quick'
    request = dict(polynomial=polynomial, variables=variables, mode=mode, search=search, compute=compute)
    if seconds is not None:
        request['seconds'] = seconds
    if monodromy:
        request['monodromy'] = True
    if zetas_only:
        request['zetas_only'] = True
    if method != 'auto':
        request['method'] = method
    return job(request, timeout=request_seconds(request))


@contextmanager
def session_decimals(unlimited):
    """Allow exact decimal input and output only during a standalone math session."""
    if not unlimited or not hasattr(sys, "set_int_max_str_digits"):
        yield
        return
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(previous)


def resolution_arguments(argv):
    if '--' not in argv:
        # Protect a leading unary minus even when flags precede the polynomial.
        options={'--vars','--mode','--method','--compute','--gpu','--quick','--exhaustive','--seconds','--json','--monodromy','--zetas-only','--help','-h'}
        for pos, token in enumerate(argv):
            if token.startswith('-') and token.partition('=')[0] not in options:
                argv=[*argv[:pos],*argv[pos+1:],'--',token]
                break
    return argv


def run_resolution(args, calculate=calculate_resolution):
    try:
        options = {'method': args.method} if getattr(args, 'method', 'auto') != 'auto' else {}
        if args.zetas_only:
            options['zetas_only'] = True
        seconds = getattr(args, 'seconds', None)
        if seconds is not None:
            if args.search == 'exhaustive':
                print('An exhaustive search has no time budget: use --seconds without --exhaustive.', file=sys.stderr)
                return 2
            options['seconds'] = seconds
        status, r = calculate(args.polynomial, [v.strip() for v in args.vars.split(',')],
                              mode=args.mode, search=args.search or ('quick' if seconds is not None else 'exhaustive'),
                              compute=args.compute, monodromy=args.monodromy or args.zetas_only, **options)
    except KeyboardInterrupt:
        print('Resolution interrupted.',file=sys.stderr)
        return 130
    if args.json: print(json.dumps(r,indent=2,ensure_ascii=False))
    elif status!=200: print(r['error'],file=sys.stderr)
    else: show(r, zetas_only=args.zetas_only)
    if status == 200 and args.zetas_only:
        return 0 if all(r.get(k + '_zeta', {}).get('status') == 'computed' for k in ('topological','monodromy')) else 3
    return 2 if status!=200 else 0 if (r['status']=='resolved' and
        (not args.monodromy or r.get('monodromy_zeta',{}).get('status')=='computed')) else 3


def _owned_progress(remove=False):
    """The progress file that the supervisor made for this run, or None; `remove` deletes it."""
    path = os.environ.get(PROGRESS_FILE) if os.environ.get(PROGRESS_OWNED) else None
    if path and remove:
        try:
            os.unlink(path)
        except OSError:
            pass
        os.environ.pop(PROGRESS_FILE, None)
    return path


def _supervised_job(data, timeout, supervisor, worker_digits):
    if supervisor == "timeout":
        # The supervisor stopped the worker at the budget and started this
        # process in its place, with the file of the worker's progress.
        progress = read_progress(os.environ.get(PROGRESS_FILE, ""))
        _owned_progress(remove=True)
        return _timed_out(data, timeout, progress=progress)
    if supervisor == "stopped":
        _owned_progress(remove=True)
        return _stopped()
    if timeout is None:
        # Only a search with a budget needs its progress afterwards.
        _owned_progress(remove=True)
    from pathlib import Path
    fd = int(supervisor)
    os.set_inheritable(fd, False)
    # The deadline begins before importing the worker and ends before
    # displaying its finished record, just as in the API supervisor.
    message = b'E' if timeout is None else b'Q' + int(timeout * 1000).to_bytes(8, sys.byteorder)
    os.write(fd, message)
    os.chdir(Path(__file__).resolve().parents[1])
    saved = sys.get_int_max_str_digits() if worker_digits is not None else None
    try:
        if worker_digits is not None:
            sys.set_int_max_str_digits(worker_digits)
        from .worker import evaluate
        result = evaluate(data)
    finally:
        if saved is not None:
            sys.set_int_max_str_digits(saved)
    os.write(fd, b'D')
    os.close(fd)
    return result


class _ResolutionParser(argparse.ArgumentParser):
    """Prepare terminal help only when the standalone command displays it."""

    def __init__(self):
        super().__init__(prog="atlas resolve", add_help=False)
        # Register the actions and their mutual exclusions together. An
        # argument group can parse them without initializing a help formatter.
        options = self.add_argument_group()
        options.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS)
        add_resolution_options(options)

    def _display_parser(self):
        parser = argparse.ArgumentParser(prog=self.prog)
        add_resolution_options(parser)
        return parser

    def format_help(self):
        return self._display_parser().format_help()

    def format_usage(self):
        return self._display_parser().format_usage()


def main(*, unlimited_decimals=False, supervisor=None):
    parser = _ResolutionParser()
    argv = resolution_arguments(sys.argv[2:])
    worker_digits = sys.get_int_max_str_digits() if hasattr(sys, 'get_int_max_str_digits') else None
    with session_decimals(unlimited_decimals):
        args, extras = parser.parse_known_args(argv)
        if extras:
            # The parent parser reports leftover arguments for subcommands.
            from .app import make_parser
            make_parser().parse_args(["resolve", *argv])
        if supervisor is None:
            return run_resolution(args)
        from functools import partial
        job = partial(_supervised_job, supervisor=supervisor, worker_digits=worker_digits)
        return run_resolution(args, partial(calculate_resolution, job=job))
