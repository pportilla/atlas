"""Command-line parsing and output for an isolated resolution at the origin."""
import argparse
from contextlib import contextmanager
import json
import subprocess
import sys

MAX_SECONDS = 55


def _timed_out(data, timeout):
    from .resolve import inconclusive
    return 200, inconclusive(data, {'description':f'The quick worker reached its {timeout:g}-second wall-clock limit.'},1000*timeout)


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


def run_job(data, timeout=MAX_SECONDS, *, owner=None):
    from .worker_process import run_worker
    try:
        return run_worker(['engine.worker'], data, timeout, owner=owner)
    except subprocess.TimeoutExpired:
        return _timed_out(data, timeout)
    except RuntimeError:
        return _stopped()


def show_zeta(zeta):
    if zeta['status'] != 'computed':
        print(f"Monodromy zeta ({zeta['status']}):", zeta['reason'])
        return
    print("Local monodromy zeta(T) =", zeta['expression'])
    print("Euler characteristic of the complex Milnor fiber:", zeta['milnor_fiber_euler_characteristic'])


def show(r, *, zetas_only=False):
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
    print("  Vertices:", ', '.join(v['id'] for v in dual['vertices']) or 'none')
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
    calc.add_argument('--mode',choices=['embedded','intrinsic'],default='embedded')
    calc.add_argument('--method',choices=['auto','weighted'],default='auto',
                      help="weighted tries weighted blowups in embedded mode. An exhaustive search continues "
                      "with ordinary resolution when the weighted method declines")
    policies=calc.add_mutually_exclusive_group()
    policies.add_argument('--quick',dest='search',action='store_const',const='quick',
                          help='use a bounded search and return exit code 3 if inconclusive')
    policies.add_argument('--exhaustive',dest='search',action='store_const',const='exhaustive',
                          help='continue until certified or interrupted (default)')
    calc.set_defaults(search='exhaustive')
    calc.add_argument('--json',action='store_true',help='print the full machine-readable atlas')
    calc.add_argument('--monodromy',action='store_true',help="compute the local monodromy zeta function of the complex Milnor fiber")
    calc.add_argument('--zetas-only', action='store_true',
                      help="compute both local zetas, allowing a plane contact certificate in place of a chart atlas")
    compute=calc.add_mutually_exclusive_group()
    compute.add_argument('--compute',choices=['cpu','auto','gpu'],help='select the CPU, automatic dispatch, or GPU acceleration')
    compute.add_argument('--gpu',dest='compute',action='store_const',const='gpu',
                         help='use available GPU kernels (equivalent to --compute gpu)')


def calculate_resolution(polynomial, variables, *, mode, search, compute=None, monodromy=False, method="auto",
                         zetas_only=False, job=run_job):
    request = dict(polynomial=polynomial, variables=variables, mode=mode, search=search, compute=compute)
    if monodromy:
        request['monodromy'] = True
    if zetas_only:
        request['zetas_only'] = True
    if method != 'auto':
        request['method'] = method
    return job(request, timeout=MAX_SECONDS if search == 'quick' else None)


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
        options={'--vars','--mode','--method','--compute','--gpu','--quick','--exhaustive','--json','--monodromy','--zetas-only','--help','-h'}
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
        status, r = calculate(args.polynomial, [v.strip() for v in args.vars.split(',')],
                              mode=args.mode, search=args.search, compute=args.compute,
                              monodromy=args.monodromy or args.zetas_only, **options)
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


def _supervised_job(data, timeout, supervisor, worker_digits):
    import os
    if supervisor == "timeout":
        return _timed_out(data, timeout)
    if supervisor == "stopped":
        return _stopped()
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
