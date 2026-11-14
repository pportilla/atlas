"""Local CLI and HTTP frontend, using only the Python standard library."""
import argparse
import codeop
from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web'
MAX_REQUEST=32768
MAX_SECONDS=55
CAPACITY=threading.BoundedSemaphore(2)


def run_job(data, timeout=MAX_SECONDS, *, owner=None):
    from .worker_process import run_worker
    try:
        return run_worker(['engine.worker'], data, timeout, owner=owner)
    except subprocess.TimeoutExpired:
        from .resolve import inconclusive
        return 200, inconclusive(data, {'description':f'The quick worker reached its {timeout:g}-second wall-clock limit.'},1000*timeout)
    except RuntimeError:
        return 422, {'error':'Worker stopped before it produced a certified result.'}


class AtlasServer(ThreadingHTTPServer):
    """Bind successfully before constructing server-owned search state."""
    def __init__(self, address, handler):
        super().__init__(address,handler)
        from .search_jobs import SearchJobs
        from .algebra_basis_store import AlgebraBasisStore
        from .worker_process import WorkerGroup
        self.search_jobs = SearchJobs()
        self.algebra_bases = AlgebraBasisStore()
        self.workers = WorkerGroup()

    def server_close(self):
        if hasattr(self, "search_jobs"):
            self.search_jobs.close()
            self.workers.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version='ResolutionAtlas/1'
    def send(self,code,data,ctype):
        self.send_response(code)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        try: self.wfile.write(data)
        except (ConnectionError,BrokenPipeError): pass

    def obj(self,code,data):
        from .exact_json import result_json
        self.send(code,result_json(data).encode(),'application/json; charset=utf-8')

    def do_GET(self):
        path=urlsplit(self.path).path
        static={'/':('index.html','text/html'),'/index.html':('index.html','text/html'),
                '/algebra.html':('algebra.html','text/html'),'/algebra.js':('algebra.js','text/javascript'),
                '/algebra-session.js':('algebra-session.js','text/javascript'),
                '/exact-json.js':('exact-json.js','text/javascript'),
                '/resolution.css':('resolution.css','text/css'),'/algebra.css':('algebra.css','text/css'),
                '/atlas-mark.svg':('atlas-mark.svg','image/svg+xml'),
                '/style.css':('style.css','text/css'),'/app.js':('app.js','text/javascript')}
        if path in ('/fonts/stix-two-text.woff2', '/fonts/stix-two-text-italic.woff2', '/fonts/instrument-sans.woff2'):
            return self.send(200, (WEB / path[1:]).read_bytes(), 'font/woff2')
        if path=='/api/health': return self.obj(200,{'status':'ok'})
        if path=='/api/searches/current':
            return self.obj(200,{'job':self.server.search_jobs.get()})
        if path.startswith('/api/searches/'):
            job=self.server.search_jobs.get(path[len('/api/searches/'):])
            return self.obj(200,job) if job else self.obj(404,{'error':'Search no longer available.'})
        if path not in static: return self.obj(404,{'error':'Unknown endpoint.'})
        name,mime=static[path]
        self.send(200,(WEB/name).read_bytes(),mime+'; charset=utf-8')

    def do_POST(self):
        path=urlsplit(self.path).path
        algebra = path in ('/api/algebra', '/api/algebra/exhaustive')
        exhaustive = path in ('/api/searches', '/api/algebra/exhaustive')
        if not algebra and path not in ('/api/resolve','/api/searches'): return self.obj(404,{'error':'Unknown endpoint.'})
        def reject(status, message, code='invalid_request'):
            if algebra:
                from .algebra import error_record
                return self.obj(status, error_record(code, message))
            return self.obj(status, {'error': message})
        if self.headers.get_content_type()!='application/json':
            return reject(415, 'Send application/json.')
        try: size=int(self.headers.get('Content-Length',''))
        except ValueError: return reject(411, 'Content-Length is required.')
        if size <= 0: return reject(413, 'Request is empty.')
        if size > MAX_REQUEST and not exhaustive:
            endpoint = '/api/algebra/exhaustive' if algebra else '/api/searches'
            return reject(413, f'Standard requests accept up to {MAX_REQUEST} bytes. Choose exhaustive mode or post to {endpoint}.')
        try:
            self.connection.settimeout(10)
            # Allocate as bytes arrive. An enormous claimed length with a short
            # body must not reserve memory for the entire claim.
            body = bytearray()
            while len(body) < size:
                chunk = self.rfile.read(min(size - len(body), 64 * 1024))
                if not chunk:
                    return reject(400, 'Incomplete request body.')
                body.extend(chunk)
            if algebra:
                from .polynomial import integer_from_decimal
                data = json.loads(body, parse_int=integer_from_decimal)
            else:
                data = json.loads(body)
            del body
        except (ValueError,OSError,RecursionError): return reject(400, 'Invalid JSON request.')
        except MemoryError: return reject(503, 'Insufficient memory to read the request.', 'worker_unavailable')
        def start_search():
            try: job=self.server.search_jobs.start(data)
            except ValueError as exc: return self.obj(400,{'error':str(exc)})
            except RuntimeError as exc: return self.obj(409,{'error':str(exc)})
            except OSError: return self.obj(503,{'error':'Could not start the search process.'})
            return self.obj(202,job)
        if path=='/api/searches' or (path=='/api/resolve' and isinstance(data,dict) and data.get('search')=='exhaustive'):
            return start_search()
        if not CAPACITY.acquire(False): return reject(429, 'Both computation slots are busy. Try again shortly.', 'busy')
        try:
            if algebra:
                timeout = None if exhaustive else MAX_SECONDS
                return self.obj(*self.server.algebra_bases.calculate(data, timeout=timeout, owner=self.server.workers))
            automatic=isinstance(data,dict) and 'search' not in data
            status,res=run_job(dict(data,search='quick') if automatic else data, owner=self.server.workers)
            if automatic and status==200:
                pending_zeta = (data.get('monodromy') is True and
                                res.get('monodromy_zeta', {}).get('status') == 'unavailable')
                if res.get('status')=='inconclusive' or pending_zeta:
                    return start_search()
            self.obj(status,res)
        finally: CAPACITY.release()

    def do_DELETE(self):
        path=urlsplit(self.path).path
        if not path.startswith('/api/searches/'): return self.obj(404,{'error':'Unknown endpoint.'})
        job=self.server.search_jobs.cancel(path[len('/api/searches/'):])
        return self.obj(200,job) if job else self.obj(404,{'error':'Search no longer available.'})


def show_zeta(zeta):
    if zeta['status'] != 'computed':
        print("Monodromy zeta unavailable:", zeta['reason'])
        return
    print("Local monodromy zeta(T) =", zeta['expression'])
    print("Euler characteristic of the complex Milnor fiber:", zeta['milnor_fiber_euler_characteristic'])


def show(r):
    print(f"{r['status'].upper()}  {r['summary']}\n{r['engine']} · {r['mode']} · {r['elapsed_ms']:g} ms")
    print(f"{r['polynomial']} = 0 at the origin in ({', '.join(r['variables'])})")
    if 'monodromy_zeta' in r:
        show_zeta(r['monodromy_zeta'])
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


def make_parser():
    from .algebra_cli import register_parser
    parser=argparse.ArgumentParser(prog='atlas',description='Exact rational ideal arithmetic and local hypersurface resolutions.')
    sub=parser.add_subparsers(dest='command',required=True)
    parser.command_names = sub.choices
    register_parser(sub)
    calc=sub.add_parser('resolve',help='resolve at the origin')
    calc.add_argument('polynomial');calc.add_argument('--vars',required=True,metavar='x,y[,z,...]')
    calc.add_argument('--mode',choices=['embedded','intrinsic'],default='embedded')
    policies=calc.add_mutually_exclusive_group()
    policies.add_argument('--quick',dest='search',action='store_const',const='quick',
                          help='use a bounded search and return exit code 3 if inconclusive')
    policies.add_argument('--exhaustive',dest='search',action='store_const',const='exhaustive',
                          help='continue until certified or interrupted (default)')
    calc.set_defaults(search='exhaustive')
    calc.add_argument('--json',action='store_true',help='print the full machine-readable atlas')
    calc.add_argument('--monodromy',action='store_true',help="compute the local monodromy zeta function of the complex Milnor fiber")
    compute=calc.add_mutually_exclusive_group()
    compute.add_argument('--compute',choices=['cpu','auto','gpu'],help='select the CPU, automatic dispatch, or GPU acceleration')
    compute.add_argument('--gpu',dest='compute',action='store_const',const='gpu',
                         help='use available GPU kernels (equivalent to --compute gpu)')
    sub.add_parser('devices',help='list available Triton and OpenCL GPU backends')
    script=sub.add_parser('run',help='run an Atlas script')
    script.add_argument('file',help='UTF-8 Atlas script or - for standard input')
    zeta=sub.add_parser('monodromy',help='compute the local monodromy zeta function from a saved resolution')
    zeta.add_argument('record',help='Atlas resolution JSON file, or - for standard input')
    zeta.add_argument('--json',action='store_true',help="print factors and A'Campo contributions")
    server=sub.add_parser('serve',help='start the local web interface')
    server.add_argument('--port',type=int,default=8765)
    return parser


def calculate_resolution(polynomial, variables, *, mode, search, compute=None, monodromy=False):
    request = dict(polynomial=polynomial, variables=variables, mode=mode, search=search, compute=compute)
    if monodromy:
        request['monodromy'] = True
    return run_job(request, timeout=MAX_SECONDS if search == 'quick' else None)


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


def run_command(parser, argv):
    if argv and argv[0]=='resolve' and '--' not in argv:
        # Protect a leading unary minus even when flags precede the polynomial.
        options={'--vars','--mode','--compute','--gpu','--quick','--exhaustive','--json','--monodromy','--help','-h'}
        for pos, token in enumerate(argv[1:], 1):
            if token.startswith('-') and token.partition('=')[0] not in options:
                argv=[*argv[:pos],*argv[pos+1:],'--',token]
                break
    args=parser.parse_args(argv)
    if args.command=='algebra':
        from .algebra_cli import run
        return run(args)
    if args.command=='run':
        from .session import run_script
        return run_script(args.file, calculate_resolution)
    if args.command=='monodromy':
        from .monodromy import monodromy_zeta
        try:
            if hasattr(sys,"set_int_max_str_digits"):
                sys.set_int_max_str_digits(0)
            if args.record=='-':
                record=json.load(sys.stdin)
            else:
                with open(args.record, encoding='utf-8') as stream:
                    record=json.load(stream)
            zeta=monodromy_zeta(record)
        except (OSError, ValueError, ArithmeticError) as error:
            print(f"Cannot compute monodromy: {error}",file=sys.stderr)
            return 2
        if args.json: print(json.dumps(zeta,indent=2,ensure_ascii=False))
        else: show_zeta(zeta)
        return 0 if zeta['status']=='computed' else 3
    if args.command=='devices':
        from .gpu_devices import available_devices
        print(json.dumps(available_devices(),indent=2))
        return 0
    if args.command=='serve':
        if not 0<=args.port<=65535: parser.error('invalid port')
        try: http=AtlasServer(('127.0.0.1',args.port),Handler)
        except OSError as exc: parser.error(f'Could not bind the local search server on port {args.port}: {exc}')
        with http:
            print(f'Resolution Atlas: http://127.0.0.1:{http.server_port}',flush=True)
            def stop_server(signum, frame):
                raise KeyboardInterrupt
            previous=signal.signal(signal.SIGTERM,stop_server)
            try: http.serve_forever()
            except KeyboardInterrupt: pass
            finally: signal.signal(signal.SIGTERM,previous)
        return 0
    try:
        status,r=calculate_resolution(args.polynomial, [v.strip() for v in args.vars.split(',')],
                                      mode=args.mode, search=args.search, compute=args.compute,
                                      monodromy=args.monodromy)
    except KeyboardInterrupt:
        print('Resolution interrupted.',file=sys.stderr)
        return 130
    if args.json: print(json.dumps(r,indent=2,ensure_ascii=False))
    elif status!=200: print(r['error'],file=sys.stderr)
    else: show(r)
    return 2 if status!=200 else 0 if (r['status']=='resolved' and
        (not args.monodromy or r.get('monodromy_zeta',{}).get('status')=='computed')) else 3


def read_expression(line):
    """Collect continued expressions and indented blocks at the prompt."""
    try:
        while codeop.compile_command(line, symbol="single") is None:
            line += "\n" + input('... ')
    except (SyntaxError, ValueError, OverflowError):
        pass  # The evaluator provides the Atlas syntax diagnostic.
    return line


def run_session(parser, panel):
    from .session import HELP, SessionError, new_session
    from .session_panel import print_workspace
    from .session_transcript import SessionTranscript
    session = new_session(calculate_resolution)
    transcript = SessionTranscript()
    try:
        import readline  # Enable arrow-key editing and in-session history for input().
    except ImportError:
        pass

    # Derive shell-command names from argparse so newly registered subcommands
    # are available here. An opening parenthesis selects function-call syntax.
    shell_names = {*parser.command_names, ":save", ":workspace", "help", "quit", "exit", "-h", "--help"}
    while True:
        try:
            panel.refresh(session.variables)
            line = input('atlas> ')
            words = line.split(maxsplit=1)
            if not words:
                continue
            tail = words[1] if len(words) == 2 else ""
            is_variable = words[0] in session.variables
            if words[0] not in shell_names or is_variable or tail.startswith(("(", "=", ".")):
                line = read_expression(line)
                try:
                    for display, value in session.run(line, transcript=transcript):
                        if display:
                            print(value)
                except SessionError as error:
                    source = line.splitlines()
                    statement = source[error.line - 1].strip() if 0 < error.line <= len(source) else ""
                    print(f"{statement}: {error}" if statement else str(error), file=sys.stderr)
                except ValueError as error:
                    # Rendering a numeric preview can exceed an explicitly
                    # configured interpreter digit limit.
                    print(str(error), file=sys.stderr)
                continue
            try:
                argv = shlex.split(line)
            except ValueError as error:
                print(f'Cannot read command: {error}', file=sys.stderr)
                continue
            if not argv:
                continue
            if argv in (['quit'], ['exit']):
                return 0
            if argv == [':workspace']:
                print_workspace(session.variables)
                continue
            if argv[0] == ':save':
                if len(argv) != 2:
                    print('Usage: :save FILE.atlas', file=sys.stderr)
                    continue
                try:
                    transcript.save(argv[1])
                    print(f'Saved session to {argv[1]}')
                except (OSError, ValueError) as error:
                    print(f'Cannot save session: {error}', file=sys.stderr)
                continue
            if argv in (['help', 'save'], ['help', ':save']):
                print('Use :save FILE.atlas to write the session. Replay it with ./atlas run FILE.atlas.')
                continue
            if argv == ['help']:
                parser.print_help()
                print('\n' + HELP)
                print('\nShow every ring and variable with :workspace.')
                print('\nSave this session with :save FILE.atlas.')
                print('\nType help COMMAND for its options or quit to leave Atlas.')
                continue
            if argv == ['help', 'script']:
                print(HELP)
                continue
            if argv[0] == 'help':
                if len(argv) != 2:
                    print('Usage: help [COMMAND]', file=sys.stderr)
                    continue
                argv = [argv[1], '--help']
            try:
                run_command(parser, argv)
            except SystemExit:
                # argparse has already printed the requested help or usage error.
                pass
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()


def main(*, unlimited_decimals=False):
    parser = make_parser()
    argv = sys.argv[1:]
    if argv:
        # The standalone parent also decodes and prints exact integer fields
        # returned by the worker, including resolution multiplicities.
        with session_decimals(unlimited_decimals):
            return run_command(parser, argv)

    from .welcome import print_welcome
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        from .session_panel import SessionPanel
        with session_decimals(unlimited_decimals), SessionPanel(sys.stdout) as panel:
            print_welcome(interactive=True)
            return run_session(parser, panel)
    print_welcome(interactive=False)
    return 0
