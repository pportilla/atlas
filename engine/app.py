"""Local CLI and HTTP frontend, using only the Python standard library."""
import argparse
import codeop
from contextlib import nullcontext
import json
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .resolve_cli import (MAX_SECONDS, add_resolution_options, calculate_resolution as _calculate_resolution,
                          needs_exhaustive_search, resolution_arguments, run_job, run_resolution,
                          session_decimals, show_zeta)

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web'
MAX_REQUEST=32768
WEB_FONTS = frozenset(f'/fonts/{face}.woff2' for face in (
    'source-code-pro', 'source-code-pro-italic', 'source-sans-3', 'source-sans-3-italic',
    'stix-two-text', 'stix-two-text-italic'))
ZOO_RECORD = re.compile(r'/zoo-data/records/[a-z0-9]+(?:-[a-z0-9]+)*\.json')
CAPACITY=threading.BoundedSemaphore(2)


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
        address = urlsplit(self.path)
        path = address.path
        if path in ('/zoo', '/zoo.html'):
            self.send_response(308)
            self.send_header('Location', '/zoo/' + ('?' + address.query if address.query else ''))
            self.end_headers()
            return
        if path.startswith('/zoo/'):
            path = '/zoo.html' if path in ('/zoo/', '/zoo/index.html') else path[len('/zoo'):]
            if path.startswith('/api/'):
                return self.obj(404, {'error': 'Unknown endpoint.'})
        static={'/':('index.html','text/html'),'/index.html':('index.html','text/html'),
                '/zoo.html':('zoo.html','text/html'),'/zoo.js':('zoo.js','text/javascript'),
                '/zoo.css':('zoo.css','text/css'),'/zoo-mark.svg':('zoo-mark.svg','image/svg+xml'),
                '/typeset.js':('typeset.js','text/javascript'),'/dual-graph.js':('dual-graph.js','text/javascript'),
                '/exact-json.js':('exact-json.js','text/javascript'),
                '/resolution.css':('resolution.css','text/css'),'/runs.css':('runs.css','text/css'),
                '/complex.css':('complex.css','text/css'),
                '/atlas-mark.svg':('atlas-mark.svg','image/svg+xml'),
                '/identity.css':('identity.css','text/css'),
                '/style.css':('style.css','text/css'),'/app.js':('app.js','text/javascript')}
        if path in WEB_FONTS:
            return self.send(200, (WEB / path[1:]).read_bytes(), 'font/woff2')
        if path in ('/zoo-data/index.json', '/zoo-data/schema.json') or ZOO_RECORD.fullmatch(path):
            resource = WEB / path[1:]
            if resource.is_file():
                return self.send(200, resource.read_bytes(), 'application/json; charset=utf-8')
            return self.obj(404, {'error': 'Unknown catalogue record.'})
        if path=='/api/health': return self.obj(200,{'status':'ok'})
        if path=='/api/searches':
            return self.obj(200,{'jobs':self.server.search_jobs.list()})
        if path=='/api/searches/current':
            return self.obj(200,{'job':self.server.search_jobs.get()})
        if path.startswith('/api/searches/'):
            job=self.server.search_jobs.get(path[len('/api/searches/'):])
            return self.obj(200,job) if job else self.obj(404,{'error':'Search no longer available.'})
        if path not in static: return self.obj(404,{'error':'Unknown endpoint.'})
        name,mime=static[path]
        if mime == 'text/html':
            from .web_brand import fill_brand
            return self.send(200, fill_brand((WEB / name).read_text()).encode(), mime + '; charset=utf-8')
        self.send(200,(WEB/name).read_bytes(),mime+'; charset=utf-8')

    def do_POST(self):
        path=urlsplit(self.path).path
        parts = path.split('/')
        zeta_job = parts[3] if len(parts) == 5 and parts[1:3] == ['api', 'searches'] and parts[4] == 'zetas' else None
        replay = zeta_job is not None or path == '/api/zetas'
        algebra = path in ('/api/algebra', '/api/algebra/exhaustive')
        exhaustive = path in ('/api/searches', '/api/algebra/exhaustive')
        if not replay and not algebra and path not in ('/api/resolve','/api/searches','/api/zoo/search'): return self.obj(404,{'error':'Unknown endpoint.'})
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
        if size > MAX_REQUEST and not exhaustive and path != '/api/zetas':
            if path == '/api/zoo/search':
                return reject(413, f'Catalogue searches accept up to {MAX_REQUEST} bytes.')
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
            if algebra or replay:
                from .polynomial import integer_from_decimal
                data = json.loads(body, parse_int=integer_from_decimal)
            else:
                data = json.loads(body)
            del body
        except (ValueError,OSError,RecursionError): return reject(400, 'Invalid JSON request.')
        except MemoryError: return reject(503, 'Insufficient memory to read the request.', 'worker_unavailable')
        if path == '/api/zoo/search':
            from .zoo_search import zoo_search
            if not isinstance(data, dict) or 'query' not in data:
                return reject(400, 'Supply a monomial query.')
            try:
                return self.obj(200, zoo_search(data['query']))
            except (TypeError, ValueError) as exc:
                return reject(400, str(exc))
            except (OSError, RuntimeError) as exc:
                return reject(503, str(exc))
        if replay:
            return self.replay_zetas(data, zeta_job)
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
            if automatic and needs_exhaustive_search(data, status, res):
                return start_search()
            self.obj(status,res)
        finally: CAPACITY.release()

    def replay_zetas(self, data, ident):
        from .resolution_zetas import selected_zetas
        if not isinstance(data, dict) or "zetas" not in data:
            return self.obj(400, {"error": "Select a zeta function."})
        try:
            kinds = selected_zetas(data)
        except ValueError as error:
            return self.obj(400, {"error": str(error)})
        if not kinds:
            return self.obj(400, {"error": "Select a zeta function."})

        lock = self.server.search_jobs.zeta_lock(ident) if ident is not None else nullcontext()
        if lock is None:
            return self.obj(404, {"error": "Search no longer available."})
        with lock:
            return self.run_zeta_replay(data, kinds, ident)

    def run_zeta_replay(self, data, kinds, ident):
        from .worker_process import run_worker
        jobs = self.server.search_jobs
        if ident is not None:
            job = jobs.get(ident)
            if job is None:
                return self.obj(404, {"error": "Search no longer available."})
            record = job.get("result")
        else:
            record = data.get("record")
        if not isinstance(record, dict):
            return self.obj(400, {"error": "Supply an Atlas resolution record."})

        companion = jobs.embedded_companion(record)
        source = {**record, "embedded_companion": companion} if companion else record
        if not CAPACITY.acquire(False):
            return self.obj(429, {"error": "Both computation slots are busy. Try again shortly."})
        try:
            status, outputs = run_worker(["engine.worker", "--zetas"], {"record": source, "zetas": list(kinds)},
                                         None, owner=self.server.workers)
        except (RuntimeError, OSError) as error:
            return self.obj(503, {"error": str(error)})
        finally:
            CAPACITY.release()
        if status != 200 or ident is None:
            return self.obj(status, outputs)
        updated = jobs.attach_zetas(ident, outputs, companion)
        return self.obj(200, updated) if updated else self.obj(404, {"error": "Search no longer available."})

    def do_DELETE(self):
        path=urlsplit(self.path).path
        if not path.startswith('/api/searches/'): return self.obj(404,{'error':'Unknown endpoint.'})
        job=self.server.search_jobs.cancel(path[len('/api/searches/'):])
        return self.obj(200,job) if job else self.obj(404,{'error':'Search no longer available.'})


def make_parser():
    from .algebra_cli import register_parser
    parser=argparse.ArgumentParser(prog='atlas',description='Exact rational ideal arithmetic and local hypersurface resolutions.')
    sub=parser.add_subparsers(dest='command',required=True)
    parser.command_names = sub.choices
    register_parser(sub)
    calc=sub.add_parser('resolve',help='resolve at the origin')
    add_resolution_options(calc)
    sub.add_parser('devices',help='list available Triton and OpenCL GPU backends')
    script=sub.add_parser('run',help='run an Atlas script')
    script.add_argument('file',help='UTF-8 Atlas script or - for standard input')
    zeta=sub.add_parser('monodromy',help='compute the local monodromy zeta function from a saved resolution')
    zeta.add_argument('record',help='Atlas resolution JSON file, or - for standard input')
    zeta.add_argument('--json',action='store_true',help="print factors and A'Campo contributions")
    incidence=sub.add_parser('recompute-dual-complex', aliases=['dual-complex'],
                             help='recompute the exceptional dual complex from saved resolution charts')
    incidence.set_defaults(command='recompute-dual-complex')
    incidence.add_argument('record',help='Atlas resolution JSON or JSON.gz file, or - for standard input')
    server=sub.add_parser('serve',help='start the local web interface')
    server.add_argument('--port',type=int,default=8765)
    return parser


def calculate_resolution(polynomial, variables, *, mode, search, compute=None, monodromy=False, method="auto", zetas_only=False):
    return _calculate_resolution(polynomial, variables, mode=mode, search=search, compute=compute, method=method,
                                 monodromy=monodromy, zetas_only=zetas_only, job=run_job)


def run_command(parser, argv):
    if argv and argv[0] == 'resolve':
        argv = ['resolve', *resolution_arguments(argv[1:])]
    args=parser.parse_args(argv)
    if args.command=='algebra':
        from .algebra_cli import run
        return run(args)
    if args.command=='run':
        from .session import run_script
        return run_script(args.file, calculate_resolution)
    if args.command=='recompute-dual-complex':
        import gzip
        from .resolution_complex import dual_complex
        try:
            if args.record=='-':
                record=json.load(sys.stdin)
            else:
                opener=gzip.open if args.record.endswith('.gz') else open
                with opener(args.record, 'rt', encoding='utf-8') as stream:
                    record=json.load(stream)
            result=dual_complex(record)
        except (OSError, ValueError, ArithmeticError, KeyError, TypeError, RuntimeError) as error:
            print(f"Cannot compute the dual complex: {error}",file=sys.stderr)
            return 2
        print(json.dumps(result,indent=2,ensure_ascii=False))
        return 0 if result['status']=='complete' else 3
    if args.command=='monodromy':
        from .resolution_zetas import replay_zetas
        try:
            if hasattr(sys,"set_int_max_str_digits"):
                sys.set_int_max_str_digits(0)
            if args.record=='-':
                record=json.load(sys.stdin)
            else:
                with open(args.record, encoding='utf-8') as stream:
                    record=json.load(stream)
            zeta=replay_zetas({"record": record, "zetas": ["monodromy"]})["monodromy_zeta"]
        except (OSError, ValueError, ArithmeticError, RuntimeError) as error:
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
    return run_resolution(args, calculate_resolution)


def read_expression(line, prompt='... '):
    """Collect continued expressions and indented blocks at the prompt."""
    try:
        while codeop.compile_command(line, symbol="single") is None:
            line += "\n" + input(prompt)
    except (SyntaxError, ValueError, OverflowError):
        pass  # The evaluator provides the Atlas syntax diagnostic.
    return line


def shell_command_names(parser):
    """The words that start a shell command at the prompt rather than an expression."""
    return {*parser.command_names, ":save", ":workspace", "quit", "exit"}


def run_session(parser, panel):
    from .cli_style import tinted_errors
    from .prompt_readline import prompt_completion
    from .session import new_session
    from .welcome import interactive_welcome
    session = new_session(calculate_resolution)
    shell_names = shell_command_names(parser)
    # The welcome plays on while the session is in use. A completion list too
    # long for the rows below it settles the map first, and so does leaving.
    with interactive_welcome(panel) as settle_welcome, tinted_errors(panel.enabled and panel.color):
        with prompt_completion(session, parser, shell_names | {"help"}, on_list=settle_welcome):
            return _session_loop(parser, panel, session, shell_names)


def _is_shell_command(words, shell_names, variables):
    """Read a line as a shell command unless its first word starts an expression."""
    tail = words[1] if len(words) == 2 else ""
    expression_tail = words[0].isidentifier() and tail.startswith(("(", "=", "."))
    return words[0] in shell_names and words[0] not in variables and not expression_tail


def _session_loop(parser, panel, session, shell_names):
    from .cli_highlight import (command_pieces, expression_pieces, live_terminal, paint_hint,
                                paint_value, repaint_entry)
    from .cli_style import paint, readline_prompt
    from .prompt_help import command_hint, help_request, show_help, topic_names
    from .session import SessionError
    from .session_expressions import assignment_syntax
    from .session_hints import error_hint
    from .session_panel import print_workspace
    from .session_transcript import SessionTranscript
    transcript = SessionTranscript()
    # Only the line editor understands the markers around a prompt's colours.
    color = panel.enabled and panel.color and "readline" in sys.modules
    prompt_pieces = (("atlas", "brand"), (">", "muted"), (" ", None))
    prompt = readline_prompt(*prompt_pieces, color=color)
    more = readline_prompt(("...", "muted"), (" ", None), color=color)
    # Colouring entries and values needs the terminal that echoed the entry.
    rich = color and live_terminal()
    functions = set(session.builtins)
    commands = shell_names | {"help"}

    def hint_text(text):
        return paint_hint(text, commands=commands, topics=topic_names(session, parser), names=session.variables,
                          functions=functions, color=rich)

    assigned = None
    while True:
        try:
            if panel.enabled:
                panel.refresh(session.variables, assigned, session.rings.active)
            assigned = None
            line = input(prompt)
            assignment = assignment_syntax(line)
            if assignment is not None and assignment[0] == "help":
                assignment = None  # Preserve the standalone "help TOPIC" command.
            if assignment is not None:
                request, shell = None, False
            else:
                words = line.split(maxsplit=1)
                if not words:
                    continue
                request = help_request(line)
                shell = request is not None or _is_shell_command(words, shell_names, session.variables)
            if rich:
                pieces = (command_pieces(line, commands=commands, topics=topic_names(session, parser)) if shell else
                          expression_pieces(line, names=session.variables, functions=functions))
                repaint_entry(sys.stdout, prompt_pieces, line, pieces, shutil.get_terminal_size().columns)
            if request is not None:
                show_help(request, session, parser, color=rich)
                continue
            if not shell:
                try:
                    if assignment is not None:
                        if session.assign(assignment, line, transcript):
                            assigned = assignment[0]
                            continue
                    else:
                        line = read_expression(line, more)
                    for display, value in session.run(line, transcript=transcript):
                        if display:
                            print(paint_value(str(value), names=session.variables, functions=functions,
                                              color=rich) if rich else value)
                except SessionError as error:
                    source = line.splitlines()
                    statement = source[error.line - 1].strip() if 0 < error.line <= len(source) else ""
                    print(f"{statement}: {error}" if statement else str(error), file=sys.stderr)
                    hint = error_hint(error, statement, session, topic_names(session, parser))
                    if hint:
                        print(hint_text(hint), file=sys.stderr)
                except ValueError as error:
                    # Rendering a numeric preview can exceed an explicitly
                    # configured interpreter digit limit.
                    print(str(error), file=sys.stderr)
                continue
            try:
                argv = shlex.split(line)
            except ValueError as error:
                print(f'Cannot read command: {error}', file=sys.stderr)
                print(hint_text(command_hint(words[:1], parser)), file=sys.stderr)
                continue
            if not argv:
                continue
            if argv[0] in ('quit', 'exit', ':workspace'):
                if len(argv) != 1:
                    print(f'Use {argv[0]} by itself. Type help {argv[0]} for an example.', file=sys.stderr)
                elif argv[0] == ':workspace':
                    print_workspace(session.variables, session.rings.active)
                else:
                    return 0
                continue
            if argv[0] == ':save':
                if len(argv) != 2:
                    print('Usage: :save FILE.atlas', file=sys.stderr)
                    continue
                try:
                    transcript.save(argv[1])
                    print(f"{paint('Saved session to', 'ok', rich)} {paint(argv[1], 'string', rich)}")
                except (OSError, ValueError) as error:
                    print(f'Cannot save session: {error}', file=sys.stderr)
                continue
            try:
                status = run_command(parser, argv)
            except SystemExit as error:
                if error.code == 2:  # argparse already printed the original usage error.
                    print(hint_text(command_hint(argv, parser)), file=sys.stderr)
                continue
            if status == 2 and argv[0] != "run":
                print(hint_text(command_hint(argv, parser)), file=sys.stderr)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()


def main(*, unlimited_decimals=False):
    parser = make_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("run", "serve"):
        # Sessions, scripts and the server start their resolution workers
        # from one preloaded engine, warmed while the first input arrives.
        # A one-shot resolve finishes before such a server could be ready.
        from . import fork_server
        with fork_server.serving(prepare_now=True):
            return _main(parser, argv, unlimited_decimals)
    return _main(parser, argv, unlimited_decimals)


def _main(parser, argv, unlimited_decimals):
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
            return run_session(parser, panel)
    print_welcome(interactive=False)
    return 0
