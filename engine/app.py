"""Local CLI and HTTP frontend, using only the Python standard library."""
import argparse
import json
import os
from pathlib import Path
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


def run_job(data, timeout=MAX_SECONDS):
    child = subprocess.Popen([sys.executable, '-m', 'engine.worker'],cwd=ROOT,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,start_new_session=True)
    try:
        out,_ = child.communicate(json.dumps(data),timeout=timeout)
    except KeyboardInterrupt:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.communicate()
        raise
    except subprocess.TimeoutExpired:
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        child.communicate()
        from .resolve import inconclusive
        return 200, inconclusive(data, {'description':f'The quick worker reached its {timeout:g}-second wall-clock limit.'},1000*timeout)
    if child.returncode:
        return 422, {'error':'Worker stopped before it produced a certified result.'}
    return json.loads(out)


class AtlasServer(ThreadingHTTPServer):
    """Bind successfully before constructing server-owned search state."""
    def __init__(self, address, handler):
        super().__init__(address,handler)
        from .search_jobs import SearchJobs
        self.search_jobs = SearchJobs()

    def server_close(self):
        if hasattr(self, "search_jobs"):
            self.search_jobs.close()
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
        self.send(code,json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8')

    def do_GET(self):
        path=urlsplit(self.path).path
        static={'/':('index.html','text/html'),'/index.html':('index.html','text/html'),
                '/style.css':('style.css','text/css'),'/app.js':('app.js','text/javascript')}
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
        if path not in ('/api/resolve','/api/searches'): return self.obj(404,{'error':'Unknown endpoint.'})
        if self.headers.get_content_type()!='application/json':
            return self.obj(415,{'error':'Send application/json.'})
        try: size=int(self.headers.get('Content-Length',''))
        except ValueError: return self.obj(411,{'error':'Content-Length is required.'})
        if not 0 < size <= MAX_REQUEST: return self.obj(413,{'error':'Request is empty or too large.'})
        try:
            self.connection.settimeout(10)
            data=json.loads(self.rfile.read(size))
        except (ValueError,OSError,RecursionError): return self.obj(400,{'error':'Invalid JSON request.'})
        def start_search():
            try: job=self.server.search_jobs.start(data)
            except ValueError as exc: return self.obj(400,{'error':str(exc)})
            except RuntimeError as exc: return self.obj(409,{'error':str(exc)})
            except OSError: return self.obj(503,{'error':'Could not start the search process.'})
            return self.obj(202,job)
        if path=='/api/searches' or (isinstance(data,dict) and data.get('search')=='exhaustive'):
            return start_search()
        if not CAPACITY.acquire(False): return self.obj(429,{'error':'Both computation slots are busy. Try again shortly.'})
        try:
            automatic=isinstance(data,dict) and 'search' not in data
            status,res=run_job(dict(data,search='quick') if automatic else data)
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


def main():
    parser=argparse.ArgumentParser(prog='atlas',description='Exact local resolutions of hypersurface germs over Q.')
    sub=parser.add_subparsers(dest='command',required=True)
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
    calc.add_argument('--compute',choices=['cpu','auto','gpu'],help='select CPU or portable OpenCL for eligible rank tests and chart transforms')
    sub.add_parser('devices',help='list available OpenCL GPUs')
    zeta=sub.add_parser('monodromy',help='compute the local monodromy zeta function from a saved resolution')
    zeta.add_argument('record',help='Atlas resolution JSON file, or - for standard input')
    zeta.add_argument('--json',action='store_true',help="print factors and A'Campo contributions")
    server=sub.add_parser('serve',help='start the local web interface')
    server.add_argument('--port',type=int,default=8765)
    argv=sys.argv[1:]
    if len(argv)>1 and argv[0]=='resolve' and argv[1].startswith('-') and argv[1] not in {
        '--vars','--mode','--compute','--quick','--exhaustive','--json','--monodromy','--help','-h','--'}:
        # argparse otherwise treats a leading unary minus as an option name.
        argv=[argv[0],*argv[2:],'--',argv[1]]
    args=parser.parse_args(argv)
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
        from .native_backend import gpu_devices
        print(json.dumps(gpu_devices(),indent=2))
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
    request={'variables':[v.strip() for v in args.vars.split(',')],'mode':args.mode,
             'polynomial':args.polynomial,'compute':args.compute,'search':args.search}
    if args.monodromy: request['monodromy']=True
    try:
        status,r=run_job(request,timeout=MAX_SECONDS if args.search=='quick' else None)
    except KeyboardInterrupt:
        print('Resolution interrupted.',file=sys.stderr)
        return 130
    if args.json: print(json.dumps(r,indent=2,ensure_ascii=False))
    elif status!=200: print(r['error'],file=sys.stderr)
    else: show(r)
    return 2 if status!=200 else 0 if (r['status']=='resolved' and
        (not args.monodromy or r.get('monodromy_zeta',{}).get('status')=='computed')) else 3
