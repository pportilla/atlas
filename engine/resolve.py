"""Dispatch only to algorithms whose hypotheses have been checked."""
import os
from pathlib import Path
from time import perf_counter
import subprocess
import sys
from .polynomial import InputError, LimitError, NonReducedError, parse, format_poly
from .native_backend import resolve_native

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'native' / 'build'


def inconclusive(request, diagnostic, elapsed_ms, polynomial=None):
    variables = request['variables']
    result = {'status':'inconclusive', 'engine':'general-search', 'can_search':True,
            'summary':'The quick search reached its budget. Continue the exhaustive search to test more blowups.',
            'diagnostic':diagnostic, 'charts':[], 'steps':[], 'schema_version':1,
            'variables':variables, 'polynomial':polynomial or request['polynomial'],
            'mode':request.get('mode','embedded'), 'base_field':'Q',
            'geometric_field':'algebraic closure of Q', 'origin':[0]*len(variables),
            'elapsed_ms':round(elapsed_ms,3),
            'dual_complex':{'status':'unavailable','convention':'Exceptional divisor in the germ at the origin.',
                            'reason':'The search has not certified an atlas yet.','vertices':[],'cells':[]}}
    if request.get('monodromy') is True:
        result['monodromy_zeta'] = {'status':'unavailable', 'reason':'The resolution has not been certified.'}
    return result


def resolve(request):
    if not isinstance(request, dict):
        raise InputError("Expected an object containing polynomial, variables, and mode.")
    variables = request.get('variables')
    mode = request.get('mode','embedded')
    if mode not in ('embedded','intrinsic'):
        raise InputError("Mode must be embedded or intrinsic.")
    compute = request.get('compute')
    if compute is not None and compute not in ('cpu','auto','gpu'):
        raise InputError('Compute must be cpu, auto, or gpu.')
    if compute == 'gpu':
        # Read when the Groebner kernel is loaded, here or in a verification
        # worker: its modular replays then run on the OpenCL device.
        os.environ['ATLAS_GROEBNER_GPU'] = '1'
    policy = request.get('search','exhaustive')
    if policy not in ('quick','exhaustive'):
        raise InputError('Search must be quick or exhaustive.')
    wants_monodromy = request.get('monodromy', False)
    if not isinstance(wants_monodromy, bool):
        raise InputError('monodromy must be true or false.')
    exhaustive = policy=='exhaustive'
    start = perf_counter()
    try:
        p = parse(request.get('polynomial'), variables, bounded=not exhaustive)
    except LimitError as error:
        if exhaustive:
            raise
        return inconclusive(request,{'description':str(error)},(perf_counter()-start)*1000)
    try:
        result = _resolve_polynomial(p,request,start)
    except NonReducedError:
        if mode=='intrinsic':
            raise
        from .nonreduced import nonreduced_resolution
        result=nonreduced_resolution(p,variables,mode,lambda support:_resolve_polynomial(support,request,start))
        if result is None:
            raise
        result.update(polynomial=format_poly(p,variables),elapsed_ms=round((perf_counter()-start)*1000,3))
    if wants_monodromy:
        from .monodromy import monodromy_zeta
        from .topological import topological_zeta
        zeta = monodromy_zeta(result, polynomial=p)
        topological = topological_zeta(result, polynomial=p, monodromy=zeta)
        if mode == 'intrinsic' and result['status'] == 'resolved' and 'unavailable' in (zeta['status'], topological['status']):
            # An intrinsic resolution can stop while the embedded divisor is
            # still tangent to the boundary. Both formulas need an embedded one.
            embedded = _resolve_polynomial(p, dict(request, mode='embedded'), start)
            if zeta['status'] == 'unavailable':
                zeta = monodromy_zeta(embedded, polynomial=p)
                zeta['source'] = 'companion embedded resolution'
            if topological['status'] == 'unavailable':
                topological = topological_zeta(embedded, polynomial=p, monodromy=zeta)
                topological['source'] = 'companion embedded resolution'
        result['monodromy_zeta'] = zeta
        result['topological_zeta'] = topological
        result['elapsed_ms'] = round((perf_counter()-start)*1000, 3)
    return result


def _resolve_polynomial(p, request, start):
    variables = request['variables']
    mode = request.get('mode','embedded')
    compute = request.get('compute')
    exhaustive = request.get('search','exhaustive')=='exhaustive'
    diagnostic = {}
    r = None
    # Keep the optimized engines in their tested representation and dimensions.
    # The unbounded ideal search works over Q in any finite ambient dimension.
    native_range = (2 <= len(variables) <= 8 and len(p)<=4096
                    and max(map(sum,p))<=128)
    singular = min(map(sum,p))>1
    if singular and mode=='embedded' and 3<=len(variables)<=8:  # plane curves keep their own engine
        # Smooth components through the origin with independent differentials
        # there (multiplicities allowed): the divisor has normal crossings at
        # the origin, and the identity is an embedded resolution near it.
        from .monodromy_special import _factor_snc, _simple_coordinates
        crossing = _simple_coordinates(p,variables) or _factor_snc(p,variables)
        if crossing is not None:
            coords=[f'u{i+1}' for i in range(len(variables))]
            r={'status':'resolved','engine':'normal-crossings-at-origin','steps':[],
               'summary':f'The divisor has normal crossings at the origin, with {len(crossing)} components through it: the identity is an embedded resolution near the origin.',
               'charts':[{'id':'U0','coordinates':coords,'map':dict(zip(variables,coords)),
                          'strict_transform':format_poly(p,coords),'exceptional':[]}],
               'certificate':{'method':'smooth components with independent differentials at the origin',
                              'components':[{'equation':str(name),'multiplicity':m} for name,m in crossing]},
               'dual_complex':{'status':'complete','convention':'Exceptional divisor of the identity.',
                               'vertices':[],'cells':[]}}
            r.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
                     mode=mode, base_field='Q', geometric_field='algebraic closure of Q',
                     origin=[0]*len(variables), elapsed_ms=round((perf_counter()-start)*1000,3))
            r['execution'] = {'requested':compute or 'environment','rank_backend':None,'chart_backend':None,'device':'CPU'}
            return r
    try:
        if len(variables)==2 and singular:
            from .plane_backend import resolve_plane
            r = resolve_plane(p,variables,mode,exhaustive=exhaustive)
        elif not native_range and not singular:
            coords=[f'u{i+1}' for i in range(len(variables))]
            r={'status':'resolved','engine':'jacobian-at-origin','steps':[],
               'summary':'The hypersurface is smooth at the origin.',
               'charts':[{'id':'U0','coordinates':coords,'map':dict(zip(variables,coords)),
                          'strict_transform':format_poly(p,coords),'exceptional':[]}],
               'certificate':{'method':'nonzero linear part'},
               'dual_complex':{'status':'complete','convention':'Exceptional divisor of the identity.',
                               'vertices':[],'cells':[]}}
        elif native_range:
            if not (BUILD / ('libresolution.dylib' if sys.platform=='darwin' else 'libresolution.so')).is_file():
                raise RuntimeError('Native engine is not built. Run make in the project directory.')
            terms = [(k,str(c)) for k,c in p.items()]
            from .chain_backend import resolve_chain
            r = resolve_chain(p,variables,mode) if singular and len(variables)==3 else None
            if r is None:
                r = resolve_native(terms,variables,mode,compute=compute,diagnostics=diagnostic)
            if r is None and len(variables)==3:
                from .toric_backend import resolve_toric
                r = resolve_toric(p,variables,mode,compute=compute)
        if r is None and len(p)==2 and 2 <= len(variables) <= 8:
            from .binomial_backend import resolve_binomial
            r = resolve_binomial(p,variables,mode,compute=compute)
        if r is None and native_range:
            from .newton_backend import resolve_newton
            # A quick request is given 55 seconds in all by the worker.
            r = resolve_newton(p,variables,mode,compute=compute,seconds=1200 if exhaustive else 20)
        few = len(variables) == 3
        if r is None and native_range and singular and (few or exhaustive and len(variables) > 3):
            # No hypothesis on the Newton polyhedron: blowups with the centres
            # of the complete rule, started at the origin. Bounded, so that
            # the search below still gets its turn. Above three variables
            # only the invariant is available, and it is slow: a quick
            # request does not wait for it.
            from .origin_backend import resolve_origin
            r = resolve_origin(p,variables,mode,seconds=(60 if exhaustive else 15) if few else 20)
    except (LimitError,subprocess.TimeoutExpired) as error:
        diagnostic['fast_path_limit']=str(error)
    if r is None:
        from .general_resolution import search
        r = search(p,variables,mode,exhaustive=exhaustive)
    if r is None:
        return inconclusive(request,diagnostic,(perf_counter()-start)*1000,format_poly(p,variables))
    r.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
             mode=mode, base_field='Q', geometric_field='algebraic closure of Q',
             origin=[0]*len(variables), elapsed_ms=round((perf_counter()-start)*1000,3))
    proof = r.get('certificate',{})
    rank = proof.get('rank_backend')
    chart_backend = proof.get('chart_batch',{}).get('backend')
    backend = chart_backend or rank or 'cpu'
    r['execution'] = {'requested':compute or 'environment','rank_backend':rank,'chart_backend':chart_backend,
        'device':'OpenCL GPU and CPU' if backend=='opencl' else 'CPU'}
    if compute=='gpu' and backend!='opencl':
        r['execution']['note']='No GPU computation was used: this path has no suitable GPU work or the requested device was unavailable. The result uses the CPU.'
    return r
