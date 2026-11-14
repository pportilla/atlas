"""Dispatch only to algorithms whose hypotheses have been checked."""
from pathlib import Path
from time import perf_counter
import subprocess
import sys
from .polynomial import InputError, LimitError, NonReducedError, parse, format_poly, rational_text
from .native_backend import resolve_native
from .resolution_progress import recorded_blowups, report
from .resolution_zetas import selected_zetas, zetas_from_strata

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
    result.update(zetas_from_strata(result, selected_zetas(request), None))
    return result


def resolve(request):
    if not isinstance(request, dict):
        raise InputError("Expected an object containing polynomial, variables, and mode.")
    compute = request.get('compute')
    if compute is not None and compute not in ('cpu','auto','gpu'):
        raise InputError('Compute must be cpu, auto, or gpu.')
    from .native_library import replay_scope
    from .newton_faces import face_search
    with replay_scope(compute) as replay, face_search(request.get('search', 'exhaustive') == 'exhaustive'):
        proofs = []
        result = _resolve(request, proofs)
        result['execution'] = _execution(compute, proofs, replay.report())
        return result


def _execution(compute, proofs, replay):
    # The companion embedded resolution can supply both zetas for an intrinsic
    # atlas. Keep its successful work visible alongside the primary resolution.
    stages = []
    for role, proof in proofs:
        batch = proof.get('chart_batch')
        if batch:
            stages.append(dict(operation='chart_transform', resolution=role, **batch))
        if 'rank_backend' in proof:
            stage = dict(operation='jacobian_rank', resolution=role, backend=proof['rank_backend'])
            for key in ('device', 'rows', 'rank', 'rank_elapsed_ms', 'rank_fallback_reason'):
                if key in proof:
                    stage[key] = proof[key]
            stages.append(stage)
    proof = proofs[0][1] if proofs else {}
    execution = {'requested': compute or 'environment', 'rank_backend': proof.get('rank_backend'),
                 'chart_backend': proof.get('chart_batch', {}).get('backend'), 'stages': stages}
    backends = {stage['backend'] for stage in stages}
    if replay:
        for backend, count in replay.get('trace', {}).get('backends', {}).items():
            stage = dict(operation='groebner_trace', backend=backend, **count)
            if replay.get('device', {}).get('backend') == backend:
                stage['device'] = replay['device']
            stages.append(stage)
            backends.add(backend)
        if replay['batches']:
            details = {key: value for key, value in replay.items() if key != 'trace'}
            if details.get('device', {}).get('backend') not in details['backends']:
                details.pop('device', None)
            stages.append(dict(operation='groebner_replay', **details))
        backends.update(replay['backends'])
    device_names = [name for key, name in [('triton', 'Triton GPU'), ('opencl', 'OpenCL GPU')] if key in backends]
    execution['device'] = ' and '.join([*device_names, 'CPU'])
    if compute == 'gpu' and not device_names:
        execution['note'] = ('The eligible GPU stages used CPU fallback. See stages for details.' if stages
                             else 'This resolution used CPU algorithms and had no eligible GPU stage.')
    return execution


def _resolve(request, proofs):
    variables = request.get('variables')
    mode = request.get('mode','embedded')
    if mode not in ('embedded','intrinsic'):
        raise InputError("Mode must be embedded or intrinsic.")
    policy = request.get('search','exhaustive')
    if policy not in ('quick','exhaustive'):
        raise InputError('Search must be quick or exhaustive.')
    kinds = selected_zetas(request)
    wants_zetas = bool(kinds)
    exhaustive = policy=='exhaustive'
    start = perf_counter()
    report("Reading the polynomial", 0)
    try:
        p = parse(request.get('polynomial'), variables, bounded=not exhaustive)
    except LimitError as error:
        if exhaustive:
            raise
        return inconclusive(request,{'description':str(error)},(perf_counter()-start)*1000)
    try:
        verified = []
        result = _resolve_polynomial(p,request,start,verified=verified)
    except NonReducedError:
        if mode=='intrinsic':
            raise
        from .weighted_record import nonreduced_resolution
        result=nonreduced_resolution(p,variables,mode,lambda support:_resolve_polynomial(support,request,start))
        if result is None:
            raise
        result.update(polynomial=format_poly(p,variables),elapsed_ms=round((perf_counter()-start)*1000,3))
    from .strata import annotate_complex, rational_chart_outputs, resolution_strata

    def attach(record):
        if record["status"] != "resolved":
            return
        report("Verifying charts and exceptional intersections", recorded_blowups(record))
        intrinsic = record["mode"] == "intrinsic"
        certified = next((value for producer, value in verified if producer is record), None)
        if certified is None:
            certified = rational_chart_outputs(record, p, intrinsic=intrinsic, integrals=wants_zetas)
        if certified is not None:
            record["atlas"], output = certified
            record["dual_complex"] = annotate_complex(record["dual_complex"], output["dual_complex"])
        else:
            from . import atlas
            from .local_blowups import GiveUp
            from .native_library import _MODE
            charts = atlas.from_record(record, p, native=_MODE not in ("check", "sympy"))
            if charts is None:
                raise RuntimeError("The resolution engine did not supply its chart ownership data.")
            factor_fibre = any(step.get("rule") == "repeated tangent" for step in record.get("steps", ()))
            try:
                output = atlas.outputs(charts, len(variables), intrinsic=intrinsic, integrals=wants_zetas,
                                       factor_fibre=factor_fibre)
            except GiveUp:
                if not intrinsic or not wants_zetas:
                    raise
                # A smooth strict transform can need a further embedded resolution.
                output = atlas.outputs(charts, len(variables), intrinsic=True, integrals=False,
                                       factor_fibre=factor_fibre)
            record["atlas"] = atlas.to_json(charts)
            record["dual_complex"] = atlas.record_complex(record, charts, output["dual_complex"])
        if "chart_sections" in record["dual_complex"].get("finite_sheet_certificate", {}):
            from .atlas_finite import geometric_chart_records
            record["charts"] = geometric_chart_records(record)
        record.setdefault("certificate", {})["atlas_verification"] = (
            "Strict transform on the owned origin fibre" if intrinsic and "strata" not in output
            else "Embedded normal crossings on the owned origin fibre")
        if "strata" in output:
            if record.get("engine") == "canonical-blowups":
                record["certificate"].update(numerical_data=output["numerical_data"], strata=output["strata"])
            return resolution_strata(record, p, computed=output)

    proofs.append(('primary', result.get('certificate', {})))
    found = attach(result)
    if wants_zetas:
        record = result
        companion = found is None and mode == "intrinsic" and result["status"] == "resolved"
        if companion:
            report("Resolving an embedded companion for the zeta calculation")
            record = _resolve_polynomial(p, dict(request, mode="embedded"), start)
            proofs.append(('companion', record.get('certificate', {})))
            found = attach(record)
            if found is not None:
                result["embedded_companion"] = {"atlas": record["atlas"]}
        report("Computing the selected zeta functions", recorded_blowups(record))
        result.update(zetas_from_strata(record, kinds, found, companion=companion))
    result['elapsed_ms'] = round((perf_counter()-start)*1000, 3)
    return result


def _resolve_polynomial(p, request, start, *, verified=None):
    report("Checking the singularity at the origin", 0)
    variables = request['variables']
    mode = request.get('mode','embedded')
    compute = request.get('compute')
    exhaustive = request.get('search','exhaustive')=='exhaustive'
    diagnostic = {}
    r = None
    # Keep the optimized engines in their tested representation and dimensions.
    # The unbounded ideal search works over Q in any finite ambient dimension.
    from .quotient_monomials import degree_bounds
    order, degree = degree_bounds(len(variables), p)
    native_range = (2 <= len(variables) <= 8 and len(p)<=4096
                    and degree<=128)
    singular = order>1
    parallel_search = singular and exhaustive and 3 <= len(variables) <= 8
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
            return r
    try:
        if len(variables)==2 and singular:
            from .plane_backend import resolve_plane
            try:
                r = resolve_plane(p,variables,mode,exhaustive=exhaustive)
            except LimitError as error:
                if not exhaustive:
                    raise
                diagnostic['plane_limit']=str(error)
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
            report("Testing ordinary and toric resolutions")
            if not (BUILD / ('libresolution.dylib' if sys.platform=='darwin' else 'libresolution.so')).is_file():
                raise RuntimeError('Native engine is not built. Run make in the project directory.')
            terms = [(k, rational_text(c)) for k,c in p.items()]
            from .chain_backend import resolve_chain
            r = resolve_chain(p,variables,mode) if singular and len(variables)==3 else None
            if r is None:
                r = resolve_native(terms,variables,mode,compute=compute,diagnostics=diagnostic)
            if r is None and len(variables)==3:
                from .toric_backend import resolve_toric
                r = resolve_toric(p,variables,mode,compute=compute)
        if r is None and len(p)==2 and 2 <= len(variables) <= 8 and 'plane_limit' not in diagnostic:
            from .binomial_backend import resolve_binomial
            r = resolve_binomial(p,variables,mode,compute=compute)
    except (LimitError,subprocess.TimeoutExpired) as error:
        diagnostic['fast_path_limit']=str(error)
    if r is None and parallel_search and mode == "embedded" and len(p) <= 3 and len(variables) <= 4:
        from .newton_trials import bounded_first_trial
        from .native_library import Unavailable
        try:
            r = bounded_first_trial(p, variables, verified=verified, integrals=bool(selected_zetas(request)))
        except (LimitError, Unavailable, OSError):
            pass  # The isolated exhaustive workers still test all three methods.
    if (r is None and parallel_search and mode == "embedded" and len(variables) == 3
            and 4 <= len(p) <= 32 and degree <= 16):
        from .newton_trials import bounded_coordinate_graph
        from .native_library import Unavailable
        try:
            r = bounded_coordinate_graph(p, variables)
        except (LimitError, Unavailable, OSError):
            pass
    # The tiny monomial Newton proposals precede this general centre in
    # both search policies. Keep their isolated retries on a packed decline.
    if r is None and native_range and singular and len(variables) >= 3 and len(p)>3:
        from .reduced_centre import smooth_reduced_graph
        r = smooth_reduced_graph(p, variables, mode)
    try:
        if r is None and not parallel_search and (native_range or exhaustive and singular and len(variables)==2):
            from .newton_backend import resolve_newton
            # A quick request is given 55 seconds in all by the worker.
            r = resolve_newton(p, variables, mode, compute=compute, seconds=1200 if exhaustive else 20,
                               defer_outputs=True, verified=verified, integrals=bool(selected_zetas(request)))
    except (LimitError,subprocess.TimeoutExpired) as error:
        diagnostic['fast_path_limit']=str(error)
    if r is None and parallel_search:
        from .exhaustive_resolution import search as exhaustive_search
        report("Searching for a verified resolution with independent methods")
        r = exhaustive_search(p, variables, mode, compute=compute)
    elif r is None and native_range and singular and len(variables) == 3:
        from .origin_backend import resolve_origin
        r = resolve_origin(p, variables, mode, seconds=15, defer_outputs=True)
    elif r is None and native_range and singular and len(variables) >= 4:
        # The repeated tangent rule has five specified centres. Let a quick
        # request try those without entering the open-ended invariant search.
        from .origin_backend import resolve_origin
        r = resolve_origin(p, variables, mode, seconds=30, defer_outputs=True, tangent_only=True)
    if r is None:
        from .general_resolution import search
        report("Searching for a blow-up centre")
        r = search(p,variables,mode,exhaustive=exhaustive)
    if r is None:
        return inconclusive(request,diagnostic,(perf_counter()-start)*1000,format_poly(p,variables))
    r.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
             mode=mode, base_field='Q', geometric_field='algebraic closure of Q',
             origin=[0]*len(variables), elapsed_ms=round((perf_counter()-start)*1000,3))
    return r
