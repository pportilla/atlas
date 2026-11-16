"""Dispatch only to algorithms whose hypotheses have been checked."""
from pathlib import Path
from time import perf_counter
import subprocess
import sys
from .polynomial import InputError, LimitError, NonReducedError, parse, format_poly, rational_text, saved_polynomial_input
from .native_backend import resolve_native
from .resolution_progress import recorded_blowups, report
from .resolution_zetas import attach_companion, finish_requested_check, selected_zetas, zetas_from_strata

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'native' / 'build'


def inconclusive(request, diagnostic, elapsed_ms, polynomial=None):
    variables = request['variables']
    result = {'status':'inconclusive', 'engine':'general-search', 'can_search':True,
            'summary':'The quick search could not certify all requested outputs. Exhaustive search can test more blowups.',
            'diagnostic':diagnostic, 'charts':[], 'steps':[], 'schema_version':1,
            'variables':variables, 'polynomial':polynomial or request['polynomial'],
            'mode':request.get('mode','embedded'), 'base_field':'Q',
            'geometric_field':'algebraic closure of Q', 'origin':[0]*len(variables),
            'elapsed_ms':round(elapsed_ms,3),
            'dual_complex':{'status':'unavailable','convention':'Exceptional divisor in the germ at the origin.',
                            'reason':'The search has not certified an atlas yet.','vertices':[],'cells':[]}}
    result.update(zetas_from_strata(result, selected_zetas(request), None))
    return result


METHODS = ("auto", "weighted", "jung")


def requested_method(request):
    """The resolution method a request selects, with the mode it implies when none is given.

    ``auto`` runs the existing dispatch. ``weighted`` tries the native weighted
    engine before the ordinary embedded search when exhaustive completion is
    requested. ``jung`` runs Jung's method for surfaces, which resolves the
    surface itself, so its default mode is intrinsic.
    """
    method = request.get("method", "auto")
    if method not in METHODS:
        raise InputError("Method must be auto, weighted or jung.")
    mode = request.get("mode")
    if mode is None:
        mode = "intrinsic" if method == "jung" else "embedded"
    return method, mode


def resolve(request):
    if not isinstance(request, dict):
        raise InputError("Expected an object containing polynomial, variables, and mode.")
    method, mode = requested_method(request)
    request = dict(request, method=method, mode=mode)
    compute = request.get('compute')
    if compute is not None and compute not in ('cpu','auto','gpu'):
        raise InputError('Compute must be cpu, auto, or gpu.')
    from .native_library import replay_scope
    from .newton_faces import face_search
    with replay_scope(compute) as replay, face_search(request.get('search', 'exhaustive') == 'exhaustive'):
        start = perf_counter()
        proofs = []
        result = _resolve(request, proofs)
        finish_requested_check(result, result, request.get("search", "exhaustive"))
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
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
    exhaustive = policy=='exhaustive'
    start = perf_counter()
    report("Reading the polynomial", 0)
    try:
        p = parse(request.get('polynomial'), variables, bounded=not exhaustive)
    except LimitError as error:
        if exhaustive:
            raise
        return inconclusive(request,{'description':str(error)},(perf_counter()-start)*1000)
    if request.get('method') == 'jung':
        return _resolve_jung(p, request, kinds, start)
    return _resolve_parsed(p, variables, mode, request, kinds, start, proofs)


@saved_polynomial_input()
def _resolve_parsed(p, variables, mode, request, kinds, start, proofs):
    # The caller's literals have passed validation. Coefficients and exponents
    # printed from this point on can be larger as a result of exact arithmetic.
    exhaustive = request.get('search', 'exhaustive') == 'exhaustive'
    wants_zetas = bool(kinds)
    if request.get('method') == 'weighted':
        return _resolve_weighted(p, request, kinds, start, proofs)
    if mode == "intrinsic":
        from .identity_atlas import reduced_at_origin
        if not reduced_at_origin(p, variables):
            raise NonReducedError("The germ is nonreduced: an intrinsic resolution requires a reduced hypersurface.")
    if request.get("zetas_only") and len(variables) == 3:
        from .contact_zetas import contact_record
        contact = contact_record(p, variables, kinds)
        if contact is not None:
            contact["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
            return contact
    from .local_blowups import GiveUp
    from .strata import annotate_complex, proposal_chart_outputs, resolution_strata
    verified = []

    def general_certificate():
        from .general_resolution import search
        report("Searching projective blowups for a certificate of the requested outputs")
        def find(poly):
            return search(poly, variables, mode, exhaustive=True, integrals=wants_zetas,
                          verified=verified)

        result = resolve_with_multiplicities(find)
        if result is None:
            raise RuntimeError("The exhaustive certificate enumerator exited without a resolution.")
        result.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
                      mode=mode, base_field="Q", geometric_field="algebraic closure of Q", origin=[0]*len(variables))
        return result

    def resolve_with_multiplicities(find):
        try:
            return find(p)
        except NonReducedError:
            if mode == "intrinsic":
                raise
            from .weighted_record import nonreduced_resolution
            result = nonreduced_resolution(p, variables, mode, find)
            if result is None:
                raise
            # A verified bundle for the reduced support carries different
            # multiplicities, even when the chart record was lifted in place.
            verified[:] = [item for item in verified if item[0] is not result]
            result.update(polynomial=format_poly(p, variables), elapsed_ms=round((perf_counter()-start)*1000, 3))
            return result

    def attach(record, deadline=None):
        if record["status"] != "resolved":
            return
        report("Verifying charts and exceptional intersections", recorded_blowups(record))
        intrinsic = record["mode"] == "intrinsic"
        certified = next((value for producer, value in verified if producer is record), None)
        if certified is None and deadline is not None:
            from .local_blowups import time_limit
            with time_limit(deadline):
                certified = proposal_chart_outputs(record, p, integrals=wants_zetas)
        elif certified is None:
            certified = proposal_chart_outputs(record, p, integrals=wants_zetas)
        record["atlas"], output = certified
        record["dual_complex"] = annotate_complex(record["dual_complex"], output["dual_complex"])
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

    # A fast proposal can have a smooth strict transform while its owned
    # charts still lack the numerical data needed for integration. Exhaustive
    # search admits a projective certificate only after its requested outputs
    # have passed verification inside the candidate's timed attempt.
    def finish(result, deadline=None):
        found = attach(result, deadline)
        if exhaustive and (result["status"] != "resolved" or
                           (not request.get("zetas_only") and result["dual_complex"]["status"] != "complete") or
                           (wants_zetas and mode == "embedded" and found is None)):
            raise GiveUp("The proposed charts did not certify all the requested outputs.")
        if found is not None and wants_zetas:
            report("Computing the selected zeta functions", recorded_blowups(result))
            return zetas_from_strata(result, kinds, found, polynomial=p)
        return None

    proposed = {}

    def verification_deadline():
        """For a quick request, when verifying the primary proposal must give way.

        Replaying a proposal's blowups can take longer than constructing them.
        The canonical fallback then keeps its own time within the worker's limit.
        """
        if exhaustive or proposed.get("engine") == "canonical-blowups" or not canonical_applicable(p, variables):
            return None
        from time import monotonic
        from .resolve_cli import MAX_SECONDS
        return monotonic() + max(0.0, MAX_SECONDS - QUICK_FALLBACK_RESERVE - (perf_counter() - start))

    def primary(poly):
        record = _resolve_polynomial(poly, request, start, verified=verified)
        proposed["engine"] = record.get("engine")
        return record

    def quick_fallback(error):
        # A fast engine can propose charts that their verifier rejects. The
        # canonical smooth-centre rule makes no assumption about the Newton
        # polyhedron or the rejected proposal, and its charts are verified
        # in the same way. It does not run twice.
        reasons = [str(error)]
        if proposed.get("engine") != "canonical-blowups" and canonical_applicable(p, variables):
            try:
                record = canonical_record(p, variables, mode, start, verified=verified, integrals=wants_zetas)
                if record is None:
                    raise GiveUp("The canonical smooth-centre rule returned no certified atlas.")
                return record, finish(record)
            except NonReducedError:
                raise
            except (GiveUp, LimitError, InputError, ArithmeticError, subprocess.TimeoutExpired) as second:
                reasons.append(f"After that proposal was rejected: {second}")
        return inconclusive(request, {"description": " ".join(reasons)}, (perf_counter()-start)*1000,
                            format_poly(p, variables)), None

    try:
        result = resolve_with_multiplicities(primary)
        outputs = finish(result, verification_deadline())
    except NonReducedError:
        raise  # Intrinsic resolution requires a reduced source.
    except (GiveUp, LimitError, InputError, ArithmeticError, subprocess.TimeoutExpired) as error:
        # The input equation, mode and origin have been validated. These
        # failures concern a solver-generated proposal, never a saved proof.
        if exhaustive:
            result = general_certificate()
            outputs = finish(result)
        else:
            result, outputs = quick_fallback(error)
    proofs.append(('primary', result.get('certificate', {})))
    if wants_zetas:
        if outputs is None and mode == "intrinsic" and result["status"] == "resolved":
            _complete_companion(result, request, kinds, proofs)
            result['elapsed_ms'] = round((perf_counter()-start)*1000, 3)
            return result
        if outputs is None and mode == "embedded" and not request.get("zetas_only"):
            from .contact_zetas import contact_zetas
            outputs = contact_zetas(p, variables, kinds)
        if outputs is None:
            outputs = zetas_from_strata(result, kinds, None)
        result.update(outputs)
    result['elapsed_ms'] = round((perf_counter()-start)*1000, 3)
    return result


def _complete_companion(record, request, kinds, proofs):
    report("Resolving an ordinary embedded companion for the selected zeta functions")
    companion_proofs = []
    embedded = _resolve(dict(request, mode="embedded", method="auto", zetas_only=True), companion_proofs)
    attach_companion(record, embedded, kinds)
    proofs.extend(("companion", proof) for _, proof in companion_proofs)


def _resolve_weighted(p, request, kinds, start, proofs):
    """Compute the weighted charts and use ordinary embedded data for any requested zetas."""
    if request.get('mode', 'embedded') != 'embedded':
        raise InputError("The weighted resolution is embedded: use the embedded mode with method weighted.")
    from .weighted_blowups import SECONDS, WeightedDeclined, weighted_resolution
    report("Resolving by weighted blowups", 0)
    seconds = SECONDS if request.get("search") == "quick" else None
    try:
        record = weighted_resolution(p, request['variables'], seconds=seconds)
    except WeightedDeclined as error:
        reason = str(error)
        if seconds is None:
            report("Continuing with ordinary embedded resolution after a weighted method decline")
            ordinary = dict(request, method="auto", zetas=list(kinds), zetas_only=False)
            record = _resolve(ordinary, proofs)
        else:
            record = inconclusive(request, {"description": reason}, (perf_counter()-start)*1000)
        record["requested_method"] = "weighted"
        record["weighted_attempt"] = {"status": "declined", "reason": reason}
        record["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        return record
    if kinds:
        _complete_companion(record, request, kinds, proofs)
    record['elapsed_ms'] = round((perf_counter() - start) * 1000, 3)
    return record


def _resolve_jung(p, request, kinds, start):
    """Use Jung's method now, after checking that the request and the surface allow it.

    The request must ask for the surface itself (intrinsic mode) in three
    variables without zeta functions, which need an embedded resolution. The
    native resolver then verifies the method's hypotheses on this surface
    before using them, and names the one that fails.
    """
    variables = request['variables']
    if len(variables) != 3:
        raise InputError("Jung's method resolves surfaces f(x,y,z) = 0: give exactly three variables, "
                         f"not {len(variables)}.")
    if request['mode'] != 'intrinsic':
        raise InputError("Jung's method resolves the surface itself, not its embedding in 3-space: "
                         "use the intrinsic mode with method jung.")
    if kinds:
        raise InputError("Jung's method gives no embedded resolution, which the zeta functions need: "
                         "drop monodromy and zetas, or use method auto.")
    from .jung import JungDeclined, jung_record
    report("Checking the hypotheses of Jung's method", 0)
    try:
        native = jung_record(None, variables, terms=p)
    except JungDeclined as error:
        raise InputError(f"Jung's method does not apply here ({error.hypothesis}): {error}") from None
    minimal = native.get('minimal_good_resolution', {})
    record = {
        'status': 'resolved', 'engine': 'jung', 'mode': 'intrinsic', 'schema_version': 1,
        'summary': native['summary'], 'polynomial': format_poly(p, variables), 'variables': variables,
        'base_field': 'Q', 'geometric_field': 'algebraic closure of Q', 'origin': [0] * len(variables),
        'charts': [], 'steps': [], 'dual_complex': native['dual_complex'],
        'invariants': minimal.get('invariants', native.get('invariants', {})),
        'certificate': {'method': "Jung's method", 'hypotheses': native['applicability']},
        'jung': native,
    }
    record['elapsed_ms'] = round((perf_counter() - start) * 1000, 3)
    return record


# Seconds that a quick request gives the canonical smooth-centre rule.
QUICK_CANONICAL_SECONDS = 20
# Seconds of the quick worker's limit kept for that rule when it must replace
# a proposal whose verification did not finish: its run and its outputs.
QUICK_FALLBACK_RESERVE = QUICK_CANONICAL_SECONDS + 5


def canonical_applicable(p, variables):
    """Whether the canonical smooth-centre rule accepts this singular germ."""
    from .quotient_monomials import degree_bounds
    order, degree = degree_bounds(len(variables), p)
    return order > 1 and 3 <= len(variables) <= 8 and len(p) <= 4096 and degree <= 128


def canonical_record(p, variables, mode, start, *, verified=None, integrals=False,
                     seconds=QUICK_CANONICAL_SECONDS):
    """The canonical smooth-centre resolution as a solver record, or None."""
    from .origin_backend import resolve_origin
    report("Resolving by blowups with smooth centres chosen by the canonical rule")
    r = resolve_origin(p, variables, mode, seconds=seconds, defer_outputs=True, verified=verified,
                       integrals=integrals)
    if r is not None:
        r.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
                 mode=mode, base_field='Q', geometric_field='algebraic closure of Q',
                 origin=[0]*len(variables), elapsed_ms=round((perf_counter()-start)*1000,3))
    return r


def _resolve_polynomial(p, request, start, *, verified=None):
    report("Checking the singularity at the origin", 0)
    variables = request['variables']
    mode = request.get('mode','embedded')
    compute = request.get('compute')
    exhaustive = request.get('search','exhaustive')=='exhaustive'
    diagnostic = {}
    fallback_details = []
    r = None
    # Keep the optimized engines in their tested representation and dimensions.
    # The unbounded ideal search works over Q in any finite ambient dimension.
    from .quotient_monomials import degree_bounds
    order, degree = degree_bounds(len(variables), p)
    native_range = (2 <= len(variables) <= 8 and len(p)<=4096
                    and degree<=128)
    singular = order>1
    parallel_search = singular and exhaustive and 3 <= len(variables) <= 8
    if singular and mode=='embedded':
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
            from .weighted_record import crossing_input_divisor
            divisor = crossing_input_divisor(p, variables, crossing)
            if divisor is not None:
                r.update(input_divisor=divisor, multiplicity=order)
        elif native_range:
            from .plane_identity import conjugate_plane_identity
            r = conjugate_plane_identity(p, variables, degree=degree,
                                         integrals=bool(selected_zetas(request)), verified=verified)
        if r is not None:
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
            elif r is None and len(variables) == 4:
                from .quadratic_fan import resolve_translated
                r = resolve_translated(p, variables, mode, compute)
        if r is None and len(p)==2 and 2 <= len(variables) <= 8 and 'plane_limit' not in diagnostic:
            from .binomial_backend import resolve_binomial
            r = resolve_binomial(p,variables,mode,compute=compute)
    except (LimitError,subprocess.TimeoutExpired) as error:
        diagnostic['fast_path_limit']=str(error)
    if r is None and native_range and len(variables) == 3:
        from .plane_arrangement import resolve_arrangement
        r = resolve_arrangement(p, variables, mode, verified, bool(selected_zetas(request)))
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
    newton_declines = {}
    try:
        if r is None and not parallel_search and (native_range or exhaustive and singular and len(variables)==2):
            from .newton_backend import resolve_newton
            # A quick request is given 55 seconds in all by the worker.
            r = resolve_newton(p, variables, mode, compute=compute, seconds=1200 if exhaustive else 20,
                               defer_outputs=True, verified=verified, integrals=bool(selected_zetas(request)),
                               diagnostics=newton_declines)
            if r is None:
                fallback_details.append("The Newton search returned no certified atlas.")
    except (LimitError,subprocess.TimeoutExpired) as error:
        diagnostic['fast_path_limit']=str(error)
        fallback_details.append(f"The Newton search reached a limit ({error}).")
    if r is None and newton_declines:
        diagnostic["newton_declines"] = newton_declines
        fallback_details.extend(newton_declines.values())
    if r is None and parallel_search:
        from .exhaustive_resolution import search as exhaustive_search
        report("Searching for a verified resolution with independent methods")
        r = exhaustive_search(p, variables, mode, compute=compute, integrals=bool(selected_zetas(request)),
                              verified=verified, zetas_only=request.get("zetas_only", False))
    elif r is None and native_range and singular and len(variables) == 3:
        from .origin_backend import resolve_origin
        r = resolve_origin(p, variables, mode, seconds=15, defer_outputs=True,
                           verified=verified, integrals=bool(selected_zetas(request)))
        if r is None:
            fallback_details.append("The successive blowup search returned no certified atlas.")
    elif r is None and native_range and singular and len(variables) >= 4:
        # The repeated tangent rule has five specified centres. When it does
        # not apply, the canonical rule chooses the centres from the
        # invariant, in a bounded time.
        from .origin_backend import resolve_origin
        r = resolve_origin(p, variables, mode, seconds=30, defer_outputs=True, tangent_only=True,
                           verified=verified, integrals=bool(selected_zetas(request)))
        if r is None:
            fallback_details.append("The repeated-tangent search returned no certified atlas.")
            r = resolve_origin(p, variables, mode, seconds=QUICK_CANONICAL_SECONDS, defer_outputs=True,
                               verified=verified, integrals=bool(selected_zetas(request)), skip_tangent=True)
            if r is None:
                fallback_details.append("The canonical smooth-centre rule returned no certified atlas.")
    if r is None:
        from .general_resolution import search
        report("Searching for a blow-up centre")
        if exhaustive:
            r = search(p, variables, mode, exhaustive=True, integrals=bool(selected_zetas(request)),
                       verified=verified, resolve_reduced_first=True)
        else:
            counts = {}
            r = search(p, variables, mode, exhaustive=False, candidate_outcomes=counts)
            if r is None:
                diagnostic["candidate_outcomes"] = counts
                fallback_details.append(f"The general search checked {sum(counts.values())} candidates "
                                        f"({counts.get('rejected', 0)} rejected, {counts.get('undecided', 0)} undecided).")
    if r is None:
        diagnostic["description"] = " ".join(filter(None, [diagnostic.get("description"), *fallback_details]))
        return inconclusive(request,diagnostic,(perf_counter()-start)*1000,format_poly(p,variables))
    r.update(schema_version=1, variables=variables, polynomial=format_poly(p, variables),
             mode=mode, base_field='Q', geometric_field='algebraic closure of Q',
             origin=[0]*len(variables), elapsed_ms=round((perf_counter()-start)*1000,3))
    return r
