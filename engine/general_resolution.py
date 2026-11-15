"""Verify an embedded resolution presented as a single projective ideal blowup.

A finite composition of blowups over affine space is itself the blowup of a
finitely generated ideal (Stacks 080B). This permits a slower complete search
in addition to the geometric algorithms. Every candidate is checked.
"""
import atexit
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import monotonic, sleep

from . import native_polynomial as np
from .candidate_field import NativeField
from .chart_expressions import chart_expressions, runtime_expressions
from .resolution_progress import report
from .reference import sp
from .reference import QQ

from . import kernel
from .blowup_ideal import (assemble_charts, blowup_charts, empty_locus, ideal_basis, ideal_subset,
                           radical_subset, resolution_charts, verify_snc)
from .native_library import current_replay_scope, replay_scope
from .rational_point import coordinate_fraction
from .polynomial import integer_text
from .reduced_centre import smooth_reduced_graph


def expression(poly, variables):
    if np.native_ring(variables):
        return np.from_terms(variables, poly)
    return sp.Add(*(sp.Rational(c.numerator,c.denominator)*sp.Mul(*(x**d for x,d in zip(variables,e)))
                    for e,c in poly.items()))


def _text(value, domain, generator="field_generator"):
    if isinstance(domain, NativeField):
        return str(np.coerce(value).xreplace({domain.parameter: np.Symbol(generator)}))
    if domain!=QQ:
        # Express every coefficient in the primitive basis. A lowered
        # coefficient such as sqrt(3) need not literally contain the
        # generator sqrt(2)+sqrt(3) as a subexpression.
        from .affine_kernel import pullbacks
        from .field_polynomial import lift, modulus
        value = sp.sympify(value)
        theta = sp.Symbol(generator)
        variables = tuple(sorted(value.free_symbols, key=sp.default_sort_key))
        ring = (*variables, theta)
        value, = pullbacks(ring, ring, ring, (lift(value, variables, domain, theta),), modulus(domain, theta))
    if isinstance(value, np.Expr):
        # Preserve spacing in certificates and their D(unit) descriptions.
        text = str(value)
        return text[:1] + text[1:].replace('+', ' + ').replace('-', ' - ')
    return str(value).replace('**','^')


def _cover(variables, ambient, pivot, pieces, domain):
    from .graph_record import _graph_request, _graph_ring
    names, text, relation = _graph_ring(variables, domain)
    return _graph_request("cover", names, ambient=[*map(text, ambient), *relation], pivot=text(pivot),
                          pieces=[list(map(text, piece)) for piece in pieces])


def _glue(charts, ideals, domain):
    """Identify irreducible affine pieces using their nonempty intersections on overlaps."""
    from .graph_record import _graph_request, _graph_ring
    from .incidence import component_classes
    keys = sorted(ideals)

    def compare(i, j):
        a, b = keys[j], keys[i]
        source, target = charts[a[0]], charts[b[0]]
        variables = tuple(dict.fromkeys((*source.variables, *target.variables)))
        names, text, relation = _graph_ring(variables, domain)
        return int(_graph_request("component_overlap", names,
                                  left_ratios=list(map(text, source.ratios)),
                                  right_ratios=list(map(text, target.ratios)),
                                  right_pivot=target.pivot_index,
                                  pieces=[*map(text, ideals[a]), *map(text, ideals[b]), *relation]))

    classes = component_classes([(ci, None) for ci, _ in keys], compare)
    groups = {}
    for group, key in zip(classes, keys):
        groups.setdefault(group, []).append(key)
    return [{ci: ideals[(ci, i)] for ci, i in keys_in_group} for keys_in_group in groups.values()]


def _component_atlas(charts, original, pieces, mode, domain, effort, generator):
    from .geometric_components import geometric_components
    from .graph_record import _graph_request, _graph_ring
    coordinates = tuple(dict.fromkeys(v for chart in charts for v in chart.variables))
    names, text, relation = _graph_ring(coordinates, domain)
    encoded = [dict(ambient=list(map(text, c.ambient)), strict=list(map(text, c.strict)),
                    pieces=[list(map(text, piece)) for piece in boundary]) for c, boundary in zip(charts, pieces)]
    tasks = _graph_request("strata_tasks", names, charts=encoded, relation=relation, dimension=len(original),
                           intrinsic=mode == "intrinsic", count=len(pieces[0]))
    strata = []
    for task in tasks:
        subset = task["subset"]
        local = {}
        for ci in task["charts"]:
            chart = charts[ci]
            carrier = chart.strict if mode == "intrinsic" else chart.ambient
            equations = (*carrier, *(g for j in subset for g in pieces[ci][j]))
            components = geometric_components(chart.variables, equations, task["dimension"], domain=domain, effort=effort)
            if components is None:
                return None
            local.update({(ci, i): ideal for i, ideal in enumerate(components)})
        groups = []
        for group in _glue(charts, local, domain):
            groups.append([dict(chart=ci, ideal=list(map(text, ideal)),
                                display=[_text(g, domain, generator) for g in ideal] if len(subset) == 1 else [])
                           for ci, ideal in group.items()])
        strata.append(dict(subset=subset, dimension=task["dimension"], groups=groups))
    answer = _graph_request("stratum_faces", names, charts=len(charts), strata=strata, relation=relation,
                            fibre=list(map(text, original)))
    divisors = answer.pop("divisors")
    for chart in divisors:
        for component in chart:
            component["equation"] = ", ".join(component["ideal"])
    convention = "Ambient exceptional divisor" if mode == "embedded" else "Reduced exceptional boundary on the smooth strict transform"
    return dict(status="complete", convention=convention+" in the algebraic germ over Qbar. Cells are geometric intersection components meeting the origin fiber.",
                vertices=answer["vertices"], cells=answer["cells"]), divisors


def _pieces(candidate, charts):
    """The boundary ideals of the candidate in the coordinates of each chart."""
    return [[tuple(eq.xreplace(dict(zip(candidate.projective,chart.ratios))) for eq in group)
             for group in candidate.boundary] for chart in charts]


def _charts(variables,f,candidate):
    """The charts of the blowup, their boundary pieces and the number of strata
    proved smooth; False when the candidate fails one of the tests."""
    domain=candidate.domain
    center=candidate.center
    # Homogeneity makes the supplied ideals agree on overlaps of D_+(U_i).
    # Verify the property here too: direct callers can supply their own candidate.
    for group in candidate.boundary:
        for g in group:
            terms = _term_rows(g, (*variables,*candidate.projective), domain)
            if terms:
                from .quotient_monomials import degree_bounds
                low, high = degree_bounds(len(candidate.projective), (e[len(variables):] for e,_ in terms))
                if low != high:
                    return False
    # The blowup must preserve the smooth locus of the original hypersurface.
    from .affine_kernel import frame_derivative
    gradient = tuple(frame_derivative(variables, f, {x:1}) for x in variables)
    if not radical_subset(variables,(f,*gradient),center,domain=QQ):
        return False
    charts=blowup_charts(variables,(),(f,),center,domain=domain)
    pieces=_pieces(candidate,charts)
    snc_checks=0
    for chart,boundary in zip(charts,pieces):
        if not _cover(chart.variables,chart.ambient,chart.pivot,boundary,domain):
            return False
        proof=verify_snc(chart.variables,chart.ambient,chart.strict,boundary,len(variables),domain=domain)
        if not proof['ok']:
            return False
        snc_checks+=len(proof['strata'])
    return charts,pieces,snc_checks


def _verify(variables,f,candidate,mode,effort,proof=None):
    """Distinguish permanent counterexamples from inconclusive projection efforts.

    `proof` is the kernel's verdict on the candidate (resolution_charts) when
    the caller already has it.
    """
    if mode not in ("embedded","intrinsic"):
        raise ValueError("Resolution mode must be embedded or intrinsic.")
    domain=candidate.domain
    center=candidate.center
    u=candidate.projective
    if (not center or all(v==0 for v in center) or len(u)!=len(center)
            or len(set(u))!=len(u) or set(u).intersection(variables)):
        return 'rejected', None
    # Try the packed rational verifier, then the general marked or wide chart operations.
    proved = proof
    if proved is None and domain==QQ:
        proved = resolution_charts(variables,f,center,u,candidate.boundary)
    if proved:
        charts,snc_checks = proved
        proved = charts,_pieces(candidate,charts),snc_checks
    if proved is None:
        proved = _charts(variables,f,candidate)
    if not proved:
        return 'rejected', None
    charts,pieces,snc_checks = proved
    return _verified_record(variables,candidate,mode,effort,charts,pieces,snc_checks)


def _verified_record(variables,candidate,mode,effort,charts,pieces,snc_checks,*,coordinate_line=False,graph_bundle=False):
    domain, center, u = candidate.domain, candidate.center, candidate.projective
    used = {v.name for chart in charts for v in chart.variables} | {v.name for v in candidate.projective}
    generator = "field_generator"
    while generator in used:
        generator += "_"
    if coordinate_line or graph_bundle:
        from .incidence import incidence, stratum_dimension
        axes = [str(x) for x in center]
        divisors = []
        for (piece,) in pieces:
            ideal = [_text(g,domain,generator) for g in piece]
            divisors.append([dict(id="E1",equation=", ".join(ideal),ideal=ideal)])
        ident = "E1" if graph_bundle else incidence("coordinate_exceptional",
            variables=[str(x) for x in variables], centre=axes,
            charts=[{"exceptional":components} for components in divisors])
        if ident is None:
            return "undecided", None
        dual = {"status":"complete", "convention":"Ambient exceptional divisor in the algebraic germ over Qbar. "
                "Cells are geometric intersection components meeting the origin fiber.",
                "vertices":[{"id":ident,"dimension":stratum_dimension(len(variables),1)}], "cells":[]}
        result = dual,divisors
    else:
        result=_component_atlas(charts,variables,pieces,mode,domain,effort,generator)
    if result is None:
        return 'undecided', None
    dual,divisors=result
    ambient_divisors = divisors
    if mode == "intrinsic":
        ambient = _component_atlas(charts, variables, pieces, "embedded", domain, effort, generator)
        if ambient is None:
            return "undecided", None
        ambient_divisors = ambient[1]
    atlas=[]
    for i,chart in enumerate(charts):
        if not graph_bundle and empty_locus(chart.variables,(*chart.ambient,*variables),domain=domain):
            continue
        # On a graph chart the ambient equations are part of the coordinate
        # ring. Present only additional strict equations when possible.
        strict=chart.strict if graph_bundle else [
            e for e in chart.strict if not ideal_subset(chart.variables,(e,),chart.ambient,domain=domain)]
        atlas.append({'id':f'G{i+1}','coordinates':[str(x) for x in chart.variables],
            'ambient_ideal':[_text(e,domain,generator) for e in chart.ambient],
            'strict_transform':[_text(e,domain,generator) for e in strict],
            'map':{str(x):str(x) for x in variables},'exceptional':divisors[i],
            **({'ambient_components': ambient_divisors[i]} if mode == "intrinsic" else {}),
            'pivot_generator':_text(chart.pivot,domain,generator),'projective_coordinates':list(map(str,chart.ratios))})
    out={'status':'resolved','engine':'verified-ideal-blowup','charts':atlas,'dual_complex':dual,
            'steps':[{'id':'B','type':'blowup of an ideal','generators':[_text(e,domain,generator) for e in center]}],
            'summary':f'Certified a projective resolution in {len(atlas)} affine charts over the origin.',
            'certificate':{'method':'saturation of the projective graph, geometric Jacobian tests and overlap incidence',
                'verified_affine_strata':snc_checks,'center_source':candidate.source,
                'projective_coordinates':[str(x) for x in candidate.projective],
                'boundary_ideals':[[_text(e,domain,generator) for e in piece] for piece in candidate.boundary]}}
    if domain!=QQ:
        primitive = domain.exact_root if isinstance(domain, NativeField) else candidate.primitive
        coefficients = domain.coefficients if isinstance(domain, NativeField) else domain.ext.minpoly.all_coeffs()
        out['coefficient_field']={'name':f'Q({generator})','generator':generator,
            'exact_root':str(primitive),'description':f'{generator} = {primitive}',
            'minimal_polynomial_coefficients':[str(c) for c in coefficients]}
    return "certified", out


def verify_candidate(variables,f,candidate,mode,effort=1):
    """Return charts and their dual complex only for a certified candidate."""
    return _verify(variables, f, candidate, mode, effort)[1]


def _pack_rational(expr):
    # Hex strings avoid the decimal digit cap in JSON, including for very long
    # coefficients appearing in unbounded certificate descriptions.
    return [hex(int(expr.numerator)), hex(int(expr.denominator))]


@lru_cache(maxsize=512)
def _term_rows(expr, variables, domain):
    """Sparse monomials and their descending power-basis coefficients."""
    from .affine_kernel import pullbacks, sparse_terms
    if domain == QQ and np.native_ring(variables):
        return tuple(sorted((e, (c,)) for e,c in np.terms(variables, expr).items()))
    from .field_polynomial import _parameter, lift, modulus

    theta = None
    if domain != QQ:
        theta = _parameter(domain)
        value = lift(expr, variables, domain, theta)
    else:
        value = expr.as_expr() if hasattr(expr, "as_expr") else chart_expressions(variables).sympify(expr)
    if theta is None or not value.has(theta):
        rows = tuple((e,(c,)) for e,c in reversed(sparse_terms(variables, value)))
    else:
        extended = (*variables, theta)
        reduced, = pullbacks(extended, extended, extended, (value,), modulus(domain, theta))
        groups = {}
        for powers, coefficient in sparse_terms(extended, reduced):
            groups.setdefault(powers[:-1], {})[powers[-1]] = coefficient
        rows = tuple((e, tuple(c.get(j, 0) for j in range(max(c), -1, -1)))
                     for e,c in sorted(groups.items()))
    return rows


def _pack_poly(poly, variables, domain=QQ):
    return [[e, [_pack_rational(c) for c in coefficients]]
            for e,coefficients in _term_rows(poly, tuple(variables), domain)]


def _unpack_poly(terms, variables, primitive=None):
    groups = {}
    for exponents, coefficients in terms:
        for j, q in enumerate(reversed(coefficients)):
            powers = tuple(exponents) + ((j,) if primitive is not None else ())
            groups[powers] = coordinate_fraction(integer_text(int(q[0],16)) + "/" + integer_text(int(q[1],16)))
    return np.from_terms((*variables, primitive) if primitive is not None else variables, groups)


def _wire_candidate(variables, f, candidate, mode, effort):
    field = None
    if isinstance(candidate.domain, NativeField):
        field = [[_pack_rational(v) for v in candidate.domain.coefficients], candidate.domain.index]
    elif candidate.domain != QQ:
        poly = candidate.domain.ext.minpoly
        from .field_coefficient import generator_expression, root_position
        primitive = generator_expression(candidate.domain.ext)
        root_index = root_position(poly, primitive)
        field = [[_pack_rational(v) for v in poly.all_coeffs()], root_index]
    return {
        "variables": [v.name for v in variables], "polynomial": _pack_poly(f,variables),
        "source": candidate.source, "center": [_pack_poly(g,variables) for g in candidate.center],
        "projective": [u.name for u in candidate.projective],
        "boundary": [[_pack_poly(g,(*variables,*candidate.projective),candidate.domain) for g in ideal]
                     for ideal in candidate.boundary],
        "field": field, "mode": mode, "effort": effort,
    }


def _read_candidate(info):
    from .resolution_search import ResolutionCandidate
    api = runtime_expressions()
    variables = tuple(map(api.Symbol, info["variables"]))
    projective = tuple(map(api.Symbol, info["projective"]))
    domain, primitive = QQ, None
    if info["field"] is not None:
        coefficients, index = info["field"]
        # Private worker payloads carry the already-established minimal
        # polynomial and marked embedding of their originating domain.
        from .candidate_field import indexed_field
        domain, primitive = indexed_field(tuple(coordinate_fraction(integer_text(int(q[0],16)) + "/" + integer_text(int(q[1],16))) for q in coefficients), index)
    center = tuple(_unpack_poly(g,variables) for g in info["center"])
    boundary = tuple(tuple(_unpack_poly(g,(*variables,*projective),primitive) for g in ideal)
                     for ideal in info["boundary"])
    candidate = ResolutionCandidate(center,projective,boundary,domain,primitive,info["source"])
    return variables, _unpack_poly(info["polynomial"],variables), candidate, info["mode"], info["effort"]


def _kernel_line(terms):
    """A packed rational polynomial in the kernel's text form."""
    return ";".join(f"{int(c[0][0],16)}/{int(c[0][1],16)} {' '.join(map(str,e))}" for e,c in terms)


def _decide(info):
    """Verify one candidate received over the pipe.

    Most candidates are rejected, and over Q the kernel says so from the
    packed polynomials alone: no expression is built for them.
    """
    if info.get("operation") == "reduced_graph_record":
        return _reduced_graph_record(info["variables"], info["mode"], info["centre"], info["proof"])
    proof = None
    if info["field"] is None:
        try:
            lines = [_kernel_line(info["polynomial"]), *map(_kernel_line,info["center"])]
            lines += [_kernel_line(g) for ideal in info["boundary"] for g in ideal]
        except (TypeError,ValueError,IndexError):
            lines = None
        if lines is not None:
            proof = kernel.verify_lines(len(info["variables"]),lines,len(info["center"]),
                                        [len(ideal) for ideal in info["boundary"]])
        if proof is False:
            return "rejected", None
    variables,f,candidate,mode,effort = _read_candidate(info)
    if proof:
        proof = assemble_charts(variables,candidate.center,proof)
    return _verify(variables,f,candidate,mode,effort,proof)


def _serve():
    # Private pipe protocol for *already generated* candidates: one JSON line
    # in, one out. Only sparse polynomial data crosses the boundary. No
    # pickling or expression evaluation.
    if hasattr(sys,"set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    # A worker must not outlive the search that started it, even when that
    # process is killed in the middle of a candidate.
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            sleep(1)
        os._exit(0)

    threading.Thread(target=watch,daemon=True).start()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        report = None
        try:
            info = json.loads(line)
            with replay_scope(**info.get("replay", {"compute": None})) as replay:
                try:
                    outcome = _decide(info)
                finally:
                    report = replay.report()
        except Exception:
            # A backend limitation is inconclusive. The scheduler can continue with
            # other certificates and will revisit this one with more resources.
            outcome = ("undecided", None)
        sys.stdout.write(json.dumps([*outcome, report],separators=(",",":"))+"\n")
        sys.stdout.flush()


class _Worker:
    """A verification process that stays up between candidates.

    Starting Python and importing SymPy costs more than most verifications,
    so a worker is killed only when a candidate exceeds its allowance.
    """
    def __init__(self):
        self.answered = 0
        self.process = subprocess.Popen([sys.executable,"-m","engine.general_resolution"],
            cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def alive(self):
        return self.process.poll() is None

    def ask(self, payload, seconds):
        """The answer line; '' when the worker failed and None when time ran out."""
        import queue
        reply = queue.SimpleQueue()

        def exchange():
            try:
                self.process.stdin.write(payload+"\n")
                self.process.stdin.flush()
                reply.put(self.process.stdout.readline())
            except (OSError,ValueError):
                reply.put("")

        threading.Thread(target=exchange,daemon=True).start()
        deadline = monotonic()+seconds
        while True:
            remaining = deadline-monotonic()
            if remaining <= 0:
                return None
            try:
                # Intervals avoid the platform's maximum representable timeout
                # under the unbounded dovetail schedule.
                return reply.get(timeout=min(remaining,3600))
            except queue.Empty:
                pass

    def close(self):
        if self.alive():
            self.process.kill()
        for stream in (self.process.stdin,self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.process.wait()


class _Workers:
    """The idle workers: the one last used and a spare already started, so
    that a candidate following a timeout does not wait for an interpreter."""
    RECYCLE = 4000  # candidates answered before a worker is replaced

    def __init__(self):
        self.lock = threading.Lock()
        self.idle = []

    def warm(self):
        with self.lock:
            if not self.idle:
                try:
                    self.idle.append(_Worker())
                except OSError:
                    pass

    def take(self, *, spare=True):
        with self.lock:
            worker = self.idle.pop() if self.idle else None  # the one used last
        if worker is not None and not worker.alive():
            worker.close()
            worker = None
        if worker is None:
            worker = _Worker()
        if spare:
            self.warm()
        return worker

    def give(self, worker):
        with self.lock:
            keep = len(self.idle) < 2 and worker.alive() and worker.answered < self.RECYCLE
            if keep:
                self.idle.append(worker)
        if not keep:
            worker.close()

    def close(self):
        with self.lock:
            idle, self.idle = self.idle, []
        for worker in idle:
            worker.close()


_workers = _Workers()
atexit.register(_workers.close)


def _attempt(variables, f, candidate, mode, effort, seconds, *, spare=True):
    """An independently killable proof attempt. Does not depend on process signals
    in the calling thread. The worker shares the outer job's cancellation group.
    """
    start = monotonic()
    try:
        info = _wire_candidate(variables,f,candidate,mode,effort)
    except Exception:
        return "undecided", None
    return _run_attempt(info, seconds-(monotonic()-start), spare=spare)


def _reduced_graph_record(names, mode, centre, proof, *, exceptional=None):
    from .resolution_search import _candidate
    variables = tuple(map(np.Symbol,names))
    candidate = _candidate(variables, kernel.exprs(variables, centre), "smooth reduced singular locus")
    charts, count = assemble_charts(variables, candidate.center, kernel.graph_proof_rows(proof))
    if exceptional is None:
        pieces = _pieces(candidate, charts)
    else:
        divisor = kernel.exprs(variables, exceptional)
        pieces = [[divisor] for chart in charts]
    state, record = _verified_record(variables, candidate, mode, 1, charts, pieces,
                                    count, graph_bundle=exceptional is not None)
    if state == "certified":
        record["certificate"]["candidates_examined"] = 1
    return state, record


def _run_attempt(info, seconds, *, spare=True):
    if seconds <= 0:
        return "undecided", None
    start = monotonic()
    replay = current_replay_scope()
    if replay is not None:
        info["replay"] = replay.settings()
    payload = json.dumps(info)
    try:
        worker = _workers.take() if spare else _workers.take(spare=False)
    except OSError:
        return "undecided", None
    output = None
    try:
        output = worker.ask(payload, seconds-(monotonic()-start))
    finally:
        if not output:
            worker.close()  # out of time, interrupted, or the process failed
    if not output:
        return "undecided", None
    worker.answered += 1
    _workers.give(worker)
    try:
        state, result, report = json.loads(output)
        if replay is not None and report:
            replay.merge(report)
    except (TypeError,ValueError):
        return "undecided", None
    if state in ("rejected","undecided","certified"):
        return state, result
    return "undecided", None


def _isolated_graph_prefix(variables, active, candidates, seconds):
    # The native worker verifies the actual scheduled candidates and can be
    # killed during a saturation. Traced requests stay on the replay-aware path.
    worker = Path(__file__).resolve().parents[1] / "native" / "build" / "graph_verify"
    if not worker.is_file():
        return None
    frames = []
    for candidate in candidates:
        rows = kernel._lines(variables, (active, *candidate.center))
        for ideal in candidate.boundary:
            rows.extend(kernel._lines((*variables, *candidate.projective), ideal))
        header = " ".join(map(str, ("verify",len(variables),len(candidate.center),
                                    len(candidate.boundary),*(len(ideal) for ideal in candidate.boundary))))
        frames.append("\n".join([header,*rows])+"\n")
    try:
        answer = subprocess.run([str(worker)], input="\f".join(frames), text=True,
                                capture_output=True, timeout=seconds)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if answer.returncode or not answer.stdout:
        return None
    proof = kernel.graph_proof_rows(answer.stdout.splitlines())
    if proof is None:
        return None
    candidate = candidates[-1]
    charts, count = assemble_charts(variables,candidate.center,proof)
    return _verified_record(variables,candidate,"embedded",1,charts,_pieces(candidate,charts),
                            count,coordinate_line=True)


def search(poly,names,mode,*,exhaustive=False,seconds=4,transverse_centre=None,prewarm_spare=True):
    from .affine_kernel import _sparse_expression, _sparse_terms, frame_function, rational_factor_rows
    from .resolution_search import priority_candidates, scheduled_candidates
    api = runtime_expressions()
    variables=tuple(api.Symbol(s) for s in names)
    f=expression(poly,variables)
    if transverse_centre is None:
        _workers.warm()  # the interpreter starts while the candidates are prepared
    # Over the germ, repeated global factors that miss zero are units. Preserve
    # them in ordinary charts, or record the unit if this fallback removes it.
    scalar, factors = rational_factor_rows(variables, f)
    active_factors, unit_factors = [], [api.Rational(scalar)]
    from .identity_atlas import origin_indices
    active_indices = set(origin_indices((dict(_sparse_terms(row)) for row, _, _ in factors), len(variables)))
    for i, (row, power, _) in enumerate(factors):
        factor = _sparse_expression(variables, row)
        if i not in active_indices:
            unit_factors.append(factor**power)
        elif power > 1:
            from .polynomial import NonReducedError
            raise NonReducedError('The germ has a repeated component through the origin. Resolution of the underlying reduced hypersurface requires its square-free equation.')
        else:
            active_factors.append(factor)
    active = api.Mul(*active_factors)
    # All removed factors are nonzero at the origin. The factorization
    # gives f/active directly, including the original scalar and powers.
    unit = frame_function(variables, api.Mul(*unit_factors))
    deadline=monotonic()+seconds
    schedule=(scheduled_candidates(variables,active) if exhaustive else
              ((c,1.25) for c in priority_candidates(variables,active)))
    attempts=0
    rejected_priority = set()
    prefix = []
    if transverse_centre is not None:
        expected = (((api.S.One,), "identity"), (variables, "origin"),
                    (tuple(variables[j] for j in transverse_centre), "singular coordinate subspace"))

    for candidate,allowance in schedule:
        seconds_left=allowance if exhaustive else min(allowance,deadline-monotonic())
        if seconds_left<=0:
            break
        # Bound memory in unbounded searches: priority candidates are finite;
        # encoded certificates may each be reached again, including after timeouts.
        priority_key = None if candidate.source.startswith("integer code ") else (candidate.center,candidate.boundary)
        if priority_key is not None and priority_key in rejected_priority:
            continue
        attempts+=1
        report(f"Checking blow-up candidate {attempts} at the proposed centre")
        if transverse_centre is None:
            state,result = _attempt(variables,active,candidate,mode,max(1,int(allowance)),seconds_left,
                                    spare=prewarm_spare)
            # A racing search already has its first verifier. A timeout or
            # undecided proof enables the replacement for subsequent attempts.
            if state == "undecided":
                prewarm_spare = True
        else:
            if attempts > len(expected) or (candidate.center, candidate.source) != expected[attempts-1]:
                return None
            prefix.append(candidate)
            if len(prefix) < len(expected):
                continue
            outcome = _isolated_graph_prefix(variables, active, prefix, seconds_left)
            if outcome is None:
                return None
            state,result = outcome
        if state=="rejected" and priority_key is not None:
            rejected_priority.add(priority_key)
        if state=="certified" and result is not None:
            result['certificate']['candidates_examined']=attempts
            if unit!=1:
                result['certificate']['unit_removed_at_origin']=_text(unit,QQ)
                result['certificate']['resolved_equation']=_text(active,QQ)
                if unit.free_symbols:
                    invert = _text(unit,QQ)
                    base = {'open':f'D({invert})','invert':[invert],
                            'description':'The removed factor is a unit on this neighborhood of the origin.'}
                    result['base_domain']=base
                    for chart in result['charts']:
                        # In graph charts the projection keeps every original coordinate.
                        chart['base_domain']=dict(base)
                        chart['invert']=list(base['invert'])
            return result
    return None


if __name__=='__main__':
    _serve()
