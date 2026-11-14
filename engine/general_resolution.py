"""Verify an embedded resolution presented as a single projective ideal blowup.

A finite composition of blowups over affine space is itself the blowup of a
finitely generated ideal (Stacks 080B). This permits a slower complete search
in addition to the geometric algorithms. Every candidate is checked.
"""
import atexit
from itertools import combinations, product
from math import prod
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import monotonic, sleep

import sympy as sp
from sympy import QQ

from . import kernel
from .blowup_ideal import (assemble_charts, blowup_charts, empty_locus, ideal_basis, ideal_subset,
                           radical_subset, resolution_charts, verify_snc)
from .groebner import CHECK


def expression(poly, variables):
    return sp.Add(*(sp.Rational(c.numerator,c.denominator)*prod(x**d for x,d in zip(variables,e))
                    for e,c in poly.items()))


def _text(value, domain, generator="field_generator"):
    if domain!=QQ:
        value=sp.sympify(value).xreplace({domain.ext.as_expr():sp.Symbol(generator)})
    return str(value).replace('**','^')


def _cover(variables, ambient, pivot, pieces, domain):
    """Equality of V(pivot) and union of the supplied closed divisor pieces."""
    for piece in pieces:
        if not radical_subset(variables,(pivot,),(*ambient,*piece),domain=domain):
            return False
    # Products generate the ideal defining the union set-theoretically.
    union=tuple(prod(fs) for fs in product(*pieces))
    return radical_subset(variables,union,(*ambient,pivot),domain=domain)


def _on_overlap(source, target, ideal):
    """Pull target equations back to source on U_target/U_source != 0."""
    scale=source.ratios[target.pivot_index]
    substitution={v:source.ratios[i]/scale for i,v in enumerate(target.ratios) if i!=target.pivot_index}
    return tuple(sp.cancel(f.xreplace(substitution)).as_numer_denom()[0] for f in ideal),scale


def _glue(charts, ideals, original, domain):
    """Identify irreducible affine pieces using their nonempty intersections on overlaps."""
    keys=sorted(ideals)
    parent={key:key for key in keys}
    def root(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]]
            key=parent[key]
        return key
    for a,b in combinations(keys,2):
        if a[0]==b[0] or root(a)==root(b):
            continue
        source,target=charts[a[0]],charts[b[0]]
        moved,scale=_on_overlap(source,target,ideals[b])
        inv=sp.Dummy('inverse_overlap')
        if not empty_locus((*source.variables,inv),
                (*ideals[a],*moved,1-inv*scale),domain=domain):
            parent[root(b)]=root(a)
    groups={}
    for key in keys:
        groups.setdefault(root(key),[]).append(key)
    # Only strata whose closure meets the special fiber survive localization
    # at the origin. Each irreducible localization is still irreducible.
    return [{ci:ideals[(ci,i)] for ci,i in keys_in_group}
            for keys_in_group in groups.values()
            if any(not empty_locus(charts[ci].variables,(*ideals[(ci,i)],*original),domain=domain)
                   for ci,i in keys_in_group)]


def _component_atlas(charts, original, pieces, mode, domain, effort, generator):
    from .geometric_components import geometric_components
    n=len(original)
    boundary_count=len(pieces[0])
    strata={}
    vertices=[]
    cells=[]
    chart_divisors=[[] for _ in charts]
    for k in range(1,min(boundary_count,n-(mode=='intrinsic'))+1):
        for subset in combinations(range(boundary_count),k):
            locals_={}
            for ci,chart in enumerate(charts):
                equations=list(chart.strict if mode=='intrinsic' else chart.ambient)
                for j in subset:
                    equations.extend(pieces[ci][j])
                if empty_locus(chart.variables,(*equations,*original),domain=domain):
                    # A chart that misses the special fiber can still be an
                    # overlap connecting two visible parts of one stratum.
                    if empty_locus(chart.variables,equations,domain=domain):
                        continue
                components=geometric_components(chart.variables,tuple(equations),
                          n-k-(mode=='intrinsic'),domain=domain,effort=effort)
                if components is None:
                    return None
                locals_.update({(ci,index):ideal for index,ideal in enumerate(components)})
            groups=_glue(charts,locals_,original,domain)
            strata[subset]=[]
            for group in groups:
                if k==1:
                    ident=f'E{len(vertices)+1}'
                    vertices.append({'id':ident,'dimension':n-k-(mode=='intrinsic')})
                    for ci,ideal in group.items():
                        display=[_text(e,domain,generator) for e in ideal]
                        chart_divisors[ci].append({'id':ident,'equation':', '.join(display),'ideal':display})
                else:
                    ci,child=next(iter(group.items()))
                    def parent_id(parent_subset):
                        matches=[ident for ident,g in strata[parent_subset] if ci in g and
                                 ideal_subset(charts[ci].variables,g[ci],child,domain=domain)]
                        if len(matches)!=1:
                            raise ArithmeticError('Stratum incidence is ambiguous.')
                        return matches[0]
                    ident=f'C{len(cells)+1}'
                    cells.append({'id':ident,'dimension':k-1,
                                  'vertices':[parent_id((j,)) for j in subset],
                                  'faces':[parent_id(s) for s in combinations(subset,k-1)]})
                strata[subset].append((ident,group))
    return {'status':'complete','convention':('Ambient exceptional divisor' if mode=='embedded'
            else 'Reduced exceptional boundary on the smooth strict transform')+
            ' in the algebraic germ over Qbar. Cells are geometric intersection components meeting the origin fiber.',
            'vertices':vertices,'cells':cells},chart_divisors


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
            monomials = sp.Poly(g,*variables,*candidate.projective,domain=domain).monoms()
            if len({sum(e[len(variables):]) for e in monomials}) > 1:
                return False
    # The blowup must preserve the smooth locus of the original hypersurface.
    if not radical_subset(variables,(f,*(f.diff(x) for x in variables)),center,domain=QQ):
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
    # Over Q the kernel decides these tests. With ATLAS_GROEBNER=check the
    # SymPy route below is taken as well and the two must agree.
    proved = proof
    if proved is None and domain==QQ:
        proved = resolution_charts(variables,f,center,u,candidate.boundary)
    if proved:
        charts,snc_checks = proved
        proved = charts,_pieces(candidate,charts),snc_checks
    if proved is None or CHECK:
        reference = _charts(variables,f,candidate)
        if proved is not None and proved != reference:
            raise AssertionError("The native verification differs from SymPy's.")
        proved = reference
    if not proved:
        return 'rejected', None
    charts,pieces,snc_checks = proved
    used = {v.name for chart in charts for v in chart.variables} | {v.name for v in candidate.projective}
    generator = "field_generator"
    while generator in used:
        generator += "_"
    result=_component_atlas(charts,variables,pieces,mode,domain,effort,generator)
    if result is None:
        return 'undecided', None
    dual,divisors=result
    atlas=[]
    for i,chart in enumerate(charts):
        if empty_locus(chart.variables,(*chart.ambient,*variables),domain=domain):
            continue
        # On a graph chart the ambient equations are part of the coordinate
        # ring. Present only additional strict equations when possible.
        strict=[e for e in chart.strict if not ideal_subset(chart.variables,(e,),chart.ambient,domain=domain)]
        atlas.append({'id':f'G{i+1}','coordinates':[str(x) for x in chart.variables],
            'ambient_ideal':[_text(e,domain,generator) for e in chart.ambient],
            'strict_transform':[_text(e,domain,generator) for e in strict],
            'map':{str(x):str(x) for x in variables},'exceptional':divisors[i],
            'pivot_generator':_text(chart.pivot,domain,generator),'projective_coordinates':list(map(str,chart.ratios))})
    out={'status':'resolved','engine':'verified-ideal-blowup','charts':atlas,'dual_complex':dual,
            'steps':[{'id':'B','type':'blowup of an ideal','generators':[_text(e,domain,generator) for e in center]}],
            'summary':f'Certified a projective resolution in {len(atlas)} affine charts over the origin.',
            'certificate':{'method':'saturation of the projective graph, geometric Jacobian tests and overlap incidence',
                'verified_affine_strata':snc_checks,'center_source':candidate.source,
                'projective_coordinates':[str(x) for x in candidate.projective],
                'boundary_ideals':[[_text(e,domain,generator) for e in piece] for piece in candidate.boundary]}}
    if domain!=QQ:
        primitive=candidate.primitive
        out['coefficient_field']={'name':f'Q({generator})','generator':generator,
            'exact_root':str(primitive),'description':f'{generator} = {primitive}',
            'minimal_polynomial_coefficients':[str(c) for c in domain.ext.minpoly.all_coeffs()]}
    return "certified", out


def verify_candidate(variables,f,candidate,mode,effort=1):
    """Return charts and their dual complex only for a certified candidate."""
    return _verify(variables, f, candidate, mode, effort)[1]


def _pack_rational(expr):
    # Hex strings avoid the decimal digit cap in JSON, including for very long
    # coefficients appearing in unbounded certificate descriptions.
    q = QQ.convert(expr)
    return [hex(int(q.numerator)), hex(int(q.denominator))]


def _unpack_rational(q):
    return sp.Rational(int(q[0],16), int(q[1],16))


def _pack_poly(poly, variables, domain=QQ):
    terms = sp.Poly(poly, *variables, domain=domain).rep.to_dict()
    return [[e, [_pack_rational(c) for c in ([v] if domain == QQ else v.to_list())]]
            for e,v in terms.items()]


def _unpack_poly(terms, variables, primitive=None):
    result = []
    for exponents, coefficients in terms:
        # Domain ANP coefficients are in descending powers of its primitive element.
        scalar = sp.S.Zero
        for q in coefficients:
            scalar = (scalar*primitive if primitive is not None else 0) + _unpack_rational(q)
        result.append(scalar * prod(v**e for v,e in zip(variables,exponents)))
    return sp.Add(*result)


def _wire_candidate(variables, f, candidate, mode, effort):
    field = None
    if candidate.domain != QQ:
        poly = candidate.domain.ext.minpoly
        primitive = candidate.domain.ext.as_expr()
        if isinstance(primitive, sp.CRootOf):
            root_index = primitive.index
        else:
            root_index = next(i for i in range(poly.degree())
                              if poly.same_root(primitive, sp.CRootOf(poly.as_expr(),i)))
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
    variables = tuple(map(sp.Symbol, info["variables"]))
    projective = tuple(map(sp.Symbol, info["projective"]))
    domain, primitive = QQ, None
    if info["field"] is not None:
        coefficients, index = info["field"]
        t = sp.Dummy("coefficient_root")
        poly = sp.Poly.from_list([_unpack_rational(q) for q in coefficients],t)
        primitive = sp.CRootOf(poly.as_expr(),index)
        domain = QQ.algebraic_field(primitive)
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
        if proof is False and not CHECK:
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
        try:
            outcome = _decide(json.loads(line))
        except Exception:
            # A backend limitation is inconclusive. The scheduler can continue with
            # other certificates and will revisit this one with more resources.
            outcome = ("undecided", None)
        sys.stdout.write(json.dumps(outcome,separators=(",",":"))+"\n")
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

    def take(self):
        with self.lock:
            worker = self.idle.pop() if self.idle else None  # the one used last
        if worker is not None and not worker.alive():
            worker.close()
            worker = None
        if worker is None:
            worker = _Worker()
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


def _attempt(variables, f, candidate, mode, effort, seconds):
    """An independently killable proof attempt. Does not depend on process signals
    in the calling thread. The worker shares the outer job's cancellation group.
    """
    start = monotonic()
    try:
        payload = json.dumps(_wire_candidate(variables,f,candidate,mode,effort))
    except Exception:
        return "undecided", None
    try:
        worker = _workers.take()
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
        state, result = json.loads(output)
    except (TypeError,ValueError):
        return "undecided", None
    if state in ("rejected","undecided","certified"):
        return state, result
    return "undecided", None


def search(poly,names,mode,*,exhaustive=False,seconds=4):
    from .resolution_search import priority_candidates, scheduled_candidates
    variables=tuple(sp.Symbol(s) for s in names)
    f=expression(poly,variables)
    _workers.warm()  # the interpreter starts while the candidates are prepared
    # Over the germ, repeated global factors that miss zero are units. Preserve
    # them in ordinary charts, or record the unit if this fallback removes it.
    factors=sp.factor_list(f,*variables)[1]
    origin={s:0 for s in variables}
    for factor,power in factors:
        if power>1 and factor.subs(origin)==0:
            from .polynomial import NonReducedError
            raise NonReducedError('The germ has a repeated component through the origin. Resolution of the underlying reduced hypersurface requires its square-free equation.')
    active=prod(factor for factor,power in factors if factor.subs(origin)==0)
    unit=sp.cancel(f/active)
    deadline=monotonic()+seconds
    schedule=(scheduled_candidates(variables,active) if exhaustive else
              ((c,1.25) for c in priority_candidates(variables,active)))
    attempts=0
    rejected_priority = set()
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
        state,result=_attempt(variables,active,candidate,mode,max(1,int(allowance)),seconds_left)
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
