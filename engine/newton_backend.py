"""A regular Newton modification of a hypersurface germ in arbitrary input dimension."""
from itertools import combinations
from math import comb, prod

from .chart_batch import transform_atlas
from .newton_faces import certify_face, edge_polynomial, exhaustive_faces
from .newton_fan import build_fan
from .polynomial import format_poly, integer_from_decimal, integer_text, LimitError


def pairing(vector, exponent):
    return sum(a*b for a, b in zip(vector, exponent))


def monomial(coordinates, powers):
    return '*'.join(name if power == 1 else f'{name}^{integer_text(power)}'
                    for name, power in zip(coordinates, powers) if power) or '1'


def shear(poly, donor, axis, scalar=1):
    """Pull back by x_donor = v_donor + scalar*v_axis."""
    if sum(e[donor]+1 for e in poly)>100_000:
        raise LimitError('The trial coordinate substitution exceeds its expansion budget.')
    from .native_library import affine_request
    from .polynomial import checked_native, sparse_row
    n = len(next(iter(poly)))
    names = [f"v{i}" for i in range(n)]
    def native():
        images = names.copy()
        images[donor] = f"({names[donor]}+({scalar})*{names[axis]})"
        ring = " ".join(names)
        row, = affine_request(["pullbacks", ring, ring, *images, "0", format_poly(poly, names)])
        return sparse_row(row)
    return checked_native(native, lambda: _shear_reference(poly, donor, axis, scalar))


def _valuations(rays, exponents):
    """Pair every Newton ray with the complete support in one integer matrix product."""
    from .native_library import affine_request, _kernel, Unavailable
    from .polynomial import checked_native
    def native():
        if not hasattr(_kernel(), "atlas_valuations_abi"):
            raise Unavailable("The lattice pairing kernel is unavailable.")
        rows = affine_request(["lattice_valuations", str(len(rays[0])), str(len(rays)),
                               *(" ".join(map(integer_text, r)) for r in (*rays, *exponents))])
        return [list(map(integer_from_decimal, row.split())) for row in rows]
    return checked_native(native, lambda: [[pairing(v, e) for e in exponents] for v in rays])


def _shear_reference(poly, donor, axis, scalar):
    result = {}
    for exponent, coefficient in poly.items():
        for k in range(exponent[donor]+1):
            new = list(exponent)
            new[donor] -= k
            new[axis] += k
            key = tuple(new)
            result[key] = result.get(key, 0) + coefficient * comb(exponent[donor], k)*scalar**k
    return {e:c for e,c in result.items() if c}


def coordinate_choices(poly, n):
    """Find convenient coordinates so every new ray maps to the origin.

    Keep the smallest axial terms first. A rational shear can expose a mixed
    term without changing the germ. Its inverse is again a linear substitution.
    """
    identity = tuple(tuple(int(i==j) for j in range(n)) for i in range(n))
    candidates = [(poly, identity, [])]
    for axis in range(n):
        next_candidates = []
        for current, matrix, changes in candidates:
            if any(e[axis] and sum(e)==e[axis] for e in current):
                next_candidates.append((current, matrix, changes))
                continue
            for donor in range(n):
                if donor == axis:
                    continue
                try:
                    changed = shear(current,donor,axis)
                except LimitError:
                    continue
                if len(changed)>4096:
                    continue
                if not any(e[axis] and sum(e)==e[axis] for e in changed):
                    continue
                new_matrix = tuple(tuple(v+row[donor] if j==axis else v
                                         for j,v in enumerate(row)) for row in matrix)
                next_candidates.append((changed,new_matrix,[*changes,(donor,axis)]))
        def score(trial):
            # The product of axial intercepts is the normalized volume of
            # their simplex. Preserve earlier improvements when processing an
            # already-convenient axis, which otherwise reshuffles the trials.
            intercepts = {}
            for exponent in trial[0]:
                support = [i for i,power in enumerate(exponent) if power]
                if len(support)==1:
                    j, = support
                    intercepts[j] = min(intercepts.get(j,exponent[j]),exponent[j])
            return -len(intercepts), prod(intercepts.values()), len(trial[0])
        candidates = sorted(next_candidates,key=score)[:8]
    yield from candidates


def subsets(cone):
    for size in range(1,len(cone)+1):
        yield from combinations(sorted(cone),size)


def _label_roots(index, edge):
    family = f'alpha{index}'
    polynomial = format_poly(edge_polynomial(edge),['T'])
    return family, {
        'degree':edge['degree'], 'polynomial':polynomial, 'direction':edge['direction'],
        'definition':f'{family}_j is root j of {polynomial} = 0, ordered by real then imaginary part, starting at 1.'
    }


def _assemble(poly, variables, mode, fan, matrix, compute, deadline=None, *, original, defer_outputs=False):
    n=len(variables)
    exponents=sorted(poly)
    rays, cones=fan['rays'],fan['cones']
    if len(cones)*(2**n-1)>750_000 or len(rays)*len(exponents)>500_000:
        raise LimitError('The Newton fan exceeds the stratum or valuation budget.')
    if len(exponents)*len(cones)>400_000:
        raise LimitError('The Newton atlas exceeds the chart expansion budget.')
    # Every cone maps to a Newton vertex. Its strata have initial forms given
    # by the common minimizers of their rays. A positive sum of the rays picks
    # exactly this common face, including strata over the origin with some
    # original coordinate rays as generators.
    valuations = _valuations(rays, exponents)
    orders = list(map(min,valuations))
    minima = [frozenset(i for i,h in enumerate(values) if h==order)
              for values,order in zip(valuations,orders)]
    face_masks = {}
    intersection_cones = set()
    for cone in cones:
        for subset in subsets(cone):
            if subset in face_masks or not any(i>=n for i in subset):
                continue
            mask = frozenset.intersection(*(minima[i] for i in subset))
            if not mask:
                raise RuntimeError('Newton chart lacks a common minimizing monomial.')
            face_masks[subset]=mask
            if min(subset)>=n:
                intersection_cones.add(subset)
    faces={}
    families={}
    degenerate=False
    for mask in sorted(set(face_masks.values()),key=lambda m:(len(m),sorted(m))):
        initial={exponents[i]:poly[exponents[i]] for i in sorted(mask)}
        proof=certify_face(initial)
        if proof['status']=='undetermined':
            return None
        # On a degenerate face the strict transform fails normal crossings at
        # the singular points of the cut it makes on the orbit. The
        # modification is kept and those points are blown up afterwards.
        degenerate = degenerate or proof['status']!='nondegenerate' 
        proof['exponents']=[list(e) for e in initial]
        if proof['face_dimension']==1:
            family,definition=_label_roots(len(families)+1,proof['edge'])
            families[family]=definition
            proof['root_family']=family
        faces[mask]=proof
    coords=[f'u{i+1}' for i in range(n)]
    # Each intersection of exceptional components has an open dense torus
    # stratum. Transversality excludes components contained in its boundary.
    # In the saturated lattice spanned by the face differences, affine
    # Lefschetz gives connectedness for a nondegenerate full-dimensional
    # torus hypersurface of Newton dimension >=2 (Danilov–Khovanskii,
    # Proposition 3.9 and Remark 3.10). Smoothness gives irreducibility,
    # and taking the product with the complementary torus preserves it.
    # A segment instead splits by its distinct roots in the primitive character.
    strata={}
    vertices=[]
    cells=[]
    # A degenerate fan is an intermediate modification. Its boundary will
    # be computed once the local blowups have completed the atlas.
    boundary_cones = () if degenerate else sorted(intersection_cones, key=lambda s: (len(s), s))
    for subset in boundary_cones:
        proof=faces[face_masks[subset]]
        dim=proof['face_dimension']
        split=mode=='intrinsic' and dim==1
        count=(0 if dim==0 else proof['edge']['degree'] if split else 1) if mode=='intrinsic' else 1
        strata[subset]=[]
        for k in range(1,count+1):
            if len(subset)==1:
                ray=subset[0]
                ident=f'E{ray-n+1}' + (f'.{k}' if split and count>1 else '')
                record={'id':ident,'ray':rays[ray],'ray_index':ray,'ambient_order':orders[ray]}
                vertices.append(record)
            else:
                ident='C'+'-'.join(str(i) for i in subset)+f'.{k}'
                def restriction(smaller):
                    matches=strata[smaller]
                    if len(matches)==1:
                        return matches[0]
                    if face_masks[smaller]!=face_masks[subset]:
                        raise RuntimeError('Incompatible nested Newton edge labels.')
                    return matches[k-1]
                record={'id':ident,'dimension':len(subset)-1,
                        'vertices':[restriction((i,)) for i in subset],
                        'faces':[restriction(s) for s in combinations(subset,len(subset)-1)]}
                cells.append(record)
            if split:
                record['root']=proof['root_family']+f'_{k}'
            strata[subset].append(ident)

    powers_by_chart, batch = transform_atlas(exponents, rays, cones, orders, valuations, compute)
    coefficients = [poly[e] for e in exponents]
    directions = list(dict.fromkeys(tuple(proof["edge"]["direction"]) for proof in faces.values()
                      if not degenerate and mode == "intrinsic" and proof["face_dimension"] == 1))
    direction_orders = dict(zip(directions, zip(*_valuations(rays, directions)))) if directions else {}
    charts=[]
    for index,(cone,powers) in enumerate(zip(cones,powers_by_chart),1):
        rr=[rays[i] for i in cone]
        minimum=[orders[i] for i in cone]
        torus_map=[monomial(coords,[ray[j] for ray in rr]) for j in range(n)]
        maps={name:'+'.join(word if c==1 else f'{c}*{word}' for c,word in zip(row,torus_map) if c)
              for name,row in zip(variables,matrix)}
        strict=dict(zip(powers,coefficients))
        divisors=[]
        for position,i in enumerate(cone):
            if i<n:
                continue
            if degenerate:
                divisors.append({"id": f"E{i-n+1}", "equation": coords[position]})
                continue
            proof=faces[minima[i]]
            ids=strata[(i,)]
            if mode=='embedded' or proof['face_dimension']!=1:
                divisors.extend({'id':ident,'equation':coords[position]} for ident in ids)
                continue
            values = direction_orders[tuple(proof['edge']['direction'])]
            powers = [values[r] for r in cone]
            positive=monomial(coords,[max(0,v) for v in powers])
            negative=monomial(coords,[max(0,-v) for v in powers])
            for k,ident in enumerate(ids,1):
                root=proof['root_family']+f'_{k}'
                ideal=[coords[position], f'{positive}-{root}*{negative}']
                divisors.append({'id':ident,'ideal':ideal,'equation':', '.join(ideal)})
        charts.append({'id':f'N{index}','coordinates':coords,'map':maps,
                       'exceptional':divisors,'strict_transform':format_poly(strict,coords),
                       'pullback_factor':monomial(coords,minimum),'ray_indices':cone})
    dual={'status':'complete','convention':('Ambient exceptional divisor' if mode=='embedded' else
          'Reduced exceptional divisor on the smooth strict transform')+' over the origin over Qbar.',
          'vertices':vertices,'cells':cells}
    result = {'status':'resolved','engine':'newton-polyhedron','charts':charts,'dual_complex':dual,
            'summary':f'Regular Newton resolution with {len(charts)} charts and {len(vertices)} geometric exceptional components.',
            'steps':[{'id':'Newton','type':'regular toric modification'}],
            'root_families':families,'fan':fan,
            'certificate':{'method':'regular Newton fan and exact torus Jacobian tests',
                'chart_batch':batch,'linear_map_to_input':matrix,'face_coordinates':variables,
                'adapted_polynomial':format_poly(poly,variables),'compact_faces':list(faces.values()),
                'description':'The map is an isomorphism off the origin. All compact faces are nondegenerate and every maximal cone has lattice determinant one. The strict transform is smooth and transverse to the toric strata in a neighborhood of the fiber over the origin.'}}
    return _blow_up_failures(result,poly,variables,mode,fan,deadline,original,defer_outputs) if degenerate else result


def _blow_up_failures(result, poly, variables, mode, fan, deadline, original, defer_outputs):
    """Complete a Newton modification with degenerate faces by blowing up the
    points of the fiber over the origin where normal crossings fail."""
    from .newton_repair import patch_record, records, strata
    from .groebner import CHECK
    answer = strata(poly,variables,fan,deadline=deadline,count_strata=CHECK)
    if answer is None or not answer['repairs']:
        return None
    repairs = answer['repairs']
    parents = {chart['id']:(i,chart) for i,chart in enumerate(result['charts'])}
    blowups = sum(len(r['blowups']) for r in repairs)
    points = sum(r['points'] for r in repairs)
    for r in repairs:
        parent_index, parent = parents[r['chart']]
        parent.setdefault('blown_up',[]).append({'id':r['id'],'ideal':r['ideal']})
        for k, chart in enumerate(r['charts'], 1):
            result['charts'].append(patch_record(chart, parent, f"{r['id']}.{k}", parent_index))
        result['steps'].append({'id':r['id'],'type':'blowups where a Newton face is degenerate','chart':r['chart'],
                                'points':r['points'],'ideal':r['ideal'],'centers':r['blowups']})
    # The fan before the blowups describes a different boundary. Its labels
    # are kept in the charts; incidences come from the completed atlas.
    result['dual_complex'] = {'status':'unavailable', 'convention':result['dual_complex']['convention'],
                              'vertices':[], 'cells':[]}
    result.update(variables=list(variables), mode=mode)
    if not defer_outputs:
        from .atlas import boundary_complex, from_record, to_json
        from .native_library import _MODE
        charts = from_record(result, original, native=_MODE == "native")
        result['atlas'] = to_json(charts)
        result['dual_complex'] = boundary_complex(charts, len(variables), mode == 'intrinsic')
    result['summary'] = (f"Regular Newton modification with {len(parents)} charts, then {blowups} blowups over "
                         f"{points} point{'s' if points!=1 else ''} where a face is degenerate.")
    proof = result['certificate']
    proof['method'] = 'regular Newton fan, exact torus Jacobian tests, and blowups certified by ideal calculations'
    proof['description'] = ('The map is an isomorphism off the origin and every maximal cone has lattice determinant one. '
        'Some compact face is degenerate: on the fiber over the origin the total transform has normal crossings except at the listed points, '
        'which are blown up until it has. Normal crossings are verified in every final chart.')
    if CHECK:
        proof['numerical_data'], proof['strata'] = records(answer)
    return result


def resolve_newton(poly, variables, mode, compute=None, seconds=120, *, defer_outputs=False):
    """A regular Newton modification, completed by blowups where a face is
    degenerate. Those blowups may use `seconds` in all, over the coordinate
    choices tried; past that the degenerate choices give no answer."""
    from time import monotonic
    deadline = monotonic()+seconds
    last_limit = None
    for transformed,matrix,_ in coordinate_choices(poly,len(variables)):
        try:
            fan=build_fan(transformed, exhaustive=exhaustive_faces())
            if not fan['isomorphism_off_origin']:
                continue
            result=_assemble(transformed,variables,mode,fan,matrix,compute,deadline,original=poly,defer_outputs=defer_outputs)
        except LimitError as error:
            last_limit = error
            continue
        if result is not None:
            return result
    if last_limit is not None:
        raise last_limit
    return None
