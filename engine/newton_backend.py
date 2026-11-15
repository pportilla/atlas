"""A regular Newton modification of a hypersurface germ in arbitrary input dimension."""
from .kernel_json import kernel_json

from .chart_batch import transform_atlas
from .binomial_kernel import _newton, signed_powers
from .newton_faces import certify_face, edge_polynomial, exhaustive_faces
from .newton_fan import build_fan
from .polynomial import format_poly, integer_from_decimal, integer_text, LimitError


def pairing(vector, exponent):
    return _valuations((vector,), (exponent,))[0][0]


def monomial(coordinates, powers):
    return '*'.join(name if power == 1 else f'{name}^{integer_text(power)}'
                    for name, power in zip(coordinates, powers) if power) or '1'


def shear(poly, donor, axis, scalar=1):
    """Pull back by x_donor = v_donor + scalar*v_axis."""
    _, _, expansion = _axial_data(poly)
    return _shear(poly, donor, axis, scalar, expansion)


def _shear(poly, donor, axis, scalar, expansion):
    if expansion[donor] > 100_000:
        raise LimitError('The trial coordinate substitution exceeds its expansion budget.')
    from .native_library import affine_request
    from .polynomial import sparse_row
    n = len(next(iter(poly)))
    names = [f"v{i}" for i in range(n)]
    images = names.copy()
    images[donor] = f"({names[donor]}+({scalar})*{names[axis]})"
    ring = " ".join(names)
    row, = affine_request(["pullbacks", ring, ring, *images, "0", format_poly(poly, names)])
    return sparse_row(row)


def _valuations(rays, exponents):
    """Pair every Newton ray with the complete support in one integer matrix product."""
    from .native_library import affine_request, _kernel, Unavailable
    if not hasattr(_kernel(), "atlas_valuations_abi"):
        raise Unavailable("The lattice pairing kernel is unavailable.")
    rows = affine_request(["lattice_valuations", str(len(rays[0])), str(len(rays)),
                           *(" ".join(map(integer_text, r)) for r in (*rays, *exponents))])
    return [list(map(integer_from_decimal, row.split())) for row in rows]


def _axial_data(poly):
    n = len(next(iter(poly)))
    header, expansion = _newton("axial", n, (len(poly),), *poly)
    volume, *intercepts = map(integer_from_decimal, header.split())
    return volume, intercepts, tuple(map(integer_from_decimal, expansion.split()))


def coordinate_choices(poly, n):
    """Find convenient coordinates so every new ray maps to the origin.

    Keep the smallest axial terms first. A rational shear can expose a mixed
    term without changing the germ. Its inverse is again a linear substitution.
    """
    identity = tuple(tuple(int(i==j) for j in range(n)) for i in range(n))
    candidates = [(poly, identity, [], _axial_data(poly))]
    for axis in range(n):
        next_candidates = []
        for candidate in candidates:
            current, matrix, changes, (_, intercepts, expansion) = candidate
            if intercepts[axis]:
                next_candidates.append(candidate)
                continue
            for donor in range(n):
                if donor == axis:
                    continue
                try:
                    changed = _shear(current, donor, axis, 1, expansion)
                except LimitError:
                    continue
                if len(changed) > 4096 or changed == current:
                    continue
                changed_data = _axial_data(changed)
                if not changed_data[1][axis]:
                    continue
                rows = _newton("shear", n, (donor, axis), *matrix)
                new_matrix = tuple(tuple(map(integer_from_decimal, row.split())) for row in rows)
                next_candidates.append((changed, new_matrix, [*changes, (donor, axis)], changed_data))
        def score(trial):
            # The product of axial intercepts is the normalized volume of
            # their simplex. Preserve earlier improvements when processing an
            # already-convenient axis, which otherwise reshuffles the trials.
            volume, intercepts, _ = trial[3]
            return intercepts.count(0), volume, len(trial[0])
        candidates = sorted(next_candidates,key=score)[:8]
    for transformed, matrix, changes, _ in candidates:
        yield transformed, matrix, changes


def _label_roots(index, edge):
    family = f'alpha{index}'
    polynomial = format_poly(edge_polynomial(edge),['T'])
    return family, {
        'degree':edge['degree'], 'polynomial':polynomial, 'direction':edge['direction'],
        'definition':f'{family}_j is root j of {polynomial} = 0, ordered by real then imaginary part, starting at 1.'
    }


def _boundary_centres(poly, variables, fan):
    """Certify the images of exceptional boundary rays by the native singular-locus test."""
    from .native_library import affine_request
    targets = tuple(dict.fromkeys(tuple(j for j, value in enumerate(fan["rays"][i]) if value)
                                  for i in fan["boundary_rays"]))
    singular = affine_request(["singular_subspaces", " ".join(variables), format_poly(poly, variables),
                               *(" ".join(map(str, axes)) for axes in targets)])
    if len(singular) != len(targets):
        return None
    return [[variables[i] for i in axes] for axes in targets]


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
    rows = _newton("faces", n, (len(rays), len(cones), len(exponents)), *valuations, *cones)
    orders, minima, face_masks = [], [], {}
    for row in rows[:len(rays)]:
        order, *mask = map(integer_from_decimal, row.split())
        orders.append(order)
        minima.append(frozenset(mask))
    for row in rows[len(rays):]:
        count, *fields = map(int, row.split())
        face_masks[tuple(fields[:count])] = frozenset(fields[count:])
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
    repair = None
    if degenerate:
        if not fan['isomorphism_off_origin']:
            return None  # The point-repair chart producer uses exceptional rays over the origin.
        from .newton_repair import strata as repair_strata

        # Repair uses the fan and adapted equation. A failed trial has no
        # atlas to return, so test its singular orbits before expanding charts.
        repair = repair_strata(poly, variables, fan, deadline=deadline, count_strata=False)
        if repair is None or not repair['repairs']:
            return None
    coords=[f'u{i+1}' for i in range(n)]
    # The repaired cover has its own divisor intersections after the blowups.
    vertices, cells = [], []
    by_ray = {}
    if not degenerate:
        face_ids = {mask: i for i, mask in enumerate(faces)}
        proofs = [(proof['face_dimension'], integer_text(proof['edge']['degree']) if 'edge' in proof else None,
                   proof.get('root_family')) for proof in faces.values()]
        orbits = [(subset, face_ids[mask]) for subset, mask in face_masks.items()]
        layout = kernel_json('atlas_torus_sections', 'newton', dimension=n, intrinsic=mode=='intrinsic',
                             proofs=proofs, orbits=orbits)
        vertices, cells = layout['vertices'], layout['cells']
        for vertex in vertices:
            i = vertex['ray_index']
            vertex.update(ray=rays[i], ambient_order=orders[i])
            by_ray.setdefault(i, []).append(vertex)

    equations, batch = transform_atlas(exponents, rays, cones, orders, valuations, compute,
                                       equations=(coords, [poly[e] for e in exponents]))
    directions = list(dict.fromkeys(tuple(proof["edge"]["direction"]) for proof in faces.values()
                      if not degenerate and mode == "intrinsic" and proof["face_dimension"] == 1))
    direction_orders = dict(zip(directions, zip(*_valuations(rays, directions)))) if directions else {}
    charts=[]
    for index, (cone, strict) in enumerate(zip(cones, equations, strict=True), 1):
        rr=[rays[i] for i in cone]
        minimum=[orders[i] for i in cone]
        torus_map=[monomial(coords,[ray[j] for ray in rr]) for j in range(n)]
        maps={name:'+'.join(word if c==1 else f'{c}*{word}' for c,word in zip(row,torus_map) if c)
              for name,row in zip(variables,matrix)}
        divisors=[]
        for position,i in enumerate(cone):
            if i<n:
                continue
            if degenerate:
                divisors.append({"id": f"E{i-n+1}", "equation": coords[position]})
                continue
            proof=faces[minima[i]]
            components=by_ray.get(i, ())
            if mode=='embedded' or proof['face_dimension']!=1:
                divisors.extend({'id':v['id'],'equation':coords[position]} for v in components)
                continue
            values = direction_orders[tuple(proof['edge']['direction'])]
            powers = [values[r] for r in cone]
            plus, minus = signed_powers(tuple(powers))
            positive, negative = monomial(coords, plus), monomial(coords, minus)
            for vertex in components:
                ident, root = vertex['id'], vertex['root']
                ideal=[coords[position], f'{positive}-{root}*{negative}']
                divisors.append({'id':ident,'ideal':ideal,'equation':', '.join(ideal)})
        charts.append({'id':f'N{index}','coordinates':coords,'map':maps,
                       'exceptional':divisors,'strict_transform':strict,
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
    return _blow_up_failures(result,variables,mode,original,defer_outputs,repair) if degenerate else result


def _blow_up_failures(result, variables, mode, original, defer_outputs, repair):
    """Attach the verified local blowups to their parent Newton charts."""
    from .newton_repair import patch_record
    repairs = repair['repairs']
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
        charts = from_record(result, original, native=True)
        result['atlas'] = to_json(charts)
        result['dual_complex'] = boundary_complex(charts, len(variables), mode == 'intrinsic')
    result['summary'] = (f"Regular Newton modification with {len(parents)} charts, then {blowups} blowups over "
                         f"{points} point{'s' if points!=1 else ''} where a face is degenerate.")
    proof = result['certificate']
    proof['method'] = 'regular Newton fan, exact torus Jacobian tests, and blowups certified by ideal calculations'
    proof['description'] = ('The initial Newton modification is an isomorphism off the origin and every maximal cone '
        'has lattice determinant one. The repair steps list the centres blown up over the degenerate faces. '
        'The final charts verify normal crossings on the owned fibre over the origin.')
    return result


def resolve_newton(poly, variables, mode, compute=None, seconds=120, *, defer_outputs=False, verified=None, integrals=False,
                   construct=True):
    """A regular Newton modification, completed by blowups where a face is
    degenerate. Those blowups may use `seconds` in all, over the coordinate
    choices tried; past that the degenerate choices give no answer."""
    from time import monotonic
    deadline = monotonic()+seconds
    last_limit = None
    if mode == 'embedded' and not all(_axial_data(poly)[1]):
        # A subdivision on the boundary is permissible if it preserves the
        # smooth locus. Each exceptional ray maps onto the coordinate subspace
        # cut out by its positive entries. For a reduced polynomial, membership
        # in the square of each corresponding ideal certifies that inclusion.
        try:
            fan = build_fan(poly, exhaustive=exhaustive_faces())
            if fan['boundary_rays']:
                centres = _boundary_centres(poly, variables, fan)
                if centres is None:
                    from .newton_trials import resolve_trials
                    trial = resolve_trials(poly, variables, verified=verified, integrals=integrals,
                                           construct=construct)
                    if trial is not None:
                        return trial
                else:
                    matrix = tuple(tuple(int(i == j) for j in range(len(variables))) for i in range(len(variables)))
                    result = _assemble(poly, variables, mode, fan, matrix, compute, deadline, original=poly,
                                       defer_outputs=defer_outputs)
                    if result is not None:
                        proof = result['certificate']
                        proof['boundary_image_coordinate_ideals'] = centres
                        proof['description'] = ('The nontrivial coordinate images of the exceptional divisors '
                            'lie in the singular locus. Every maximal cone has lattice determinant one. '
                            'The strict transform is smooth and transverse to the toric strata over the origin.')
                        proof['exceptional_orbit_faces'] = proof.pop('compact_faces')
                        return result
        except LimitError as error:
            last_limit = error
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
