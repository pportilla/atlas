"""A regular Newton modification of a hypersurface germ in arbitrary input dimension."""
from functools import cache
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
    from .fan_centres import singular_ray_images
    return singular_ray_images(format_poly(poly, variables), variables, (fan["rays"][i] for i in fan["boundary_rays"]))


def check_fan_budget(fan, terms):
    """Raise LimitError for a fan whose strata or chart equations exceed the budgets of the atlas."""
    n = len(fan['rays'][0])
    rays, cones = fan['rays'], fan['cones']
    if len(cones)*(2**n-1)>750_000 or len(rays)*terms>500_000:
        raise LimitError('The Newton fan exceeds the stratum or valuation budget.')
    if terms*len(cones)>400_000:
        raise LimitError('The Newton atlas exceeds the chart expansion budget.')


def fan_faces(poly, fan):
    """The orders of the rays and the face of common minimizers on every stratum of the fan.

    Every cone maps to a Newton vertex. Its strata have initial forms given
    by the common minimizers of their rays. A positive sum of the rays picks
    exactly this common face, including strata over the origin with some
    original coordinate rays as generators.
    """
    n = len(fan['rays'][0])
    exponents=sorted(poly)
    rays, cones=fan['rays'],fan['cones']
    check_fan_budget(fan, len(exponents))
    valuations = _valuations(rays, exponents)
    rows = _newton("face_orders", n, (len(rays), len(cones), len(exponents)), *valuations, *cones)
    orders, exceptional_orders, minima, face_masks = [], [], [], {}
    for row in rows[:len(rays)]:
        order, exceptional_order, *mask = map(integer_from_decimal, row.split())
        orders.append(order)
        exceptional_orders.append(exceptional_order)
        minima.append(frozenset(mask))
    for row in rows[len(rays):]:
        count, *fields = map(int, row.split())
        face_masks[tuple(fields[:count])] = frozenset(fields[count:])
    return exponents, valuations, orders, exceptional_orders, minima, face_masks


def _assemble(poly, variables, mode, fan, matrix, compute, deadline=None, *, original, defer_outputs=False,
              diagnostics=None, before_repair=None, repair_points=True):
    n=len(variables)
    rays, cones=fan['rays'],fan['cones']
    exponents, valuations, orders, exceptional_orders, minima, face_masks = fan_faces(poly, fan)
    faces={}
    families={}
    degenerate=False
    for mask in sorted(set(face_masks.values()),key=lambda m:(len(m),sorted(m))):
        initial={exponents[i]:poly[exponents[i]] for i in sorted(mask)}
        proof=certify_face(initial)
        if proof['status']=='undetermined':
            if diagnostics is not None:
                diagnostics.setdefault("face", "A Newton face could not be certified.")
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
        if not repair_points:
            return None
        if not fan['isomorphism_off_origin']:
            if diagnostics is not None:
                diagnostics.setdefault("relative_repair", "A degenerate face needs repair on a fan with centres extending beyond the origin.")
            return None  # The point-repair chart producer uses exceptional rays over the origin.
        from .newton_repair import strata as repair_strata

        # Repair uses the fan and adapted equation. A failed trial has no
        # atlas to return, so test its singular orbits before expanding charts.
        repair = repair_strata(poly, variables, fan, deadline=deadline, count_strata=False,
                               diagnostics=diagnostics, before_points=before_repair)
        if repair is not None and "alternative" in repair:
            return repair["alternative"]
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

    equations, batch = transform_atlas(exponents, rays, cones, exceptional_orders, valuations, compute,
                                       equations=(coords, [poly[e] for e in exponents]))
    directions = list(dict.fromkeys(tuple(proof["edge"]["direction"]) for proof in faces.values()
                      if not degenerate and mode == "intrinsic" and proof["face_dimension"] == 1))
    direction_orders = dict(zip(directions, zip(*_valuations(rays, directions)))) if directions else {}
    charts=[]
    for index, (cone, strict) in enumerate(zip(cones, equations, strict=True), 1):
        rr=[rays[i] for i in cone]
        minimum=[exceptional_orders[i] for i in cone]
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
        _repaired_boundary(result, original)
    result['summary'] = (f"Regular Newton modification with {len(parents)} charts, then {blowups} blowups over "
                         f"{points} point{'s' if points!=1 else ''} where a face is degenerate.")
    proof = result['certificate']
    proof['method'] = 'regular Newton fan, exact torus Jacobian tests, and blowups certified by ideal calculations'
    proof['description'] = ('The initial Newton modification is an isomorphism off the origin and every maximal cone '
        'has lattice determinant one. The repair steps list the centres blown up over the degenerate faces. '
        'The final charts verify normal crossings on the owned fibre over the origin.')
    return result


def _repaired_boundary(record, polynomial):
    from .atlas import boundary_complex, from_record, to_json
    charts = from_record(record, polynomial, native=True)
    record['atlas'] = to_json(charts)
    record['dual_complex'] = boundary_complex(charts, len(record['variables']), record['mode'] == 'intrinsic')


def _decline_reason(error):
    text = str(error).strip() or type(error).__name__
    return text if text.endswith(".") else text + "."


# From this number of variables on the search for adapted coordinates is part
# of the Newton search. With fewer the blowups from the origin are tried
# first: there they take under a second on the germs concerned and give the
# smaller model, as one blowup for a cusp cylinder in sheared coordinates.
# The dispatcher asks for the adapted frames after them.
ADAPTED_FRAMES_DIMENSION = 4


def resolve_newton(poly, variables, mode, compute=None, seconds=120, *, defer_outputs=False, verified=None, integrals=False,
                   construct=True, diagnostics=None, frames=None):
    """A regular Newton modification, completed by blowups where a face is
    degenerate. Those blowups may use `seconds` in all, over the coordinate
    choices tried; past that the degenerate choices give no answer.

    `frames` selects the constructions: "linear" for the fans in linear
    frames with their point repairs, "adapted" for the triangular changes of
    coordinates alone, "all" for both, with the adapted frames after the
    linear fans that need no blowups and before the point repairs. By
    default the adapted frames are included from ADAPTED_FRAMES_DIMENSION
    variables on.

    Returns None when the Newton method does not apply to this germ, for the
    caller to hand it over to another method. Each construction is tried on
    its own: one that declines leaves its reason in `diagnostics` and the
    next one runs. Nothing a declined construction built is returned or
    added to `verified`. A LimitError is raised at the end only if a budget
    was reached and no construction applied.
    """
    if frames is None:
        frames = "all" if len(variables) >= ADAPTED_FRAMES_DIMENSION else "linear"
    if frames not in ("all", "linear", "adapted"):
        raise ValueError("The Newton frames are all, linear or adapted.")
    import subprocess
    from time import monotonic
    from .centres import GiveUp
    from .polynomial import InputError, NonReducedError
    deadline = monotonic()+seconds
    last_limit = None
    kept = len(verified) if verified is not None else 0

    def fan_of(terms):
        # A fan that is not finished by the deadline of the search is of no use to it.
        return build_fan(terms, exhaustive=exhaustive_faces(), seconds=deadline-monotonic())

    def attempt(name, construct_record):
        """One construction: its record, or None after recording why it does not apply."""
        nonlocal last_limit
        try:
            return construct_record()
        except NonReducedError:
            raise  # A property of the input, which no method resolves in this mode.
        except LimitError as error:
            last_limit = error
        except (GiveUp, InputError, ArithmeticError, subprocess.TimeoutExpired) as error:
            if diagnostics is not None:
                diagnostics.setdefault(name, _decline_reason(error))
        # A verified bundle of this construction belongs to a record that is not returned.
        if verified is not None:
            del verified[kept:]
        return None

    def adapted():
        # A triangular polynomial substitution can take a boundary image off
        # the smooth locus or straighten a factor of a degenerate face while
        # the coordinate subspaces of the singular locus stay in place.
        from .newton_frames import resolve_adapted
        return resolve_adapted(poly, variables, mode, compute, deadline, defer_outputs=defer_outputs,
                               diagnostics=diagnostics)

    def declined():
        if last_limit is not None:
            raise last_limit
        return None

    if frames == "adapted":
        return attempt("adapted_frame", adapted) or declined()
    nonconvenient = attempt("support", lambda: not all(_axial_data(poly)[1]))
    if nonconvenient is None:
        return declined()
    identity = tuple(tuple(int(i == j) for j in range(len(variables))) for i in range(len(variables)))
    if mode == 'embedded' and nonconvenient:
        # A subdivision on the boundary is permissible if it preserves the
        # smooth locus. Each exceptional ray maps onto the coordinate subspace
        # cut out by its positive entries. For a reduced polynomial, membership
        # in the square of each corresponding ideal certifies that inclusion.
        fan = attempt("fan", lambda: fan_of(poly))

        def hyperplanes():
            from .newton_trials import hyperplane_fan
            return hyperplane_fan(poly, variables, fan, verified, integrals)

        def monomial_trials():
            from .newton_trials import resolve_trials
            return resolve_trials(poly, variables, verified=verified, integrals=integrals, construct=construct)

        def boundary_fan(centres):
            result = _assemble(poly, variables, mode, fan, identity, compute, deadline, original=poly,
                               defer_outputs=defer_outputs, diagnostics=diagnostics)
            if result is not None:
                proof = result['certificate']
                proof['boundary_image_coordinate_ideals'] = centres
                proof['description'] = ('The nontrivial coordinate images of the exceptional divisors '
                    'lie in the singular locus. Every maximal cone has lattice determinant one. '
                    'The strict transform is smooth and transverse to the toric strata over the origin.')
                proof['exceptional_orbit_faces'] = proof.pop('compact_faces')
            return result

        if fan is not None:
            trial = attempt("hyperplane_fan", hyperplanes) if 4 <= len(variables) <= 8 else None
            if trial is not None:
                return trial
            if fan['boundary_rays']:
                found = attempt("boundary_images", lambda: (_boundary_centres(poly, variables, fan),))
                if found is not None:
                    centres, = found
                    trial = (attempt("monomial_trials", monomial_trials) if centres is None
                             else attempt("boundary_fan", lambda: boundary_fan(centres)))
                    if trial is not None:
                        return trial
    # Failure of embedded transversality can still leave a smooth strict
    # transform. Test its own boundary before constructing point blowups.
    # All adapted-coordinate candidates share the same original equation.
    @cache
    def intrinsic_trial():
        from .newton_trials import resolve_trials
        return attempt("intrinsic_fan", lambda: resolve_trials(poly, variables, mode=mode, verified=verified,
                                                               integrals=integrals))

    fans = {}

    def linear_frame(index, transformed, matrix, repair_points):
        if index not in fans:
            fans[index] = fan_of(transformed)
        fan = fans[index]
        if not fan['isomorphism_off_origin']:
            return None
        return _assemble(transformed,variables,mode,fan,matrix,compute,deadline,original=poly,
                         defer_outputs=defer_outputs, diagnostics=diagnostics,
                         before_repair=intrinsic_trial if mode == 'intrinsic' else None,
                         repair_points=repair_points)

    choices = attempt("linear_frames", lambda: list(coordinate_choices(poly,len(variables)))) or ()
    # With adapted frames in the search, a fan that needs no blowups comes
    # before the point repairs of a degenerate one: the repairs can use the
    # whole time of the search before they decline.
    passes = (False, None, True) if frames == "all" else (True,)
    for repair_points in passes:
        if repair_points is None:
            result = attempt("adapted_frame", adapted)
            if result is not None:
                return result
            continue
        for index, (transformed,matrix,_) in enumerate(choices):
            result = attempt("linear_frame", lambda: linear_frame(index, transformed, matrix, repair_points))
            if result is not None:
                return result
    if mode == 'intrinsic' and nonconvenient:
        trial = intrinsic_trial()
        if trial is not None:
            return trial
    if len(variables) == 3:
        def translated():
            from .quadratic_fan import resolve_translated
            return resolve_translated(poly, variables, mode, compute, ramification=True)
        result = attempt("translated_fan", translated)
        if result is not None:
            return result
    return declined()
