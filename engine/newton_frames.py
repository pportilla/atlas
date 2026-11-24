"""Coordinates adapted to the singular locus before a Newton modification.

A Newton fan is certified in given coordinates only if every exceptional ray
has its image in the singular locus and every face on a stratum over the
origin is nondegenerate. Two triangular changes of coordinates repair a
frame in which this fails for a reason that depends on the coordinates:

  - A boundary ray maps onto a coordinate subspace L of the hypersurface that
    is not in its singular locus. The substitution x_j = X_j + (prod x_i)^k,
    over the free coordinates of L, takes L off the hypersurface and fixes
    every coordinate subspace of the singular locus.
  - A degenerate face polynomial has a factor c*x_j + m. The substitution
    x_j = X_j - m/c makes that factor a coordinate.

Each step is an automorphism of affine space that fixes the origin and has
Jacobian determinant one. The native kernel proposes the steps, computes the
pullbacks and certifies the faces. The search over sequences of steps is
depth first, by the number of obstructions that a step leaves. The fan of
the adapted polynomial is verified in the adapted coordinates, and every
chart map is composed with the steps.
"""
from time import monotonic

from .kernel_json import kernel_json
from .native_library import affine_request
from .polynomial import LimitError, encode_sparse_row, format_poly, integer_text, parse_expression, sparse_row
from .strata import InvalidChart

METHOD = "coordinates adapted to the singular locus"
# Each step removes one obstruction of the current fan. The bound stops a
# sequence that keeps exposing new ones.
MAXIMUM_STEPS = 12
# Seconds for the search of a frame, which certifies the faces of each fan it tries.
FRAME_SECONDS = 8
# Frames tried in all, and substitutions proposed from one frame.
MAXIMUM_FRAMES = 64
MAXIMUM_CHOICES = 8


def _choices(operation, poly, n, **operands):
    return kernel_json("atlas_resolution_candidate", operation, width=n, row=encode_sparse_row(poly), **operands)


def _smooth_images(poly, names, fan):
    """The boundary images outside the singular locus, as lists of free coordinates."""
    from .fan_centres import smooth_ray_images
    if not fan["boundary_rays"]:
        return []
    return smooth_ray_images(format_poly(poly, names), names, (fan["rays"][i] for i in fan["boundary_rays"]))


def _degenerate_faces(poly, fan, limit=None):
    """The faces on the strata of the fan that are not certified nondegenerate."""
    from .newton_backend import fan_faces
    from .newton_faces import certify_face
    exponents, *_, face_masks = fan_faces(poly, fan)
    faces = []
    for mask in sorted(set(face_masks.values()), key=lambda m: (len(m), sorted(m))):
        if len(mask) < 3:
            continue  # A binomial has no repeated factor.
        if limit is not None and monotonic() > limit:
            raise LimitError("The search for adapted coordinates exceeded its time budget.")
        initial = {exponents[i]: poly[exponents[i]] for i in sorted(mask)}
        if certify_face(initial)["status"] != "nondegenerate":
            faces.append(initial)
    return faces


class _Frame:
    """A polynomial with its fan, and the obstructions of that fan when they are needed."""

    def __init__(self, poly, names, limit=None):
        from .newton_fan import TRIAL_SECONDS, build_fan
        from .newton_faces import exhaustive_faces
        if limit is not None and monotonic() > limit:
            raise LimitError("The search for adapted coordinates exceeded its time budget.")
        self.poly, self.names, self.limit = poly, names, limit
        # The fan is bounded by the memory that its charts need and by the
        # time that the search has left.
        seconds = TRIAL_SECONDS if limit is None else max(1.0, min(TRIAL_SECONDS, limit - monotonic()))
        self.fan = build_fan(poly, exhaustive=exhaustive_faces(), seconds=seconds, chart_terms=len(poly))
        self.smooth = _smooth_images(poly, names, self.fan)
        self._degenerate = None

    @property
    def degenerate(self):
        if self._degenerate is None:
            self._degenerate = _degenerate_faces(self.poly, self.fan, self.limit)
        return self._degenerate

    def boundary_choices(self):
        # The first rays of the fan are the normals of the Newton polyhedron.
        normals = self.fan["rays"][:self.fan.get("newton_ray_count", len(self.fan["rays"]))]
        return _choices("boundary_translations", self.poly, len(self.names), subspaces=self.smooth,
                        normals=[[integer_text(value) for value in ray] for ray in normals])

    def face_choices(self):
        found, seen = [], set()
        for initial in self.degenerate:
            for choice in _choices("face_translations", self.poly, len(self.names), face=encode_sparse_row(initial)):
                key = (choice["axis"], choice["shift"])
                if key not in seen:
                    seen.add(key)
                    found.append(choice)
            if len(found) >= MAXIMUM_CHOICES:
                break
        return found[:MAXIMUM_CHOICES]



def _order(frame, children):
    """Sort the frames after one step, the most promising first.

    Certifying the faces of a fan is the expensive part of the search, so it
    is done only where it decides the order. A substitution with equality
    on a normal can make a face degenerate: it is compared with its frame.
    A frame without boundary images to treat has its faces certified in any
    case. The other frames are taken to keep the degenerate faces they had.
    """
    known = None

    def before():
        nonlocal known
        if known is None:
            known = len(frame.degenerate)
        return known

    ranked = []
    for index, (choice, child, path) in enumerate(children):
        try:
            if frame.smooth:
                faces = len(child.degenerate) if choice.get("tie") else before() if any(
                    other.get("tie") for other, _, _ in children) else 0
            else:
                faces = len(child.degenerate) if not child.smooth else before() - 1
        except LimitError:
            if frame.limit is not None and monotonic() > frame.limit:
                raise
            continue  # The faces of this frame exceed the budgets of an atlas.
        ranked.append(((faces, len(child.smooth), len(child.fan["cones"]), index), child, path))
    return [(child, path) for _, child, path in sorted(ranked, key=lambda item: item[0])]


def adapted_frame(poly, names, seconds=FRAME_SECONDS):
    """The steps, the adapted polynomial and its fan, or None when no sequence of steps is found.

    In the frame that is returned every boundary ray of the fan has its image
    in the singular locus and every face on a stratum is nondegenerate. The
    search goes depth first. From each frame it tries first the step that
    leaves the fewest degenerate faces: a substitution can bring a singular
    point of a face into its torus, and a later step then has to straighten
    the factor it exposed. Raises LimitError when the first fan exceeds its
    budgets or the search its time.
    """
    limit = None if seconds is None else monotonic() + seconds
    root = _Frame(poly, names, limit)
    seen = {frozenset(poly.items())}
    pending = [(root, [])]
    tried = 0
    while pending:
        frame, steps = pending.pop()
        if not frame.smooth:
            try:
                if not frame.degenerate:
                    return (steps, frame.poly, frame.fan) if steps else None
            except LimitError:
                if frame is root or limit is not None and monotonic() > limit:
                    raise
                continue  # The faces of this frame exceed the budgets of an atlas.
        if len(steps) == MAXIMUM_STEPS:
            continue
        children = []
        for choice in (frame.boundary_choices() if frame.smooth else frame.face_choices()):
            adapted = sparse_row(choice["adapted"])
            key = frozenset(adapted.items())
            if key in seen or tried == MAXIMUM_FRAMES:
                continue
            seen.add(key)
            tried += 1
            try:
                child = _Frame(adapted, names, limit)
            except LimitError:
                if limit is not None and monotonic() > limit:
                    raise
                continue  # The fan of this frame exceeds the budgets of an atlas.
            step = {"coordinate": names[choice["axis"]], "offset": format_poly(sparse_row(choice["shift"]), names)}
            children.append((choice, child, [*steps, step]))
        # The best child is on top of the stack, and the others wait for a dead end.
        pending.extend(reversed(_order(frame, children)))
    return None


def translation_steps(translation, names):
    """Validate the triangular steps of a saved coordinate change."""
    steps = translation.get("steps") if isinstance(translation, dict) else None
    if not isinstance(steps, list) or not steps or len(steps) > 64:
        raise InvalidChart("An adapted coordinate change needs its list of steps.")
    for step in steps:
        if (not isinstance(step, dict) or step.get("coordinate") not in names
                or not isinstance(step.get("offset"), str)):
            raise InvalidChart("A coordinate step needs its coordinate and polynomial offset.")
    return steps


def adapted_polynomial(polynomial, names, steps):
    """The pullback of the input polynomial by the steps, checked to be triangular and to fix the origin."""
    rows = [{"axis": names.index(step["coordinate"]),
             "shift": parse_expression(step["offset"], names, as_row=True)} for step in steps]
    answer = _choices("translation_chain", polynomial, len(names), steps=rows)
    if answer is None:
        raise InvalidChart("A coordinate step is not a triangular substitution that fixes the origin.")
    return sparse_row(answer["adapted"])


def compose_map(chart, names, steps, *, inverse=False):
    """The chart projection in the input coordinates, or back in the adapted ones."""
    projection = chart["map"]
    if set(projection) != set(names):
        raise InvalidChart("The translated fan chart needs one image per input coordinate.")
    coords = " ".join(chart["coordinates"])
    header = " ".join(names)
    projection = dict(projection)
    # The adapted polynomial is the pullback by the first step, then the
    # second: the last step is the first applied to a point of the chart.
    for step in (steps if inverse else reversed(steps)):
        axis = step["coordinate"]
        image = f"{axis}{'-' if inverse else '+'}({step['offset']})"
        row, = affine_request(["pullbacks", coords, header, *(projection[name] for name in names), "0", image])
        projection[axis] = format_poly(sparse_row(row), chart["coordinates"])
    return projection


def _certificate(proof, names, steps):
    proof = dict(proof, fan_coordinate_frame="adapted")
    if "boundary_image_coordinate_ideals" in proof:
        centres = proof.pop("boundary_image_coordinate_ideals")
        proof["adapted_boundary_image_coordinate_ideals"] = centres
        images = compose_map({"coordinates": names, "map": dict(zip(names, names))}, names, steps, inverse=True)
        proof["boundary_image_ideals"] = [[images[coordinate] for coordinate in centre] for centre in centres]
    return proof


def resolve_adapted(poly, names, mode, compute=None, deadline=None, *, defer_outputs=False, diagnostics=None):
    """A Newton modification in adapted coordinates, with chart maps to the input coordinates."""
    from .identity_atlas import reduced_at_origin
    from .newton_backend import _assemble, _boundary_centres
    if not reduced_at_origin(poly, names):
        # A repeated factor that a step turns into a coordinate would pass
        # for a nondegenerate face. The divisor with multiplicities is
        # resolved through its reduced support, which comes back here.
        if diagnostics is not None:
            diagnostics.setdefault("adapted_frame", "Adapted coordinates are searched for a reduced germ.")
        return None
    try:
        frame = adapted_frame(poly, names)
    except LimitError as error:
        if diagnostics is not None:
            diagnostics.setdefault("adapted_frame", f"The search for adapted coordinates stopped: {error}")
        return None
    if frame is None:
        if diagnostics is not None:
            diagnostics.setdefault("adapted_frame", "No triangular change of coordinates puts the exceptional "
                                   "images in the singular locus with nondegenerate faces.")
        return None
    steps, adapted, fan = frame
    n = len(names)
    result = None
    if mode == "embedded" or fan["isomorphism_off_origin"]:
        centres = _boundary_centres(adapted, names, fan) if fan["boundary_rays"] else []
        if centres is None:
            return None
        identity = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
        result = _assemble(adapted, names, mode, fan, identity, compute, deadline, original=adapted,
                           defer_outputs=defer_outputs, diagnostics=diagnostics, repair_points=False)
        if result is not None and not fan["isomorphism_off_origin"]:
            proof = result["certificate"]
            proof["boundary_image_coordinate_ideals"] = centres
            proof["description"] = ("The nontrivial coordinate images of the exceptional divisors "
                                    "lie in the singular locus. Every maximal cone has lattice determinant one. "
                                    "The strict transform is smooth and transverse to the toric strata over the origin.")
            proof["exceptional_orbit_faces"] = proof.pop("compact_faces")
    if result is None and mode == "intrinsic":
        from .newton_trials import _try
        # The fan of the frame is the only candidate here, whatever its size.
        result = _try(adapted, names, adapted, mode=mode, fan=fan, construct=False, defer=False)
    if result is None:
        return None
    for chart in result["charts"]:
        chart["map"] = compose_map(chart, names, steps)
    translation = {"method": METHOD, "steps": steps}
    proof = result["certificate"] = _certificate(result["certificate"], names, steps)
    proof.update(coordinate_translation=translation, adapted_polynomial=format_poly(adapted, names))
    proof["description"] = ("A sequence of triangular polynomial substitutions, each fixing the origin with "
                            "Jacobian determinant one, gives coordinates in which the exceptional images lie "
                            "in the singular locus and the faces over the origin are nondegenerate. "
                            + proof["description"])
    return result


def adapted_outputs(record, polynomial, translation, verify, options):
    """Verify the fan in the adapted coordinates after undoing the steps on every chart map."""
    names = record["variables"]
    steps = translation_steps(translation, names)
    adapted = adapted_polynomial(polynomial, names, steps)
    if record.get("input_divisor"):
        raise InvalidChart("An adapted coordinate change requires a reduced fan record.")
    proof = record.get("certificate", {})
    certificate = {k: v for k, v in proof.items() if k != "coordinate_translation"}
    if "adapted_polynomial" in certificate and parse_expression(certificate["adapted_polynomial"], names) != adapted:
        raise InvalidChart("The adapted polynomial differs from the translated source.")
    if "adapted_boundary_image_coordinate_ideals" in certificate or "boundary_image_ideals" in certificate:
        centres = certificate.pop("adapted_boundary_image_coordinate_ideals", None)
        declared = certificate.pop("boundary_image_ideals", None)
        if centres is not None:
            certificate["boundary_image_coordinate_ideals"] = centres
        expected = _certificate(dict(boundary_image_coordinate_ideals=centres or []), names, steps)
        if declared is not None and (centres is None or declared != expected["boundary_image_ideals"]):
            raise InvalidChart("The declared boundary images differ from the adapted fan.")
    certificate.pop("fan_coordinate_frame", None)
    normalized_record = {**record, "certificate": certificate}
    if "polynomial" in normalized_record:
        normalized_record["polynomial"] = format_poly(adapted, names)
    for key in ("atlas", "charts"):
        if key not in record:
            continue
        normalized = []
        for chart in record[key]:
            torus = chart.get("torus", {})
            if key == "atlas" and torus.get("coordinate_translation") != translation:
                raise InvalidChart("The charts disagree on the translation to the input coordinates.")
            item = dict(chart, map=compose_map(chart, names, steps, inverse=True))
            if torus:
                item["torus"] = {k: v for k, v in torus.items() if k != "coordinate_translation"}
            normalized.append(item)
        normalized_record[key] = normalized
    verified = verify(normalized_record, adapted, **options)
    if verified is None:
        return None
    atlas, outputs = verified
    if "atlas" in record:
        return record["atlas"], outputs
    # The fibre ideal generated by the adapted coordinates equals the ideal
    # generated by the translated ones because each step fixes the origin.
    return [dict(item, map=chart["map"], torus={**item["torus"], "coordinate_translation": translation})
            for item, chart in zip(atlas, record["charts"], strict=True)], outputs
