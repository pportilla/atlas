"""A'Campo strata above the origin for point blowups over the complex numbers."""
from collections import Counter


def _plane(result):
    # Intrinsic point blowups can stop before the curve is transverse to the
    # exceptional boundary. The embedded atlas certifies all the punctures.
    if result.get("mode") == "intrinsic":
        return None
    dual = result["dual_complex"]
    vertices = dual["vertices"]
    labels = {v["id"] for v in vertices}
    if dual["status"] != "complete" or not labels or any(
            "total_multiplicity" not in v or "self_intersection" not in v for v in vertices):
        return None
    meetings = Counter()
    for cell in dual["cells"]:
        pair = cell["vertices"]
        if len(set(pair)) != 2 or len(pair) != 2 or not labels.issuperset(pair):
            return None
        meetings.update(pair)
    marked = Counter()
    for branch in result["branches"]:
        axes = branch["ambient_components"]
        if len(axes) != 1 or axes[0] not in labels:
            return None
        # Each branch record is one geometric point. It may appear in two
        # overlapping affine charts, or be conjugate to another branch.
        marked[axes[0]] += 1
    return [{"component":v["id"], "multiplicity":v["total_multiplicity"],
             "euler_characteristic":2-meetings[v["id"]]-marked[v["id"]],
             "exceptional_intersection_points":meetings[v["id"]],
             "strict_transform_intersection_points":marked[v["id"]]}
            for v in vertices]


def _ordinary(result):
    steps = result["steps"]
    if len(steps) != 1 or steps[0].get("type") != "point blowup":
        return None
    step = steps[0]
    degree = step.get("reduced_support_multiplicity",step.get("multiplicity"))
    dimension = len(result["tangent_coordinates"])
    if not degree or dimension < 2:
        return None
    # The exceptional fiber is P^(n−1) and the strict support meets it in
    # its smooth projective tangent hypersurface H_d. Gauss–Bonnet gives
    # χ(P^(n−1) \ H_d) as the coefficient of h^(n−1) in (1+h)^n/(1+dh).
    chi = (1-(1-degree)**dimension)//degree
    return [{"component":step["exceptional"], "multiplicity":result["multiplicity"],
             "euler_characteristic":chi, "projective_dimension":dimension-1,
             "tangent_degree":degree}]


def blowup_contributions(result):
    """Return (N_i, χ(E_i°∩π⁻¹(0))) records for the supported point blowups."""
    if result.get("status") != "resolved":
        return None
    if result["engine"] == "C++ / FLINT plane point blowups":
        return _plane(result)
    if result["engine"] == "native-ordinary":
        return _ordinary(result)
    return None
