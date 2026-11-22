"""Transport charts of the native central plane arrangement resolution."""

from .kernel_json import kernel_json
from .point_geometry import encode_terms
from .strata import InvalidChart


def _construction(poly, names, mode):
    return kernel_json(
        "atlas_plane_arrangement",
        "resolve",
        variables=names,
        terms=encode_terms(poly),
        intrinsic=mode == "intrinsic",
    )


def _outputs(proof, integrals):
    output = {"dual_complex": proof["dual_complex"]}
    if integrals:
        output.update(numerical_data=proof["numerical_data"], strata=proof["strata"])
    return proof["atlas"], output


def resolve_arrangement(poly, names, mode, verified, integrals):
    proof = _construction(poly, names, mode)
    if proof is None:
        return None
    result = dict(
        status="resolved",
        engine="central-plane-arrangement",
        summary=(
            f"Blow up the vertex of the plane arrangement and resolve the projective line arrangement "
            f"on its exceptional plane ({len(proof['charts'])} affine charts)."
        ),
        certificate={
            "method": "Blowups of intersection flats of a reduced rational plane arrangement",
            "description": "The projective directions of the intersection lines determine disjoint "
            "centres after blowing up the vertex. On their exceptional lines the incident planes "
            "meet at distinct tangent directions.",
        },
        steps=proof["steps"],
        charts=proof["charts"],
        dual_complex=proof["dual_complex"],
    )
    if verified is not None:
        verified.append((result, _outputs(proof, integrals)))
    return result


def arrangement_outputs(record, poly, integrals):
    proof = _construction(poly, record["variables"], record["mode"])
    if proof is None or not (record.get("atlas") or record.get("charts")):
        raise InvalidChart("Missing plane arrangement or its chart cover.")
    if any(record[key] != proof[key] for key in ("steps", "atlas", "charts") if key in record):
        raise InvalidChart("The saved arrangement charts or centres differ from the defining equation.")
    return _outputs(proof, integrals)
