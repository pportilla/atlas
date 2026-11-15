"""Certify owned affine charts, including implicit and finite-field presentations."""


def native_general_outputs(record, serialized_charts, *, complex=True, intrinsic=False, integrals=True):
    from .atlas import from_json, outputs
    from .local_blowups import GiveUp
    charts = from_json(serialized_charts, native=True)
    try:
        return outputs(charts, len(record["variables"]), complex, intrinsic=intrinsic, integrals=integrals)
    except GiveUp:
        if not (intrinsic and integrals):
            raise
        return outputs(charts, len(record["variables"]), complex, intrinsic=True, integrals=False)
