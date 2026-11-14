"""Transport rational and marked polynomial rows for chart verification and geometric components."""

from .affine_kernel import _sparse_expression
from .kernel_json import kernel_json
from .reference import QQ


class ComponentRing:
    """Serialize coordinates and a marked coefficient field in one shared sparse ring."""
    def __init__(self, variables, domain):
        self.variables = tuple(variables)
        self.domain = domain
        self.ring = self.variables
        self.relation = None
        if domain != QQ:
            from .field_polynomial import _parameter, modulus
            from .field_wide import _expanded_row
            self.theta = _parameter(domain)
            self.ring = (*self.variables, self.theta)
            self.relation = _expanded_row(self.ring, modulus(domain, self.theta))

    def row(self, value):
        from .field_wide import _expanded_row
        if self.relation is not None:
            from .field_polynomial import lift
            value = lift(value, self.variables, self.domain, self.theta)
        return _expanded_row(self.ring, value)

    def expression(self, row):
        if self.relation is None:
            return _sparse_expression(self.variables, row)
        from .field_polynomial import _line_expression
        return _line_expression(row, self.variables, self.domain)

    def request(self, operation, **data):
        return kernel_json("atlas_component_geometry", operation, width=len(self.variables), relation=self.relation, **data)
