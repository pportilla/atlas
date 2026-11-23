"""Subject pages and source pointers for the session functions in the reference manual.

The session supplies the signatures, descriptions and examples. Each link here
points to a particular piece of the implementation or a test that demonstrates
an observable result. A source needle identifies a line without baking volatile
line numbers into the reference.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    label: str
    path: str
    symbol: str | None = None
    needle: str | None = None


@dataclass(frozen=True)
class Entry:
    sources: tuple[Link, ...]
    tests: tuple[Link, ...]


def native(label, file, needle):
    return Link(label, f"native/{file}.cpp", needle=needle)


def session(label, file, needle):
    return Link(label, f"engine/{file}.py", needle=needle)


def example(label, file, symbol):
    return Link(label, f"tests/test_{file}.py", symbol="test_" + symbol)


basis_example = example("Reduce with a saved basis and check membership", "session_algebra",
                        "reuse_basis_and_polynomial_in_original_coordinates")
ideal_example = example("Compare intersection, quotient and saturation", "ideal_session",
                        "prompt_and_script_preserve_result_rings")
ideal_arithmetic_example = example("Use ideal arithmetic at the prompt and in a script", "ideal_operators",
                                   "arithmetic_composes_and_preserves_the_parent")
power_example = example("Compare ordinary and symbolic squares of the three coordinate axes",
                        "ideal_power_interface", "reuse_at_prompt_and_in_scripts")
quotient_example = example("Compare global and local quotient bases", "local_algebra_session",
                           "quotient_bases_and_cusp_invariants_at_both_entry_points")
standard_example = example("Check local membership and its unit certificate", "local_standard_basis_session",
                           "local_membership_and_unit_certificate_at_prompt_and_in_script")
hilbert_example = example("Read cumulative and graded lengths and multiplicity of a cusp",
                          "session_hilbert_samuel", "cusp_at_prompt_and_in_script")
tangent_example = example("Check an exact tangent direction at a rational point",
                          "session_zariski_tangent", "fractional_tangent_direction_at_prompt_and_in_script")
complex_example = example("Recover both edges of the cusp graph from the saved charts",
                          "resolution_complex", "archived_record_replays_without_cached_incidence")
resolve_example = example("Follow the cusp branch to its charts and read the divisor multiplicity",
                          "cli_session", "script_follows_cusp_from_branch_to_charts")

basis_at_origin = native("Local standard basis by homogenization", "local_basis",
                         "Basis certified = global_basis(rows")
global_basis = native("Reduced Gröbner basis", "groebner", "void Engine::compute(")
quotient_at_point = native("Jacobian and local quotient algebras at a point", "local_point_quotient",
                           "PointQuotient quotient_at_point(")
quotient_length = native("Length and basis from the leading monomials", "monomial_quotient",
                         "local::PointQuotient local::monomial_point_quotient(")
series_numerator = native("Hilbert numerator from standard monomials", "monomial_quotient", "Polynomial numerator(")
point_dimension = native("Local Krull dimension and the smoothness criterion", "point_geometry",
                         "DimensionResult dimension_at_point(")
tangent_space = native("Jacobian row space and its nullspace", "point_geometry", "Json::Value row_space(")
cone_forms = native("Initial forms of a local standard basis", "point_geometry", 'if (op == "cone_forms")')
complex_sources = (
    native("Cells and faces of exceptional intersections", "atlas_incidence", "Value cells() const"),
    session("Chart and toric replay paths", "resolution_complex", "def dual_complex("),
)
builtin_registration = session("Python built-in registered in the session", "session", '"len": len, "print": print')

# Each subject page of the reference manual owns its functions' canonical names,
# in reading order. Aliases keep their anchors but share the canonical card,
# including its example, implementation and tests.
_GROUPS = {
    "docs/reference/rings.md": {
        "ring": Entry(
            (session("Declare a ring and its generators", "session_polynomial", "def ring(*names, coeffs=QQ, fresh=False):"),),
            (example("Use the coordinates of a declared ring", "session_ring_names",
                     "ring_returns_the_session_parent_of_the_declared_coordinates"),)),
        "use_ring": Entry(
            (session("Choose the active ring", "session_rings", "def use_ring(ring):"),),
            (example("Read a formula and construct an ideal in the active ring", "session_ring_names",
                     "active_ring_reads_strings_and_holds_ideals_of_numbers"),)),
        "active_ring": Entry(
            (session("Inspect the active ring", "session_rings", "def active_ring():"),),
            (example("See the selected ring in the workspace", "session_ring_names",
                     "workspace_marks_the_active_ring"),)),
        "ring_name": Entry(
            (session("Look up a live ring in the workspace", "session_rings", "def ring_name(ring):"),),
            (example("Follow ring names through renaming and aliasing", "session_ring_names",
                     "ring_name_follows_live_bindings"),)),
        "delete_ring": Entry(
            (session("Delete a ring and its dependent names", "session_rings", "def delete_ring(ring):"),),
            (example("Check which bindings survive deletion", "session_ring_names",
                     "deleting_a_ring_removes_everything_that_depends_on_it"),)),
        "coordinates": Entry(
            (session("Coordinate declarations and parent rings", "session_polynomial", "def coordinates(*names, coeffs=QQ, fresh=False):"),),
            (example("Redeclare coordinates and retain the same ring", "cli_session",
                     "coordinate_redeclaration_reuses_the_parent_at_prompt_and_in_script"),)),
        "modular_arithmetic": Entry(
            (native("Residue arithmetic and lifting", "modular_batch", "Json::Value calculate("),),
            (example("Verify reconstructed residues and simple roots", "modular_arithmetic",
                     "lifts_and_reconstruction"),)),
        "matrix_analysis": Entry(
            (native("Exact matrix analysis and system solving", "point_geometry", "Json::Value matrix_analysis("),),
            (example("Check all three solution statuses", "matrix_analysis", "solution_status_and_subspaces"),)),
        "polynomial": Entry(
            (native("Polynomial expression parser", "affine", "class RecordParser {"),),
            (example("Read formulas in the original and projected rings", "session_algebra",
                     "quoted_polynomial_uses_original_or_projected_coordinates"),)),
        "absolute_factor": Entry(
            (native("Geometric factors from transverse sections", "absolute_factorization",
                    "std::vector<GeometricPolynomial> absolute_factors("),),
            (example("Factor a cubic and combine two quadratic extensions", "absolute_factorization",
                     "univariate_cubic_and_two_independent_quadratic_extensions"),)),
        "polynomial_arithmetic": Entry(
            (session("Choose operands in their live ring", "session_algebra", "def polynomial_arithmetic("),),
            (example("Compare native CPU and GPU requests", "polynomial_arithmetic_modes",
                     "exact_actions_and_live_parent"),)),
        "hensel_factors": Entry(
            (native("Coprime Hensel correction and verification", "hensel_factors", 'Json::Value hensel('),),
            (example("Verify the lifted factors and their live ring", "hensel_factors", "quadratic_and_ownership"),)),
        "cancel_fraction": Entry(
            (native("Normalize the coprime numerator and denominator", "triton_factor_division", 'if (op == "cancel-fraction")'),),
            (example("Cancel rational functions and retain their live parent", "gpu_factor_division",
                     "reduced_fraction_zero_wide_and_ownership"),)),
        "polynomial_division": Entry(
            (native("Exact binomial division", "triton_factor_division", "bool triton_binomial_division("),),
            (example("Check quotient, remainder and ownership", "gpu_factor_division", "public_results_and_ring_ownership"),)),
        "extended_gcd": Entry(
            (native("Exact Euclidean and Bézout computation", "triton_factor_division", "void xgcd("),),
            (example("Reconstruct the monic GCD", "gpu_factor_division", "public_results_and_ring_ownership"),)),
        "squarefree_factor": Entry(
            (native("Square-free factors grouped by multiplicity", "polynomial_factors", "const char* squarefree_factors("),),
            (example("Check repeated rational and geometric factors", "gpu_factor_division", "squarefree_groups_multiplicities"),)),
        "factor": Entry(
            (native("Ordered factorization over the rationals", "polynomial_factors", "const char* ordered_factors("),),
            (example("Reconstruct the input from its unit and factors", "factor_session",
                     "prompt_and_script_reconstruct_the_polynomial"),)),
        "gcd": Entry(
            (native("Monic polynomial GCD", "polynomial_gcd", "void monic_polynomial_gcd("),),
            (example("Check the leading coefficient in each order", "polynomial_gcd",
                     "monic_in_the_declared_order"),)),
        "resultant": Entry(
            (native("Resultant in a chosen coordinate", "affine", "Polynomial resultant("),),
            (example("Eliminate each of two coordinates in turn", "resultant_session",
                     "choose_either_coordinate_at_prompt_and_in_scripts"),)),
        "quotient_ring": Entry(
            (native("Arithmetic of residue classes", "quotient_ring", "Value Ring::evaluate("),),
            (example("Multiply and invert classes modulo xy − 1 and modulo x²",
                     "session_quotient_ring", "prompt_and_script"),)),
        "hom": Entry(
            (native("Simultaneous polynomial substitution", "affine", "Polynomial compose("),
             session("Generator images and quotient relations", "session_maps", "def hom(")),
            (example("Substitute, swap generators and evaluate over QQ", "session_maps",
                     "simultaneous_substitution_and_wide_inputs_at_both_entry_points"),
             example("Check that the images kill the source relations", "session_maps",
                     "foreign_bindings_and_source_relations"))),
    },
    "docs/reference/ideals.md": {
        "ideal": Entry(
            (session("Ideal declaration in an ambient ring", "session_algebra", "def ideal(*generators"),),
            (example("Declare ideals in polynomial and scalar rings", "ideal_session",
                     "live_parents_and_rational_scalar_embeddings_at_both_entry_points"),)),
        "groebner": Entry((global_basis,), (basis_example,)),
        "normal_form": Entry(
            (native("Reduction over the rationals", "groebner", "char* atlas_groebner_reduce("),),
            (example("Compare the same remainder across interfaces", "algebra_interface",
                     "normal_form_through_shell_prompt_script_and_http"),)),
        "membership": Entry(
            (native("Membership by reduction", "groebner", "bool Engine::contains("),), (basis_example,)),
        "eliminate": Entry(
            (native("Elimination in a chosen coordinate order", "ordered_ideals", "Basis eliminate_step("),),
            (example("Recover the surviving coordinates and eliminate down to QQ", "session_algebra",
                     "elimination_supplies_surviving_coordinates_and_scalar_field"),)),
        "intersection": Entry(
            (native("Ideal intersection", "ordered_ideals", "Basis intersect("),), (ideal_example,)),
        "quotient": Entry(
            (native("Ideal quotient", "ordered_ideals", "Basis quotient("),), (ideal_example,)),
        "saturation": Entry(
            (native("Saturation by a polynomial", "ordered_ideals", 'else if(op=="saturate")'),), (ideal_example,)),
        "ideal_sum": Entry(
            (native("Combine the generators of two ideals", "ideal_arithmetic", "Basis sum("),),
            (ideal_arithmetic_example,)),
        "ideal_product": Entry(
            (native("Form pairwise products of generators", "ideal_arithmetic", "Basis product("),),
            (ideal_arithmetic_example,)),
        "ideal_power": Entry(
            (native("Ordinary ideal powers", "ideal_powers", "PowerResult ordinary("),), (power_example,)),
        "symbolic_power": Entry(
            (native("Symbolic powers of radical ideals", "ideal_powers", "PowerResult symbolic("),), (power_example,)),
    },
    "docs/reference/components.md": {
        "radical": Entry(
            (native("Radical of a rational ideal", "primary_decomposition", "Basis radical(Basis rows"),),
            (example("Reuse the radical of a mixed-dimensional ideal", "session_radical",
                     "prompt_and_script_reuse_the_radical_of_a_mixed_dimensional_ideal"),)),
        "primary_decomposition": Entry(
            (native("Primary components and associated primes", "primary_decomposition", "Components decompose("),),
            (example("Reconstruct an ideal with an embedded prime", "primary_decomposition_interface",
                     "prompt_and_script_reconstruct_an_ideal_with_an_embedded_prime"),)),
        "localize": Entry(
            (native("Contraction after localization at a prime", "localization", "Basis contract("),),
            (example("Inspect a localized ideal and test fractions", "localization_interface",
                     "structured_ring_fractions_and_ideal_at_prompt_and_in_script"),)),
        "normalization": Entry(
            (native("Integral closure of a domain", "normalization", "Presentation normalize_domain("),),
            (example("Recover the cusp parametrization and its inverse fraction", "normalization_interface",
                     "cusp_images_and_inverse_fractions_at_prompt_and_in_script"),)),
    },
    "docs/reference/dimension.md": {
        "dimension": Entry(
            (native("Dimension from leading monomial supports", "affine",
                    "return report_dimension(ring.n - component_cover_height"),),
            (example("Read dimensions after elimination and for the empty scheme", "ideal_session",
                     "prompt_and_script_preserve_result_rings"),)),
        "quotient_basis": Entry((quotient_length, global_basis), (quotient_example,)),
        "multiplicity": Entry(
            (quotient_length, global_basis),
            (example("Compare a finite global length with infinite colength", "session_multiplicity",
                     "global_length_and_infinite_colength_at_both_entry_points"),)),
        "hilbert_series": Entry(
            (series_numerator,),
            (example("Read the signed Hilbert numerator of (x², y³)", "session_hilbert_series",
                     "prompt_and_script_expose_signed_sparse_terms"),)),
    },
    "docs/reference/local-algebra.md": {
        "local_standard_basis": Entry(
            (basis_at_origin,
             native("Standard basis at a rational point", "local_point_basis", "Basis standard_basis_at_point(")),
            (standard_example,)),
        "local_normal_form": Entry(
            (native("Local division and unit certificate", "local_division", "NormalForm normal_form("),
             native("Normal form at a rational point", "local_point_normal_form", "NormalForm normal_form_at_point(")),
            (standard_example,)),
        "local_quotient_basis": Entry((quotient_at_point,), (quotient_example,)),
        "hilbert_samuel_function": Entry(
            (native("Cumulative and graded Hilbert–Samuel values", "monomial_quotient", "std::string function_values("),),
            (hilbert_example,)),
        "hilbert_samuel_multiplicity": Entry(
            (native("Multiplicity from the monomial staircase", "monomial_quotient", "Series analyze("),),
            (hilbert_example,)),
    },
    "docs/reference/tangent-geometry.md": {
        "tangent_cone": Entry(
            (cone_forms, basis_at_origin),
            (example("Recover the double tangent line of a translated cusp", "session_tangent_cone",
                     "translated_cusp_at_prompt_and_in_script"),)),
        "tangent_cone_hilbert_series": Entry(
            (cone_forms, series_numerator),
            (example("Read the graded series at a cone vertex", "session_tangent_cone",
                     "series_at_cone_vertex_exports_exact_degrees_and_replays"),)),
        "zariski_tangent_space": Entry((tangent_space,), (tangent_example,)),
        "embedding_dimension": Entry((tangent_space,), (tangent_example,)),
        "local_dimension": Entry(
            (point_dimension,),
            (example("Compare local dimensions on different components", "local_dimension_interface",
                     "different_components_http_cli_and_replay"),)),
        "local_smoothness": Entry(
            (point_dimension,),
            (example("Distinguish a junction from a smooth branch", "local_dimension_interface",
                     "junction_and_smooth_branch_above_translation_cap"),)),
    },
    "docs/reference/germ-invariants.md": {
        "local_jacobian_algebra": Entry(
            (quotient_at_point,),
            (example("Read the cusp's derivatives and its Jacobian algebra basis", "local_algebra_session",
                     "quotient_bases_and_cusp_invariants_at_both_entry_points"),)),
        "milnor_number": Entry(
            (quotient_at_point,),
            (example("Check the Milnor number at a translated cusp", "session_point_invariants",
                     "translated_cusp_at_prompt_and_in_script"),)),
        "tjurina_number": Entry(
            (quotient_at_point,),
            (example("Compare Milnor number 12 with Tjurina number 11", "local_algebra_session",
                     "tjurina_number_also_imposes_the_original_equation"),)),
        "ae_codimension": Entry(
            (native("Extended left-right codimension and finite-jet certificates", "ae_codimension",
                    "Json::Value ae_codimension("),
             native("Semigroup gaps of a parametrized monomial curve", "ae_special",
                    "std::optional<Json::Value> monomial_curve(")),
            (example("Check codimension 1 after rescaling and reordering a cusp parametrization",
                     "ae_codimension_interface", "cusp_ordered_components_and_replay"),
             example("Inspect the lower bound when the calculation is inconclusive",
                     "ae_codimension_interface", "inconclusive_record_preserves_last_proved_lower_bound"))),
    },
    "docs/reference/resolution.md": {
        "resolve": Entry(
            (native("Point blowups of plane curve germs", "plane", "void visit(Task&"),
             native("Weighted blowups along points and bad curves", "weighted_resolution",
                    "CurvePlan plan_curve(const Chart& chart"),
             session("Dispatch a resolution from the session", "session", "def resolve(polynomial")),
            (resolve_example,
             example("Blow up a double conic along its curve with method weighted", "weighted_resolution",
                     "a_double_conic_is_blown_up_along_its_curve"))),
        "monodromy_check": Entry(
            (native("Pole orders and monodromy witnesses", "monodromy", "Json::Value pole_data("),
             native("Transverse compositions on critical components", "monodromy",
                    "Json::Value transverse_compositions("),
             native("Decide a nearby eigenvalue using local Bernstein–Sato roots", "monodromy_local_b",
                    "Json::Value local_eigenvalues(")),
            (example("Compare pole orders and nearby witnesses", "session_monodromy",
                     "signed_orders_h0_and_nearby_witnesses_at_both_entry_points"),
             example("Witness an eigenvalue on singular lines without rational points", "session_monodromy",
                     "transverse_composition_witnesses_at_prompt_and_in_scripts"))),
        "recompute_dual_complex": Entry(complex_sources, (complex_example,)),
        "dual_complex": Entry(complex_sources, (complex_example,)),
        "zoo_search": Entry(
            (native("Match monomials under a shared coordinate injection", "zoo_search", "bool assign_axes("),),
            (example("Find the same catalogue records at the prompt and in a script", "zoo_search",
                     "prompt_and_script"),)),
    },
    "docs/reference/output.md": {
        "print": Entry((builtin_registration,), (resolve_example,)),
        "len": Entry((builtin_registration,), (resolve_example,)),
        "print_json": Entry(
            (session("Serialize session values as a JSON record", "session", "def print_json("),),
            (example("Serialize a large exact integer and rational", "session_monodromy",
                     "json_preserves_large_integers_and_existing_floats"),)),
        "algebra_request": Entry(
            (session("Dispatch a saved request and restore the result's ring", "session_algebra",
                     "def algebra_request("),),
            (example("Restore a Jacobian algebra with its basis and derivatives", "local_algebra_session",
                     "algebra_request_restores_basis_and_derivatives_in_its_ring"),)),
    },
}

TOPICS = {page: tuple(entries) for page, entries in _GROUPS.items()}
ENTRIES = {name: entry for entries in _GROUPS.values() for name, entry in entries.items()}
ALIASES = {"saturate": "saturation", "localization": "localize"}
