"""First displayed-constant logarithmic reconstruction cases."""

from sympy import Poly, Rational, log, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicFunctionFieldReduction,
    AlgebraicIntegrationStatus,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    integrate_algebraic,
    integrate_algebraic_with_constant_base_change,
)
from sympy_risch.algebraic_geometry.base_change import (
    build_algebraic_constant_base_change,
)

x, y = symbols("x y")


def algebraic_tower(modulus, constants=QQ):
    """Return the standard ``C(x)(y)`` tower and its power basis."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return tower, extension, IntegralBasis.power_basis(extension)


def test_integrate_algebraic_reconstructs_a_ramified_logarithm() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        1 / x,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.is_elementary
    assert result.reconstructs
    assert result.remainder == extension.field.zero
    assert result.antiderivative == 2 * log(y)
    assert result.logarithmic_argument == extension.field.generator
    assert result.logarithmic_coefficient == QQ.convert(2)


def test_integrate_algebraic_reconstructs_split_finite_residues() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        1 / (x - 1),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert result.remainder == extension.field.zero
    assert result.antiderivative == log(x - 1)


def test_rational_residue_at_a_higher_degree_place_is_not_rejected() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        2 * x / (x**2 + 1),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert result.antiderivative == log(x**2 + 1)
    assert result.residue_divisor is not None
    assert any(place.degree == 4 for place in result.residue_divisor.support)


def test_fractional_rational_content_scales_the_logarithmic_term() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        1 / (6 * x),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert result.logarithmic_coefficient == QQ.convert(1 / 3)
    assert result.antiderivative == log(y) / 3


def test_displayed_algebraic_constants_produce_multiple_logarithmic_terms() -> None:
    """Coordinate descent over QQ(sqrt(2)) keeps independent log directions."""
    constants = QQ.algebraic_field(sqrt(2))
    tower, extension, basis = algebraic_tower(y**2 - x, constants)

    result = integrate_algebraic(
        extension.field.from_sympy(sqrt(2) / x + 1 / (x - 1)),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert result.remainder == extension.field.zero
    assert len(result.logarithmic_terms) == 2
    assert result.logarithmic_coefficient is None
    assert {
        constants.to_sympy(term.coefficient) for term in result.logarithmic_terms
    } == {1, 2 * sqrt(2)}
    assert result.antiderivative is not None


def test_positive_genus_algebraic_constant_torsion_is_automatic(
    monkeypatch,
) -> None:
    r"""The order-two class ``P - infinity`` reconstructs ``log(x-a)/2``."""
    import sympy_risch.algebraic.logarithmic as logarithmic

    constants = QQ.algebraic_field(sqrt(2))
    tower, extension, basis = algebraic_tower(
        y**2 - x * (x - sqrt(2)) * (x - 1),
        constants,
    )
    integrand = extension.field.from_sympy(
        (1 + sqrt(2)) / (2 * (x - sqrt(2))),
    )
    original_prepare = logarithmic.prepare_algebraic_torsion_reductions
    original_reduce_divisor = AlgebraicFunctionFieldReduction.reduce_divisor
    preparation_count = 0
    divisor_reduction_count = 0

    def counted_prepare(model, **kwargs):
        nonlocal preparation_count
        preparation_count += 1
        return original_prepare(model, **kwargs)

    def counted_reduce_divisor(reduction, divisor):
        nonlocal divisor_reduction_count
        divisor_reduction_count += 1
        return original_reduce_divisor(reduction, divisor)

    monkeypatch.setattr(
        logarithmic,
        "prepare_algebraic_torsion_reductions",
        counted_prepare,
    )
    monkeypatch.setattr(
        AlgebraicFunctionFieldReduction,
        "reduce_divisor",
        counted_reduce_divisor,
    )

    result = integrate_algebraic(integrand, tower.derivation(), basis)

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.remainder == extension.field.zero
    assert preparation_count == 1
    assert divisor_reduction_count == 2
    assert len(result.logarithmic_terms) == 2
    assert all(
        term.principality.divisor_class_search is not None
        and term.principality.divisor_class_search.order == 2
        for term in result.logarithmic_terms
    )
    expected = (1 + sqrt(2)) * log(x - sqrt(2)) / 2
    assert (result.antiderivative - expected).expand() == 0


def test_target_algebraic_constant_logarithm_uses_exact_field_quotient() -> None:
    """A certified principal witness remains usable over QQ(alpha)(x)."""
    source_tower, source_extension, _source_basis = algebraic_tower(y**2 - x)
    constant = symbols("constant")
    change = build_algebraic_constant_base_change(
        AlgebraicFunctionField(source_extension),
        source_tower.derivation(),
        Poly(constant**2 - Rational(1, 18), constant, domain=QQ),
    )
    assert change.certified
    assert change.target_extension is not None
    assert change.target_derivation is not None

    target = change.target_extension
    result = integrate_algebraic(
        change.embed(source_extension.field.from_sympy(y / (x - 2))),
        change.target_derivation,
        IntegralBasis.power_basis(target),
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert len(result.logarithmic_terms) == 1
    assert result.logarithmic_derivative != target.field.zero
    assert result.remainder == target.field.zero
    assert result.exactness is not None and result.exactness.is_exact
    assert result.exact_correction - target.field.from_sympy(2 * y) == target.field.zero


def test_selected_constant_base_change_can_finish_an_algebraic_log_integral() -> None:
    """Recollecting the target places yields an elementary answer in C'(x, y)."""
    source_tower, source_extension, source_basis = algebraic_tower(y**2 - x)
    constant = symbols("constant")

    result = integrate_algebraic_with_constant_base_change(
        source_extension.field.from_sympy(y / (x * (x - 2))),
        source_tower.derivation(),
        source_basis,
        Poly(constant**2 - Rational(1, 18), constant, domain=QQ),
    )

    assert (
        result.source_result.status
        is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    )
    assert result.source_reconstructs
    assert result.base_change is not None
    assert result.base_change.certified
    assert result.base_change.target_extension is not None
    assert result.target_result is not None
    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.is_elementary
    assert result.reconstructs
    assert (
        result.target_result.remainder == result.base_change.target_extension.field.zero
    )
    assert len(result.target_result.logarithmic_terms) == 1
    assert result.antiderivative is not None


def test_profile_base_change_selects_a_compatible_constant_factor() -> None:
    """One profile factor can supply every residue constant on this curve."""
    source_tower, source_extension, source_basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic_with_constant_base_change(
        source_extension.field.from_sympy(y / (x * (x - 2))),
        source_tower.derivation(),
        source_basis,
    )

    assert result.base_change is not None
    assert result.base_change.certified
    (candidate,) = result.base_change.factors
    assert candidate == Poly(
        candidate.gens[0] ** 2 - Rational(1, 2),
        candidate.gens[0],
        domain=QQ,
    )
    assert result.target_result is not None
    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.is_elementary
    assert result.reconstructs


def test_profile_base_change_combines_two_independent_residue_fields() -> None:
    """Two quadratic residue fields are integrated in their exact compositum."""
    source_tower, source_extension, source_basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic_with_constant_base_change(
        source_extension.field.from_sympy(y / ((x - 2) * (x - 3))),
        source_tower.derivation(),
        source_basis,
    )

    assert result.base_change is not None
    assert result.base_change.certified
    assert len(result.steps) == 1
    assert len(result.base_change.factors) == 2
    assert result.base_change.constant_compositum is not None
    assert len(result.base_change.constant_compositum.steps) == 2
    assert result.base_change.target_constants is not None
    assert result.base_change.target_constants.ext.minpoly.degree() == 4
    assert result.target_result is not None
    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative is not None


def test_profile_base_change_reports_algebraic_coefficient_boundary() -> None:
    """A non-rational profile factor remains unsupported without an exception."""
    constants = QQ.algebraic_field(sqrt(2))
    tower, extension, basis = algebraic_tower(y**2 - x, constants)

    result = integrate_algebraic_with_constant_base_change(
        extension.field.from_sympy(y / (x - sqrt(2))),
        tower.derivation(),
        basis,
    )

    assert (
        result.source_result.status
        is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    )
    assert result.status is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    assert result.steps == ()
    assert "no new rationally defined irreducible factor" in result.detail


def test_logarithmic_reconstruction_adds_an_exact_residue_free_correction() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        y / (x - 1),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.is_elementary
    assert result.reconstructs
    assert result.logarithmic_argument is not None
    assert result.logarithmic_derivative != extension.field.zero
    assert result.remainder == extension.field.zero
    assert result.exactness is not None and result.exactness.is_exact
    assert result.exact_correction == 2 * extension.field.generator
    assert result.antiderivative is not None


def test_zero_residues_remain_for_first_and_second_kind_exactness() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x**3 + x)

    result = integrate_algebraic(
        extension.field.one / extension.field.generator,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    assert not result.is_elementary
    assert result.reconstructs
    assert result.logarithmic_argument is None
    assert result.logarithmic_derivative == extension.field.zero
    assert result.remainder == extension.field.one / extension.field.generator
    assert result.exactness is not None
    assert result.exactness.complete_negative


def test_nonrational_residues_request_compatible_constant_base_change() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        y / ((x - 2) * (x - 8)),
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    assert not result.is_elementary
    assert result.reconstructs
    assert result.residue_divisor is None
    assert result.logarithmic_argument is None
    assert result.remainder == extension.field.from_sympy(
        y / ((x - 2) * (x - 8)),
    )


def test_nonprincipal_residue_divisor_is_an_integration_obstruction() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x**3 + x - 1)

    result = integrate_algebraic(
        y / x,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicIntegrationStatus.UNSUPPORTED
    assert not result.is_elementary
    assert result.reconstructs
    assert result.principality_decision is not None
    assert result.principality_decision.complete_negative
    assert result.logarithmic_argument is None
    assert result.remainder == extension.field.from_sympy(y / x)
