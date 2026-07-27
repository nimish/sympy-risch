"""First displayed-constant logarithmic reconstruction cases."""

from sympy import Poly, log, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicIntegrationStatus,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    integrate_algebraic,
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
        max_x_degree=0,
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
        max_x_degree=0,
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
        max_x_degree=0,
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
        max_x_degree=0,
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
        max_x_degree=1,
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


def test_logarithmic_reconstruction_retains_a_residue_free_remainder() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)

    result = integrate_algebraic(
        y / (x - 1),
        tower.derivation(),
        basis,
        max_x_degree=1,
    )

    assert result.status is AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    assert not result.is_elementary
    assert result.reconstructs
    assert result.logarithmic_argument is not None
    assert result.logarithmic_derivative != extension.field.zero
    assert result.remainder != extension.field.zero
    assert result.antiderivative is None


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


def test_principal_bound_miss_is_not_an_integration_obstruction() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x**3 + x - 1)

    result = integrate_algebraic(
        y / x,
        tower.derivation(),
        basis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicIntegrationStatus.INCOMPLETE
    assert not result.is_elementary
    assert result.reconstructs
    assert result.principal_search is not None
    assert result.logarithmic_argument is None
    assert result.remainder == extension.field.from_sympy(y / x)
