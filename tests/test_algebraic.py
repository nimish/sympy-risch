import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    hermite_reduce_algebraic,
)


x, y = symbols("x y")


def algebraic_tower(modulus):
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(
        Poly(modulus, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    return tower, extension, IntegralBasis.power_basis(extension)


def test_algebraic_extension_differentiates_a_square_root() -> None:
    tower, extension, _ = algebraic_tower(y**2 - x)
    field = extension.field
    parent_element = extension.parent_field.from_sympy(1/x)

    assert tower.derivation()(field.generator) == field.from_sympy(y/(2*x))
    assert tower.derivation()(parent_element) == field.from_sympy(-1/x**2)


def test_algebraic_extension_differentiates_a_mixed_cubic_basis() -> None:
    tower, extension, _ = algebraic_tower(y**3 - y - x)
    field = extension.field
    delta = 27*x**2 - 4

    assert tower.derivation()(field.generator) == field.from_sympy(
        (4 + 9*x*y - 6*y**2)/delta,
    )
    assert tower.derivation()(field.generator**2) == field.from_sympy(
        (-12*x - 4*y + 18*x*y**2)/delta,
    )


def test_algebraic_extension_validates_the_minimal_polynomial() -> None:
    parent = QQ.frac_field(x)

    with pytest.raises(TypeError, match="SymPy Poly"):
        AlgebraicExtension(y**2 - x)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="degree at least two"):
        AlgebraicExtension(Poly(y - x, y, domain=parent))
    with pytest.raises(ValueError, match="monic"):
        AlgebraicExtension(Poly(2*y**2 - x, y, domain=parent))
    with pytest.raises(ValueError, match="square-free"):
        AlgebraicExtension(Poly(y**2, y, domain=parent))
    with pytest.raises(ValueError, match="irreducible"):
        AlgebraicExtension(Poly(y**2 - 1, y, domain=parent))


def test_integral_basis_round_trip() -> None:
    _, extension, basis = algebraic_tower(y**3 - y - x)
    parent = extension.parent_field
    coordinates = (
        parent.from_sympy(1/x),
        parent.from_sympy(x + 1),
        parent.from_sympy(2),
    )

    element = basis.element(coordinates)

    assert basis.coordinates(element) == coordinates
    representation = basis.fraction(element)
    assert representation.denominator == Poly(x, x, domain=QQ)
    assert representation.numerators == (
        Poly(1, x, domain=QQ),
        Poly(x**2 + x, x, domain=QQ),
        Poly(2*x, x, domain=QQ),
    )


def test_algebraic_hermite_reduces_a_quadratic_example() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    field = extension.field
    integrand = field.from_sympy(y/x**2)

    exact, remainder = hermite_reduce_algebraic(
        integrand,
        tower.derivation(),
        basis,
    )

    assert exact == field.from_sympy(-2*y/x)
    assert remainder == field.zero
    assert tower.derivation()(exact) + remainder == integrand


def test_algebraic_hermite_reduces_the_cubic_walkthrough() -> None:
    tower, extension, basis = algebraic_tower(y**3 - y - x)
    field = extension.field
    delta = 27*x**2 - 4
    numerator = (
        27*x**3 - 39*x**2 + 4
        + (4 - 4*x - 18*x**2)*y
        + (4 - 6*x - 9*x**2)*y**2
    )
    integrand = field.from_sympy(numerator/(x**2*delta))

    exact, remainder = hermite_reduce_algebraic(
        integrand,
        tower.derivation(),
        basis,
    )

    assert exact == field.from_sympy((1 + y + y**2)/x)
    assert remainder == field.from_sympy(1/x)
    assert tower.derivation()(exact) + remainder == integrand
    assert basis.fraction(remainder).is_square_free


def test_algebraic_hermite_accepts_a_nonpower_integral_basis() -> None:
    tower, extension, _ = algebraic_tower(y**3 - y - x)
    field = extension.field
    basis = IntegralBasis(
        extension,
        (field.one, field.generator, field.generator + field.generator**2),
    )
    delta = 27*x**2 - 4
    numerator = (
        27*x**3 - 39*x**2 + 4
        + (4 - 4*x - 18*x**2)*y
        + (4 - 6*x - 9*x**2)*y**2
    )
    integrand = field.from_sympy(numerator/(x**2*delta))

    exact, remainder = hermite_reduce_algebraic(
        integrand,
        tower.derivation(),
        basis,
    )

    assert exact == field.from_sympy((1 + y + y**2)/x)
    assert remainder == field.from_sympy(1/x)


def test_algebraic_hermite_reduces_multiple_repeated_factors() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    field = extension.field
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        (1 + y)/(x + 1)**2 + (2 - y)/(x + 2)**3,
    )
    planted_remainder = field.from_sympy(1/(x + 3))
    integrand = derivation(planted_exact) + planted_remainder

    exact, remainder = hermite_reduce_algebraic(integrand, derivation, basis)

    assert derivation(exact) + remainder == integrand
    assert basis.fraction(remainder).is_square_free


@given(
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=2, max_value=4),
)
@settings(max_examples=25, deadline=None)
def test_generated_algebraic_reductions_reconstruct_the_integrand(a, b, power) -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    field = extension.field
    derivation = tower.derivation()
    planted_exact = field.from_sympy((a + b*y)/(x + 1)**(power - 1))
    planted_remainder = field.from_sympy(1/(x + 2))
    integrand = derivation(planted_exact) + planted_remainder

    exact, remainder = hermite_reduce_algebraic(integrand, derivation, basis)

    assert derivation(exact) + remainder == integrand
    assert basis.fraction(remainder).is_square_free
