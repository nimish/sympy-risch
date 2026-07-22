from math import prod

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    hermite_reduce_algebraic,
    lazy_hermite_reduce_algebraic,
    recognize_simple_radical,
    simple_radical_integral_basis,
)


x, y = symbols("x y")
distinct_linear_shifts = st.tuples(
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=-3, max_value=3),
).filter(lambda shifts: shifts[0] != shifts[1])


@st.composite
def hidden_radical_parameters(draw):
    factor_count = draw(st.integers(min_value=1, max_value=2))
    shifts = draw(st.lists(
        st.integers(min_value=-3, max_value=3),
        min_size=factor_count + 1,
        max_size=factor_count + 1,
        unique=True,
    ))
    return shifts[:-1], shifts[-1]


def algebraic_tower(modulus, constants=QQ):
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
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


def test_simple_radical_recognition_normalizes_hidden_square_factors() -> None:
    _, extension, _ = algebraic_tower(y**2 - x**2*(x + 1))
    field = extension.field

    radical = recognize_simple_radical(extension)

    assert radical is not None
    assert radical.degree == 2
    assert radical.radicand == extension.parent_field.from_sympy(x**2*(x + 1))
    assert radical.extracted_factor == Poly(x, x, domain=QQ)
    assert radical.normalized_radicand == Poly(x + 1, x, domain=QQ)
    assert radical.normalized_generator == field.from_sympy(y/x)
    assert radical.normalized_generator**2 == field.from_sympy(x + 1)


def test_simple_radical_basis_handles_a_rational_radicand() -> None:
    _, extension, _ = algebraic_tower(y**3 - x/(x + 1))
    field = extension.field
    z = field.from_sympy(y*(x + 1))

    radical = recognize_simple_radical(extension)
    basis = simple_radical_integral_basis(extension)

    assert radical is not None
    assert radical.extracted_factor == Poly(1, x, domain=QQ)
    assert radical.normalized_radicand == Poly(x*(x + 1)**2, x, domain=QQ)
    assert radical.normalized_generator == z
    assert basis.elements == (
        field.one,
        z,
        field.from_sympy(y**2*(x + 1)),
    )


def test_simple_radical_recognition_rejects_a_mixed_minimal_polynomial() -> None:
    _, extension, _ = algebraic_tower(y**3 - y - x)

    assert recognize_simple_radical(extension) is None
    with pytest.raises(ValueError, match="not a simple radical"):
        simple_radical_integral_basis(extension)


def test_algebraic_hermite_uses_the_simple_radical_integral_basis() -> None:
    tower, extension, _ = algebraic_tower(y**2 - x**2*(x + 1))
    field = extension.field
    derivation = tower.derivation()
    basis = simple_radical_integral_basis(extension)
    planted_exact = field.from_sympy(y/x**2)
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_algebraic(integrand, derivation, basis)

    assert derivation(exact) + remainder == integrand
    assert remainder == field.zero
    assert basis.fraction(remainder).is_square_free


def test_lazy_hermite_exposes_a_hidden_quadratic_integral_element() -> None:
    # y is integral, but (1, y) is incomplete because z = y/x is also
    # integral: z**2 = x + 1.  In that basis y/x**2 = z/x has only a simple
    # pole, so refinement alone completes the reduction.
    tower, extension, power_basis = algebraic_tower(y**2 - x**2*(x + 1))
    field = extension.field
    integrand = field.from_sympy(y/x**2)

    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        integrand,
        tower.derivation(),
        power_basis,
    )

    assert exact == field.zero
    assert remainder == integrand
    assert refined_basis.fraction(field.from_sympy(y/x)).denominator.degree() == 0
    assert refined_basis.fraction(remainder).denominator == Poly(x, x, domain=QQ)


def test_lazy_hermite_reduces_after_quadratic_basis_refinement() -> None:
    # With z = y/x and z**2 = x + 1,
    #
    #   z/x**2 = D(-z/x) + z/(2*x*(x + 1)).
    #
    # The first term is returned as exact and the remainder has squarefree
    # denominator x*(x + 1) in the refined basis.
    tower, extension, power_basis = algebraic_tower(y**2 - x**2*(x + 1))
    field = extension.field
    derivation = tower.derivation()
    integrand = field.from_sympy(y/x**3)

    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        integrand,
        derivation,
        power_basis,
    )

    assert exact == field.from_sympy(-y/x**2)
    assert remainder == field.from_sympy(y/(2*x**2*(x + 1)))
    assert refined_basis.fraction(remainder).denominator == Poly(
        x*(x + 1),
        x,
        domain=QQ,
    )
    assert derivation(exact) + remainder == integrand


def test_lazy_algebraic_hermite_refines_bronsteins_example() -> None:
    # Bronstein, ISSAC'98 tutorial, Examples 1--2.  The failed solution has
    # denominator Q = x while V = x.  Theorem 2 finds an integral element and
    # enlarges (1, y, y**2, y**3) to the module generated by
    # (1, y, y**2, y**3/x), where the integrand's denominator is squarefree.
    tower, extension, power_basis = algebraic_tower(
        y**4 + (x**2 + x)*y - x**2,
    )
    field = extension.field
    derivation = tower.derivation()
    integrand = field.from_sympy(y**3/x**2)

    with pytest.raises(ValueError, match="not an integral basis"):
        hermite_reduce_algebraic(integrand, derivation, power_basis)

    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        integrand,
        derivation,
        power_basis,
    )

    expected_basis = IntegralBasis(
        extension,
        (
            field.one,
            field.generator,
            field.generator**2,
            field.from_sympy(y**3/x),
        ),
    )
    assert all(
        expected_basis.fraction(element).denominator.degree() == 0
        for element in refined_basis.elements
    )
    assert all(
        refined_basis.fraction(element).denominator.degree() == 0
        for element in expected_basis.elements
    )
    assert exact == field.zero
    assert remainder == integrand
    assert refined_basis.fraction(remainder).denominator == Poly(x, x, domain=QQ)
    assert derivation(exact) + remainder == integrand


@given(
    st.integers(min_value=2, max_value=4),
    distinct_linear_shifts,
)
@settings(max_examples=12, deadline=None)
def test_generated_simple_radicals_remove_exact_power_factors(degree, shifts) -> None:
    power_shift, radicand_shift = shifts
    power_factor = x + power_shift
    reduced_radicand = x + radicand_shift
    _, extension, _ = algebraic_tower(
        y**degree - power_factor**degree*reduced_radicand,
    )
    field = extension.field

    radical = recognize_simple_radical(extension)
    basis = simple_radical_integral_basis(extension)

    assert radical is not None
    assert radical.extracted_factor == Poly(power_factor, x, domain=QQ)
    assert radical.normalized_radicand == Poly(reduced_radicand, x, domain=QQ)
    assert radical.normalized_generator == field.from_sympy(y/power_factor)
    assert basis.elements == tuple(
        field.from_sympy((y/power_factor)**power)
        for power in range(degree)
    )


@given(
    st.integers(min_value=2, max_value=4),
    hidden_radical_parameters(),
)
@settings(max_examples=18, deadline=None)
def test_generated_lazy_reductions_rediscover_hidden_radicals(
    degree,
    parameters,
) -> None:
    # Generate z**degree = h and a squarefree p with one or two factors, then
    # hide the integral generator as y = p*z.  The power basis in y misses z.
    factor_shifts, radicand_shift = parameters
    power_factor = prod(x + shift for shift in factor_shifts)
    reduced_radicand = x + radicand_shift
    tower, extension, power_basis = algebraic_tower(
        y**degree - power_factor**degree*reduced_radicand,
    )
    field = extension.field
    derivation = tower.derivation()
    hidden_generator = field.from_sympy(y/power_factor)

    # y/p**2 = z/p only appears to have a double pole before refinement.
    simple_pole = field.from_sympy(y/power_factor**2)
    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        simple_pole,
        derivation,
        power_basis,
    )

    assert exact == field.zero
    assert remainder == simple_pole
    assert refined_basis.fraction(hidden_generator).denominator.degree() == 0
    assert refined_basis.fraction(remainder).denominator == Poly(
        power_factor,
        x,
        domain=QQ,
    )

    # Since z' = z/(degree*h),
    # z*p'/p**2 = D(-z/p) + z/(degree*h*p).
    repeated_pole = field.from_sympy(
        y*power_factor.diff(x)/power_factor**3,
    )
    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        repeated_pole,
        derivation,
        power_basis,
    )

    assert exact == field.from_sympy(-y/power_factor**2)
    assert remainder == field.from_sympy(
        y/(degree*reduced_radicand*power_factor**2),
    )
    assert refined_basis.fraction(remainder).denominator == Poly(
        power_factor*reduced_radicand,
        x,
        domain=QQ,
    )
    assert derivation(exact) + remainder == repeated_pole
    assert refined_basis.fraction(remainder).is_square_free


@given(
    st.integers(min_value=2, max_value=4),
    distinct_linear_shifts,
)
@settings(max_examples=6, deadline=None)
def test_generated_lazy_reductions_handle_mixed_algebraic_relations(
    degree,
    shifts,
) -> None:
    # Start with the non-radical relation z**degree - z - h = 0 and hide z as
    # y = p*z.  Multiplication by p**degree gives the transformed monic
    # polynomial below, which contains a nonzero linear y term.  Unlike a
    # simple radical, Dz = Dh/(degree*z**(degree - 1) - 1) mixes basis powers.
    power_shift, relation_shift = shifts
    power_factor = x + power_shift
    relation_term = x + relation_shift
    tower, extension, power_basis = algebraic_tower(
        y**degree
        - power_factor**(degree - 1)*y
        - relation_term*power_factor**degree,
    )
    field = extension.field
    derivation = tower.derivation()
    hidden_integral_element = field.from_sympy(
        y**(degree - 1)/power_factor**(degree - 1),
    )
    integrand = field.from_sympy(y**(degree - 1)/power_factor**degree)

    assert recognize_simple_radical(extension) is None
    with pytest.raises(ValueError, match="not an integral basis"):
        hermite_reduce_algebraic(integrand, derivation, power_basis)

    exact, remainder, refined_basis = lazy_hermite_reduce_algebraic(
        integrand,
        derivation,
        power_basis,
    )

    assert exact == field.zero
    assert remainder == integrand
    assert refined_basis.fraction(hidden_integral_element).denominator.degree() == 0
    assert refined_basis.fraction(remainder).denominator == Poly(
        power_factor,
        x,
        domain=QQ,
    )
    assert derivation(exact) + remainder == integrand


def test_algebraic_hermite_over_an_algebraic_constant_field() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower, extension, basis = algebraic_tower(y**2 - x, constants)
    field = extension.field
    derivation = tower.derivation()
    integrand = field.from_sympy((sqrt(2) + y)/x**2)

    exact, remainder = hermite_reduce_algebraic(integrand, derivation, basis)

    assert exact == field.from_sympy(-sqrt(2)/x - 2*y/x)
    assert remainder == field.zero
    assert derivation(exact) - integrand == field.zero


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
