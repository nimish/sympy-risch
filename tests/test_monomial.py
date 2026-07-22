import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    DifferentialTower,
    MonomialExtension,
    hermite_reduce_monomial,
    hermite_reduce_rational_recursive,
    split_normal_special,
)


x, t = symbols("x t")


def monomial_tower(generator_derivative) -> DifferentialTower:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    return rational.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=rational.field()),
        ),
    )


def test_exponential_normal_special_decomposition() -> None:
    tower = monomial_tower(t)
    parent = tower.field(0)
    polynomial = Poly(6*t**2*(t - 1)**3*(t + 1), t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(6*(t - 1)**3*(t + 1), t, domain=parent)
    assert special == Poly(t**2, t, domain=parent)
    assert normal*special == polynomial


def test_tangent_normal_special_decomposition() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(t**2*(1 + t**2)**3, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(t**2, t, domain=parent)
    assert special == Poly((1 + t**2)**3, t, domain=parent)


def test_coefficient_derivatives_expose_a_special_new_constant() -> None:
    # With Dx = Dt = 1, D(t - x) = 0.  This checks that decomposition uses
    # the total tower derivative rather than only differentiation in t.
    tower = monomial_tower(1)
    parent = tower.field(0)
    polynomial = Poly((t - x)**2*(t + 1)**3, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly((t + 1)**3, t, domain=parent)
    assert special == Poly((t - x)**2, t, domain=parent)


def test_general_polynomial_monomials_use_the_same_decomposition() -> None:
    tower = monomial_tower(t**3)
    parent = tower.field(0)
    polynomial = Poly(t**4*(t - 1)**2, t, domain=parent)

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly((t - 1)**2, t, domain=parent)
    assert special == Poly(t**4, t, domain=parent)


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=0, max_value=5),
    st.integers(min_value=0, max_value=5),
)
def test_generated_exponential_decompositions_reconstruct_the_input(
    coefficient,
    special_multiplicity,
    normal_multiplicity,
) -> None:
    tower = monomial_tower(t)
    parent = tower.field(0)
    polynomial = Poly(
        coefficient*t**special_multiplicity*(t - 1)**normal_multiplicity,
        t,
        domain=parent,
    )

    normal, special = split_normal_special(polynomial, tower.derivation())

    assert normal == Poly(
        coefficient*(t - 1)**normal_multiplicity,
        t,
        domain=parent,
    )
    assert special == Poly(t**special_multiplicity, t, domain=parent)
    assert normal*special == polynomial


def test_normal_special_decomposition_validates_its_polynomial() -> None:
    tower = monomial_tower(1/x)

    with pytest.raises(TypeError, match="SymPy Poly"):
        split_normal_special(t, tower.derivation())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero polynomial"):
        split_normal_special(
            Poly(0, t, domain=tower.field(0)),
            tower.derivation(),
        )
    with pytest.raises(ValueError, match="generator does not match"):
        split_normal_special(Poly(x + 1, x, domain=QQ), tower.derivation())


def denominator_parts(element, tower: DifferentialTower) -> tuple[Poly, Poly]:
    extension = tower.levels[tower.top]
    assert isinstance(extension, MonomialExtension)
    _, denominator_expression = element.as_expr().as_numer_denom()
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    return split_normal_special(denominator, tower.derivation())


def assert_normal_part_is_square_free(
    element,
    tower: DifferentialTower,
) -> None:
    normal, _ = denominator_parts(element, tower)
    assert all(
        multiplicity == 1
        for _, multiplicity in normal.sqf_list()[1]
    )


def test_monomial_hermite_reduces_a_logarithmic_repeated_pole() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(x*t**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == field.from_sympy(-1/t)
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand


def test_monomial_hermite_preserves_an_exponential_special_part() -> None:
    tower = monomial_tower(t)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(t**2*(t - 1)**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)

    assert derivation(exact) + remainder == integrand
    assert normal == Poly(t - 1, t, domain=tower.field(0))
    assert special == Poly(t**2, t, domain=tower.field(0))


def test_monomial_hermite_preserves_a_tangent_special_part() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(t**2*(1 + t**2)**2))

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)

    assert derivation(exact) + remainder == integrand
    assert normal == Poly(1, t, domain=tower.field(0))
    assert special == Poly((1 + t**2)**2, t, domain=tower.field(0))


@pytest.mark.parametrize("expression", [t**3 + x*t, 1/t**3, 1/(t - 1)])
def test_monomial_hermite_leaves_polynomial_and_special_terms_for_later(
    expression,
) -> None:
    tower = monomial_tower(t)
    field = tower.field()
    integrand = field.from_sympy(expression)

    exact, remainder = hermite_reduce_monomial(
        integrand,
        tower.derivation(),
    )

    assert exact == field.zero
    assert remainder == integrand


@given(st.integers(min_value=1, max_value=6))
def test_generated_logarithmic_exact_derivatives_are_recovered(power) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(t**(-power))
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero


def test_monomial_hermite_requires_a_monomial_level() -> None:
    constants = DifferentialTower(QQ)

    with pytest.raises(TypeError, match="monomial level"):
        hermite_reduce_monomial(QQ.one, constants.derivation())


@given(
    generator_derivative=st.sampled_from((1/x, t, 1 + t**2, t**3 + x)),
    left_power=st.integers(min_value=1, max_value=5),
    right_power=st.integers(min_value=1, max_value=5),
    coefficient=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_planted_derivatives_across_monomial_families(
    generator_derivative,
    left_power,
    right_power,
    coefficient,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        (coefficient*x + 1)/(t - 1)**left_power
        + (x - coefficient)/(t + 1)**right_power,
    )
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand


@given(
    family=st.sampled_from(
        (
            (t, t),
            (1 + t**2, 1 + t**2),
            (t**3, t),
        ),
    ),
    special_power=st.integers(min_value=1, max_value=4),
    normal_power=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_generated_mixed_special_and_normal_denominators(
    family,
    special_power,
    normal_power,
) -> None:
    generator_derivative, special_factor = family
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        1/(special_factor**special_power*(t - 1)**normal_power),
    )
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)
    normal, special = denominator_parts(remainder, tower)
    expected_special = Poly(
        special_factor**special_power,
        t,
        domain=tower.field(0),
    )

    assert derivation(exact) + remainder == integrand
    assert all(
        multiplicity == 1
        for _, multiplicity in normal.sqf_list()[1]
    )
    assert expected_special.rem(special).is_zero

    second_exact, second_remainder = hermite_reduce_monomial(
        remainder,
        derivation,
    )
    assert second_exact == field.zero
    assert second_remainder == remainder


@given(
    left_power=st.integers(min_value=2, max_value=5),
    right_power=st.integers(min_value=2, max_value=5),
    linear=st.integers(min_value=-5, max_value=5),
    constant=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_monomial_reducer_agrees_with_classical_rational_hermite_reduction(
    left_power,
    right_power,
    linear,
    constant,
) -> None:
    tower = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    integrand = (linear*x + constant) / (
        (x - 1)**left_power*(x + 1)**right_power
    )

    exact, remainder = hermite_reduce_monomial(
        integrand,
        tower.derivation(),
    )
    rational_exact, rational_remainder = hermite_reduce_rational_recursive(
        integrand,
        x,
    )

    assert cancel(exact.as_expr() - rational_exact) == 0
    assert cancel(remainder.as_expr() - rational_remainder) == 0
    assert_normal_part_is_square_free(remainder, tower)


def test_monomial_hermite_handles_multiple_multiplicities_and_lower_terms() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy(
        (x + 1)/(t - 2)**4 + (x - 1)/(t + 2)**2,
    )
    planted_remainder = field.from_sympy(
        t**2 + 1/x**2 + 1/(t - 1),
    )
    integrand = derivation(planted_exact) + planted_remainder

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == planted_remainder
    assert derivation(exact) + remainder == integrand
    assert_normal_part_is_square_free(remainder, tower)


@pytest.mark.parametrize(
    ("generator_derivative", "normal_factor"),
    [
        (1/x, t**2 + x),
        (t, t**2 + t + 1),
        (1 + t**2, t**2 + t + 2),
        (t**3 + x, t**2 + x*t + 1),
    ],
)
def test_monomial_hermite_reduces_nonlinear_normal_factors(
    generator_derivative,
    normal_factor,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact = field.from_sympy((x + 1)/normal_factor**3)
    integrand = derivation(planted_exact)

    exact, remainder = hermite_reduce_monomial(integrand, derivation)

    assert exact == planted_exact
    assert remainder == field.zero
    assert derivation(exact) + remainder == integrand
