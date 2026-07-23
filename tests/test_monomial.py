import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    bronstein_residue_criterion,
    hermite_reduce_monomial,
    hermite_reduce_rational_recursive,
    polynomial_reduce_monomial,
    reduce_monomial_reduced_function,
    residue_reduce_monomial,
    split_normal_special,
)


x, t = symbols("x t")


def monomial_tower(generator_derivative) -> DifferentialTower:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
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


def test_tangent_polynomial_reduction() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(t**4, t, domain=parent)

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert exact == Poly(t**3/3 - t, t, domain=parent)
    assert remainder == Poly(1, t, domain=parent)
    assert tower.derivation().polynomial(exact) + remainder == polynomial


def test_polynomial_reduction_differentiates_lower_field_coefficients() -> None:
    tower = monomial_tower(1 + t**2)
    parent = tower.field(0)
    polynomial = Poly(x*t**3, t, domain=parent)

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert exact == Poly(x*t**2/2 - t/2, t, domain=parent)
    assert remainder == Poly(-x*t + 1/2, t, domain=parent)
    assert tower.derivation().polynomial(exact) + remainder == polynomial


@pytest.mark.parametrize(
    "generator_derivative",
    [1 + t**2, t**3 + x, x*t**2 + t + 1],
)
def test_nonlinear_polynomial_reduction_reaches_the_degree_bound(
    generator_derivative,
) -> None:
    tower = monomial_tower(generator_derivative)
    parent = tower.field(0)
    polynomial = Poly((x + 1)*t**8 + x*t**5 + t + 1, t, domain=parent)
    delta = tower.levels[-1].generator_derivative.degree()

    exact, remainder = polynomial_reduce_monomial(
        polynomial,
        tower.derivation(),
    )

    assert tower.derivation().polynomial(exact) + remainder == polynomial
    assert remainder.is_zero or remainder.degree() < delta

    second_exact, second_remainder = polynomial_reduce_monomial(
        remainder,
        tower.derivation(),
    )
    assert second_exact.is_zero
    assert second_remainder == remainder


def test_nonlinear_polynomial_reduction_handles_zero_and_low_degree() -> None:
    tower = monomial_tower(t**3 + x)
    parent = tower.field(0)

    for polynomial in (
        Poly(0, t, domain=parent),
        Poly(x*t**2 + 1, t, domain=parent),
    ):
        exact, remainder = polynomial_reduce_monomial(
            polynomial,
            tower.derivation(),
        )

        assert exact.is_zero
        assert remainder == polynomial


@pytest.mark.parametrize("generator_derivative", [1/x, t, t + 1])
def test_nonlinear_polynomial_reduction_rejects_linear_families(
    generator_derivative,
) -> None:
    tower = monomial_tower(generator_derivative)

    with pytest.raises(ValueError, match=r"degree\(Dt\) >= 2"):
        polynomial_reduce_monomial(
            Poly(t**3, t, domain=tower.field(0)),
            tower.derivation(),
        )


def test_nonlinear_polynomial_reduction_validates_the_polynomial() -> None:
    tower = monomial_tower(1 + t**2)

    with pytest.raises(TypeError, match="SymPy Poly"):
        polynomial_reduce_monomial(t**3, tower.derivation())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="generator does not match"):
        polynomial_reduce_monomial(
            Poly(x**3, x, domain=QQ),
            tower.derivation(),
        )


@given(
    generator_derivative=st.sampled_from(
        (1 + t**2, t**3 + x, x*t**2 + t + 1),
    ),
    coefficient=st.integers(min_value=-4, max_value=4),
    constant=st.integers(min_value=-4, max_value=4),
    exponent=st.integers(min_value=0, max_value=7),
    remainder_linear=st.integers(min_value=-4, max_value=4),
    remainder_constant=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_nonlinear_polynomial_reductions_reconstruct_input(
    generator_derivative,
    coefficient,
    constant,
    exponent,
    remainder_linear,
    remainder_constant,
) -> None:
    tower = monomial_tower(generator_derivative)
    parent = tower.field(0)
    derivation = tower.derivation()
    planted_exact = Poly(
        (coefficient*x + constant)*t**exponent,
        t,
        domain=parent,
    )
    planted_remainder = Poly(
        remainder_linear*t + remainder_constant,
        t,
        domain=parent,
    )
    polynomial = derivation.polynomial(planted_exact) + planted_remainder

    exact, remainder = polynomial_reduce_monomial(polynomial, derivation)

    delta = tower.levels[-1].generator_derivative.degree()
    assert derivation.polynomial(exact) + remainder == polynomial
    assert remainder.is_zero or remainder.degree() < delta


def test_tangent_reduced_function_extracts_the_logarithm_at_infinity() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == 1
    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_coefficient == tower.field(0).convert(QQ(1, 2))
    assert reduction.logarithmic_argument == Poly(
        1 + t**2,
        t,
        domain=tower.field(0),
    )
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert not reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


def test_reduced_function_handles_a_nonunit_derivative_leading_coefficient() -> None:
    tower = monomial_tower(x*(1 + t**2))
    field = tower.field()
    integrand = field.from_sympy(x*t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_coefficient == tower.field(0).convert(QQ(1, 2))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero


@given(
    scale=st.integers(min_value=-5, max_value=5).filter(bool),
    exponent=st.integers(min_value=1, max_value=4),
)
def test_structure_theorems_reject_new_constant_counterexamples(
    scale,
    exponent,
) -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(scale*t**(exponent + 1), t, domain=rational.field()),
            ConstantFieldStatus.ENLARGED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    generator = field.from_sympy(t)
    new_constant = field.from_sympy(
        x + 1/(exponent*scale*t**exponent),
    )
    integrand = field.from_sympy(scale*x*t**exponent)
    normal_factor = field.from_sympy(t - 1)
    logarithmic_integrand = (
        new_constant*derivation(normal_factor)/normal_factor
    )

    assert derivation(new_constant) == field.zero
    # D(new_constant*log(t) - x/exponent) equals the polynomial integrand.
    assert (
        new_constant*derivation(generator)/generator
        - field.from_sympy(Rational(1, exponent))
        == integrand
    )
    with pytest.raises(ValueError, match=r"Const\(K\(t\)\) = Const\(K\)"):
        reduce_monomial_reduced_function(integrand, derivation)
    with pytest.raises(ValueError, match=r"Const\(K\(t\)\) = Const\(K\)"):
        bronstein_residue_criterion(logarithmic_integrand, derivation)


def test_reduced_function_certifies_a_nonconstant_infinity_residue() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(x*t + 3)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_coefficient == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue == tower.field(0).from_sympy(x)
    assert reduction.infinity_residue_derivative == 1
    assert reduction.logarithmic_argument is None
    assert reduction.logarithmic_derivative == field.zero
    assert reduction.remainder == integrand
    assert reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


def test_reduced_function_uses_its_special_denominator() -> None:
    tower = monomial_tower(t**3)
    field = tower.field()
    integrand = field.from_sympy(2*t**2 + 1/t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.special_numerator == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert reduction.special_denominator == Poly(
        t,
        t,
        domain=tower.field(0),
    )
    assert reduction.logarithmic_coefficient == 2
    assert reduction.logarithmic_argument == reduction.special_denominator
    assert reduction.logarithmic_derivative == field.from_sympy(2*t**2)
    assert reduction.remainder == field.from_sympy(1/t)


def test_reduced_function_can_use_a_supplied_special_polynomial() -> None:
    # q = t - x is special because Dq = t*q, but it is not a factor of Dt.
    tower = monomial_tower(t**2 - x*t + 1)
    parent = tower.field(0)
    field = tower.field()
    special = Poly(t - x, t, domain=parent)

    reduction = reduce_monomial_reduced_function(
        field.from_sympy(t),
        tower.derivation(),
        special,
    )

    assert reduction.logarithmic_coefficient == 1
    assert reduction.logarithmic_argument == special
    assert reduction.logarithmic_derivative == field.from_sympy(t)
    assert reduction.remainder == field.zero


def test_reduced_function_defers_when_no_special_polynomial_is_known() -> None:
    tower = monomial_tower(t**2 + x)
    field = tower.field()
    integrand = field.from_sympy(t)

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.infinity_residue == 1
    assert reduction.infinity_residue_derivative == 0
    assert reduction.logarithmic_argument is None
    assert reduction.remainder == integrand
    assert not reduction.has_infinity_obstruction
    assert reduction.needs_special_polynomial


def test_reduced_function_leaves_lower_degree_and_special_parts() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(4*t + 3 + 1/(1 + t**2))

    reduction = reduce_monomial_reduced_function(
        integrand,
        tower.derivation(),
    )

    assert reduction.logarithmic_coefficient == 2
    assert reduction.logarithmic_derivative == field.from_sympy(4*t)
    assert reduction.remainder == field.from_sympy(3 + 1/(1 + t**2))
    assert reduction.logarithmic_derivative + reduction.remainder == integrand


def test_reduced_function_requires_the_previous_reductions() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()

    with pytest.raises(ValueError, match="normal residues"):
        reduce_monomial_reduced_function(
            field.from_sympy(t + 1/(t - 1)),
            derivation,
        )
    with pytest.raises(ValueError, match="polynomial reduction"):
        reduce_monomial_reduced_function(field.from_sympy(t**2), derivation)
    with pytest.raises(ValueError, match="must divide"):
        reduce_monomial_reduced_function(
            field.from_sympy(t),
            derivation,
            Poly(t - 1, t, domain=tower.field(0)),
        )


@given(
    example=st.sampled_from(
        (
            (1 + t**2, 1 + t**2),
            (t**3, t),
            (t**3, t**2),
        ),
    ),
    logarithmic_coefficient=st.integers(min_value=-4, max_value=4),
    background=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_generated_reduced_function_logarithms_reconstruct_input(
    example,
    logarithmic_coefficient,
    background,
) -> None:
    generator_derivative, special_expression = example
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    special = field.from_sympy(special_expression)
    planted_logarithmic_derivative = (
        logarithmic_coefficient*derivation(special)/special
    )
    integrand = planted_logarithmic_derivative + background

    reduction = reduce_monomial_reduced_function(integrand, derivation)

    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    assert reduction.remainder == field.from_sympy(background)
    assert not reduction.has_infinity_obstruction
    assert not reduction.needs_special_polynomial


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


def test_residue_algorithms_require_preserved_constants() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(1/x, t, domain=rational.field()),
        ),
    )
    integrand = tower.field().from_sympy(1/t)

    with pytest.raises(ValueError, match="status is unknown"):
        bronstein_residue_criterion(integrand, tower.derivation())
    with pytest.raises(ValueError, match="status is unknown"):
        residue_reduce_monomial(integrand, tower.derivation())


def test_bronstein_criterion_rejects_the_logarithmic_integral_residue() -> None:
    # With t = log(x), 1/t has residue x because D(t) = 1/x.  Since Dx != 0,
    # Liouville's theorem rules out an elementary logarithmic term.
    tower = monomial_tower(1/x)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/t),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - x, z, domain=tower.field(0))
    assert certificate.kappa_residue_polynomial == Poly(
        -1,
        z,
        domain=tower.field(0),
    )
    assert not certificate.all_residues_constant
    assert certificate.factors[0].constant_factor == Poly(
        1,
        z,
        domain=tower.field(0),
    )
    assert certificate.factors[0].nonconstant_factor == Poly(
        z - x,
        z,
        domain=tower.field(0),
    )


def test_bronstein_criterion_accepts_log_log_residue() -> None:
    # 1/(x*t) has the constant residue 1 and integrates to log(t).
    tower = monomial_tower(1/x)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/(x*t)),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1/x, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - 1, z, domain=tower.field(0))
    assert certificate.kappa_residue_polynomial.is_zero
    assert certificate.all_residues_constant
    assert certificate.factors[0].constant_factor == Poly(
        z - 1,
        z,
        domain=tower.field(0),
    )
    assert certificate.factors[0].nonconstant_factor == Poly(
        1,
        z,
        domain=tower.field(0),
    )


def test_bronstein_criterion_splits_constant_and_nonconstant_roots() -> None:
    tower = monomial_tower(1/x)
    z = symbols("z")
    # The residues at t=0 and t=1 are respectively 1 and x.
    remainder = tower.field().from_sympy(
        1/(x*t) + 1/(t - 1),
    )

    certificate = bronstein_residue_criterion(
        remainder,
        tower.derivation(),
        z,
    )

    factor = certificate.factors[0]
    assert certificate.residue_polynomial == Poly(
        (z - 1)*(z - x),
        z,
        domain=tower.field(0),
    )
    assert factor.constant_factor == Poly(z - 1, z, domain=tower.field(0))
    assert factor.nonconstant_factor == Poly(z - x, z, domain=tower.field(0))
    assert not certificate.all_residues_constant


def test_bronstein_criterion_isolates_normal_from_special_fraction() -> None:
    # For Dt=t, t is special and t-1 is normal:
    # 1/(t*(t-1)) = 1/(t-1) - 1/t.
    tower = monomial_tower(t)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(1/(t*(t - 1))),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator == Poly(1, t, domain=tower.field(0))
    assert certificate.normal_denominator == Poly(t - 1, t, domain=tower.field(0))
    assert certificate.residue_polynomial == Poly(z - 1, z, domain=tower.field(0))
    assert certificate.all_residues_constant


def test_bronstein_criterion_ignores_polynomial_and_special_only_parts() -> None:
    tower = monomial_tower(t)
    z = symbols("z")

    certificate = bronstein_residue_criterion(
        tower.field().from_sympy(t**2 + 1/t),
        tower.derivation(),
        z,
    )

    assert certificate.normal_numerator.is_zero
    assert certificate.normal_denominator.is_one
    assert certificate.residue_polynomial.is_one
    assert certificate.kappa_residue_polynomial.is_zero
    assert certificate.factors == ()
    assert certificate.all_residues_constant


def test_bronstein_criterion_requires_prior_hermite_reduction() -> None:
    tower = monomial_tower(1/x)

    with pytest.raises(ValueError, match="Hermite-reduced normal denominator"):
        bronstein_residue_criterion(
            tower.field().from_sympy(1/t**2),
            tower.derivation(),
        )
    with pytest.raises(ValueError, match="new over the tower field"):
        bronstein_residue_criterion(
            tower.field().from_sympy(1/t),
            tower.derivation(),
            x,
        )


@pytest.mark.parametrize(
    ("generator_derivative", "normal_factor"),
    [
        (1/x, t),
        (t, t - 1),
        (1 + t**2, t),
        (t**3 + x, t - 1),
    ],
)
def test_bronstein_criterion_across_monomial_families(
    generator_derivative,
    normal_factor,
) -> None:
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    logarithmic_derivative = tower.derivation()(field.from_sympy(normal_factor))
    logarithmic_derivative /= field.from_sympy(normal_factor)

    constant_certificate = bronstein_residue_criterion(
        2*logarithmic_derivative,
        tower.derivation(),
    )
    variable_certificate = bronstein_residue_criterion(
        field.from_sympy(x)*logarithmic_derivative,
        tower.derivation(),
    )

    assert constant_certificate.all_residues_constant
    assert not variable_certificate.all_residues_constant


@given(
    constant_residue=st.integers(min_value=-3, max_value=3),
    slope=st.integers(min_value=-3, max_value=3),
    intercept=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_bronstein_criterion_matches_planted_residues(
    constant_residue,
    slope,
    intercept,
) -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    variable_residue = slope*x + intercept
    remainder = field.from_sympy(
        constant_residue/(x*t)
        + variable_residue/(x*(t - 1)),
    )

    certificate = bronstein_residue_criterion(
        remainder,
        tower.derivation(),
    )

    assert certificate.all_residues_constant is (slope == 0)


def test_residue_reduction_extracts_log_log_term() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*t))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand == Poly(t, t, domain=tower.field(0))
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_preserves_nonconstant_residue_obstruction() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(1/t)

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert reduction.logarithmic_parts == ()
    assert reduction.logarithmic_derivative == field.zero
    assert reduction.remainder == integrand
    assert not reduction.all_normal_residues_removed


def test_residue_reduction_extracts_only_constant_residue_roots() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*t) + 1/(t - 1))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.as_expr() == t
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == field.from_sympy(1/(x*t))
    assert reduction.remainder == field.from_sympy(1/(t - 1))
    assert not reduction.all_normal_residues_removed

    remaining_certificate = bronstein_residue_criterion(
        reduction.remainder,
        tower.derivation(),
        z,
    )
    assert remaining_certificate.factors[0].constant_factor.is_one
    assert remaining_certificate.factors[0].nonconstant_factor == Poly(
        z - x,
        z,
        domain=tower.field(0),
    )


def test_residue_reduction_traces_algebraic_constant_residues() -> None:
    # The residues are +/-i/2.  Their encoded logarithmic sum differentiates
    # to the input without constructing either algebraic constant explicitly.
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    integrand = field.from_sympy(1/(x*(t**2 + 1)))

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.as_expr() == t + 2*z
    assert residue_factor == Poly(z**2 + 1/4, z, domain=tower.field(0))
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_traces_degree_five_constant_residues() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    integrand = field.from_sympy(
        (t**2 + 1)/(x*(t**5 + t + 1)),
    )

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand.degree() == 1
    assert residue_factor.degree() == 5
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero
    assert reduction.all_normal_residues_removed


def test_residue_reduction_groups_repeated_equal_residues() -> None:
    tower = monomial_tower(1/x)
    field = tower.field()
    z = symbols("z")
    normal_factor = field.from_sympy(t*(t - 1))
    integrand = tower.derivation()(normal_factor)/normal_factor

    reduction = residue_reduce_monomial(integrand, tower.derivation(), z)

    assert len(reduction.logarithmic_parts) == 1
    logand, residue_factor = reduction.logarithmic_parts[0]
    assert logand == Poly(t*(t - 1), t, domain=tower.field(0))
    assert residue_factor == Poly(z - 1, z, domain=tower.field(0))
    assert reduction.certificate.factors[0].multiplicity == 2
    assert reduction.logarithmic_derivative == integrand
    assert reduction.remainder == field.zero


def test_residue_reduction_leaves_only_exponential_special_terms() -> None:
    tower = monomial_tower(t)
    field = tower.field()
    integrand = field.from_sympy(1/(t*(t - 1)))

    reduction = residue_reduce_monomial(integrand, tower.derivation())

    assert reduction.logarithmic_derivative == field.from_sympy(t/(t - 1))
    assert reduction.remainder == field.from_sympy(-1 - 1/t)
    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    remaining_certificate = bronstein_residue_criterion(
        reduction.remainder,
        tower.derivation(),
    )
    assert remaining_certificate.normal_denominator.is_one


@given(
    example=st.sampled_from(
        (
            (1/x, t**2 + 1),
            (t, t**2 - 1),
            (1 + t**2, t),
            (t**3 + x, t - 1),
        ),
    ),
    residue_value=st.integers(min_value=-4, max_value=4),
    background_constant=st.integers(min_value=-4, max_value=4),
    background_linear=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_residue_reductions_reconstruct_planted_logarithms(
    example,
    residue_value,
    background_constant,
    background_linear,
) -> None:
    generator_derivative, normal_factor_expression = example
    tower = monomial_tower(generator_derivative)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(normal_factor_expression)
    planted_logarithmic_derivative = (
        residue_value*derivation(normal_factor)/normal_factor
    )
    background = field.from_sympy(background_linear*t + background_constant)
    integrand = planted_logarithmic_derivative + background

    reduction = residue_reduce_monomial(integrand, derivation)

    assert reduction.logarithmic_derivative == planted_logarithmic_derivative
    assert reduction.remainder == background
    assert reduction.logarithmic_derivative + reduction.remainder == integrand
    assert reduction.all_normal_residues_removed
