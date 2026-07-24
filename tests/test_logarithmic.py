"""Rational logarithmic-derivative decisions and certificates."""

from math import gcd

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, cancel, sqrt, symbols
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from sympy_risch import (
    solve_rational_exact_derivative,
    solve_rational_logarithmic_derivative,
)


x = symbols("x")


def factor_certificates(result):
    """Index factor certificates by their polynomial expressions."""
    return {
        certificate.factor.as_expr(): certificate
        for certificate in result.factors
    }


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        (1, 1),
        (x, x),
        (1/x, 1/x),
        (x**2/(x - 1)**3, x**2/(x - 1)**3),
        (
            x**2*(x**2 + 1)**2/(x - 1)**3,
            x**2*(x**2 + 1)**2/(x - 1)**3,
        ),
        ((x**5 + x + 1)**7, (x**5 + x + 1)**7),
    ],
)
def test_rational_logarithmic_solver_recovers_canonical_arguments(
    argument,
    expected,
) -> None:
    integrand = cancel(cancel(argument).diff(x)/argument)

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert result.has_solution
    assert cancel(result.argument - expected) == 0
    assert cancel(result.argument.diff(x)/result.argument - integrand) == 0
    assert not result.has_polynomial_obstruction
    assert not result.has_repeated_pole_obstruction
    assert not result.has_nonlogarithmic_residue_obstruction
    assert not result.has_nonintegral_residue_obstruction
    assert result.minimal_radical_power == 1


def test_rational_logarithmic_solver_records_factor_valuations() -> None:
    integrand = 2/x - 3/(x - 1) + 4*x/(x**2 + 1)

    result = solve_rational_logarithmic_derivative(integrand, x)
    certificates = factor_certificates(result)

    assert result.has_solution
    assert certificates[x].multiplicity == QQ(2)
    assert certificates[x - 1].multiplicity == QQ(-3)
    assert certificates[x**2 + 1].multiplicity == QQ(2)
    for certificate in result.factors:
        assert certificate.has_logarithmic_residue
        assert certificate.has_integral_multiplicity
        assert certificate.partial_numerator == (
            certificate.factor.diff().mul_ground(
                certificate.multiplicity,
            )
        )


def test_logarithmic_and_ordinary_exactness_are_distinct() -> None:
    logarithmic_only = solve_rational_logarithmic_derivative(1/x, x)
    ordinary_failure = solve_rational_exact_derivative(1/x, x)
    ordinary_only = solve_rational_exact_derivative(-1/x**2, x)
    logarithmic_failure = solve_rational_logarithmic_derivative(-1/x**2, x)

    assert logarithmic_only.has_solution
    assert logarithmic_only.argument == x
    assert not ordinary_failure.has_solution
    assert ordinary_only.has_solution
    assert ordinary_only.antiderivative == 1/x
    assert not logarithmic_failure.has_solution
    assert logarithmic_failure.has_repeated_pole_obstruction


def test_fractional_valuation_is_retained_as_a_radical_certificate() -> None:
    result = solve_rational_logarithmic_derivative(1/(2*x), x)
    certificate = result.factors[0]

    assert not result.has_solution
    assert result.argument is None
    assert certificate.factor == Poly(x, x, domain=QQ)
    assert certificate.multiplicity == QQ(1, 2)
    assert certificate.has_logarithmic_residue
    assert not certificate.has_integral_multiplicity
    assert result.has_nonintegral_residue_obstruction
    assert not result.has_nonlogarithmic_residue_obstruction
    assert result.minimal_radical_power == 2


def test_nonlogarithmic_residue_is_distinguished_from_fractional_valuation() -> None:
    result = solve_rational_logarithmic_derivative(1/(x**2 + 1), x)
    certificate = result.factors[0]

    assert not result.has_solution
    assert certificate.factor == Poly(x**2 + 1, x, domain=QQ)
    assert certificate.partial_numerator == Poly(1, x, domain=QQ)
    assert certificate.multiplicity is None
    assert not certificate.has_logarithmic_residue
    assert result.has_nonlogarithmic_residue_obstruction
    assert not result.has_nonintegral_residue_obstruction
    assert result.minimal_radical_power is None


@pytest.mark.parametrize(
    "integrand",
    [
        1/x**2,
        (x + 1)/x**3,
        1/x**2 + 1/(x - 1),
    ],
)
def test_repeated_poles_are_logarithmic_derivative_obstructions(
    integrand,
) -> None:
    result = solve_rational_logarithmic_derivative(integrand, x)

    assert not result.has_solution
    assert result.argument is None
    assert result.has_repeated_pole_obstruction
    assert any(
        multiplicity > 1
        for _, multiplicity in result.denominator_factors
    )


@pytest.mark.parametrize(
    "integrand",
    [
        1,
        x**3 - 2*x,
        1 + 1/x,
        x + 2/x - 3/(x - 1),
    ],
)
def test_polynomial_parts_are_logarithmic_derivative_obstructions(
    integrand,
) -> None:
    result = solve_rational_logarithmic_derivative(integrand, x)

    assert not result.has_solution
    assert result.argument is None
    assert result.has_polynomial_obstruction
    assert not result.polynomial_part.is_zero


def test_input_cancellation_occurs_before_pole_analysis() -> None:
    integrand = (x + 1)/(x*(x + 1))

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert result.has_solution
    assert result.argument == x
    assert result.denominator_factors == (
        (Poly(x, x, domain=QQ), 1),
    )


@given(
    x_valuation=st.integers(min_value=-6, max_value=6),
    linear_valuation=st.integers(min_value=-6, max_value=6),
    quadratic_valuation=st.integers(min_value=-6, max_value=6),
)
@settings(max_examples=45, deadline=None)
def test_generated_integer_valuations_reconstruct_rational_arguments(
    x_valuation,
    linear_valuation,
    quadratic_valuation,
) -> None:
    argument = cancel(
        x**x_valuation
        * (x - 1)**linear_valuation
        * (x**2 + x + 1)**quadratic_valuation
    )
    integrand = cancel(argument.diff(x)/argument)

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert result.has_solution
    assert cancel(result.argument/argument - 1) == 0
    assert cancel(result.argument.diff(x)/result.argument - integrand) == 0
    for certificate in result.factors:
        assert certificate.has_integral_multiplicity


@given(
    numerator_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=1,
        max_size=5,
    ),
    denominator_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=45, deadline=None)
def test_generated_arbitrary_rational_arguments_are_recognized(
    numerator_coefficients,
    denominator_coefficients,
) -> None:
    numerator = sum(
        coefficient*x**degree
        for degree, coefficient in enumerate(numerator_coefficients)
    )
    denominator = sum(
        coefficient*x**degree
        for degree, coefficient in enumerate(denominator_coefficients)
    )
    assume(numerator != 0 and denominator != 0)
    argument = cancel(numerator/denominator)
    integrand = cancel(argument.diff(x)/argument)

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert result.has_solution
    assert cancel(result.argument.diff(x)/result.argument - integrand) == 0
    assert cancel((result.argument/argument).diff(x)) == 0
    for certificate in result.factors:
        assert certificate.has_integral_multiplicity


@given(
    numerator=st.integers(min_value=-8, max_value=8).filter(bool),
    denominator=st.integers(min_value=2, max_value=8),
    root=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=35, deadline=None)
def test_generated_fractional_valuations_are_rejected_but_preserved(
    numerator,
    denominator,
    root,
) -> None:
    valuation = Rational(numerator, denominator)
    assume(valuation.q > 1)
    integrand = valuation/(x - root)

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert not result.has_solution
    assert result.has_nonintegral_residue_obstruction
    assert len(result.factors) == 1
    certificate = result.factors[0]
    assert certificate.multiplicity == QQ(valuation.p, valuation.q)
    assert certificate.has_logarithmic_residue
    assert not certificate.has_integral_multiplicity


@given(
    numerator=st.integers(min_value=1, max_value=7),
    denominator=st.integers(min_value=2, max_value=7),
)
@settings(max_examples=25, deadline=None)
def test_minimal_radical_multiplier_is_visible_in_factor_certificates(
    numerator,
    denominator,
) -> None:
    common = gcd(numerator, denominator)
    reduced_numerator = numerator//common
    reduced_denominator = denominator//common
    assume(reduced_denominator > 1)
    integrand = Rational(reduced_numerator, reduced_denominator)/x

    result = solve_rational_logarithmic_derivative(integrand, x)
    multiplicity = result.factors[0].multiplicity

    assert not result.has_solution
    assert QQ.denom(multiplicity) == reduced_denominator
    assert result.minimal_radical_power == reduced_denominator
    scaled = solve_rational_logarithmic_derivative(
        reduced_denominator*integrand,
        x,
    )
    assert scaled.has_solution
    assert scaled.argument == x**reduced_numerator


def test_mixed_obstructions_are_reported_independently() -> None:
    integrand = x + 1/x**2 + Rational(1, 2)/(x - 1)

    result = solve_rational_logarithmic_derivative(integrand, x)

    assert not result.has_solution
    assert result.has_polynomial_obstruction
    assert result.has_repeated_pole_obstruction
    assert result.argument is None


@pytest.mark.parametrize(
    "integrand",
    [
        sqrt(2)/x,
        1/(x + symbols("y")),
    ],
)
def test_nonrational_coefficients_are_rejected(integrand) -> None:
    with pytest.raises((CoercionFailed, PolynomialError)):
        solve_rational_logarithmic_derivative(integrand, x)


@pytest.mark.parametrize("variable", [x + 1, 1, "x", None])
def test_nonsymbol_differentiation_variables_are_rejected(variable) -> None:
    with pytest.raises(
        TypeError,
        match="differentiation variable must be a Symbol",
    ):
        solve_rational_logarithmic_derivative(1/x, variable)
