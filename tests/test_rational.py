from math import prod

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Abs, Poly, Rational, atan, cancel, log, sqrt, symbols

from sympy_risch import (
    hermite_reduce_rational,
    hermite_reduce_rational_recursive,
    integrate_rational,
    integrate_rational_real,
    rothstein_trager_residue_polynomial,
    solve_rational_exact_derivative,
)

x, z = symbols("x z")


@st.composite
def rational_functions_with_repeated_factors(draw):
    factor_count = draw(st.integers(min_value=1, max_value=3))
    roots = draw(
        st.lists(
            st.integers(min_value=-4, max_value=4),
            min_size=factor_count,
            max_size=factor_count,
            unique=True,
        )
    )
    multiplicities = draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=factor_count,
            max_size=factor_count,
        )
    )
    assume(any(multiplicity > 1 for multiplicity in multiplicities))
    denominator = Poly(
        prod(
            (x - root) ** multiplicity
            for root, multiplicity in zip(roots, multiplicities, strict=True)
        ),
        x,
        domain="QQ",
    )
    coefficients = draw(
        st.lists(
            st.integers(min_value=-4, max_value=4),
            min_size=denominator.degree(),
            max_size=denominator.degree(),
        )
    )
    assume(any(coefficients))
    numerator = Poly(
        sum(coefficient * x**degree for degree, coefficient in enumerate(coefficients)),
        x,
        domain="QQ",
    )
    assume(numerator.gcd(denominator).degree() == 0)
    return numerator.as_expr() / denominator.as_expr()


@pytest.mark.parametrize(
    "integrand",
    [
        0,
        x**3 - 2 * x + 1,
        1 / x,
        1 / x**3,
        1 / (x**2 * (x + 1)),
        (2 * x + 3) / (x**2 + 1) ** 2,
    ],
)
def test_hermite_reduction_identity_and_square_free_remainder(integrand) -> None:
    exact_part, remainder = hermite_reduce_rational(integrand, x)

    assert cancel(exact_part.diff(x) + remainder - integrand) == 0

    if remainder != 0:
        _, denominator = cancel(remainder).as_numer_denom()
        denominator = Poly(denominator, x, domain="QQ")
        assert denominator.gcd(denominator.diff()).degree() == 0


def test_hermite_reduction_example() -> None:
    exact_part, remainder = hermite_reduce_rational(1 / (x**2 * (x + 1)), x)

    assert exact_part == -1 / x
    assert cancel(remainder + 1 / (x * (x + 1))) == 0


def test_recursive_hermite_reduction_lowers_one_factor_at_a_time() -> None:
    integrand = 1 / (x**4 * (x + 1) ** 3 * (x + 2) ** 2)

    exact_part, remainder = hermite_reduce_rational_recursive(integrand, x)

    assert cancel(exact_part.diff(x) + remainder - integrand) == 0
    _, denominator = cancel(remainder).as_numer_denom()
    denominator = Poly(denominator, x, domain="QQ")
    assert denominator.gcd(denominator.diff()).degree() == 0


@pytest.mark.parametrize(
    "integrand",
    [0, x**3 - 2 * x + 1, 1 / x, 1 / x**3],
)
def test_recursive_hermite_reduction_handles_base_cases(integrand) -> None:
    recursive = hermite_reduce_rational_recursive(integrand, x)
    global_reduction = hermite_reduce_rational(integrand, x)

    assert all(
        cancel(recursive_part - global_part) == 0
        for recursive_part, global_part in zip(
            recursive,
            global_reduction,
            strict=True,
        )
    )


@given(rational_functions_with_repeated_factors())
@settings(max_examples=40, deadline=None)
def test_recursive_and_horowitz_ostrogradsky_reductions_agree(integrand) -> None:
    global_exact, global_remainder = hermite_reduce_rational(integrand, x)
    recursive_exact, recursive_remainder = hermite_reduce_rational_recursive(
        integrand,
        x,
    )

    assert cancel(recursive_exact.diff(x) + recursive_remainder - integrand) == 0
    assert cancel(recursive_exact - global_exact) == 0
    assert cancel(recursive_remainder - global_remainder) == 0

    second_exact, second_remainder = hermite_reduce_rational(
        global_remainder,
        x,
    )
    assert second_exact == 0
    assert cancel(second_remainder - global_remainder) == 0


@pytest.mark.parametrize(
    "antiderivative",
    [
        0,
        x**5 - 3 * x + 2,
        1 / x,
        (x + 1) / (x - 2) ** 3,
        (x**3 + 1) / (x**2 + 1),
    ],
)
def test_rational_exact_derivative_solver_recovers_planted_solutions(
    antiderivative,
) -> None:
    integrand = cancel(antiderivative).diff(x)

    result = solve_rational_exact_derivative(integrand, x)

    assert result.has_solution
    assert result.antiderivative == result.exact_part
    assert result.obstruction == 0
    assert cancel(result.exact_part.diff(x) - integrand) == 0


@pytest.mark.parametrize(
    "integrand",
    [
        1 / x,
        1 / (x**2 + 1),
        x / (x**2 + 1),
        1 / x + 2 / (x - 1),
    ],
)
def test_rational_exact_derivative_solver_returns_simple_pole_obstruction(
    integrand,
) -> None:
    result = solve_rational_exact_derivative(integrand, x)

    assert not result.has_solution
    assert result.antiderivative is None
    assert result.exact_part == 0
    assert cancel(result.obstruction - integrand) == 0

    numerator, denominator = cancel(result.obstruction).as_numer_denom()
    denominator = Poly(denominator, x, domain="QQ")
    assert numerator != 0
    assert denominator.gcd(denominator.diff()).degree() == 0


@given(
    polynomial_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=1,
        max_size=6,
    ),
    pole_root=st.integers(min_value=-4, max_value=4),
    pole_power=st.integers(min_value=1, max_value=5),
    pole_numerator=st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_rational_derivatives_have_zero_obstruction(
    polynomial_coefficients,
    pole_root,
    pole_power,
    pole_numerator,
) -> None:
    antiderivative = (
        sum(
            coefficient * x**degree
            for degree, coefficient in enumerate(polynomial_coefficients)
        )
        + pole_numerator / (x - pole_root) ** pole_power
    )
    integrand = cancel(antiderivative.diff(x))

    result = solve_rational_exact_derivative(integrand, x)

    assert result.has_solution
    assert result.antiderivative is not None
    assert cancel(result.antiderivative.diff(x) - integrand) == 0


@given(
    exact_root=st.integers(min_value=-5, max_value=-1),
    obstruction_root=st.integers(min_value=1, max_value=5),
    exact_numerator=st.integers(min_value=-4, max_value=4),
    exact_power=st.integers(min_value=1, max_value=5),
    residue=st.integers(min_value=-5, max_value=5).filter(bool),
)
@settings(max_examples=40, deadline=None)
def test_generated_simple_poles_are_exact_derivative_obstructions(
    exact_root,
    obstruction_root,
    exact_numerator,
    exact_power,
    residue,
) -> None:
    planted_exact_part = exact_numerator / (x - exact_root) ** exact_power
    planted_obstruction = residue / (x - obstruction_root)
    integrand = cancel(planted_exact_part.diff(x) + planted_obstruction)

    result = solve_rational_exact_derivative(integrand, x)

    assert not result.has_solution
    assert result.antiderivative is None
    assert cancel(result.exact_part.diff(x) + result.obstruction - integrand) == 0
    assert cancel(result.obstruction - planted_obstruction) == 0

    second = solve_rational_exact_derivative(result.obstruction, x)
    assert second.exact_part == 0
    assert cancel(second.obstruction - result.obstruction) == 0


def test_rothstein_trager_residue_polynomial() -> None:
    residue_polynomial = rothstein_trager_residue_polynomial(
        -1 / (x * (x + 1)),
        x,
        z,
    )

    assert residue_polynomial.monic() == Poly(z**2 - 1, z, domain="QQ")


@pytest.mark.parametrize(
    "integrand",
    [
        0,
        x**3 - 2 * x + 1,
        1 / x,
        1 / x**3,
        1 / (x**2 * (x + 1)),
        1 / (x**2 + 1),
        (2 * x + 3) / (x**2 + 1) ** 2,
        Rational(3, 2) / (x - 1),
    ],
)
def test_integrate_rational_differentiates_to_input(integrand) -> None:
    result = integrate_rational(integrand, x)

    assert cancel(result.diff(x) - integrand) == 0


def test_integrate_rational_example() -> None:
    result = integrate_rational(1 / (x**2 * (x + 1)), x)

    assert result.expand(log=True, force=True) == -1 / x - log(x) + log(x + 1)


def test_rothstein_trager_rejects_repeated_denominator() -> None:
    with pytest.raises(ValueError, match="square-free denominator"):
        rothstein_trager_residue_polynomial(1 / x**2, x, z)


def test_rothstein_trager_requires_a_distinct_residue_symbol() -> None:
    with pytest.raises(ValueError, match="must differ"):
        rothstein_trager_residue_polynomial(1 / (x + 1), x, x)
    with pytest.raises(TypeError, match="must be a Symbol"):
        rothstein_trager_residue_polynomial(1 / (x + 1), x, 0)  # type: ignore[arg-type]


def test_integrate_rational_real_uses_arctangent() -> None:
    assert integrate_rational_real(1 / (x**2 + 1), x) == atan(x)


def test_integrate_rational_real_uses_real_logarithm() -> None:
    assert integrate_rational_real(1 / (x - 1), x) == log(Abs(x - 1))


def test_integrate_rational_real_handles_real_algebraic_roots() -> None:
    result = integrate_rational_real(1 / (x**2 - 2), x)

    assert result == (log(Abs(2 * x - 2 * sqrt(2))) - log(Abs(2 * x + 2 * sqrt(2)))) / (
        2 * sqrt(2)
    )


@pytest.mark.parametrize(
    "integrand",
    [
        1 / (x**2 + 1),
        x / (x**2 + 1),
        1 / (x - 1),
        1 / (x**2 - 2),
        1 / (x**2 * (x + 1)),
        (2 * x + 3) / (x**2 + 1) ** 2,
        (x + 2) / ((x - 1) * (x**2 + 1)),
    ],
)
def test_integrate_rational_real_differentiates_to_input(integrand) -> None:
    result = integrate_rational_real(integrand, x)
    result_without_absolute_values = result.replace(Abs, lambda argument: argument)

    assert cancel(result_without_absolute_values.diff(x) - integrand) == 0


def test_integrate_rational_real_rejects_higher_degree_factor() -> None:
    with pytest.raises(NotImplementedError, match="degree at most two"):
        integrate_rational_real(1 / (x**3 + x + 1), x)
