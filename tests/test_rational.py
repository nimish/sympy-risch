import pytest

from sympy import Abs, Poly, Rational, atan, cancel, log, sqrt, symbols

from sympy_risch import (
    hermite_reduce_rational,
    integrate_rational,
    integrate_rational_real,
    rothstein_trager_residue_polynomial,
)


x, z = symbols("x z")


@pytest.mark.parametrize(
    "integrand",
    [
        0,
        x**3 - 2*x + 1,
        1/x,
        1/x**3,
        1/(x**2*(x + 1)),
        (2*x + 3)/(x**2 + 1)**2,
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
    exact_part, remainder = hermite_reduce_rational(1/(x**2*(x + 1)), x)

    assert exact_part == -1/x
    assert cancel(remainder + 1/(x*(x + 1))) == 0


def test_rothstein_trager_residue_polynomial() -> None:
    residue_polynomial = rothstein_trager_residue_polynomial(
        -1/(x*(x + 1)),
        x,
        z,
    )

    assert residue_polynomial.monic() == Poly(z**2 - 1, z, domain="QQ")


@pytest.mark.parametrize(
    "integrand",
    [
        0,
        x**3 - 2*x + 1,
        1/x,
        1/x**3,
        1/(x**2*(x + 1)),
        1/(x**2 + 1),
        (2*x + 3)/(x**2 + 1)**2,
        Rational(3, 2)/(x - 1),
    ],
)
def test_integrate_rational_differentiates_to_input(integrand) -> None:
    result = integrate_rational(integrand, x)

    assert cancel(result.diff(x) - integrand) == 0


def test_integrate_rational_example() -> None:
    result = integrate_rational(1/(x**2*(x + 1)), x)

    assert result.expand(log=True, force=True) == -1/x - log(x) + log(x + 1)


def test_rothstein_trager_rejects_repeated_denominator() -> None:
    with pytest.raises(ValueError, match="square-free denominator"):
        rothstein_trager_residue_polynomial(1/x**2, x, z)


def test_integrate_rational_real_uses_arctangent() -> None:
    assert integrate_rational_real(1/(x**2 + 1), x) == atan(x)


def test_integrate_rational_real_uses_real_logarithm() -> None:
    assert integrate_rational_real(1/(x - 1), x) == log(Abs(x - 1))


def test_integrate_rational_real_handles_real_algebraic_roots() -> None:
    result = integrate_rational_real(1/(x**2 - 2), x)

    assert result == (
        log(Abs(2*x - 2*sqrt(2)))
        - log(Abs(2*x + 2*sqrt(2)))
    )/(2*sqrt(2))


@pytest.mark.parametrize(
    "integrand",
    [
        1/(x**2 + 1),
        x/(x**2 + 1),
        1/(x - 1),
        1/(x**2 - 2),
        1/(x**2*(x + 1)),
        (2*x + 3)/(x**2 + 1)**2,
        (x + 2)/((x - 1)*(x**2 + 1)),
    ],
)
def test_integrate_rational_real_differentiates_to_input(integrand) -> None:
    result = integrate_rational_real(integrand, x)
    result_without_absolute_values = result.replace(Abs, lambda argument: argument)

    assert cancel(result_without_absolute_values.diff(x) - integrand) == 0


def test_integrate_rational_real_rejects_higher_degree_factor() -> None:
    with pytest.raises(NotImplementedError, match="degree at most two"):
        integrate_rational_real(1/(x**3 + x + 1), x)
