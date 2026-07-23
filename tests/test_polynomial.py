import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sympy import Poly, Rational, symbols
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from sympy_risch import integrate_polynomial


x, a = symbols("x a")


@pytest.mark.parametrize(
    ("integrand", "expected"),
    [
        (0, 0),
        (5, 5*x),
        (3*x**2 + 2*x + 1, x**3 + x**2 + x),
        (x**4 - 2*x**2 + 7, x**5/5 - 2*x**3/3 + 7*x),
        (Rational(3, 2)*x, Rational(3, 4)*x**2),
    ],
)
def test_integrate_polynomial(integrand, expected) -> None:
    result = integrate_polynomial(integrand, x)

    assert result == expected
    assert result.diff(x) == integrand


def test_integrate_polynomial_accepts_poly() -> None:
    result = integrate_polynomial(Poly(x**2 + 1, x, domain="QQ"), x)

    assert result == x**3/3 + x


def test_integrate_polynomial_rejects_rational_function() -> None:
    with pytest.raises(PolynomialError):
        integrate_polynomial(1/x, x)


def test_integrate_polynomial_rejects_non_rational_coefficient() -> None:
    with pytest.raises(CoercionFailed):
        integrate_polynomial(a*x, x)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-20, max_value=20),
            st.integers(min_value=1, max_value=10),
        ),
        min_size=0,
        max_size=15,
    ),
)
@settings(max_examples=40, deadline=None)
def test_generated_rational_polynomials_differentiate_back(coefficients) -> None:
    integrand = sum(
        Rational(numerator, denominator)*x**degree
        for degree, (numerator, denominator) in enumerate(coefficients)
    )

    antiderivative = integrate_polynomial(integrand, x)

    assert antiderivative.diff(x) == integrand
    assert not antiderivative.has(a)
