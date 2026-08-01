"""Public Risch integration decisions."""

from sympy import diff, exp, log, sin, symbols

from sympy_risch import (
    ElementaryPrimitive,
    NonElementary,
    NotSupported,
    integrate_risch,
)
from sympy_risch.expression import ExpressionUnsupportedReason

x = symbols("x")


def test_integrate_risch_returns_a_verified_elementary_primitive() -> None:
    integrand = 1 / (x * log(x))

    result = integrate_risch(integrand, x)

    assert isinstance(result, ElementaryPrimitive)
    assert result.reconstructs
    assert result.primitive == log(log(x))
    assert diff(result.primitive, x) == integrand


def test_integrate_risch_preserves_a_non_elementarity_proof() -> None:
    result = integrate_risch(exp(x**2), x)

    assert isinstance(result, NonElementary)
    assert result.obstruction.remainder != 0


def test_integrate_risch_preserves_an_unsupported_grammar_boundary() -> None:
    result = integrate_risch(sin(x), x)

    assert isinstance(result, NotSupported)
    assert result.reason is ExpressionUnsupportedReason.UNSUPPORTED_FUNCTION


def test_integrate_risch_does_not_rewrite_a_dependent_branch_sensitive_call() -> None:
    expression = log(exp(x), evaluate=False)

    result = integrate_risch(expression, x)

    assert isinstance(result, NotSupported)
    assert result.reason is ExpressionUnsupportedReason.DEPENDENT_EXTENSION
