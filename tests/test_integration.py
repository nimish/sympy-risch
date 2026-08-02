"""Public Risch integration decisions."""

from dataclasses import replace

import pytest
from sympy import Rational, cancel, diff, exp, log, sin, sqrt, symbols

from sympy_risch import (
    AlgebraicExactnessLiouvilleObstruction,
    ElementaryPrimitive,
    NonElementary,
    NotSupported,
    integrate_risch,
)
from sympy_risch.expression import (
    ExpressionIntegrationOutcome,
    ExpressionUnsupportedReason,
)

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


@pytest.mark.parametrize(
    ("integrand", "primitive"),
    (
        (x ** Rational(1, 3), 3 * x ** Rational(4, 3) / 4),
        (1 / (2 * (x + sqrt(x))), log(sqrt(x) + 1)),
    ),
)
def test_integrate_risch_dispatches_direct_algebraic_expressions(
    integrand,
    primitive,
) -> None:
    result = integrate_risch(integrand, x)

    assert isinstance(result, ElementaryPrimitive)
    assert result.reconstructs
    assert result.primitive == primitive
    assert cancel(diff(result.primitive, x) - integrand) == 0


def test_integrate_risch_preserves_an_algebraic_liouville_obstruction() -> None:
    result = integrate_risch(1 / sqrt(x**3 - x), x)

    assert isinstance(result, NonElementary)
    assert isinstance(result.obstruction, AlgebraicExactnessLiouvilleObstruction)
    assert result.certificate.native_result is not None
    assert result.certificate.native_result.reconstructs

    with pytest.raises(ValueError, match="public outcome"):
        replace(
            result.certificate,
            outcome=ExpressionIntegrationOutcome.ELEMENTARY,
            antiderivative=x,
        )


def test_direct_algebraic_dispatch_builds_global_geometry_only_when_needed(
    monkeypatch,
) -> None:
    from sympy_risch.algebraic_geometry import AlgebraicFunctionField

    original_init = AlgebraicFunctionField.__init__
    construction_count = 0

    def counted_init(self, *args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(AlgebraicFunctionField, "__init__", counted_init)

    for degree in (2, 3, 5, 11):
        integrand = 1 / (degree * (x + x ** Rational(degree - 1, degree)))
        result = integrate_risch(integrand, x)

        assert isinstance(result, ElementaryPrimitive)
        assert result.primitive == log(x ** Rational(1, degree) + 1)
        assert cancel(diff(result.primitive, x) - integrand) == 0
    assert construction_count == 0

    positive_genus = integrate_risch(1 / sqrt(x**3 - x), x)

    assert isinstance(positive_genus, NonElementary)
    assert construction_count == 1


def test_mixed_algebraic_tower_remains_an_explicit_boundary() -> None:
    result = integrate_risch(sqrt(x + log(x)), x)

    assert isinstance(result, NotSupported)
    assert result.reason is ExpressionUnsupportedReason.INTEGRATION_BACKEND


def test_rational_radical_parameter_avoids_a_constant_base_change() -> None:
    result = integrate_risch(sqrt(x) / (x - 2), x)

    assert isinstance(result, ElementaryPrimitive)
    assert result.reconstructs
    assert result.primitive.free_symbols == {x}
    assert cancel(diff(result.primitive, x) - sqrt(x) / (x - 2)) == 0
