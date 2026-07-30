"""Consumer-facing type checks for exact public input boundaries."""

from typing import assert_type

from sympy import Expr, Symbol

from sympy_risch._typing import FieldElement
from sympy_risch.polynomial import integrate_polynomial
from sympy_risch.rde import (
    LimitedIntegrationResult,
    ParametricRischDifferentialEquationResult,
    RationalRischDifferentialEquationResult,
    solve_limited_integration,
    solve_parametric_risch_differential_equation,
    solve_rational_risch_differential_equation,
)
from sympy_risch.tower import Derivation

x = Symbol("x")

assert_type(integrate_polynomial(1, x), Expr)
assert_type(
    solve_rational_risch_differential_equation(1, 0, x),
    RationalRischDifferentialEquationResult,
)


def check_native_limited_integration_inputs(
    value: FieldElement,
    derivation: Derivation,
) -> None:
    """Type-check the native field elements used by the documented API."""
    assert_type(
        solve_limited_integration(value, (value,), derivation),
        LimitedIntegrationResult,
    )
    assert_type(
        solve_parametric_risch_differential_equation(value, (value,), derivation),
        ParametricRischDifferentialEquationResult,
    )
