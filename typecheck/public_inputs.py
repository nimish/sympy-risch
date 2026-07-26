"""Consumer-facing type checks for exact public input boundaries."""

from typing import assert_type

from sympy import Expr, Symbol

from sympy_risch.polynomial import integrate_polynomial
from sympy_risch.rde import (
    RationalRischDifferentialEquationResult,
    solve_rational_risch_differential_equation,
)

x = Symbol("x")

assert_type(integrate_polynomial(1, x), Expr)
assert_type(
    solve_rational_risch_differential_equation(1, 0, x),
    RationalRischDifferentialEquationResult,
)
