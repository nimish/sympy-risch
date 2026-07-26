"""Independent matrix oracle for polynomial Risch differential equations."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from sympy_risch.rde import (
    PolynomialRDEObstruction,
    PolynomialRischDifferentialEquationResult,
    _coerce_polynomial_rde_inputs,
    _solve_trivial_polynomial_rde,
)


def _polynomial_rde_system(
    a: Poly,
    b: Poly,
    degree_bound: int,
) -> tuple[DomainMatrix, DomainMatrix]:
    """Build the exact coefficient system for ``Dy + a*y = b``."""
    variable = a.gens[0]
    columns = []
    for degree in range(degree_bound + 1):
        monomial = Poly(variable**degree, variable, domain=QQ)
        columns.append(monomial.diff() + a * monomial)

    equation_count = (
        max(
            b.degree(),
            max(column.degree() for column in columns),
        )
        + 1
    )
    system = DomainMatrix(
        [
            [QQ.convert(column.nth(degree)) for column in columns]
            for degree in range(equation_count)
        ],
        (equation_count, degree_bound + 1),
        QQ,
    )
    right_hand_side = DomainMatrix(
        [[QQ.convert(b.nth(degree))] for degree in range(equation_count)],
        (equation_count, 1),
        QQ,
    )
    return system, right_hand_side


def solve_polynomial_risch_differential_equation_matrix(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> PolynomialRischDifferentialEquationResult:
    r"""Solve ``Dy + a*y = b`` with an independent exact matrix method.

    This deliberately slower reference implementation searches every
    polynomial degree through ``degree(b)`` and uses exact rank comparison.
    It cross-checks the leading-term recurrence without sharing that
    algorithm's forced-degree calculation.
    """
    a_polynomial, b_polynomial = _coerce_polynomial_rde_inputs(a, b, x)
    trivial_result = _solve_trivial_polynomial_rde(
        a_polynomial,
        b_polynomial,
    )
    if trivial_result is not None:
        return trivial_result

    forced_degree = b_polynomial.degree() - a_polynomial.degree()
    search_degree = b_polynomial.degree()
    system, right_hand_side = _polynomial_rde_system(
        a_polynomial,
        b_polynomial,
        search_degree,
    )
    try:
        solution_numerators, solution_denominator = system.solve_den(
            right_hand_side,
            method="rref",
        )
    except DMNonInvertibleMatrixError as exc:
        coefficient_rank = system.rank()
        augmented_rank = system.hstack(right_hand_side).rank()
        if augmented_rank <= coefficient_rank:
            raise ValueError(
                "polynomial Risch equation unexpectedly lacks a unique solution"
            ) from exc
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            max(-1, forced_degree),
            None,
            PolynomialRDEObstruction.INCONSISTENT_SYSTEM,
            coefficient_rank,
            augmented_rank,
        )

    coefficients = [
        row[0] / solution_denominator for row in solution_numerators.to_list()
    ]
    solution = Poly.from_dict(
        {(degree,): coefficient for degree, coefficient in enumerate(coefficients)},
        x,
        domain=QQ,
    )
    if solution.diff() + a_polynomial * solution != b_polynomial:
        raise ValueError("matrix polynomial Risch equation reconstruction failed")

    coefficient_rank = system.rank()
    augmented_rank = system.hstack(right_hand_side).rank()
    if coefficient_rank != search_degree + 1 or augmented_rank != coefficient_rank:
        raise ValueError("matrix polynomial Risch equation returned invalid ranks")

    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        forced_degree,
        solution,
        None,
        coefficient_rank,
        augmented_rank,
    )
