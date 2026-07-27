"""Independent matrix oracles for rational scalar and coupled RDEs."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly, cancel

from sympy_risch.rde.rational import _as_qq_rational

from .models import BoundedCoupledRDEMatrixResult, BoundedRationalRDEMatrixResult


def solve_bounded_rational_rde_matrix(
    coefficient: Expr | Poly,
    right_hand_side: Expr | Poly,
    x: Symbol,
    denominator: Expr | Poly,
    numerator_degree_bound: int,
) -> BoundedRationalRDEMatrixResult:
    r"""Solve a bounded rational RDE ansatz by independent exact linear algebra.

    The candidate is ``y = p/denominator`` with
    ``degree(p) <= numerator_degree_bound``.  This implementation deliberately
    performs no weak normalization, denominator reduction, SPDE, or
    leading-term recurrence and is intended as a differential-test oracle.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    if (
        not isinstance(numerator_degree_bound, int)
        or isinstance(numerator_degree_bound, bool)
        or numerator_degree_bound < 0
    ):
        raise ValueError("the numerator degree bound must be a nonnegative int")

    coefficient, _, _ = _as_qq_rational(coefficient, x)
    right_hand_side, _, _ = _as_qq_rational(right_hand_side, x)
    denominator = Poly(denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the ansatz denominator must be nonzero")
    denominator = denominator.monic()

    column_expressions = []
    for degree in range(numerator_degree_bound + 1):
        candidate = x**degree / denominator.as_expr()
        column_expressions.append(cancel(candidate.diff(x) + coefficient * candidate))

    rational_expressions = [*column_expressions, right_hand_side]
    reduced_expressions = [
        _as_qq_rational(expression, x) for expression in rational_expressions
    ]
    common_denominator = Poly(1, x, domain=QQ)
    for _, _, expression_denominator in reduced_expressions:
        common_denominator = common_denominator.lcm(expression_denominator)

    cleared_polynomials = [
        Poly(
            cancel(expression * common_denominator.as_expr()),
            x,
            domain=QQ,
        )
        for expression in rational_expressions
    ]
    column_polynomials = cleared_polynomials[:-1]
    rhs_polynomial = cleared_polynomials[-1]
    equation_degree = max(
        (
            polynomial.degree()
            for polynomial in cleared_polynomials
            if not polynomial.is_zero
        ),
        default=0,
    )
    equation_count = equation_degree + 1
    unknown_count = numerator_degree_bound + 1
    system = DomainMatrix(
        [
            [QQ.convert(polynomial.nth(degree)) for polynomial in column_polynomials]
            for degree in range(equation_count)
        ],
        (equation_count, unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        [[QQ.convert(rhs_polynomial.nth(degree))] for degree in range(equation_count)],
        (equation_count, 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedRationalRDEMatrixResult(
            denominator,
            numerator_degree_bound,
            None,
            coefficient_rank,
            augmented_rank,
        )

    reduced_matrix, pivots = augmented.rref()
    reduced_rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = reduced_rows[row][unknown_count]
    numerator = Poly.from_dict(
        {(degree,): value for degree, value in enumerate(coefficients)},
        x,
        domain=QQ,
    )
    solution = cancel(numerator.as_expr() / denominator.as_expr())
    if cancel(solution.diff(x) + coefficient * solution - right_hand_side) != 0:
        raise ValueError("bounded matrix RDE reconstruction failed")
    return BoundedRationalRDEMatrixResult(
        denominator,
        numerator_degree_bound,
        solution,
        coefficient_rank,
        augmented_rank,
    )


def solve_bounded_coupled_rde_matrix(
    lambda_coefficient: Expr | Poly,
    real_right_hand_side: Expr | Poly,
    imaginary_right_hand_side: Expr | Poly,
    x: Symbol,
    denominator: Expr | Poly,
    numerator_degree_bound: int,
) -> BoundedCoupledRDEMatrixResult:
    r"""Solve a bounded coupled rational ansatz by exact block linear algebra.

    This deliberately shares no denominator or leading-recurrence logic with
    the production coupled solver; it is a differential-test oracle for a
    caller-supplied common denominator and numerator degree bound.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    if (
        not isinstance(numerator_degree_bound, int)
        or isinstance(numerator_degree_bound, bool)
        or numerator_degree_bound < 0
    ):
        raise ValueError("the numerator degree bound must be a nonnegative int")

    lambda_expression, lambda_numerator, _ = _as_qq_rational(
        lambda_coefficient,
        x,
    )
    if lambda_numerator.is_zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs, _, _ = _as_qq_rational(real_right_hand_side, x)
    imaginary_rhs, _, _ = _as_qq_rational(imaginary_right_hand_side, x)
    denominator = Poly(denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the ansatz denominator must be nonzero")
    denominator = denominator.monic()

    basis = [
        x**degree / denominator.as_expr()
        for degree in range(numerator_degree_bound + 1)
    ]
    real_columns: list[Expr] = []
    imaginary_columns: list[Expr] = []
    for candidate in basis:
        real_columns.append(cancel(candidate.diff(x)))
        imaginary_columns.append(cancel(lambda_expression * candidate))
    for candidate in basis:
        real_columns.append(cancel(-lambda_expression * candidate))
        imaginary_columns.append(cancel(candidate.diff(x)))

    rational_expressions = [
        *real_columns,
        *imaginary_columns,
        real_rhs,
        imaginary_rhs,
    ]
    common_denominator = Poly(1, x, domain=QQ)
    for expression in rational_expressions:
        _, _, expression_denominator = _as_qq_rational(expression, x)
        common_denominator = common_denominator.lcm(expression_denominator)
    cleared = [
        Poly(
            cancel(expression * common_denominator.as_expr()),
            x,
            domain=QQ,
        )
        for expression in rational_expressions
    ]
    unknown_count = 2 * (numerator_degree_bound + 1)
    cleared_real_columns = cleared[:unknown_count]
    cleared_imaginary_columns = cleared[unknown_count : 2 * unknown_count]
    cleared_real_rhs = cleared[-2]
    cleared_imaginary_rhs = cleared[-1]
    equation_degree = max(
        (polynomial.degree() for polynomial in cleared if not polynomial.is_zero),
        default=0,
    )
    coefficient_rows = []
    rhs_rows = []
    for columns, rhs_polynomial in (
        (cleared_real_columns, cleared_real_rhs),
        (cleared_imaginary_columns, cleared_imaginary_rhs),
    ):
        for degree in range(equation_degree + 1):
            coefficient_rows.append(
                [QQ.convert(column.nth(degree)) for column in columns]
            )
            rhs_rows.append([QQ.convert(rhs_polynomial.nth(degree))])

    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        rhs_rows,
        (len(rhs_rows), 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedCoupledRDEMatrixResult(
            denominator,
            numerator_degree_bound,
            None,
            None,
            coefficient_rank,
            augmented_rank,
        )

    reduced_matrix, pivots = augmented.rref()
    reduced_rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = reduced_rows[row][unknown_count]
    component_count = numerator_degree_bound + 1
    real_numerator = Poly.from_dict(
        {
            (degree,): value
            for degree, value in enumerate(coefficients[:component_count])
        },
        x,
        domain=QQ,
    )
    imaginary_numerator = Poly.from_dict(
        {
            (degree,): value
            for degree, value in enumerate(coefficients[component_count:])
        },
        x,
        domain=QQ,
    )
    real_solution = cancel(real_numerator.as_expr() / denominator.as_expr())
    imaginary_solution = cancel(imaginary_numerator.as_expr() / denominator.as_expr())
    if (
        cancel(
            real_solution.diff(x) - lambda_expression * imaginary_solution - real_rhs
        )
        != 0
        or cancel(
            imaginary_solution.diff(x)
            + lambda_expression * real_solution
            - imaginary_rhs
        )
        != 0
    ):
        raise ValueError("bounded coupled matrix RDE reconstruction failed")
    return BoundedCoupledRDEMatrixResult(
        denominator,
        numerator_degree_bound,
        real_solution,
        imaginary_solution,
        coefficient_rank,
        augmented_rank,
    )
