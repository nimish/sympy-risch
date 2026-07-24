r"""Risch differential equations over exact polynomial domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from .polynomial import integrate_polynomial


class PolynomialRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in ``QQ(x)``."""

    DEGREE = "degree"
    INCONSISTENT_SYSTEM = "inconsistent_system"


@dataclass(frozen=True, slots=True)
class PolynomialRischDifferentialEquationResult:
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``QQ[x]``.

    A rational solution cannot have a finite pole: at a pole of ``y``,
    ``Dy`` has strictly higher order than ``a*y`` and cannot be cancelled by
    polynomial ``b``.  It is therefore enough to search for polynomial
    solutions.

    When ``a`` and ``b`` are nonzero, leading degrees force

    ``degree(y) = degree(b) - degree(a)``.

    ``degree_bound`` stores this forced degree.  The value ``-1`` is the
    degree certificate that no nonzero polynomial can solve a nonzero right
    hand side.  For an inconsistent coefficient system, the two ranks certify
    that the augmented matrix has larger rank.
    """

    a: Poly
    b: Poly
    degree_bound: int
    solution: Poly | None
    obstruction: PolynomialRDEObstruction | None
    coefficient_rank: int | None = None
    augmented_rank: int | None = None

    @property
    def has_solution(self) -> bool:
        """Whether the differential equation has a rational solution."""
        return self.solution is not None

    @property
    def solution_expression(self) -> Expr | None:
        """Return the canonical polynomial solution as a SymPy expression."""
        if self.solution is None:
            return None
        return self.solution.as_expr()

    @property
    def has_degree_obstruction(self) -> bool:
        """Whether leading degrees alone prove nonexistence."""
        return self.obstruction is PolynomialRDEObstruction.DEGREE

    @property
    def has_inconsistent_system_obstruction(self) -> bool:
        """Whether exact rank comparison proves the coefficient system fails."""
        return self.obstruction is PolynomialRDEObstruction.INCONSISTENT_SYSTEM


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
        columns.append(monomial.diff() + a*monomial)

    equation_count = max(
        b.degree(),
        max(column.degree() for column in columns),
    ) + 1
    system = DomainMatrix(
        [
            [
                QQ.convert(column.nth(degree))
                for column in columns
            ]
            for degree in range(equation_count)
        ],
        (equation_count, degree_bound + 1),
        QQ,
    )
    right_hand_side = DomainMatrix(
        [
            [QQ.convert(b.nth(degree))]
            for degree in range(equation_count)
        ],
        (equation_count, 1),
        QQ,
    )
    return system, right_hand_side


def solve_polynomial_risch_differential_equation(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> PolynomialRischDifferentialEquationResult:
    r"""Solve ``Dy + a*y = b`` for rational ``y`` when ``a,b`` are in ``QQ[x]``.

    This is the ``D = d/dt`` specialization of Bronstein,
    *Symbolic Integration I*, Section 6.5, particularly Lemma 6.5.1(i) and
    ``PolyRischDENoCancel``.  In this specialization the derivative always
    lowers polynomial degree, so the leading term of ``a*y`` cannot cancel.

    The output is a decision certificate, not merely an ansatz search:

    * finite-pole analysis proves every rational solution is polynomial;
    * leading degrees either determine a finite degree or prove failure;
    * exact linear algebra over ``QQ`` decides the remaining coefficients.

    When ``a = 0``, the equation is ordinary polynomial integration and the
    integration constant is normalized to zero.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    a_polynomial = Poly(a, x, domain=QQ)
    b_polynomial = Poly(b, x, domain=QQ)

    if b_polynomial.is_zero:
        zero = Poly(0, x, domain=QQ)
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            0,
            zero,
            None,
        )

    if a_polynomial.is_zero:
        solution = Poly(
            integrate_polynomial(b_polynomial, x),
            x,
            domain=QQ,
        )
        if solution.diff() != b_polynomial:
            raise ValueError("polynomial Risch equation integration failed")
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            b_polynomial.degree() + 1,
            solution,
            None,
        )

    degree_bound = b_polynomial.degree() - a_polynomial.degree()
    if degree_bound < 0:
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            -1,
            None,
            PolynomialRDEObstruction.DEGREE,
        )

    system, right_hand_side = _polynomial_rde_system(
        a_polynomial,
        b_polynomial,
        degree_bound,
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
            degree_bound,
            None,
            PolynomialRDEObstruction.INCONSISTENT_SYSTEM,
            coefficient_rank,
            augmented_rank,
        )

    coefficients = [
        row[0]/solution_denominator
        for row in solution_numerators.to_list()
    ]
    solution = Poly.from_dict(
        {
            (degree,): coefficient
            for degree, coefficient in enumerate(coefficients)
        },
        x,
        domain=QQ,
    )
    if solution.diff() + a_polynomial*solution != b_polynomial:
        raise ValueError(
            "polynomial Risch equation reconstruction failed"
        )

    # solve_den returns only for a consistent system with a unique solution,
    # so both ranks equal the number of coefficient unknowns.  Recomputing two
    # RREFs here would duplicate the dominant linear-algebra work.
    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        degree_bound,
        solution,
        None,
        degree_bound + 1,
        degree_bound + 1,
    )
