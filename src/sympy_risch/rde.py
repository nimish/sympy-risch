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
    NONZERO_REMAINDER = "nonzero_remainder"
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
    hand side.

    The recurrence solver certifies a later failure with ``partial_solution``
    and ``residual`` satisfying

    ``b = D(partial_solution) + a*partial_solution + residual``,

    where the nonzero residual has degree smaller than ``a`` and therefore
    cannot be cancelled by another polynomial term.  The independent matrix
    solver instead records coefficient and augmented ranks.
    """

    a: Poly
    b: Poly
    degree_bound: int
    solution: Poly | None
    obstruction: PolynomialRDEObstruction | None
    coefficient_rank: int | None = None
    augmented_rank: int | None = None
    partial_solution: Poly | None = None
    residual: Poly | None = None

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

    @property
    def has_nonzero_remainder_obstruction(self) -> bool:
        """Whether leading-term reduction leaves an uncancellable remainder."""
        return self.obstruction is PolynomialRDEObstruction.NONZERO_REMAINDER


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


def _coerce_polynomial_rde_inputs(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> tuple[Poly, Poly]:
    """Coerce a polynomial Risch equation to its exact ``QQ[x]`` domain."""
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    return Poly(a, x, domain=QQ), Poly(b, x, domain=QQ)


def _solve_trivial_polynomial_rde(
    a: Poly,
    b: Poly,
) -> PolynomialRischDifferentialEquationResult | None:
    """Handle the zero cases shared by both independent solvers."""
    x = a.gens[0]

    if b.is_zero:
        zero = Poly(0, x, domain=QQ)
        return PolynomialRischDifferentialEquationResult(
            a,
            b,
            0,
            zero,
            None,
        )

    if a.is_zero:
        solution = Poly(
            integrate_polynomial(b, x),
            x,
            domain=QQ,
        )
        if solution.diff() != b:
            raise ValueError("polynomial Risch equation integration failed")
        return PolynomialRischDifferentialEquationResult(
            a,
            b,
            b.degree() + 1,
            solution,
            None,
        )

    return None


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
    The solver therefore cancels the forced leading term repeatedly instead
    of constructing a coefficient matrix.

    The output is a decision certificate, not merely an ansatz search:

    * finite-pole analysis proves every rational solution is polynomial;
    * leading degrees either determine a finite degree or prove failure;
    * each recurrence step is forced and strictly lowers the residual degree.

    When ``a = 0``, the equation is ordinary polynomial integration and the
    integration constant is normalized to zero.
    """
    a_polynomial, b_polynomial = _coerce_polynomial_rde_inputs(a, b, x)
    trivial_result = _solve_trivial_polynomial_rde(
        a_polynomial,
        b_polynomial,
    )
    if trivial_result is not None:
        return trivial_result

    degree_bound = b_polynomial.degree() - a_polynomial.degree()
    if degree_bound < 0:
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            -1,
            None,
            PolynomialRDEObstruction.DEGREE,
            partial_solution=Poly(0, x, domain=QQ),
            residual=b_polynomial,
        )

    solution = Poly(0, x, domain=QQ)
    residual = b_polynomial
    while not residual.is_zero:
        candidate_degree = residual.degree() - a_polynomial.degree()
        if candidate_degree < 0:
            if (
                solution.diff()
                + a_polynomial*solution
                + residual
                != b_polynomial
            ):
                raise ValueError(
                    "polynomial Risch equation reduction invariant failed"
                )
            return PolynomialRischDifferentialEquationResult(
                a_polynomial,
                b_polynomial,
                degree_bound,
                None,
                PolynomialRDEObstruction.NONZERO_REMAINDER,
                partial_solution=solution,
                residual=residual,
            )

        term = Poly.from_dict(
            {
                (candidate_degree,): (
                    residual.LC()/a_polynomial.LC()
                ),
            },
            x,
            domain=QQ,
        )
        new_residual = residual - term.diff() - a_polynomial*term
        if (
            not new_residual.is_zero
            and new_residual.degree() >= residual.degree()
        ):
            raise ValueError(
                "polynomial Risch equation residual degree did not decrease"
            )
        solution += term
        residual = new_residual

    if solution.diff() + a_polynomial*solution != b_polynomial:
        raise ValueError(
            "polynomial Risch equation reconstruction failed"
        )
    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        degree_bound,
        solution,
        None,
    )


def solve_polynomial_risch_differential_equation_matrix(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> PolynomialRischDifferentialEquationResult:
    r"""Solve ``Dy + a*y = b`` with an independent exact matrix method.

    This deliberately slower reference implementation searches every
    polynomial degree through ``degree(b)`` and uses exact rank comparison.
    It is retained to cross-check the leading-term recurrence without sharing
    that algorithm's forced-degree calculation.
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
            "matrix polynomial Risch equation reconstruction failed"
        )

    coefficient_rank = system.rank()
    augmented_rank = system.hstack(right_hand_side).rank()
    if (
        coefficient_rank != search_degree + 1
        or augmented_rank != coefficient_rank
    ):
        raise ValueError(
            "matrix polynomial Risch equation returned invalid ranks"
        )

    # The matrix searches beyond the forced degree on purpose.  The extra
    # solution coefficients are zero, but the reported ranks describe the
    # complete independently searched system.
    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        forced_degree,
        solution,
        None,
        coefficient_rank,
        augmented_rank,
    )
