r"""Risch differential equations over exact polynomial domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly, cancel, invert

from .polynomial import integrate_polynomial
from .rational import (
    hermite_reduce_rational,
    solve_rational_logarithmic_derivative,
)


class PolynomialRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in ``QQ(x)``."""

    DEGREE = "degree"
    NONZERO_REMAINDER = "nonzero_remainder"
    INCONSISTENT_SYSTEM = "inconsistent_system"


class RationalRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in ``QQ(x)``."""

    DENOMINATOR = "denominator"
    SPDE_DEGREE = "spde_degree"
    SPDE_DIVISIBILITY = "spde_divisibility"
    POLYNOMIAL = "polynomial"


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


@dataclass(frozen=True, slots=True)
class RationalRischDifferentialEquationResult:
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``QQ(x)``.

    ``normalization_factor`` records the weak normalization

    ``a_normal = a - D(normalization_factor)/normalization_factor``.

    When denominator reduction succeeds, every solution is represented as

    ``y = q/(normalization_factor*denominator_bound)``

    for a polynomial ``q`` of degree at most ``degree_bound``.  A successful
    SPDE reduction finishes with ``polynomial_result``.  On a denominator or
    SPDE divisibility failure, ``obstruction_polynomial`` and
    ``obstruction_remainder`` retain the failed exact division.
    """

    coefficient: Expr
    right_hand_side: Expr
    solution: Expr | None
    obstruction: RationalRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: Expr
    normalized_right_hand_side: Expr
    denominator_bound: Poly | None = None
    degree_bound: int | None = None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def has_solution(self) -> bool:
        """Whether the Risch differential equation has a rational solution."""
        return self.solution is not None

    @property
    def has_denominator_obstruction(self) -> bool:
        """Whether the normal-denominator theorem proves nonexistence."""
        return self.obstruction is RationalRDEObstruction.DENOMINATOR

    @property
    def has_spde_obstruction(self) -> bool:
        """Whether SPDE degree or divisibility proves nonexistence."""
        return self.obstruction in {
            RationalRDEObstruction.SPDE_DEGREE,
            RationalRDEObstruction.SPDE_DIVISIBILITY,
        }

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether the final polynomial RDE has no solution."""
        return self.obstruction is RationalRDEObstruction.POLYNOMIAL


@dataclass(frozen=True, slots=True)
class BoundedRationalRDEMatrixResult:
    """Exact matrix result for a caller-supplied rational ansatz."""

    denominator: Poly
    numerator_degree_bound: int
    solution: Expr | None
    coefficient_rank: int
    augmented_rank: int

    @property
    def has_solution(self) -> bool:
        """Whether the bounded rational ansatz contains a solution."""
        return self.solution is not None


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


def _as_qq_rational(
    value: Expr | Poly,
    x: Symbol,
) -> tuple[Expr, Poly, Poly]:
    """Return an expression and coprime numerator/denominator in ``QQ[x]``."""
    if isinstance(value, Poly):
        value = value.as_expr()
    numerator, denominator = cancel(value).as_numer_denom()
    numerator_polynomial = Poly(numerator, x, domain=QQ)
    denominator_polynomial = Poly(denominator, x, domain=QQ)
    coefficient, numerator_polynomial, denominator_polynomial = (
        numerator_polynomial.cancel(denominator_polynomial)
    )
    numerator_polynomial = numerator_polynomial.mul_ground(coefficient)
    expression = cancel(
        numerator_polynomial.as_expr()/denominator_polynomial.as_expr()
    )
    return expression, numerator_polynomial, denominator_polynomial


def _weakly_normalize_rational_rde_coefficient(
    coefficient: Expr,
    x: Symbol,
) -> tuple[Poly, Expr]:
    r"""Return ``q`` and ``coefficient - Dq/q`` with no positive residues."""
    _, residue_part = hermite_reduce_rational(coefficient, x)
    logarithmic_result = solve_rational_logarithmic_derivative(
        residue_part,
        x,
    )
    normalizer = Poly(1, x, domain=QQ)
    for factor in logarithmic_result.factors:
        if factor.multiplicity is None:
            continue
        multiplicity = QQ.convert(factor.multiplicity)
        if (
            QQ.denom(multiplicity) == 1
            and QQ.numer(multiplicity) > 0
        ):
            normalizer *= factor.factor**int(QQ.numer(multiplicity))

    normalized = cancel(
        coefficient - normalizer.diff().as_expr()/normalizer.as_expr()
    )
    if cancel(
        normalized
        + normalizer.diff().as_expr()/normalizer.as_expr()
        - coefficient
    ) != 0:
        raise ValueError("weak normalization reconstruction failed")
    return normalizer, normalized


@dataclass(frozen=True, slots=True)
class _NormalDenominatorReduction:
    denominator_bound: Poly
    derivative_coefficient: Poly | None
    solution_coefficient: Poly | None
    right_hand_side: Poly | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None


def _reduce_normal_denominator(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
) -> _NormalDenominatorReduction:
    r"""Reduce a weakly normalized equation to ``A*Dq + B*q = C``.

    This is the ``D = d/dx`` specialization of Bronstein's normal
    denominator theorem.  Every nonconstant polynomial in ``QQ[x]`` is
    normal, so no separate special-denominator step remains.
    """
    _, coefficient_numerator, coefficient_denominator = _as_qq_rational(
        coefficient,
        x,
    )
    _, rhs_numerator, rhs_denominator = _as_qq_rational(
        right_hand_side,
        x,
    )

    shared_denominator = coefficient_denominator.gcd(rhs_denominator)
    denominator_bound = (
        rhs_denominator.gcd(rhs_denominator.diff())
        .exquo(
            shared_denominator.gcd(shared_denominator.diff())
        )
        .monic()
    )

    derivative_coefficient = coefficient_denominator*denominator_bound
    solution_coefficient = (
        denominator_bound*coefficient_numerator
        - coefficient_denominator*denominator_bound.diff()
    )
    multiplier = (
        coefficient_denominator
        * denominator_bound
        * denominator_bound
    )
    quotient, remainder = multiplier.div(rhs_denominator)
    if not remainder.is_zero:
        return _NormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_denominator,
            remainder,
        )

    transformed_rhs = quotient*rhs_numerator
    return _NormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )


def _base_polynomial_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
) -> int:
    r"""Bound polynomial solutions of ``A*Dq + B*q = C`` over ``QQ[x]``."""
    if right_hand_side.is_zero:
        return 0

    derivative_degree = derivative_coefficient.degree()
    rhs_degree = right_hand_side.degree()
    if solution_coefficient.is_zero:
        dominant_degree = derivative_degree - 1
    else:
        dominant_degree = max(
            solution_coefficient.degree(),
            derivative_degree - 1,
        )
    bound = max(0, rhs_degree - dominant_degree)

    if (
        not solution_coefficient.is_zero
        and solution_coefficient.degree() == derivative_degree - 1
    ):
        cancellation_degree = QQ.convert(
            -solution_coefficient.LC()/derivative_coefficient.LC()
        )
        if QQ.denom(cancellation_degree) == 1:
            bound = max(
                bound,
                int(QQ.numer(cancellation_degree)),
                rhs_degree - solution_coefficient.degree(),
            )
    return bound


@dataclass(frozen=True, slots=True)
class _SPDEResult:
    solution: Poly | None
    obstruction: RationalRDEObstruction | None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_special_polynomial_differential_equation(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
) -> _SPDEResult:
    r"""Apply Rothstein's SPDE reduction, then the polynomial RDE solver."""
    x = derivative_coefficient.gens[0]
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = Poly(1, x, domain=QQ)
    beta = Poly(0, x, domain=QQ)
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient*beta.diff()
                + original_solution_coefficient*beta
                != original_rhs
            ):
                raise ValueError("SPDE zero-tail reconstruction failed")
            return _SPDEResult(beta, None)

        if current_bound < 0:
            return _SPDEResult(
                None,
                RationalRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _SPDEResult(
                None,
                RationalRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse_leading_coefficient = QQ.one/derivative_coefficient.LC()
            normalized_coefficient = solution_coefficient.mul_ground(
                inverse_leading_coefficient
            )
            normalized_rhs = right_hand_side.mul_ground(
                inverse_leading_coefficient
            )
            polynomial_result = solve_polynomial_risch_differential_equation(
                normalized_coefficient,
                normalized_rhs,
                x,
            )
            if not polynomial_result.has_solution:
                return _SPDEResult(
                    None,
                    RationalRDEObstruction.POLYNOMIAL,
                    polynomial_result,
                )
            tail = polynomial_result.solution
            if (
                not tail.is_zero
                and tail.degree() > current_bound
            ):
                return _SPDEResult(
                    None,
                    RationalRDEObstruction.SPDE_DEGREE,
                    polynomial_result,
                )
            solution = alpha*tail + beta
            if (
                original_derivative_coefficient*solution.diff()
                + original_solution_coefficient*solution
                != original_rhs
            ):
                raise ValueError("SPDE reconstruction failed")
            return _SPDEResult(
                solution,
                None,
                polynomial_result,
            )

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side*inverse).rem(
            derivative_coefficient
        )
        quotient = (
            right_hand_side - solution_coefficient*residue
        ).exquo(derivative_coefficient)

        right_hand_side = quotient - residue.diff()
        solution_coefficient += derivative_coefficient.diff()
        current_bound -= derivative_coefficient.degree()
        beta += alpha*residue
        alpha *= derivative_coefficient


def solve_rational_risch_differential_equation(
    coefficient: Expr | Poly,
    right_hand_side: Expr | Poly,
    x: Symbol,
) -> RationalRischDifferentialEquationResult:
    r"""Decide ``Dy + coefficient*y = right_hand_side`` in ``QQ(x)``.

    The algorithm is the base-field specialization of Bronstein's Chapter 6
    Risch differential equation:

    1. weakly normalize the coefficient by removing positive integer
       logarithmic residues;
    2. compute the forced normal denominator of a rational solution;
    3. bound the remaining polynomial numerator;
    4. apply Rothstein's SPDE reduction;
    5. finish with :func:`solve_polynomial_risch_differential_equation`.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    coefficient, _, _ = _as_qq_rational(coefficient, x)
    right_hand_side, _, _ = _as_qq_rational(right_hand_side, x)

    normalization_factor, normalized_coefficient = (
        _weakly_normalize_rational_rde_coefficient(coefficient, x)
    )
    normalized_rhs = cancel(
        normalization_factor.as_expr()*right_hand_side
    )

    if normalized_rhs == 0:
        zero = S.Zero
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            zero,
            None,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            Poly(1, x, domain=QQ),
            0,
        )

    denominator_reduction = _reduce_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        x,
    )
    if not denominator_reduction.succeeded:
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            None,
            RationalRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            obstruction_polynomial=(
                denominator_reduction.obstruction_polynomial
            ),
            obstruction_remainder=(
                denominator_reduction.obstruction_remainder
            ),
        )

    denominator_bound = denominator_reduction.denominator_bound
    derivative_coefficient = denominator_reduction.derivative_coefficient
    solution_coefficient = denominator_reduction.solution_coefficient
    transformed_rhs = denominator_reduction.right_hand_side
    degree_bound = _base_polynomial_rde_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )
    spde_result = _solve_special_polynomial_differential_equation(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
    )
    if spde_result.solution is None:
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_bound,
            degree_bound,
            spde_result.polynomial_result,
            spde_result.obstruction_polynomial,
            spde_result.obstruction_remainder,
        )

    solution = cancel(
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_bound.as_expr()
        )
    )
    if cancel(
        solution.diff(x)
        + coefficient*solution
        - right_hand_side
    ) != 0:
        raise ValueError("rational Risch equation reconstruction failed")
    return RationalRischDifferentialEquationResult(
        coefficient,
        right_hand_side,
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        normalized_rhs,
        denominator_bound,
        degree_bound,
        spde_result.polynomial_result,
    )


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
        candidate = x**degree/denominator.as_expr()
        column_expressions.append(
            cancel(candidate.diff(x) + coefficient*candidate)
        )

    rational_expressions = [*column_expressions, right_hand_side]
    reduced_expressions = [
        _as_qq_rational(expression, x)
        for expression in rational_expressions
    ]
    common_denominator = Poly(1, x, domain=QQ)
    for _, _, expression_denominator in reduced_expressions:
        common_denominator = common_denominator.lcm(expression_denominator)

    cleared_polynomials = [
        Poly(
            cancel(expression*common_denominator.as_expr()),
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
            [
                QQ.convert(polynomial.nth(degree))
                for polynomial in column_polynomials
            ]
            for degree in range(equation_count)
        ],
        (equation_count, unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        [
            [QQ.convert(rhs_polynomial.nth(degree))]
            for degree in range(equation_count)
        ],
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
        {
            (degree,): value
            for degree, value in enumerate(coefficients)
        },
        x,
        domain=QQ,
    )
    solution = cancel(numerator.as_expr()/denominator.as_expr())
    if cancel(
        solution.diff(x)
        + coefficient*solution
        - right_hand_side
    ) != 0:
        raise ValueError("bounded matrix RDE reconstruction failed")
    return BoundedRationalRDEMatrixResult(
        denominator,
        numerator_degree_bound,
        solution,
        coefficient_rank,
        augmented_rank,
    )
