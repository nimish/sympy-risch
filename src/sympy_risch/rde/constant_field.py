r"""Rational Risch differential equations over finite constant extensions."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel, invert

from ._common import _integer_constant
from .models import (
    PolynomialRDEObstruction,
    PolynomialRischDifferentialEquationResult,
    RationalRDEObstruction,
)


def _as_domain_rational(
    value: Expr | Poly,
    x: Symbol,
    domain: Domain,
) -> tuple[Expr, Poly, Poly]:
    """Return a reduced rational function over an exact constant field."""
    if isinstance(value, Poly):
        value = value.as_expr()
    field = domain.frac_field(x)
    element = field.from_sympy(value)
    numerator_polynomial = Poly.from_dict(
        element.numer.to_dict(),
        x,
        domain=domain,
    )
    denominator_polynomial = Poly.from_dict(
        element.denom.to_dict(),
        x,
        domain=domain,
    )
    expression = field.to_sympy(element)
    return expression, numerator_polynomial, denominator_polynomial


def _weakly_normalize_over_constant_field(
    coefficient: Expr,
    x: Symbol,
    domain: Domain,
) -> tuple[Poly, Expr]:
    r"""Apply Bronstein weak normalization over ``domain(x)``.

    For ``D=d/dx``, an irreducible factor ``p`` occurring simply in the
    denominator has logarithmic residue

    ``numerator / ((denominator/p)*Dp) mod p``.

    Every positive integer residue ``n`` is removed by subtracting
    ``D(p**n)/p**n``.  Higher-order poles need no adjustment.
    """
    coefficient, numerator, denominator = _as_domain_rational(
        coefficient,
        x,
        domain,
    )
    normalizer = Poly(1, x, domain=domain)
    for factor, multiplicity in denominator.factor_list()[1]:
        if multiplicity != 1:
            continue
        cofactor = denominator.exquo(factor)
        local_denominator = (cofactor * factor.diff()).rem(factor)
        residue = (numerator * invert(local_denominator, factor)).rem(factor)
        if residue.degree() > 0:
            continue
        integer_residue = _integer_constant(residue.nth(0), domain)
        if integer_residue is not None and integer_residue > 0:
            normalizer *= factor**integer_residue

    normalized = cancel(
        coefficient - normalizer.diff().as_expr() / normalizer.as_expr()
    )
    if (
        cancel(
            normalized
            + normalizer.diff().as_expr() / normalizer.as_expr()
            - coefficient
        )
        != 0
    ):
        raise ValueError("constant-field weak normalization failed")
    return normalizer.monic(), normalized


@dataclass(frozen=True, slots=True)
class _DomainNormalDenominatorReduction:
    denominator_bound: Poly
    derivative_coefficient: Poly | None
    solution_coefficient: Poly | None
    right_hand_side: Poly | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None

    @property
    def equation(self) -> tuple[Poly, Poly, Poly] | None:
        """Return the transformed polynomial equation when reduction succeeds."""
        if (
            self.derivative_coefficient is None
            or self.solution_coefficient is None
            or self.right_hand_side is None
        ):
            return None
        return (
            self.derivative_coefficient,
            self.solution_coefficient,
            self.right_hand_side,
        )


def _reduce_normal_denominator_over_constant_field(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
    domain: Domain,
) -> _DomainNormalDenominatorReduction:
    """Apply Bronstein's normal-denominator theorem over ``domain(x)``."""
    _, coefficient_numerator, coefficient_denominator = _as_domain_rational(
        coefficient,
        x,
        domain,
    )
    _, rhs_numerator, rhs_denominator = _as_domain_rational(
        right_hand_side,
        x,
        domain,
    )
    shared_denominator = coefficient_denominator.gcd(rhs_denominator)
    denominator_bound = (
        rhs_denominator.gcd(rhs_denominator.diff())
        .exquo(shared_denominator.gcd(shared_denominator.diff()))
        .monic()
    )
    derivative_coefficient = coefficient_denominator * denominator_bound
    solution_coefficient = (
        denominator_bound * coefficient_numerator
        - coefficient_denominator * denominator_bound.diff()
    )
    multiplier = coefficient_denominator * denominator_bound * denominator_bound
    quotient, remainder = multiplier.div(rhs_denominator)
    if not remainder.is_zero:
        return _DomainNormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_denominator,
            remainder,
        )
    return _DomainNormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        quotient * rhs_numerator,
    )


def _domain_polynomial_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
) -> int:
    """Bound solutions of ``A*Dq+B*q=C`` over an exact constant field."""
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
        domain = derivative_coefficient.domain
        cancellation_degree = _integer_constant(
            domain.exquo(
                -domain.convert(solution_coefficient.LC()),
                domain.convert(derivative_coefficient.LC()),
            ),
            domain,
        )
        if cancellation_degree is not None:
            bound = max(
                bound,
                cancellation_degree,
                rhs_degree - solution_coefficient.degree(),
            )
    return bound


def _solve_polynomial_rde_over_constant_field(
    coefficient: Poly,
    right_hand_side: Poly,
) -> PolynomialRischDifferentialEquationResult:
    """Solve ``Dq+a*q=b`` by the forced leading recurrence."""
    x = coefficient.gens[0]
    domain = coefficient.domain
    zero = Poly(0, x, domain=domain)
    if right_hand_side.is_zero:
        return PolynomialRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            0,
            zero,
            None,
        )
    if coefficient.is_zero:
        solution = Poly.from_dict(
            {
                (degree + 1,): domain.exquo(
                    domain.convert(value),
                    domain.convert(degree + 1),
                )
                for (degree,), value in right_hand_side.terms()
            },
            x,
            domain=domain,
        )
        return PolynomialRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            right_hand_side.degree() + 1,
            solution,
            None,
        )

    degree_bound = right_hand_side.degree() - coefficient.degree()
    if degree_bound < 0:
        return PolynomialRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            -1,
            None,
            PolynomialRDEObstruction.DEGREE,
            partial_solution=zero,
            residual=right_hand_side,
        )
    solution = zero
    residual = right_hand_side
    while not residual.is_zero:
        candidate_degree = residual.degree() - coefficient.degree()
        if candidate_degree < 0:
            return PolynomialRischDifferentialEquationResult(
                coefficient,
                right_hand_side,
                degree_bound,
                None,
                PolynomialRDEObstruction.NONZERO_REMAINDER,
                partial_solution=solution,
                residual=residual,
            )
        term = Poly.from_dict(
            {
                (candidate_degree,): domain.exquo(
                    domain.convert(residual.LC()),
                    domain.convert(coefficient.LC()),
                ),
            },
            x,
            domain=domain,
        )
        new_residual = residual - term.diff() - coefficient * term
        if not new_residual.is_zero and new_residual.degree() >= residual.degree():
            raise ValueError("constant-field polynomial RDE degree did not decrease")
        solution += term
        residual = new_residual
    if solution.diff() + coefficient * solution != right_hand_side:
        raise ValueError("constant-field polynomial RDE reconstruction failed")
    return PolynomialRischDifferentialEquationResult(
        coefficient,
        right_hand_side,
        degree_bound,
        solution,
        None,
    )


@dataclass(frozen=True, slots=True)
class _DomainSPDEResult:
    solution: Poly | None
    obstruction: RationalRDEObstruction | None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_spde_over_constant_field(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
) -> _DomainSPDEResult:
    """Apply Rothstein SPDE reduction over an exact constant field."""
    x = derivative_coefficient.gens[0]
    domain = derivative_coefficient.domain
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = Poly(1, x, domain=domain)
    beta = Poly(0, x, domain=domain)
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient * beta.diff()
                + original_solution_coefficient * beta
                != original_rhs
            ):
                raise ValueError("constant-field SPDE zero-tail reconstruction failed")
            return _DomainSPDEResult(beta, None)
        if current_bound < 0:
            return _DomainSPDEResult(
                None,
                RationalRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _DomainSPDEResult(
                None,
                RationalRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse_leading_coefficient = domain.exquo(
                domain.one,
                domain.convert(derivative_coefficient.LC()),
            )
            normalized_coefficient = solution_coefficient.mul_ground(
                inverse_leading_coefficient
            )
            normalized_rhs = right_hand_side.mul_ground(inverse_leading_coefficient)
            polynomial_result = _solve_polynomial_rde_over_constant_field(
                normalized_coefficient,
                normalized_rhs,
            )
            if polynomial_result.solution is None:
                return _DomainSPDEResult(
                    None,
                    RationalRDEObstruction.POLYNOMIAL,
                    polynomial_result,
                )
            tail = polynomial_result.solution
            if not tail.is_zero and tail.degree() > current_bound:
                return _DomainSPDEResult(
                    None,
                    RationalRDEObstruction.SPDE_DEGREE,
                    polynomial_result,
                )
            solution = alpha * tail + beta
            if (
                original_derivative_coefficient * solution.diff()
                + original_solution_coefficient * solution
                != original_rhs
            ):
                raise ValueError("constant-field SPDE reconstruction failed")
            return _DomainSPDEResult(solution, None, polynomial_result)

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side * inverse).rem(derivative_coefficient)
        quotient = (right_hand_side - solution_coefficient * residue).exquo(
            derivative_coefficient
        )
        right_hand_side = quotient - residue.diff()
        solution_coefficient += derivative_coefficient.diff()
        current_bound -= derivative_coefficient.degree()
        beta += alpha * residue
        alpha *= derivative_coefficient


@dataclass(frozen=True, slots=True)
class _DomainRDEResult:
    solution: Expr | None
    obstruction: RationalRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: Expr
    denominator_bound: Poly
    degree_bound: int | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None


def _solve_rde_over_constant_field(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
    domain: Domain,
) -> _DomainRDEResult:
    """Solve a scalar rational RDE over ``domain(x)``."""
    coefficient, _, _ = _as_domain_rational(coefficient, x, domain)
    right_hand_side, _, _ = _as_domain_rational(
        right_hand_side,
        x,
        domain,
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_over_constant_field(
            coefficient,
            x,
            domain,
        )
    )
    normalized_rhs = cancel(normalization_factor.as_expr() * right_hand_side)
    if normalized_rhs == 0:
        return _DomainRDEResult(
            S.Zero,
            None,
            normalization_factor,
            normalized_coefficient,
            Poly(1, x, domain=domain),
            0,
        )

    denominator_reduction = _reduce_normal_denominator_over_constant_field(
        normalized_coefficient,
        normalized_rhs,
        x,
        domain,
    )
    equation = denominator_reduction.equation
    if equation is None:
        return _DomainRDEResult(
            None,
            RationalRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            denominator_reduction.denominator_bound,
            None,
            denominator_reduction.obstruction_polynomial,
            denominator_reduction.obstruction_remainder,
        )
    (
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    ) = equation
    degree_bound = _domain_polynomial_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )
    spde_result = _solve_spde_over_constant_field(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
    )
    if spde_result.solution is None:
        return _DomainRDEResult(
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            denominator_reduction.denominator_bound,
            degree_bound,
            obstruction_polynomial=spde_result.obstruction_polynomial,
            obstruction_remainder=spde_result.obstruction_remainder,
            polynomial_result=spde_result.polynomial_result,
        )
    solution = cancel(
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_reduction.denominator_bound.as_expr()
        )
    )
    if cancel(solution.diff(x) + coefficient * solution - right_hand_side) != 0:
        raise ValueError("constant-field rational RDE reconstruction failed")
    return _DomainRDEResult(
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        denominator_reduction.denominator_bound,
        degree_bound,
        polynomial_result=spde_result.polynomial_result,
    )
