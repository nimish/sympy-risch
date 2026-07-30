r"""Coupled Risch differential equations and hypertangent real forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..admission import RischOperation, assess_risch_operation
from ..constant_fields import (
    finite_constant_field_profile,
    gaussian_constant_field,
)
from ..constant_proofs import require_preserved_constant_field
from ..tower import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
    _rebase_monomial_derivation,
)
from ._common import _field_element_is_zero, _maximum_polynomial_degree
from ._tower_polynomials import (
    _as_primitive_polynomial,
    _positive_integer_in_domain,
)
from .constant_field import _solve_rde_over_constant_field
from .hyperexponential import (
    _hyperexponential_laurent_coefficients,
    _hyperexponential_laurent_rde_extension,
    _hyperexponential_laurent_term,
)
from .models import (
    CoupledHyperexponentialLaurentRDEResult,
    CoupledHyperexponentialLaurentRDEStep,
    CoupledHypertangentPolynomialRDECorrection,
    CoupledHypertangentPolynomialRDEMode,
    CoupledHypertangentPolynomialRDEResult,
    CoupledRationalRDEResult,
    CoupledRDEObstruction,
    CoupledRischDifferentialEquationResult,
    HyperexponentialLaurentRDEObstruction,
    HyperexponentialRDEObstruction,
    PrimitiveRDEObstruction,
    RationalRDEObstruction,
)
from .rational import _as_qq_rational
from .scalar import _solve_admitted_risch_differential_equation

if TYPE_CHECKING:
    from ..gaussian import HypertangentGaussianNormalForm


def _conjugate_gaussian_polynomial(polynomial: Poly) -> Poly:
    """Conjugate coefficients of a polynomial over ``QQ_I``."""
    x = polynomial.gens[0]
    return Poly(
        polynomial.as_expr().xreplace({I: -I}),
        x,
        domain=QQ_I,
    )


def _real_candidate_denominator(
    normalization_factor: Poly,
    denominator_bound: Poly,
) -> tuple[Poly, int]:
    """Return the least real common denominator and numerator-degree shift."""
    x = normalization_factor.gens[0]
    gaussian_denominator = normalization_factor * denominator_bound
    conjugate_denominator = _conjugate_gaussian_polynomial(gaussian_denominator)
    real_common = gaussian_denominator.lcm(conjugate_denominator).monic()
    real_coefficients = {}
    for monomial, coefficient in real_common.terms():
        coefficient = QQ_I.convert(coefficient)
        if coefficient.y != 0:
            raise ValueError(
                "Gaussian common denominator unexpectedly has nonreal coefficients"
            )
        real_coefficients[monomial] = QQ.convert(coefficient.x)
    real_denominator = Poly.from_dict(
        real_coefficients,
        x,
        domain=QQ,
    ).monic()
    numerator_degree_shift = real_common.degree() - gaussian_denominator.degree()
    return real_denominator, numerator_degree_shift


def _split_gaussian_rational(
    value: Expr,
    x: Symbol,
) -> tuple[Expr, Expr]:
    """Split a member of ``QQ_I(x)`` into canonical real rational parts."""
    conjugate = cancel(value.xreplace({I: -I}))
    real_part = cancel((value + conjugate) / 2)
    imaginary_part = cancel((value - conjugate) / (2 * I))
    real_part, _, _ = _as_qq_rational(real_part, x)
    imaginary_part, _, _ = _as_qq_rational(imaginary_part, x)
    return real_part, imaginary_part


def _solve_coupled_rational_rotation(
    lambda_coefficient: Expr,
    real_rhs: Expr,
    imaginary_rhs: Expr,
    x: Symbol,
) -> CoupledRationalRDEResult:
    """Solve a rational rotation through one Gaussian scalar RDE."""
    scalar_result = _solve_rde_over_constant_field(
        I * lambda_coefficient,
        real_rhs + I * imaginary_rhs,
        x,
        QQ_I,
    )
    if real_rhs == 0 and imaginary_rhs == 0:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            S.Zero,
            S.Zero,
            None,
            Poly(1, x, domain=QQ),
            0,
            gaussian_normalization_factor=(scalar_result.normalization_factor),
            gaussian_normalized_coefficient=(scalar_result.normalized_coefficient),
            gaussian_denominator_bound=(scalar_result.denominator_bound),
        )
    real_denominator, numerator_degree_shift = _real_candidate_denominator(
        scalar_result.normalization_factor,
        scalar_result.denominator_bound,
    )
    real_degree_bound = (
        None
        if scalar_result.degree_bound is None
        else scalar_result.degree_bound + numerator_degree_shift
    )
    obstruction_map = {
        RationalRDEObstruction.DENOMINATOR: (CoupledRDEObstruction.DENOMINATOR),
        RationalRDEObstruction.SPDE_DEGREE: CoupledRDEObstruction.DEGREE,
        RationalRDEObstruction.SPDE_DIVISIBILITY: (
            CoupledRDEObstruction.NONZERO_REMAINDER
        ),
        RationalRDEObstruction.POLYNOMIAL: (CoupledRDEObstruction.NONZERO_REMAINDER),
    }
    if scalar_result.solution is None:
        if scalar_result.obstruction is None:
            raise ValueError("a failed scalar RDE lacks its obstruction")
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            obstruction_map[scalar_result.obstruction],
            real_denominator,
            real_degree_bound,
            gaussian_normalization_factor=(scalar_result.normalization_factor),
            gaussian_normalized_coefficient=(scalar_result.normalized_coefficient),
            gaussian_denominator_bound=(scalar_result.denominator_bound),
            gaussian_obstruction=scalar_result.obstruction,
            gaussian_obstruction_polynomial=(scalar_result.obstruction_polynomial),
            gaussian_obstruction_remainder=(scalar_result.obstruction_remainder),
        )

    real_solution, imaginary_solution = _split_gaussian_rational(
        scalar_result.solution,
        x,
    )
    if (
        cancel(
            real_solution.diff(x) - lambda_coefficient * imaginary_solution - real_rhs
        )
        != 0
        or cancel(
            imaginary_solution.diff(x)
            + lambda_coefficient * real_solution
            - imaginary_rhs
        )
        != 0
    ):
        raise ValueError("rational-rotation coupled RDE reconstruction failed")
    return CoupledRationalRDEResult(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        real_solution,
        imaginary_solution,
        None,
        real_denominator,
        real_degree_bound,
        gaussian_normalization_factor=scalar_result.normalization_factor,
        gaussian_normalized_coefficient=(scalar_result.normalized_coefficient),
        gaussian_denominator_bound=scalar_result.denominator_bound,
    )


def solve_coupled_rational_rde(
    lambda_coefficient: Expr | Poly,
    real_right_hand_side: Expr | Poly,
    imaginary_right_hand_side: Expr | Poly,
    x: Symbol,
) -> CoupledRationalRDEResult:
    r"""Solve a real coupled RDE with rational rotation coefficient.

    This decides

    ``Dp - lambda*q = u`` and ``Dq + lambda*p = v``

    in ``QQ(x)`` for nonzero ``lambda``.  Polynomial ``lambda`` uses the
    direct real denominator theorem and two-coordinate leading recurrence.
    Rational ``lambda`` is transformed to

    ``Dz + i*lambda*z = u + i*v``

    over ``QQ_I(x)``.  Bronstein weak normalization, normal-denominator
    reduction, SPDE, and a forced polynomial recurrence decide that scalar
    equation, after which its solution is split exactly back into ``p,q``.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    lambda_expression, lambda_numerator, lambda_denominator = _as_qq_rational(
        lambda_coefficient, x
    )
    if lambda_numerator.is_zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs, real_numerator, real_denominator = _as_qq_rational(
        real_right_hand_side,
        x,
    )
    imaginary_rhs, imaginary_numerator, imaginary_denominator = _as_qq_rational(
        imaginary_right_hand_side, x
    )
    if lambda_denominator.degree() > 0:
        return _solve_coupled_rational_rotation(
            lambda_expression,
            real_rhs,
            imaginary_rhs,
            x,
        )
    lambda_coefficient = Poly(lambda_expression, x, domain=QQ)

    common_denominator = real_denominator.lcm(imaginary_denominator)
    common_real_numerator = real_numerator * common_denominator.exquo(real_denominator)
    common_imaginary_numerator = imaginary_numerator * common_denominator.exquo(
        imaginary_denominator
    )
    # At a finite pole, differentiation raises the order while multiplication
    # by polynomial lambda does not.  A right-hand-side pole of order e
    # therefore forces a solution pole of order e - 1.  For a common
    # denominator D, h = gcd(D, D') has precisely those exponents.  The test
    # D | h**2 rejects a simple pole before any coefficient solving.
    denominator_bound = common_denominator.gcd(common_denominator.diff()).monic()
    quotient, denominator_remainder = (denominator_bound * denominator_bound).div(
        common_denominator
    )
    if not denominator_remainder.is_zero:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            CoupledRDEObstruction.DENOMINATOR,
            denominator_bound,
            obstruction_polynomial=common_denominator,
            obstruction_remainder=denominator_remainder,
        )

    # Write p=P/h and q=Q/h and clear h**2:
    #
    # h*P' - h'*P - lambda*h*Q = U
    # h*Q' - h'*Q + lambda*h*P = V.
    transformed_real_rhs = quotient * common_real_numerator
    transformed_imaginary_rhs = quotient * common_imaginary_numerator
    rotation = lambda_coefficient * denominator_bound
    residual_degree = _maximum_polynomial_degree(
        transformed_real_rhs,
        transformed_imaginary_rhs,
    )
    if residual_degree is None:
        zero = S.Zero
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            zero,
            zero,
            None,
            denominator_bound,
            0,
        )

    degree_bound = residual_degree - rotation.degree()
    real_numerator_solution = Poly(0, x, domain=QQ)
    imaginary_numerator_solution = Poly(0, x, domain=QQ)
    real_residual = transformed_real_rhs
    imaginary_residual = transformed_imaginary_rhs
    if degree_bound < 0:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            CoupledRDEObstruction.DEGREE,
            denominator_bound,
            degree_bound,
            real_numerator_solution,
            imaginary_numerator_solution,
            real_residual,
            imaginary_residual,
        )

    current_bound = degree_bound
    while not (real_residual.is_zero and imaginary_residual.is_zero):
        residual_degree = _maximum_polynomial_degree(
            real_residual,
            imaginary_residual,
        )
        if residual_degree is None:
            raise ValueError("a nonzero coupled residual has no degree")
        candidate_degree = residual_degree - rotation.degree()
        if candidate_degree < 0 or candidate_degree > current_bound:
            if (
                denominator_bound * real_numerator_solution.diff()
                - denominator_bound.diff() * real_numerator_solution
                - rotation * imaginary_numerator_solution
                + real_residual
                != transformed_real_rhs
                or denominator_bound * imaginary_numerator_solution.diff()
                - denominator_bound.diff() * imaginary_numerator_solution
                + rotation * real_numerator_solution
                + imaginary_residual
                != transformed_imaginary_rhs
            ):
                raise ValueError("coupled RDE reduction invariant failed")
            return CoupledRationalRDEResult(
                lambda_coefficient,
                real_rhs,
                imaginary_rhs,
                None,
                None,
                CoupledRDEObstruction.NONZERO_REMAINDER,
                denominator_bound,
                degree_bound,
                real_numerator_solution,
                imaginary_numerator_solution,
                real_residual,
                imaginary_residual,
            )

        real_leading_residual = real_residual.nth(residual_degree)
        imaginary_leading_residual = imaginary_residual.nth(residual_degree)
        # The rotation terms have one degree more than the derivative terms
        # (at least), so their invertible 2-by-2 leading matrix uniquely fixes
        # the next coefficients of P and Q.
        real_term = Poly.from_dict(
            {(candidate_degree,): (imaginary_leading_residual / rotation.LC())},
            x,
            domain=QQ,
        )
        imaginary_term = Poly.from_dict(
            {(candidate_degree,): (-real_leading_residual / rotation.LC())},
            x,
            domain=QQ,
        )
        new_real_residual = (
            real_residual
            - denominator_bound * real_term.diff()
            + denominator_bound.diff() * real_term
            + rotation * imaginary_term
        )
        new_imaginary_residual = (
            imaginary_residual
            - denominator_bound * imaginary_term.diff()
            + denominator_bound.diff() * imaginary_term
            - rotation * real_term
        )
        new_degree = _maximum_polynomial_degree(
            new_real_residual,
            new_imaginary_residual,
        )
        if new_degree is not None and new_degree >= residual_degree:
            raise ValueError("coupled RDE residual degree did not decrease")

        real_numerator_solution += real_term
        imaginary_numerator_solution += imaginary_term
        real_residual = new_real_residual
        imaginary_residual = new_imaginary_residual
        current_bound = candidate_degree - 1

    real_solution = cancel(
        real_numerator_solution.as_expr() / denominator_bound.as_expr()
    )
    imaginary_solution = cancel(
        imaginary_numerator_solution.as_expr() / denominator_bound.as_expr()
    )
    lambda_expression = lambda_coefficient.as_expr()
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
        raise ValueError("coupled rational RDE reconstruction failed")
    return CoupledRationalRDEResult(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        real_solution,
        imaginary_solution,
        None,
        denominator_bound,
        degree_bound,
    )


def solve_hyperexponential_coupled_laurent_rde(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledHyperexponentialLaurentRDEResult:
    r"""Solve a coupled Laurent RDE directly at a hyperexponential level.

    This is the modewise form of Bronstein's ``CoupledDECancelExp``. If
    ``Dt = eta*t`` and both matrix coefficients belong to the lower field,
    the coefficient pair at power ``m`` satisfies a lower coupled system with
    diagonal coefficient ``a + m*eta``. No Gaussian tower is constructed by
    this recurrence.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    diagonal_element, diagonal_terms = _hyperexponential_laurent_coefficients(
        diagonal_coefficient,
        extension,
        "diagonal coefficient",
    )
    rotation_element, rotation_terms = _hyperexponential_laurent_coefficients(
        rotation_coefficient,
        extension,
        "rotation coefficient",
    )
    if any(power != 0 for power in diagonal_terms):
        raise NotImplementedError(
            "direct hyperexponential coupled cancellation requires the "
            "diagonal coefficient to belong to the lower field"
        )
    if any(power != 0 for power in rotation_terms):
        raise NotImplementedError(
            "direct hyperexponential coupled cancellation requires the "
            "rotation coefficient to belong to the lower field"
        )
    real_rhs, real_terms = _hyperexponential_laurent_coefficients(
        real_right_hand_side,
        extension,
        "real right-hand side",
    )
    imaginary_rhs, imaginary_terms = _hyperexponential_laurent_coefficients(
        imaginary_right_hand_side,
        extension,
        "imaginary right-hand side",
    )
    diagonal_lower = diagonal_terms.get(0, parent.zero)
    rotation_lower = rotation_terms.get(0, parent.zero)
    eta = parent.convert(extension.generator_derivative.nth(1))
    lower_derivation = derivation.lower()
    partial_real = field.zero
    partial_imaginary = field.zero
    real_residual = field.zero
    imaginary_residual = field.zero
    steps: list[CoupledHyperexponentialLaurentRDEStep] = []

    for power in sorted(set(real_terms) | set(imaginary_terms)):
        real_coefficient = real_terms.get(power, parent.zero)
        imaginary_coefficient = imaginary_terms.get(power, parent.zero)
        shifted_diagonal = diagonal_lower + parent.convert(power) * eta
        lower_result = _solve_general_coupled_risch_differential_equation(
            shifted_diagonal,
            rotation_lower,
            real_coefficient,
            imaginary_coefficient,
            lower_derivation,
        )
        steps.append(
            CoupledHyperexponentialLaurentRDEStep(
                power,
                real_coefficient,
                imaginary_coefficient,
                shifted_diagonal,
                lower_result,
            )
        )
        if lower_result.has_solution:
            if (
                lower_result.real_solution is None
                or lower_result.imaginary_solution is None
            ):
                raise ValueError("a solved lower coupled RDE lacks its solution pair")
            partial_real += _hyperexponential_laurent_term(
                lower_result.real_solution,
                power,
                extension,
            )
            partial_imaginary += _hyperexponential_laurent_term(
                lower_result.imaginary_solution,
                power,
                extension,
            )
        else:
            real_residual += _hyperexponential_laurent_term(
                real_coefficient,
                power,
                extension,
            )
            imaginary_residual += _hyperexponential_laurent_term(
                imaginary_coefficient,
                power,
                extension,
            )

    has_obstruction = real_residual != field.zero or imaginary_residual != field.zero
    result = CoupledHyperexponentialLaurentRDEResult(
        derivation,
        diagonal_element,
        rotation_element,
        real_rhs,
        imaginary_rhs,
        None if has_obstruction else partial_real,
        None if has_obstruction else partial_imaginary,
        partial_real,
        partial_imaginary,
        real_residual,
        imaginary_residual,
        tuple(steps),
        CoupledRDEObstruction.LOWER_FIELD if has_obstruction else None,
    )
    if not result.reconstructs:
        raise ValueError(
            "direct hyperexponential coupled recurrence failed to reconstruct"
        )
    return result


def _hypertangent_extension(derivation: Derivation) -> MonomialExtension:
    """Validate a hypertangent top level for coupled polynomial solving."""
    if derivation.level < 1:
        raise TypeError(
            "hypertangent coupled RDE solving requires a nonconstant lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "hypertangent coupled RDE solving does not support algebraic tower levels"
        )
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError("hypertangent coupled RDE solving requires Dt = eta*(1+t**2)")
    require_preserved_constant_field(
        derivation,
        "hypertangent coupled RDE solving",
    )
    return extension


def _polynomial_at_i_coordinates(
    polynomial: Poly,
) -> tuple[FieldElement, FieldElement]:
    """Return the two real coordinates of ``polynomial(i)``."""
    domain = polynomial.domain
    real = domain.zero
    imaginary = domain.zero
    for (power,), coefficient in polynomial.terms():
        value = domain.convert(coefficient)
        phase = power % 4
        if phase == 0:
            real += value
        elif phase == 1:
            imaginary += value
        elif phase == 2:
            real -= value
        else:
            imaginary -= value
    return real, imaginary


def _zero_hypertangent_coupled_result(
    derivation: Derivation,
    mode: CoupledHypertangentPolynomialRDEMode,
    diagonal_coefficient: Poly,
    rotation_coefficient: FieldElement,
    real_right_hand_side: Poly,
    imaginary_right_hand_side: Poly,
    degree_bound: int,
    obstruction: CoupledRDEObstruction,
    *,
    boundary_result: CoupledRischDifferentialEquationResult | None = None,
    recursive_result: CoupledHypertangentPolynomialRDEResult | None = None,
) -> CoupledHypertangentPolynomialRDEResult:
    """Build a reconstructing failure certificate with zero partial solution."""
    zero = real_right_hand_side.zero
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        mode,
        diagonal_coefficient,
        rotation_coefficient,
        real_right_hand_side,
        imaginary_right_hand_side,
        degree_bound,
        None,
        None,
        zero,
        zero,
        real_right_hand_side,
        imaginary_right_hand_side,
        obstruction,
        boundary_result=boundary_result,
        recursive_result=recursive_result,
    )
    if not result.reconstructs:
        raise ValueError("hypertangent coupled failure certificate did not reconstruct")
    return result


def solve_hypertangent_coupled_cancellation_rde(
    base_diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
    degree_bound: int,
) -> CoupledHypertangentPolynomialRDEResult:
    r"""Implement Bronstein's ``CoupledDECancelTan`` recurrence.

    With ``Dt = eta*(1+t**2)``, this decides polynomial solutions of degree at
    most ``n`` for the cancellation-shaped system whose diagonal is
    ``b0 - n*eta*t`` and whose rotation coefficient is ``b2`` in the lower
    field. Evaluation at the place ``t = i`` creates one lower coupled
    system. Exact division by ``t - i`` is performed in real coordinates
    using ``t**2 + 1`` before recurring with bound ``n - 1``.
    """
    if (
        not isinstance(degree_bound, int)
        or isinstance(degree_bound, bool)
        or degree_bound < 0
    ):
        raise ValueError("the degree bound must be a nonnegative int")
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    eta = parent.convert(extension.generator_derivative.nth(2))
    base_diagonal = _coerce(parent, base_diagonal_coefficient)
    rotation = _coerce(parent, rotation_coefficient)
    real_rhs = _as_primitive_polynomial(
        real_right_hand_side,
        extension,
    )
    imaginary_rhs = _as_primitive_polynomial(
        imaginary_right_hand_side,
        extension,
    )
    diagonal = Poly(
        parent.to_sympy(base_diagonal) - degree_bound * parent.to_sympy(eta) * t,
        t,
        domain=parent,
    )
    maximum_rhs_degree = _maximum_polynomial_degree(real_rhs, imaginary_rhs)
    if maximum_rhs_degree is not None and maximum_rhs_degree > degree_bound:
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            CoupledRDEObstruction.DEGREE,
        )

    if degree_bound == 0:
        boundary_result = _solve_general_coupled_risch_differential_equation(
            base_diagonal,
            rotation,
            parent.convert(real_rhs.nth(0)),
            parent.convert(imaginary_rhs.nth(0)),
            derivation.lower(),
        )
        if not boundary_result.has_solution:
            return _zero_hypertangent_coupled_result(
                derivation,
                CoupledHypertangentPolynomialRDEMode.CANCELLATION,
                diagonal,
                rotation,
                real_rhs,
                imaginary_rhs,
                degree_bound,
                CoupledRDEObstruction.LOWER_FIELD,
                boundary_result=boundary_result,
            )
        if (
            boundary_result.real_solution is None
            or boundary_result.imaginary_solution is None
        ):
            raise ValueError("a solved tangent boundary system lacks its solution pair")
        real_solution = Poly(
            parent.to_sympy(boundary_result.real_solution),
            t,
            domain=parent,
        )
        imaginary_solution = Poly(
            parent.to_sympy(boundary_result.imaginary_solution),
            t,
            domain=parent,
        )
        zero = real_rhs.zero
        result = CoupledHypertangentPolynomialRDEResult(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            real_solution,
            imaginary_solution,
            real_solution,
            imaginary_solution,
            zero,
            zero,
            None,
            boundary_result=boundary_result,
        )
        if not result.reconstructs:
            raise ValueError("tangent cancellation base case failed to reconstruct")
        return result

    real_at_i, real_at_i_imaginary = _polynomial_at_i_coordinates(real_rhs)
    imaginary_at_i_real, imaginary_at_i = _polynomial_at_i_coordinates(imaginary_rhs)
    boundary_real = real_at_i - imaginary_at_i
    boundary_imaginary = real_at_i_imaginary + imaginary_at_i_real
    shifted_rotation = rotation - parent.convert(degree_bound) * eta
    boundary_result = _solve_general_coupled_risch_differential_equation(
        base_diagonal,
        shifted_rotation,
        boundary_real,
        boundary_imaginary,
        derivation.lower(),
    )
    if not boundary_result.has_solution:
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            CoupledRDEObstruction.LOWER_FIELD,
            boundary_result=boundary_result,
        )
    if (
        boundary_result.real_solution is None
        or boundary_result.imaginary_solution is None
    ):
        raise ValueError("a solved tangent boundary system lacks its solution pair")
    s1 = parent.convert(boundary_result.real_solution)
    s2 = parent.convert(boundary_result.imaginary_solution)
    n_eta = parent.convert(degree_bound) * eta
    real_numerator = (
        real_rhs
        - Poly(parent.to_sympy(boundary_real), t, domain=parent)
        + Poly(
            parent.to_sympy(n_eta) * (parent.to_sympy(s1) * t + parent.to_sympy(s2)),
            t,
            domain=parent,
        )
    )
    imaginary_numerator = (
        imaginary_rhs
        - Poly(parent.to_sympy(boundary_imaginary), t, domain=parent)
        + Poly(
            parent.to_sympy(n_eta) * (parent.to_sympy(s2) * t - parent.to_sympy(s1)),
            t,
            domain=parent,
        )
    )
    top_generator = Poly(t, t, domain=parent)
    special = Poly(t**2 + 1, t, domain=parent)
    reduced_real, real_remainder = (
        top_generator * real_numerator - imaginary_numerator
    ).div(special)
    reduced_imaginary, imaginary_remainder = (
        real_numerator + top_generator * imaginary_numerator
    ).div(special)
    if not real_remainder.is_zero or not imaginary_remainder.is_zero:
        raise ValueError("tangent cancellation numerator is not divisible by t - i")

    recursive_result = solve_hypertangent_coupled_cancellation_rde(
        base_diagonal,
        rotation + eta,
        reduced_real,
        reduced_imaginary,
        derivation,
        degree_bound - 1,
    )
    if not recursive_result.has_solution:
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            recursive_result.obstruction or CoupledRDEObstruction.LOWER_FIELD,
            boundary_result=boundary_result,
            recursive_result=recursive_result,
        )
    if (
        recursive_result.real_solution is None
        or recursive_result.imaginary_solution is None
    ):
        raise ValueError("a solved recursive tangent system lacks its solution pair")
    real_solution = (
        recursive_result.real_solution * top_generator
        + recursive_result.imaginary_solution
        + Poly(parent.to_sympy(s1), t, domain=parent)
    )
    imaginary_solution = (
        recursive_result.imaginary_solution * top_generator
        - recursive_result.real_solution
        + Poly(parent.to_sympy(s2), t, domain=parent)
    )
    if (
        real_solution.degree() > degree_bound
        or imaginary_solution.degree() > degree_bound
    ):
        raise ValueError("tangent cancellation recurrence exceeded its degree bound")
    zero = real_rhs.zero
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        CoupledHypertangentPolynomialRDEMode.CANCELLATION,
        diagonal,
        rotation,
        real_rhs,
        imaginary_rhs,
        degree_bound,
        real_solution,
        imaginary_solution,
        real_solution,
        imaginary_solution,
        zero,
        zero,
        None,
        boundary_result=boundary_result,
        recursive_result=recursive_result,
    )
    if not result.reconstructs:
        raise ValueError("tangent cancellation recurrence failed to reconstruct")
    return result


def _solve_hypertangent_coupled_no_cancellation(
    diagonal_coefficient: FieldElement,
    rotation_coefficient: FieldElement,
    real_right_hand_side: Poly,
    imaginary_right_hand_side: Poly,
    derivation: Derivation,
) -> CoupledHypertangentPolynomialRDEResult:
    """Solve the lower-coefficient nonlinear polynomial branch."""
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    eta = parent.convert(extension.generator_derivative.nth(2))
    diagonal = Poly(
        parent.to_sympy(diagonal_coefficient),
        t,
        domain=parent,
    )
    zero = real_right_hand_side.zero
    real_solution = zero
    imaginary_solution = zero
    real_residual = real_right_hand_side
    imaginary_residual = imaginary_right_hand_side
    corrections: list[CoupledHypertangentPolynomialRDECorrection] = []
    maximum_degree = _maximum_polynomial_degree(
        real_right_hand_side,
        imaginary_right_hand_side,
    )
    degree_bound = max(0, (maximum_degree or 0) - 1)

    while True:
        residual_degree = _maximum_polynomial_degree(
            real_residual,
            imaginary_residual,
        )
        if residual_degree is None or residual_degree <= 0:
            break
        correction_degree = residual_degree - 1
        if correction_degree <= 0:
            result = CoupledHypertangentPolynomialRDEResult(
                derivation,
                CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
                diagonal,
                rotation_coefficient,
                real_right_hand_side,
                imaginary_right_hand_side,
                degree_bound,
                None,
                None,
                real_solution,
                imaginary_solution,
                real_residual,
                imaginary_residual,
                CoupledRDEObstruction.DEGREE,
                tuple(corrections),
            )
            if not result.reconstructs:
                raise ValueError(
                    "tangent no-cancellation obstruction did not reconstruct"
                )
            return result
        divisor = parent.convert(correction_degree) * eta
        real_coefficient = parent.convert(real_residual.nth(residual_degree)) / divisor
        imaginary_coefficient = (
            parent.convert(imaginary_residual.nth(residual_degree)) / divisor
        )
        real_correction = Poly.from_dict(
            {(correction_degree,): real_coefficient},
            t,
            domain=parent,
        )
        imaginary_correction = Poly.from_dict(
            {(correction_degree,): imaginary_coefficient},
            t,
            domain=parent,
        )
        new_real_residual = (
            real_residual
            - derivation.polynomial(real_correction)
            - diagonal * real_correction
            + imaginary_correction.mul_ground(rotation_coefficient)
        )
        new_imaginary_residual = (
            imaginary_residual
            - derivation.polynomial(imaginary_correction)
            - real_correction.mul_ground(rotation_coefficient)
            - diagonal * imaginary_correction
        )
        new_degree = _maximum_polynomial_degree(
            new_real_residual,
            new_imaginary_residual,
        )
        if new_degree is not None and new_degree >= residual_degree:
            raise ValueError("tangent no-cancellation recurrence did not lower degree")
        corrections.append(
            CoupledHypertangentPolynomialRDECorrection(
                correction_degree,
                real_coefficient,
                imaginary_coefficient,
                real_residual,
                imaginary_residual,
                new_real_residual,
                new_imaginary_residual,
            )
        )
        real_solution += real_correction
        imaginary_solution += imaginary_correction
        real_residual = new_real_residual
        imaginary_residual = new_imaginary_residual

    lower_result = _solve_general_coupled_risch_differential_equation(
        diagonal_coefficient,
        rotation_coefficient,
        parent.convert(real_residual.nth(0)),
        parent.convert(imaginary_residual.nth(0)),
        derivation.lower(),
    )
    if not lower_result.has_solution:
        result = CoupledHypertangentPolynomialRDEResult(
            derivation,
            CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
            diagonal,
            rotation_coefficient,
            real_right_hand_side,
            imaginary_right_hand_side,
            degree_bound,
            None,
            None,
            real_solution,
            imaginary_solution,
            real_residual,
            imaginary_residual,
            CoupledRDEObstruction.LOWER_FIELD,
            tuple(corrections),
            boundary_result=lower_result,
        )
        if not result.reconstructs:
            raise ValueError("tangent lower-field obstruction did not reconstruct")
        return result
    if lower_result.real_solution is None or lower_result.imaginary_solution is None:
        raise ValueError("a solved tangent constant system lacks its solution pair")
    real_solution += Poly(
        parent.to_sympy(lower_result.real_solution),
        t,
        domain=parent,
    )
    imaginary_solution += Poly(
        parent.to_sympy(lower_result.imaginary_solution),
        t,
        domain=parent,
    )
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
        diagonal,
        rotation_coefficient,
        real_right_hand_side,
        imaginary_right_hand_side,
        degree_bound,
        real_solution,
        imaginary_solution,
        real_solution,
        imaginary_solution,
        zero,
        zero,
        None,
        tuple(corrections),
        boundary_result=lower_result,
    )
    if not result.reconstructs:
        raise ValueError("tangent no-cancellation recurrence failed to reconstruct")
    return result


def solve_hypertangent_coupled_polynomial_rde(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledHypertangentPolynomialRDEResult:
    r"""Decide implemented polynomial coupled branches for ``Dt=eta(1+t²)``.

    Lower-field matrix coefficients use the forced nonlinear leading
    recurrence. A diagonal of the form ``b0 - n*eta*t`` with positive integer
    ``n`` dispatches to Bronstein's direct cancellation recurrence.
    """
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    diagonal = _as_primitive_polynomial(
        diagonal_coefficient,
        extension,
    )
    rotation = _as_primitive_polynomial(
        rotation_coefficient,
        extension,
    )
    real_rhs = _as_primitive_polynomial(
        real_right_hand_side,
        extension,
    )
    imaginary_rhs = _as_primitive_polynomial(
        imaginary_right_hand_side,
        extension,
    )
    if not rotation.is_zero and rotation.degree() > 0:
        raise NotImplementedError(
            "hypertangent polynomial coupled solving currently requires "
            "the rotation coefficient to belong to the lower field"
        )
    rotation_lower = parent.convert(rotation.nth(0))
    if diagonal.is_zero or diagonal.degree() <= 0:
        return _solve_hypertangent_coupled_no_cancellation(
            parent.convert(diagonal.nth(0)),
            rotation_lower,
            real_rhs,
            imaginary_rhs,
            derivation,
        )
    if diagonal.degree() != 1:
        raise NotImplementedError(
            "the hypertangent diagonal coefficient is outside the "
            "implemented cancellation shapes"
        )
    eta = parent.convert(extension.generator_derivative.nth(2))
    cancellation_degree = _positive_integer_in_domain(
        -parent.convert(diagonal.nth(1)) / eta,
        parent,
    )
    if cancellation_degree is None:
        raise NotImplementedError(
            "the hypertangent diagonal coefficient is not b0 - n*eta*t "
            "for a positive integer n"
        )
    return solve_hypertangent_coupled_cancellation_rde(
        parent.convert(diagonal.nth(0)),
        rotation_lower,
        real_rhs,
        imaginary_rhs,
        derivation,
        cancellation_degree,
    )


def _coupled_obstruction_from_scalar(
    obstruction: (
        RationalRDEObstruction
        | PrimitiveRDEObstruction
        | HyperexponentialRDEObstruction
        | HyperexponentialLaurentRDEObstruction
    ),
) -> CoupledRDEObstruction:
    """Map a scalar Gaussian RDE obstruction to the real coupled contract."""
    if obstruction in {
        RationalRDEObstruction.DENOMINATOR,
        PrimitiveRDEObstruction.DENOMINATOR,
        HyperexponentialRDEObstruction.DENOMINATOR,
    }:
        return CoupledRDEObstruction.DENOMINATOR
    if obstruction in {
        RationalRDEObstruction.SPDE_DEGREE,
        PrimitiveRDEObstruction.SPDE_DEGREE,
        HyperexponentialRDEObstruction.SPDE_DEGREE,
    }:
        return CoupledRDEObstruction.DEGREE
    return CoupledRDEObstruction.NONZERO_REMAINDER


def _require_supported_coupled_tower(
    derivation: Derivation,
) -> None:
    """Replay admission for a real tower before Gaussian scalar transport."""
    admission = assess_risch_operation(derivation, RischOperation.COUPLED_RDE)
    if not admission.accepted:
        if admission.reason is None:
            raise RuntimeError("unsupported coupled RDE admission omitted its reason")
        raise NotImplementedError(
            "recursive coupled RDEs require an admitted scalar tower prefix: "
            f"{admission.reason.value}: {admission.detail}"
        )
    profile = finite_constant_field_profile(derivation.tower.constants)
    if profile is None or not profile.is_real:
        raise NotImplementedError(
            "real coupled RDE solving requires a provably real finite "
            "algebraic constant field"
        )


def _coupled_tower_is_gaussian_safe(derivation: Derivation) -> bool:
    """Whether adjoining ``i`` preserves every active monomial family."""
    return all(
        isinstance(extension, MonomialExtension)
        and extension.family
        in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
        }
        for extension in derivation.tower.levels[1 : derivation.level + 1]
    )


@dataclass(frozen=True, slots=True)
class _GaussianCoupledTransport:
    """One reusable exact complexification of a fixed real tower position."""

    source_derivation: Derivation
    target_derivation: Derivation
    normal_form: HypertangentGaussianNormalForm | None


def _build_gaussian_coupled_transport(
    derivation: Derivation,
) -> _GaussianCoupledTransport:
    """Build Gaussian coordinates after source admission has been checked."""
    normal_form = None
    if _coupled_tower_is_gaussian_safe(derivation):
        target_derivation = _rebase_monomial_derivation(
            derivation,
            gaussian_constant_field(derivation.tower.constants),
        )
    else:
        from ..gaussian import build_hypertangent_gaussian_normal_form

        normal_form = build_hypertangent_gaussian_normal_form(derivation)
        target_derivation = normal_form.target_derivation
    return _GaussianCoupledTransport(
        derivation,
        target_derivation,
        normal_form,
    )


def _prepare_gaussian_coupled_transport(
    derivation: Derivation,
) -> _GaussianCoupledTransport:
    """Validate and complexify one source position for several coupled solves."""
    _require_supported_coupled_tower(derivation)
    return _build_gaussian_coupled_transport(derivation)


def _has_direct_hyperexponential_laurent_shape(
    diagonal_coefficient: FieldElement,
    rotation_coefficient: FieldElement,
    real_right_hand_side: FieldElement,
    imaginary_right_hand_side: FieldElement,
    extension: MonomialExtension,
) -> bool:
    """Whether Chapter 8 modewise cancellation accepts all four inputs."""
    try:
        _, diagonal_terms = _hyperexponential_laurent_coefficients(
            diagonal_coefficient,
            extension,
            "diagonal coefficient",
        )
        _, rotation_terms = _hyperexponential_laurent_coefficients(
            rotation_coefficient,
            extension,
            "rotation coefficient",
        )
        _hyperexponential_laurent_coefficients(
            real_right_hand_side,
            extension,
            "real right-hand side",
        )
        _hyperexponential_laurent_coefficients(
            imaginary_right_hand_side,
            extension,
            "imaginary right-hand side",
        )
    except NotImplementedError:
        return False
    return all(power == 0 for power in diagonal_terms) and all(
        power == 0 for power in rotation_terms
    )


def _split_gaussian_tower_element(
    value: FieldElement,
    gaussian_field: Domain,
    real_field: Domain,
) -> tuple[FieldElement, FieldElement]:
    """Split a Gaussian tower element into exact real-field coordinates."""
    expression = gaussian_field.to_sympy(value)
    conjugate = expression.xreplace({I: -I})
    real_expression = cancel((expression + conjugate) / 2)
    imaginary_expression = cancel((expression - conjugate) / (2 * I))
    real_part = _coerce(real_field, real_expression)
    imaginary_part = _coerce(real_field, imaginary_expression)
    reconstructed = gaussian_field.from_sympy(
        real_field.to_sympy(real_part) + I * real_field.to_sympy(imaginary_part)
    )
    if not _field_element_is_zero(
        reconstructed - value,
        gaussian_field,
    ):
        raise ValueError("Gaussian tower coordinates failed to reconstruct")
    return real_part, imaginary_part


def _solve_general_coupled_risch_differential_equation(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
    *,
    gaussian_transport: _GaussianCoupledTransport | None = None,
) -> CoupledRischDifferentialEquationResult:
    """Solve the general real two-coordinate RDE over supported tower shapes."""
    if not isinstance(derivation, Derivation):
        raise TypeError("coupled RDE solving requires a Derivation")
    if derivation.level < 0:
        raise TypeError("coupled RDE solving requires a nonconstant field")
    if gaussian_transport is None:
        _require_supported_coupled_tower(derivation)
    elif not gaussian_transport.source_derivation.has_same_position(derivation):
        raise ValueError(
            "a Gaussian coupled transport belongs to a different tower position"
        )

    real_field = derivation.field
    diagonal_element = _coerce(real_field, diagonal_coefficient)
    rotation_element = _coerce(real_field, rotation_coefficient)
    real_rhs = _coerce(real_field, real_right_hand_side)
    imaginary_rhs = _coerce(real_field, imaginary_right_hand_side)

    if derivation.level > 0:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "coupled RDE solving does not support algebraic tower levels"
            )
        if (
            extension.family is MonomialFamily.HYPEREXPONENTIAL
            and _has_direct_hyperexponential_laurent_shape(
                diagonal_element,
                rotation_element,
                real_rhs,
                imaginary_rhs,
                extension,
            )
        ):
            direct_result = solve_hyperexponential_coupled_laurent_rde(
                diagonal_element,
                rotation_element,
                real_rhs,
                imaginary_rhs,
                derivation,
            )
            result = CoupledRischDifferentialEquationResult(
                derivation=derivation,
                diagonal_coefficient=diagonal_element,
                lambda_coefficient=rotation_element,
                real_right_hand_side=real_rhs,
                imaginary_right_hand_side=imaginary_rhs,
                real_solution=direct_result.real_solution,
                imaginary_solution=direct_result.imaginary_solution,
                obstruction=direct_result.obstruction,
                direct_result=direct_result,
            )
            if result.has_solution and not result.reconstructs:
                raise ValueError(
                    "direct hyperexponential coupled result failed to reconstruct"
                )
            return result
        if extension.family is MonomialFamily.HYPERTANGENT:
            try:
                direct_result = solve_hypertangent_coupled_polynomial_rde(
                    diagonal_element,
                    rotation_element,
                    real_rhs,
                    imaginary_rhs,
                    derivation,
                )
            except NotImplementedError:
                # The terminal recurrence is complete only for its stated
                # polynomial shapes.  General rational cases are sent
                # through the certified Cayley normal form below.
                pass
            else:
                real_solution = (
                    _coerce(real_field, direct_result.real_solution.as_expr())
                    if direct_result.real_solution is not None
                    else None
                )
                imaginary_solution = (
                    _coerce(
                        real_field,
                        direct_result.imaginary_solution.as_expr(),
                    )
                    if direct_result.imaginary_solution is not None
                    else None
                )
                result = CoupledRischDifferentialEquationResult(
                    derivation=derivation,
                    diagonal_coefficient=diagonal_element,
                    lambda_coefficient=rotation_element,
                    real_right_hand_side=real_rhs,
                    imaginary_right_hand_side=imaginary_rhs,
                    real_solution=real_solution,
                    imaginary_solution=imaginary_solution,
                    obstruction=direct_result.obstruction,
                    direct_result=direct_result,
                )
                if result.has_solution and not result.reconstructs:
                    raise ValueError(
                        "direct hypertangent coupled result failed to reconstruct"
                    )
                return result

    transport = gaussian_transport or _build_gaussian_coupled_transport(derivation)
    gaussian_normal_form = transport.normal_form
    gaussian_derivation = transport.target_derivation
    gaussian_field = gaussian_derivation.field
    if gaussian_normal_form is None:
        gaussian_diagonal = _coerce(
            gaussian_field,
            real_field.to_sympy(diagonal_element),
        )
        gaussian_rotation = _coerce(
            gaussian_field,
            real_field.to_sympy(rotation_element),
        )
        gaussian_rhs = _coerce(
            gaussian_field,
            real_field.to_sympy(real_rhs) + I * real_field.to_sympy(imaginary_rhs),
        )
    else:
        gaussian_diagonal = gaussian_normal_form.forward(diagonal_element)
        gaussian_rotation = gaussian_normal_form.forward(rotation_element)
        gaussian_rhs = gaussian_normal_form.forward(
            real_rhs
        ) + gaussian_field.from_sympy(I) * gaussian_normal_form.forward(
            imaginary_rhs,
        )
    # `_require_supported_coupled_tower` independently replayed the source
    # tower's no-new-constants evidence.  The Gaussian target is an exact
    # algebraic extension of constants (or a checked Cayley isomorphism), so
    # re-running proposal admission on the transported metadata would discard
    # that proof at recursive levels.  Solve inside the already admitted
    # coordinate change instead.
    scalar_result = _solve_admitted_risch_differential_equation(
        gaussian_diagonal + gaussian_field.from_sympy(I) * gaussian_rotation,
        gaussian_rhs,
        gaussian_derivation,
    )
    if scalar_result.solution is None:
        if scalar_result.obstruction is None:
            raise ValueError("a failed scalar RDE lacks its obstruction")
        return CoupledRischDifferentialEquationResult(
            derivation=derivation,
            diagonal_coefficient=diagonal_element,
            lambda_coefficient=rotation_element,
            real_right_hand_side=real_rhs,
            imaginary_right_hand_side=imaginary_rhs,
            real_solution=None,
            imaginary_solution=None,
            obstruction=_coupled_obstruction_from_scalar(scalar_result.obstruction),
            scalar_result=scalar_result,
            gaussian_normal_form=gaussian_normal_form,
        )

    if gaussian_normal_form is None:
        real_solution, imaginary_solution = _split_gaussian_tower_element(
            scalar_result.solution,
            gaussian_field,
            real_field,
        )
    else:
        real_solution, imaginary_solution = gaussian_normal_form.split_inverse(
            scalar_result.solution,
        )
    result = CoupledRischDifferentialEquationResult(
        derivation=derivation,
        diagonal_coefficient=diagonal_element,
        lambda_coefficient=rotation_element,
        real_right_hand_side=real_rhs,
        imaginary_right_hand_side=imaginary_rhs,
        real_solution=real_solution,
        imaginary_solution=imaginary_solution,
        obstruction=None,
        scalar_result=scalar_result,
        gaussian_normal_form=gaussian_normal_form,
    )
    if not result.reconstructs:
        raise ValueError("recursive coupled RDE reconstruction failed")
    return result


def solve_coupled_risch_differential_equation(
    lambda_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledRischDifferentialEquationResult:
    r"""Solve ``Dp-lambda*q=u`` and ``Dq+lambda*p=v`` recursively.

    Primitive and non-Laurent hyperexponential cases use the equivalent scalar
    equation over an immutable exact ``C(i)`` rebase. General rational
    hypertangent cases use a certified Cayley normal form whose top generator
    is hyperexponential, then map the scalar answer back to real coordinates.
    Hyperexponential Laurent cancellation and implemented hypertangent
    polynomial branches retain the direct real Chapter 8 recurrences.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("coupled RDE solving requires a Derivation")
    if derivation.level < 0:
        raise TypeError("coupled RDE solving requires a nonconstant field")
    field = derivation.field
    rotation = _coerce(field, lambda_coefficient)
    if rotation == field.zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    return _solve_general_coupled_risch_differential_equation(
        field.zero,
        rotation,
        real_right_hand_side,
        imaginary_right_hand_side,
        derivation,
    )
