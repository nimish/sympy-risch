r"""Risch differential equations at primitive monomial levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy.polys.polytools import Poly, invert

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..tower import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._common import (
    _field_element_is_zero,
    _field_elements_equal,
    _integer_constant,
)
from ._tower_polynomials import (
    _as_primitive_fraction,
    _as_primitive_polynomial,
    _positive_integer_rational,
)
from .models import (
    PrimitivePolynomialRDEObstruction,
    PrimitivePolynomialRischDifferentialEquationResult,
    PrimitiveRDEObstruction,
    PrimitiveRischDifferentialEquationResult,
)
from .rational import _NormalDenominatorReduction

if TYPE_CHECKING:
    from ..monomial import RadicalLogarithmicDerivativeResult


def _primitive_rde_extension(
    derivation: Derivation,
) -> MonomialExtension:
    """Validate a primitive top level with the no-new-constants property."""
    if derivation.level < 0:
        raise TypeError("a primitive RDE requires a differential extension")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "primitive RDE solving does not support algebraic tower levels"
        )
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError("primitive RDE solving requires Dt in the lower field")
    require_preserved_constant_field(derivation, "primitive RDE solving")
    return extension


def _weakly_normalize_primitive_rde_coefficient(
    coefficient: object,
    derivation: Derivation,
) -> tuple[Poly, FieldElement]:
    r"""Return ``q`` and ``coefficient - Dq/q`` weakly normalized in ``t``.

    At a normal irreducible factor ``p``, weak normalization removes the
    simple logarithmic residue only when it is a positive integer ``n``.
    Multiplying ``q`` by ``p**n`` subtracts ``n*Dp/p`` from the coefficient.

    A primitive monomial with preserved constants has no nonconstant special
    factors, so the complete principal-part signature supplies every factor
    that must be inspected.
    """
    extension = _primitive_rde_extension(derivation)
    field = extension.field
    element = _coerce(field, coefficient)
    _, _, denominator = _as_primitive_fraction(element, extension)
    if denominator.degree() == 0:
        return (
            Poly(
                1,
                extension.generator,
                domain=extension.parent_field,
            ),
            element,
        )

    # Local import avoids the module cycle: primitive monomial reduction uses
    # the RDE dispatcher defined in this module.
    from ..monomial import primitive_logarithmic_derivative_signature

    signature = primitive_logarithmic_derivative_signature(
        element,
        derivation,
    )
    normalizer = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    for factor in signature.factors:
        if factor.logarithmic_multiplicity is None:
            continue
        multiplicity = _positive_integer_rational(
            factor.logarithmic_multiplicity,
        )
        if multiplicity is not None:
            normalizer *= factor.factor**multiplicity
    normalizer = normalizer.monic()
    if normalizer.degree() == 0:
        return normalizer, element

    normalizer_element = _coerce(field, normalizer.as_expr())
    normalized = element - derivation(normalizer_element) / normalizer_element
    if not _field_elements_equal(
        normalized + derivation(normalizer_element) / normalizer_element,
        element,
        field,
    ):
        raise ValueError("primitive weak normalization failed to reconstruct")

    normalized_signature = primitive_logarithmic_derivative_signature(
        normalized,
        derivation,
    )
    for factor in normalized_signature.factors:
        if (
            factor.logarithmic_multiplicity is not None
            and _positive_integer_rational(
                factor.logarithmic_multiplicity,
            )
            is not None
        ):
            raise ValueError(
                "primitive weak normalization left a positive integer residue"
            )
    return normalizer, normalized


def _reduce_primitive_normal_denominator(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _NormalDenominatorReduction:
    r"""Reduce a weakly normalized primitive RDE to ``A*Dq+B*q=C``.

    Every nonconstant polynomial is normal in a primitive extension with
    preserved constants. Thus Bronstein's normal-denominator formula uses the
    complete top-generator denominators and no special-factor component.
    Formal differentiation detects their repeated factors, while ``D(h)`` in
    the transformed equation is the total tower derivative.
    """
    extension = _primitive_rde_extension(derivation)
    _, coefficient_numerator, coefficient_denominator = _as_primitive_fraction(
        coefficient, extension
    )
    _, rhs_numerator, rhs_denominator = _as_primitive_fraction(
        right_hand_side,
        extension,
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
        - coefficient_denominator * derivation.polynomial(denominator_bound)
    )
    multiplier = coefficient_denominator * denominator_bound * denominator_bound
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
    return _NormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        quotient * rhs_numerator,
    )


def _primitive_limited_integration_relation(
    value: object,
    derivation: Derivation,
) -> tuple[int, FieldElement] | None:
    r"""Find ``m`` and ``z`` with ``value = m*Dt + Dz``, if supported.

    Reducing a constant polynomial at the current primitive level computes
    exactly the quotient by ``D(K) + Const(K)*Dt``. The degree-bound theorem
    additionally requires the coefficient of ``Dt`` to be an integer.
    """
    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    element = parent.convert(value)
    if _field_element_is_zero(element, parent):
        return 0, parent.zero

    # See the cycle note in primitive weak normalization above.
    from ..monomial import polynomial_reduce_primitive

    reduction = polynomial_reduce_primitive(
        Poly(
            element,
            extension.generator,
            domain=parent,
        ),
        derivation,
    )
    if reduction.has_obstruction or len(reduction.steps) != 1:
        return None
    step = reduction.steps[0]
    integer_multiple = _integer_constant(
        step.eta_multiple,
        derivation.tower.constants,
    )
    if integer_multiple is None:
        return None
    exact_part = parent.convert(step.exact_coefficient)
    eta = parent.convert(extension.generator_derivative.nth(0))
    if not _field_elements_equal(
        derivation.lower()(exact_part) + parent.convert(integer_multiple) * eta,
        element,
        parent,
    ):
        raise ValueError("primitive limited-integration relation is invalid")
    return integer_multiple, exact_part


def _primitive_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    derivation: Derivation,
) -> int:
    r"""Implement Bronstein's primitive polynomial RDE degree bound.

    This is ``RdeBoundDegreePrim`` from Section 6.3. Besides the generic
    degree estimate, the two possible leading cancellations require one
    limited-integration decision and, at equal degrees, one logarithmic
    derivative decision in the lower field.
    """
    if derivative_coefficient.is_zero:
        raise ValueError("the derivative coefficient must be nonzero")
    if right_hand_side.is_zero:
        return 0

    degree_a = derivative_coefficient.degree()
    degree_b = solution_coefficient.degree() if not solution_coefficient.is_zero else -1
    degree_c = right_hand_side.degree()
    if degree_b > degree_a:
        bound = max(0, degree_c - degree_b)
    else:
        bound = max(0, degree_c - degree_a + 1)

    parent = derivative_coefficient.domain
    if not solution_coefficient.is_zero and degree_b == degree_a - 1:
        alpha = parent.convert(-solution_coefficient.LC() / derivative_coefficient.LC())
        relation = _primitive_limited_integration_relation(
            alpha,
            derivation,
        )
        if relation is not None:
            bound = max(bound, relation[0])

    if degree_b == degree_a:
        alpha = parent.convert(-solution_coefficient.LC() / derivative_coefficient.LC())
        from ..monomial import solve_radical_logarithmic_derivative

        logarithmic_result = solve_radical_logarithmic_derivative(
            alpha,
            derivation.lower(),
        )
        if logarithmic_result.has_solution and logarithmic_result.minimal_power == 1:
            z = parent.convert(logarithmic_result.argument)
            dz = derivation.lower()(z)
            transformed = derivative_coefficient.mul_ground(
                dz
            ) + solution_coefficient.mul_ground(z)
            # ``mul_ground`` can expose the leading coefficient as a SymPy
            # expression even though this polynomial lives over ``parent``.
            # Normalize it before field division: otherwise an
            # ``Expr / FracElement`` operation leaks out for valid primitive RDEs.
            beta = -parent.convert(transformed.LC()) / (
                z * parent.convert(derivative_coefficient.LC())
            )
            relation = _primitive_limited_integration_relation(
                beta,
                derivation,
            )
            if relation is not None:
                bound = max(bound, relation[0])
    return bound


def _solve_primitive_polynomial_exact_case(
    coefficient_polynomial: Poly,
    rhs_polynomial: Poly,
    derivation: Derivation,
    multiplier: object,
    logarithmic_result: RadicalLogarithmicDerivativeResult | None = None,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Solve a polynomial cancellation case by exact differentiation.

    If the coefficient is zero, ``multiplier`` is one.  Otherwise the caller
    has certified ``coefficient = D(multiplier)/multiplier``, so

    ``D(y) + coefficient*y = c``

    is equivalent to ``D(multiplier*y) = multiplier*c``.
    """
    from ..exact import solve_exact_derivative

    extension = _primitive_rde_extension(derivation)
    field = extension.field
    zero = rhs_polynomial.zero
    multiplier_element = _coerce(field, multiplier)
    transformed_rhs = multiplier_element * _coerce(field, rhs_polynomial.as_expr())
    exact_result = solve_exact_derivative(
        transformed_rhs,
        derivation,
    )
    degree_bound = 0 if rhs_polynomial.is_zero else rhs_polynomial.degree() + 1
    if not exact_result.has_solution:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            degree_bound,
            None,
            PrimitivePolynomialRDEObstruction.EXACT_DERIVATIVE,
            zero,
            rhs_polynomial,
            logarithmic_derivative_result=logarithmic_result,
            exact_derivative_result=exact_result,
        )

    solution_element = exact_result.antiderivative / multiplier_element
    solution = _as_primitive_polynomial(
        solution_element,
        extension,
    )
    if not solution.is_zero and solution.degree() > degree_bound:
        raise ValueError("primitive exact cancellation exceeded its degree bound")
    if (
        derivation.polynomial(solution) + coefficient_polynomial * solution
        != rhs_polynomial
    ):
        raise ValueError("primitive exact cancellation reconstruction failed")
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        zero,
        logarithmic_derivative_result=logarithmic_result,
        exact_derivative_result=exact_result,
    )


def _solve_primitive_polynomial_lower_coefficient(
    coefficient_polynomial: Poly,
    rhs_polynomial: Poly,
    derivation: Derivation,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Apply Bronstein's ``PolyRischDECancelPrim`` recurrence."""
    from ..monomial import solve_radical_logarithmic_derivative
    from .scalar import solve_risch_differential_equation

    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    lower_derivation = derivation.lower()
    coefficient = parent.convert(coefficient_polynomial.nth(0))
    zero = rhs_polynomial.zero

    if _field_element_is_zero(coefficient, parent):
        return _solve_primitive_polynomial_exact_case(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
            parent.one,
        )

    logarithmic_result = solve_radical_logarithmic_derivative(
        coefficient,
        lower_derivation,
    )
    if logarithmic_result.has_solution and logarithmic_result.minimal_power == 1:
        return _solve_primitive_polynomial_exact_case(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
            logarithmic_result.argument,
            logarithmic_result,
        )

    if rhs_polynomial.is_zero:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            0,
            zero,
            None,
            zero,
            zero,
            logarithmic_derivative_result=logarithmic_result,
        )

    degree_bound = rhs_polynomial.degree()
    solution = zero
    residual = rhs_polynomial
    lower_results = []
    while not residual.is_zero:
        degree = residual.degree()
        lower_rhs = parent.convert(residual.LC())
        lower_result = solve_risch_differential_equation(
            coefficient,
            lower_rhs,
            lower_derivation,
        )
        lower_results.append(lower_result)
        if not lower_result.has_solution:
            if (
                derivation.polynomial(solution)
                + coefficient_polynomial * solution
                + residual
                != rhs_polynomial
            ):
                raise ValueError("primitive cancellation obstruction invariant failed")
            return PrimitivePolynomialRischDifferentialEquationResult(
                coefficient_polynomial,
                rhs_polynomial,
                degree_bound,
                None,
                PrimitivePolynomialRDEObstruction.LOWER_FIELD,
                solution,
                residual,
                lower_field_results=tuple(lower_results),
                logarithmic_derivative_result=logarithmic_result,
            )

        correction = Poly.from_dict(
            {
                (degree,): _coerce(parent, lower_result.solution),
            },
            extension.generator,
            domain=parent,
        )
        new_residual = (
            residual
            - derivation.polynomial(correction)
            - coefficient_polynomial * correction
        )
        if not new_residual.is_zero and new_residual.degree() >= residual.degree():
            raise ValueError("primitive cancellation residual degree did not decrease")
        solution += correction
        residual = new_residual

    if (
        derivation.polynomial(solution) + coefficient_polynomial * solution
        != rhs_polynomial
    ):
        raise ValueError("primitive cancellation reconstruction failed")
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        residual,
        lower_field_results=tuple(lower_results),
        logarithmic_derivative_result=logarithmic_result,
    )


def solve_primitive_polynomial_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Decide a polynomial RDE at a primitive tower level.

    For ``Dt`` in ``K``, solve

    ``D(y) + a*y = b``, with ``a,b`` in ``K[t]``.

    Positive-degree coefficients use the no-cancellation leading recurrence.
    Lower-field coefficients use Bronstein's cancellation recurrence,
    recursively solving coefficient equations in ``K``. Preservation of
    constants excludes special finite poles, so the polynomial decision also
    covers all rational solutions when both inputs are polynomial.
    """
    extension = _primitive_rde_extension(derivation)
    coefficient_polynomial = _as_primitive_polynomial(
        coefficient,
        extension,
    )
    rhs_polynomial = _as_primitive_polynomial(
        right_hand_side,
        extension,
    )
    if coefficient_polynomial.is_zero or coefficient_polynomial.degree() <= 0:
        return _solve_primitive_polynomial_lower_coefficient(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
        )

    zero = Poly(
        0,
        extension.generator,
        domain=extension.parent_field,
    )
    if rhs_polynomial.is_zero:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            0,
            zero,
            None,
            zero,
            zero,
        )

    degree_bound = rhs_polynomial.degree() - coefficient_polynomial.degree()
    if degree_bound < 0:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            -1,
            None,
            PrimitivePolynomialRDEObstruction.DEGREE,
            zero,
            rhs_polynomial,
        )

    solution = zero
    residual = rhs_polynomial
    while not residual.is_zero:
        correction_degree = residual.degree() - coefficient_polynomial.degree()
        if correction_degree < 0:
            if (
                derivation.polynomial(solution)
                + coefficient_polynomial * solution
                + residual
                != rhs_polynomial
            ):
                raise ValueError(
                    "primitive polynomial RDE obstruction invariant failed"
                )
            return PrimitivePolynomialRischDifferentialEquationResult(
                coefficient_polynomial,
                rhs_polynomial,
                degree_bound,
                None,
                PrimitivePolynomialRDEObstruction.NONZERO_REMAINDER,
                solution,
                residual,
            )

        correction = Poly.from_dict(
            {(correction_degree,): (residual.LC() / coefficient_polynomial.LC())},
            extension.generator,
            domain=extension.parent_field,
        )
        new_residual = (
            residual
            - derivation.polynomial(correction)
            - coefficient_polynomial * correction
        )
        if not new_residual.is_zero and new_residual.degree() >= residual.degree():
            raise ValueError(
                "primitive polynomial RDE residual degree did not decrease"
            )
        solution += correction
        residual = new_residual

    if (
        derivation.polynomial(solution) + coefficient_polynomial * solution
        != rhs_polynomial
    ):
        raise ValueError("primitive polynomial RDE reconstruction failed")
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        residual,
    )


@dataclass(frozen=True, slots=True)
class _PrimitiveSPDEResult:
    """Internal numerator result after primitive-field SPDE reduction."""

    solution: Poly | None
    obstruction: PrimitiveRDEObstruction | None
    polynomial_result: PrimitivePolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_primitive_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
    derivation: Derivation,
) -> _PrimitiveSPDEResult:
    """Apply Rothstein SPDE over a primitive coefficient field."""
    extension = _primitive_rde_extension(derivation)
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = derivative_coefficient.one
    beta = derivative_coefficient.zero
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient * derivation.polynomial(beta)
                + original_solution_coefficient * beta
                != original_rhs
            ):
                raise ValueError("primitive SPDE zero-tail reconstruction failed")
            return _PrimitiveSPDEResult(beta, None)
        if current_bound < 0:
            return _PrimitiveSPDEResult(
                None,
                PrimitiveRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _PrimitiveSPDEResult(
                None,
                PrimitiveRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse = extension.parent_field.one / extension.parent_field.convert(
                derivative_coefficient.LC()
            )
            normalized_coefficient = solution_coefficient.mul_ground(inverse)
            normalized_rhs = right_hand_side.mul_ground(inverse)
            polynomial_result = solve_primitive_polynomial_risch_differential_equation(
                normalized_coefficient,
                normalized_rhs,
                derivation,
            )
            if not polynomial_result.has_solution:
                return _PrimitiveSPDEResult(
                    None,
                    PrimitiveRDEObstruction.POLYNOMIAL,
                    polynomial_result,
                )
            tail = polynomial_result.solution
            if tail is None:
                raise ValueError("a solved primitive polynomial RDE lacks its solution")
            if not tail.is_zero and tail.degree() > current_bound:
                return _PrimitiveSPDEResult(
                    None,
                    PrimitiveRDEObstruction.SPDE_DEGREE,
                    polynomial_result,
                )
            solution = alpha * tail + beta
            if (
                original_derivative_coefficient * derivation.polynomial(solution)
                + original_solution_coefficient * solution
                != original_rhs
            ):
                raise ValueError("primitive SPDE reconstruction failed")
            return _PrimitiveSPDEResult(
                solution,
                None,
                polynomial_result,
            )

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side * inverse).rem(derivative_coefficient)
        quotient = (right_hand_side - solution_coefficient * residue).exquo(
            derivative_coefficient
        )
        right_hand_side = quotient - derivation.polynomial(residue)
        solution_coefficient += derivation.polynomial(derivative_coefficient)
        current_bound -= derivative_coefficient.degree()
        beta += alpha * residue
        alpha *= derivative_coefficient


def _solve_primitive_rde_exact_case(
    coefficient: FieldElement,
    right_hand_side: FieldElement,
    normalized_coefficient: FieldElement,
    normalized_right_hand_side: FieldElement,
    derivation: Derivation,
    normalization_factor: Poly,
    multiplier: object,
    logarithmic_result: RadicalLogarithmicDerivativeResult | None = None,
) -> PrimitiveRischDifferentialEquationResult:
    """Solve a weakly normalized primitive RDE by exact differentiation."""
    from ..exact import solve_exact_derivative

    extension = _primitive_rde_extension(derivation)
    field = extension.field
    one = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    multiplier_element = _coerce(field, multiplier)
    exact_result = solve_exact_derivative(
        multiplier_element * normalized_right_hand_side,
        derivation,
    )
    if not exact_result.has_solution:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient,
            right_hand_side=right_hand_side,
            solution=None,
            obstruction=PrimitiveRDEObstruction.EXACT_DERIVATIVE,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_right_hand_side,
            denominator_bound=one,
            logarithmic_derivative_result=logarithmic_result,
            exact_derivative_result=exact_result,
        )

    normalized_solution = exact_result.antiderivative / multiplier_element
    solution = normalized_solution / normalization_element
    if not _field_elements_equal(
        derivation(solution) + coefficient * solution,
        right_hand_side,
        field,
    ):
        raise ValueError("primitive exact RDE reconstruction failed")
    return PrimitiveRischDifferentialEquationResult(
        coefficient=coefficient,
        right_hand_side=right_hand_side,
        solution=solution,
        obstruction=None,
        normalization_factor=normalization_factor,
        normalized_coefficient=normalized_coefficient,
        normalized_right_hand_side=normalized_right_hand_side,
        denominator_bound=one,
        logarithmic_derivative_result=logarithmic_result,
        exact_derivative_result=exact_result,
    )


def solve_primitive_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> PrimitiveRischDifferentialEquationResult:
    r"""Decide an RDE with rational coefficient over primitive ``K(t)``.

    This implements the Chapter 6 sequence in Bronstein, *Symbolic
    Integration I*: weak normalization, normal denominator, primitive degree
    bound, Rothstein SPDE, and the polynomial RDE. Polynomial coefficients
    retain the existing no-cancellation and recursive lower-field
    cancellation paths.
    """
    extension = _primitive_rde_extension(derivation)
    field = extension.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_primitive_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    normalized_rhs = normalization_element * rhs_element
    one = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    if _field_element_is_zero(normalized_rhs, field):
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=field.zero,
            obstruction=None,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=one,
            degree_bound=0,
        )

    _, normalized_numerator, normalized_denominator = _as_primitive_fraction(
        normalized_coefficient, extension
    )
    normalized_polynomial = None
    if normalized_denominator.degree() == 0:
        inverse = extension.parent_field.one / extension.parent_field.convert(
            normalized_denominator.LC()
        )
        normalized_polynomial = normalized_numerator.mul_ground(inverse)

    coefficient_logarithmic_result = None
    coefficient_is_lower = normalized_polynomial is not None and (
        normalized_polynomial.is_zero or normalized_polynomial.degree() <= 0
    )
    if coefficient_is_lower:
        coefficient_lower = extension.parent_field.convert(normalized_polynomial.nth(0))
        if _field_element_is_zero(
            coefficient_lower,
            extension.parent_field,
        ):
            return _solve_primitive_rde_exact_case(
                coefficient_element,
                rhs_element,
                normalized_coefficient,
                normalized_rhs,
                derivation,
                normalization_factor,
                extension.parent_field.one,
            )

        from ..monomial import solve_radical_logarithmic_derivative

        coefficient_logarithmic_result = solve_radical_logarithmic_derivative(
            coefficient_lower,
            derivation.lower(),
        )
        if (
            coefficient_logarithmic_result.has_solution
            and coefficient_logarithmic_result.minimal_power == 1
        ):
            return _solve_primitive_rde_exact_case(
                coefficient_element,
                rhs_element,
                normalized_coefficient,
                normalized_rhs,
                derivation,
                normalization_factor,
                coefficient_logarithmic_result.argument,
                coefficient_logarithmic_result,
            )

    denominator_reduction = _reduce_primitive_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        derivation,
    )
    if not denominator_reduction.succeeded:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=None,
            obstruction=PrimitiveRDEObstruction.DENOMINATOR,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=denominator_reduction.denominator_bound,
            obstruction_polynomial=(denominator_reduction.obstruction_polynomial),
            obstruction_remainder=(denominator_reduction.obstruction_remainder),
            logarithmic_derivative_result=(coefficient_logarithmic_result),
        )

    denominator_bound = denominator_reduction.denominator_bound
    derivative_coefficient = denominator_reduction.derivative_coefficient
    solution_coefficient = denominator_reduction.solution_coefficient
    transformed_rhs = denominator_reduction.right_hand_side
    if (
        derivative_coefficient is None
        or solution_coefficient is None
        or transformed_rhs is None
    ):
        raise ValueError("successful primitive denominator reduction lacks an equation")
    degree_bound = _primitive_rde_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        derivation,
    )
    spde_result = _solve_primitive_spde(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
        derivation,
    )
    if spde_result.solution is None:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=None,
            obstruction=spde_result.obstruction,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=denominator_bound,
            degree_bound=degree_bound,
            polynomial_result=spde_result.polynomial_result,
            obstruction_polynomial=spde_result.obstruction_polynomial,
            obstruction_remainder=spde_result.obstruction_remainder,
            logarithmic_derivative_result=(coefficient_logarithmic_result),
        )

    solution = _coerce(
        field,
        spde_result.solution.as_expr()
        / (normalization_factor.as_expr() * denominator_bound.as_expr()),
    )
    if not _field_elements_equal(
        derivation(solution) + coefficient_element * solution,
        rhs_element,
        field,
    ):
        raise ValueError("primitive RDE reconstruction failed")
    return PrimitiveRischDifferentialEquationResult(
        coefficient=coefficient_element,
        right_hand_side=rhs_element,
        solution=solution,
        obstruction=None,
        normalization_factor=normalization_factor,
        normalized_coefficient=normalized_coefficient,
        normalized_right_hand_side=normalized_rhs,
        denominator_bound=denominator_bound,
        degree_bound=degree_bound,
        polynomial_result=spde_result.polynomial_result,
        logarithmic_derivative_result=coefficient_logarithmic_result,
    )
