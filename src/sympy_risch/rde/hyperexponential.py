r"""Risch differential equations at hyperexponential monomial levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy.core.symbol import Dummy
from sympy.polys.polytools import Poly, invert, resultant

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..reduction.fractions import split_normal_special
from ..tower import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._common import _field_element_is_zero, _field_elements_equal
from ._tower_polynomials import (
    _as_primitive_fraction,
    _as_primitive_polynomial,
    _positive_integer_in_domain,
)
from .models import (
    HyperexponentialLaurentRDECorrection,
    HyperexponentialLaurentRDEDirection,
    HyperexponentialLaurentRDEObstruction,
    HyperexponentialLaurentRDEStep,
    HyperexponentialLaurentRischDifferentialEquationResult,
    HyperexponentialRDEObstruction,
    HyperexponentialRischDifferentialEquationResult,
)

if TYPE_CHECKING:
    from ..dependency import ParametricLogarithmicDerivativeResult


def _weakly_normalize_hyperexponential_rde_coefficient(
    coefficient: object,
    derivation: Derivation,
) -> tuple[Poly, FieldElement]:
    r"""Apply Bronstein's general weak-normalizer resultant construction.

    Let ``d1`` be the product of the normal irreducible denominator factors
    that occur exactly once. If ``a`` is the partial numerator of ``f`` at
    ``d1``, the positive integer roots of

    ``resultant_t(a - z*D(d1), d1)``

    are precisely the logarithmic residues that weak normalization removes.
    For every such root ``n``, the factor
    ``gcd(a - n*D(d1), d1)**n`` is included in the normalizer.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    element, numerator, denominator = _as_primitive_fraction(
        coefficient,
        extension,
    )
    normal_denominator, _ = split_normal_special(denominator, derivation)
    repeated_part = normal_denominator.gcd(normal_denominator.diff())
    squarefree_part = normal_denominator.exquo(repeated_part)
    simple_normal_part = squarefree_part.exquo(
        squarefree_part.gcd(repeated_part)
    ).monic()
    one = Poly(1, extension.generator, domain=parent)
    if simple_normal_part.degree() == 0:
        return one, element

    cofactor = denominator.exquo(simple_normal_part)
    partial_numerator = (numerator * invert(cofactor, simple_normal_part)).rem(
        simple_normal_part
    )
    simple_derivative = derivation.polynomial(simple_normal_part)
    residue_symbol = Dummy("z")
    residue_polynomial = Poly(
        resultant(
            (
                partial_numerator.as_expr()
                - residue_symbol * simple_derivative.as_expr()
            ),
            simple_normal_part.as_expr(),
            extension.generator,
        ),
        residue_symbol,
        domain=parent,
    )

    normalizer = one
    for root in residue_polynomial.ground_roots():
        multiplicity = _positive_integer_in_domain(root, parent)
        if multiplicity is None:
            continue
        factor = (
            partial_numerator
            - simple_derivative.mul_ground(parent.convert(multiplicity))
        ).gcd(simple_normal_part)
        normalizer *= factor**multiplicity
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
        raise ValueError("hyperexponential weak normalization failed to reconstruct")
    return normalizer, normalized


@dataclass(frozen=True, slots=True)
class _HyperexponentialNormalDenominatorReduction:
    """Internal result of Bronstein's normal-denominator theorem."""

    denominator_bound: Poly
    derivative_coefficient: FieldElement | None
    solution_coefficient: FieldElement | None
    right_hand_side: FieldElement | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None


def _reduce_hyperexponential_normal_denominator(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _HyperexponentialNormalDenominatorReduction:
    r"""Bound all normal factors in a hyperexponential RDE solution.

    This is ``RdeNormalDenominator`` from Bronstein, Section 6.1. The returned
    equation ``A*Dq + B*q = C`` has only special powers of ``t`` in ``B`` and
    ``C``; every normalized solution is ``q/h`` where ``h`` is the returned
    denominator bound.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    _, _, coefficient_denominator = _as_primitive_fraction(
        coefficient,
        extension,
    )
    _, _, rhs_denominator = _as_primitive_fraction(
        right_hand_side,
        extension,
    )
    coefficient_normal, _ = split_normal_special(
        coefficient_denominator,
        derivation,
    )
    rhs_normal, _ = split_normal_special(rhs_denominator, derivation)
    shared_normal = coefficient_normal.gcd(rhs_normal)
    denominator_bound = (
        rhs_normal.gcd(rhs_normal.diff())
        .exquo(shared_normal.gcd(shared_normal.diff()))
        .monic()
    )
    divisibility_target = coefficient_normal * denominator_bound**2
    _, remainder = divisibility_target.div(rhs_normal)
    if not remainder.is_zero:
        return _HyperexponentialNormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_normal,
            remainder,
        )

    coefficient_element = _coerce(field, coefficient)
    rhs_element = _coerce(field, right_hand_side)
    normal_element = _coerce(field, coefficient_normal.as_expr())
    bound_element = _coerce(field, denominator_bound.as_expr())
    derivative_coefficient = normal_element * bound_element
    solution_coefficient = (
        derivative_coefficient * coefficient_element
        - normal_element * derivation(bound_element)
    )
    transformed_rhs = normal_element * bound_element * bound_element * rhs_element
    return _HyperexponentialNormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )


def _hyperexponential_laurent_rde_extension(
    derivation: Derivation,
) -> MonomialExtension:
    """Validate a hyperexponential top level with unchanged constants."""
    if derivation.level < 1:
        raise TypeError(
            "hyperexponential Laurent RDE solving requires a nonconstant lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "hyperexponential Laurent RDE solving does not support "
            "algebraic tower levels"
        )
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError("hyperexponential Laurent RDE solving requires Dt = eta*t")
    require_preserved_constant_field(
        derivation,
        "hyperexponential Laurent RDE solving",
    )
    return extension


def _hyperexponential_laurent_coefficients(
    value: object,
    extension: MonomialExtension,
    role: str,
) -> tuple[FieldElement, dict[int, FieldElement]]:
    """Return the finite Laurent expansion of a top-field element."""
    element, numerator, denominator = _as_primitive_fraction(
        value,
        extension,
    )
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise NotImplementedError(
            f"the hyperexponential Laurent RDE {role} must be a Laurent "
            "polynomial in the top generator"
        )
    (denominator_power,), denominator_coefficient = denominator_terms[0]
    parent = extension.parent_field
    scale = parent.one / parent.convert(denominator_coefficient)
    coefficients = {
        numerator_power - denominator_power: (parent.convert(coefficient) * scale)
        for (numerator_power,), coefficient in numerator.terms()
        if not _field_element_is_zero(parent.convert(coefficient), parent)
    }
    return element, coefficients


def _hyperexponential_laurent_term(
    coefficient: FieldElement,
    power: int,
    extension: MonomialExtension,
) -> FieldElement:
    """Embed one lower-field Laurent term in the top field."""
    return _coerce(
        extension.field,
        extension.parent_field.to_sympy(coefficient) * extension.generator**power,
    )


def _hyperexponential_boundary_power(
    constant_coefficient: FieldElement,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> tuple[ParametricLogarithmicDerivativeResult, int | None]:
    r"""Decide ``-a_0 = m*eta + D(v)/v`` for an ordinary integer ``m``."""
    from ..dependency import solve_parametric_logarithmic_derivative

    result = solve_parametric_logarithmic_derivative(
        -constant_coefficient,
        extension,
        lower_derivation,
    )
    if result.is_unsupported or result.is_inconclusive:
        raise NotImplementedError(
            "hyperexponential Laurent support bounds require a complete "
            "parametric logarithmic-derivative decision in the lower field"
        )
    if result.has_solution and result.n == 1:
        if result.m is None:
            raise ValueError("a solved boundary relation lacks its integer power")
        return result, result.m
    return result, None


@dataclass(frozen=True, slots=True)
class _HyperexponentialSpecialDenominatorReduction:
    """Internal output of ``RdeSpecialDenomExp``."""

    derivative_coefficient: Poly
    solution_coefficient: Poly
    right_hand_side: Poly
    solution_denominator_power: int
    clearing_power: int
    boundary_result: ParametricLogarithmicDerivativeResult | None = None
    exceptional_power: int | None = None


def _hyperexponential_special_denominator(
    derivative_coefficient: object,
    solution_coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _HyperexponentialSpecialDenominatorReduction:
    r"""Apply Bronstein's hyperexponential special-denominator reduction.

    For ``A*Dq + B*q = C``, all remaining denominators are powers of the
    special polynomial ``t``. The valuation bound ``n <= 0`` implies
    ``r = q*t**(-n)`` is polynomial. A final power ``N`` clears the transformed
    equation

    ``A*Dr + (B + n*A*Dt/t)*r = C*t**(-n)``.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    derivative_polynomial = _as_primitive_polynomial(
        derivative_coefficient,
        extension,
    )
    if _field_element_is_zero(
        derivative_polynomial.eval(parent.zero),
        parent,
    ):
        raise ValueError(
            "the special-denominator derivative coefficient must be "
            "coprime to the hyperexponential generator"
        )
    solution_element, solution_terms = _hyperexponential_laurent_coefficients(
        solution_coefficient,
        extension,
        "special-denominator solution coefficient",
    )
    rhs_element, rhs_terms = _hyperexponential_laurent_coefficients(
        right_hand_side,
        extension,
        "special-denominator right-hand side",
    )
    if not rhs_terms:
        raise ValueError(
            "special-denominator reduction requires a nonzero right-hand side"
        )

    solution_valuation = min(solution_terms) if solution_terms else None
    rhs_valuation = min(rhs_terms)
    bounded_solution_valuation = min(0, solution_valuation or 0)
    valuation_bound = min(
        0,
        rhs_valuation - bounded_solution_valuation,
    )
    boundary_result = None
    exceptional_power = None
    if solution_valuation == 0:
        alpha = -solution_terms[0] / parent.convert(derivative_polynomial.nth(0))
        boundary_result, exceptional_power = _hyperexponential_boundary_power(
            -alpha,
            extension,
            derivation.lower(),
        )
        if exceptional_power is not None:
            valuation_bound = min(valuation_bound, exceptional_power)

    clearing_power = max(
        0,
        -solution_valuation if solution_valuation is not None else 0,
        valuation_bound - rhs_valuation,
    )
    field = extension.field
    t_power = _coerce(field, t**clearing_power)
    eta = parent.convert(extension.generator_derivative.nth(1))
    eta_element = _coerce(field, parent.to_sympy(eta))
    derivative_element = _coerce(
        field,
        derivative_polynomial.as_expr(),
    )
    transformed_derivative = derivative_element * t_power
    transformed_solution = (
        solution_element
        + field.convert(valuation_bound) * derivative_element * eta_element
    ) * t_power
    transformed_rhs = rhs_element * _coerce(
        field, t ** (clearing_power - valuation_bound)
    )
    derivative_output = _as_primitive_polynomial(
        transformed_derivative,
        extension,
    )
    solution_output = _as_primitive_polynomial(
        transformed_solution,
        extension,
    )
    rhs_output = _as_primitive_polynomial(transformed_rhs, extension)
    return _HyperexponentialSpecialDenominatorReduction(
        derivative_output,
        solution_output,
        rhs_output,
        -valuation_bound,
        clearing_power,
        boundary_result,
        exceptional_power,
    )


def _hyperexponential_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    derivation: Derivation,
) -> tuple[
    int,
    ParametricLogarithmicDerivativeResult | None,
    int | None,
]:
    r"""Implement Bronstein's ``RdeBoundDegreeExp``."""
    if derivative_coefficient.is_zero:
        raise ValueError("the derivative coefficient must be nonzero")
    if right_hand_side.is_zero:
        return 0, None, None

    degree_a = derivative_coefficient.degree()
    degree_b = solution_coefficient.degree() if not solution_coefficient.is_zero else -1
    degree_c = right_hand_side.degree()
    bound = max(0, degree_c - max(degree_b, degree_a))
    boundary_result = None
    exceptional_power = None
    if degree_a == degree_b:
        parent = derivative_coefficient.domain
        alpha = parent.convert(-solution_coefficient.LC() / derivative_coefficient.LC())
        extension = _hyperexponential_laurent_rde_extension(derivation)
        boundary_result, exceptional_power = _hyperexponential_boundary_power(
            -alpha,
            extension,
            derivation.lower(),
        )
        if exceptional_power is not None:
            bound = max(bound, exceptional_power)
    return bound, boundary_result, exceptional_power


@dataclass(frozen=True, slots=True)
class _HyperexponentialSPDEResult:
    """Internal numerator result after hyperexponential SPDE."""

    solution: Poly | None
    obstruction: HyperexponentialRDEObstruction | None
    laurent_result: HyperexponentialLaurentRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_hyperexponential_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
    derivation: Derivation,
) -> _HyperexponentialSPDEResult:
    """Apply Rothstein SPDE and finish with the complete Laurent solver."""
    extension = _hyperexponential_laurent_rde_extension(derivation)
    parent = extension.parent_field
    original_derivative = derivative_coefficient
    original_solution = solution_coefficient
    original_rhs = right_hand_side
    alpha = derivative_coefficient.one
    beta = derivative_coefficient.zero
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative * derivation.polynomial(beta)
                + original_solution * beta
                != original_rhs
            ):
                raise ValueError(
                    "hyperexponential SPDE zero-tail reconstruction failed"
                )
            return _HyperexponentialSPDEResult(beta, None)
        if current_bound < 0:
            return _HyperexponentialSPDEResult(
                None,
                HyperexponentialRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _HyperexponentialSPDEResult(
                None,
                HyperexponentialRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse = parent.one / parent.convert(derivative_coefficient.LC())
            normalized_coefficient = solution_coefficient.mul_ground(inverse)
            normalized_rhs = right_hand_side.mul_ground(inverse)
            laurent_result = solve_hyperexponential_laurent_risch_differential_equation(
                normalized_coefficient,
                normalized_rhs,
                derivation,
            )
            if not laurent_result.has_solution:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.LAURENT,
                    laurent_result,
                )
            tail_element = laurent_result.solution
            if tail_element is None:
                raise ValueError(
                    "a solved hyperexponential Laurent RDE lacks its solution"
                )
            _, tail_numerator, tail_denominator = _as_primitive_fraction(
                tail_element,
                extension,
            )
            if tail_denominator.degree() > 0:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.SPDE_DEGREE,
                    laurent_result,
                )
            tail = tail_numerator.mul_ground(
                parent.one / parent.convert(tail_denominator.LC())
            )
            if not tail.is_zero and tail.degree() > current_bound:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.SPDE_DEGREE,
                    laurent_result,
                )
            numerator_solution = alpha * tail + beta
            if (
                original_derivative * derivation.polynomial(numerator_solution)
                + original_solution * numerator_solution
                != original_rhs
            ):
                raise ValueError("hyperexponential SPDE reconstruction failed")
            return _HyperexponentialSPDEResult(
                numerator_solution,
                None,
                laurent_result,
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


def _hyperexponential_laurent_support_bounds(
    coefficient: dict[int, FieldElement],
    right_hand_side: dict[int, FieldElement],
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> tuple[
    int,
    int,
    ParametricLogarithmicDerivativeResult | None,
    int | None,
]:
    r"""Apply Bronstein's valuations at ``t=0`` and ``t=infinity``.

    This is the standardized ``D(y) + a*y = b`` specialization of Sections
    6.2 and 6.3. The returned interval contains the Laurent support of every
    solution. Equality is forced at an edge where multiplication by a
    nonzero power of ``t`` dominates differentiation.
    """
    if not coefficient or not right_hand_side:
        raise ValueError("Laurent support bounds require nonzero inputs")

    minimum_coefficient_power = min(coefficient)
    maximum_coefficient_power = max(coefficient)
    minimum_rhs_power = min(right_hand_side)
    maximum_rhs_power = max(right_hand_side)
    boundary_result = None
    exceptional_power = None

    if minimum_coefficient_power == 0 or maximum_coefficient_power == 0:
        boundary_result, exceptional_power = _hyperexponential_boundary_power(
            coefficient[0],
            extension,
            lower_derivation,
        )

    if minimum_coefficient_power < 0:
        minimum_solution_power = min(
            0,
            minimum_rhs_power - minimum_coefficient_power,
        )
    elif minimum_coefficient_power > 0:
        minimum_solution_power = min(0, minimum_rhs_power)
    else:
        candidates = [0, minimum_rhs_power]
        if exceptional_power is not None:
            candidates.append(exceptional_power)
        minimum_solution_power = min(candidates)

    if maximum_coefficient_power > 0:
        maximum_solution_power = max(
            0,
            maximum_rhs_power - maximum_coefficient_power,
        )
    elif maximum_coefficient_power < 0:
        maximum_solution_power = max(0, maximum_rhs_power)
    else:
        candidates = [0, maximum_rhs_power]
        if exceptional_power is not None:
            candidates.append(exceptional_power)
        maximum_solution_power = max(candidates)

    return (
        minimum_solution_power,
        maximum_solution_power,
        boundary_result,
        exceptional_power,
    )


def _solve_hyperexponential_diagonal_laurent_rde(
    coefficient_element: FieldElement,
    lower_coefficient: FieldElement,
    rhs_element: FieldElement,
    coefficients: dict[int, FieldElement],
    derivation: Derivation,
    extension: MonomialExtension,
) -> HyperexponentialLaurentRischDifferentialEquationResult:
    """Solve the diagonal ``a in K`` base case mode by mode."""
    from .scalar import solve_risch_differential_equation

    field = extension.field
    parent = extension.parent_field
    eta = parent.convert(extension.generator_derivative.nth(1))
    lower_derivation = derivation.lower()
    partial_solution = field.zero
    residual = field.zero
    steps: list[HyperexponentialLaurentRDEStep] = []

    for power in sorted(coefficients):
        rhs_coefficient = coefficients[power]
        shifted_coefficient = lower_coefficient + parent.convert(power) * eta
        lower_result = solve_risch_differential_equation(
            shifted_coefficient,
            rhs_coefficient,
            lower_derivation,
        )
        solution_coefficient = lower_result.solution
        term = _hyperexponential_laurent_term(
            rhs_coefficient,
            power,
            extension,
        )
        if solution_coefficient is None:
            residual += term
        else:
            partial_solution += _hyperexponential_laurent_term(
                solution_coefficient,
                power,
                extension,
            )
        steps.append(
            HyperexponentialLaurentRDEStep(
                power,
                rhs_coefficient,
                shifted_coefficient,
                solution_coefficient,
                lower_result,
            )
        )

    if not _field_elements_equal(
        derivation(partial_solution)
        + coefficient_element * partial_solution
        + residual,
        rhs_element,
        field,
    ):
        raise ValueError(
            "hyperexponential diagonal RDE decomposition failed to reconstruct"
        )
    has_obstruction = not _field_element_is_zero(residual, field)
    powers = tuple(coefficients)
    return HyperexponentialLaurentRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        eta,
        None if has_obstruction else partial_solution,
        (
            HyperexponentialLaurentRDEObstruction.LOWER_FIELD
            if has_obstruction
            else None
        ),
        partial_solution,
        residual,
        tuple(steps),
        min(powers) if powers else 0,
        max(powers) if powers else 0,
    )


def solve_hyperexponential_laurent_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> HyperexponentialLaurentRischDifferentialEquationResult:
    r"""Decide ``D(y) + a*y = b`` in the Laurent branch of ``K(t)``.

    Both ``a`` and ``b`` may be Laurent polynomials. Bronstein's valuations
    at zero and infinity first bound the support of every solution. If ``a``
    has a positive highest power, its leading product term forces corrections
    downward. Otherwise a negative lowest power forces corrections upward.
    The terminal ``a in K`` case is diagonal and recursively solves

    ``D(y_j) + (a + j*eta)*y_j = b_j``.

    Every exit retains ``partial_solution`` and ``residual`` satisfying

    ``b = D(partial_solution) + a*partial_solution + residual``.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    coefficient_element, coefficient_terms = _hyperexponential_laurent_coefficients(
        coefficient,
        extension,
        "coefficient",
    )
    rhs_element, coefficients = _hyperexponential_laurent_coefficients(
        right_hand_side,
        extension,
        "right-hand side",
    )
    eta = parent.convert(extension.generator_derivative.nth(1))
    if not coefficients:
        return HyperexponentialLaurentRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            eta,
            field.zero,
            None,
            field.zero,
            field.zero,
            (),
            0,
            0,
        )

    if not coefficient_terms or set(coefficient_terms) == {0}:
        return _solve_hyperexponential_diagonal_laurent_rde(
            coefficient_element,
            coefficient_terms.get(0, parent.zero),
            rhs_element,
            coefficients,
            derivation,
            extension,
        )

    (
        minimum_power_bound,
        maximum_power_bound,
        boundary_result,
        exceptional_power,
    ) = _hyperexponential_laurent_support_bounds(
        coefficient_terms,
        coefficients,
        extension,
        derivation.lower(),
    )
    partial_solution = field.zero
    residual = rhs_element
    corrections: list[HyperexponentialLaurentRDECorrection] = []
    if minimum_power_bound > maximum_power_bound:
        raise ValueError("hyperexponential Laurent support bounds are inconsistent")

    maximum_coefficient_power = max(coefficient_terms)
    minimum_coefficient_power = min(coefficient_terms)
    if maximum_coefficient_power > 0:
        direction = HyperexponentialLaurentRDEDirection.DESCENDING
        active_bound = maximum_power_bound
        edge_power = maximum_coefficient_power
        edge_coefficient = coefficient_terms[edge_power]
    else:
        direction = HyperexponentialLaurentRDEDirection.ASCENDING
        active_bound = minimum_power_bound
        edge_power = minimum_coefficient_power
        edge_coefficient = coefficient_terms[edge_power]

    while not _field_element_is_zero(residual, field):
        _, residual_terms = _hyperexponential_laurent_coefficients(
            residual,
            extension,
            "recurrence residual",
        )
        if not residual_terms:
            raise ValueError("a nonzero Laurent residual has no terms")
        if direction is HyperexponentialLaurentRDEDirection.DESCENDING:
            residual_edge_power = max(residual_terms)
            correction_power = residual_edge_power - edge_power
            inside_support = minimum_power_bound <= correction_power <= active_bound
        else:
            residual_edge_power = min(residual_terms)
            correction_power = residual_edge_power - edge_power
            inside_support = active_bound <= correction_power <= maximum_power_bound
        if not inside_support:
            break

        correction_coefficient = residual_terms[residual_edge_power] / edge_coefficient
        correction = _hyperexponential_laurent_term(
            correction_coefficient,
            correction_power,
            extension,
        )
        new_residual = (
            residual - derivation(correction) - coefficient_element * correction
        )
        if not _field_element_is_zero(new_residual, field):
            _, new_terms = _hyperexponential_laurent_coefficients(
                new_residual,
                extension,
                "recurrence residual",
            )
            if (
                direction is HyperexponentialLaurentRDEDirection.DESCENDING
                and max(new_terms) >= residual_edge_power
            ):
                raise ValueError(
                    "descending Laurent RDE recurrence did not lower degree"
                )
            if (
                direction is HyperexponentialLaurentRDEDirection.ASCENDING
                and min(new_terms) <= residual_edge_power
            ):
                raise ValueError(
                    "ascending Laurent RDE recurrence did not raise valuation"
                )
        corrections.append(
            HyperexponentialLaurentRDECorrection(
                correction_power,
                correction_coefficient,
                correction,
                residual,
                new_residual,
            )
        )
        partial_solution += correction
        residual = new_residual
        if direction is HyperexponentialLaurentRDEDirection.DESCENDING:
            active_bound = correction_power - 1
        else:
            active_bound = correction_power + 1

    if not _field_elements_equal(
        derivation(partial_solution)
        + coefficient_element * partial_solution
        + residual,
        rhs_element,
        field,
    ):
        raise ValueError(
            "hyperexponential Laurent RDE decomposition failed to reconstruct"
        )
    has_obstruction = not _field_element_is_zero(residual, field)
    return HyperexponentialLaurentRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        eta,
        None if has_obstruction else partial_solution,
        (
            HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
            if has_obstruction
            else None
        ),
        partial_solution,
        residual,
        (),
        minimum_power_bound,
        maximum_power_bound,
        direction,
        tuple(corrections),
        boundary_result,
        exceptional_power,
    )


def solve_hyperexponential_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> HyperexponentialRischDifferentialEquationResult:
    r"""Decide ``D(y) + a*y = b`` over hyperexponential ``K(t)``.

    This implements the complete Chapter 6 rational frontend for the
    hyperexponential case: general weak normalization, normal-denominator
    reduction, the special ``t``-denominator valuation bound, the
    hyperexponential polynomial degree bound, and Rothstein SPDE. The terminal
    equation is delegated to the complete Laurent-coefficient solver.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
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
        _weakly_normalize_hyperexponential_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    normalized_rhs = normalization_element * rhs_element
    one = Poly(1, extension.generator, domain=parent)
    if _field_element_is_zero(normalized_rhs, field):
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            field.zero,
            None,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            one,
            special_denominator_power=0,
            special_clearing_power=0,
            degree_bound=0,
        )

    denominator_reduction = _reduce_hyperexponential_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        derivation,
    )
    if not denominator_reduction.succeeded:
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            None,
            HyperexponentialRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            obstruction_polynomial=(denominator_reduction.obstruction_polynomial),
            obstruction_remainder=(denominator_reduction.obstruction_remainder),
        )
    derivative_element = denominator_reduction.derivative_coefficient
    solution_element = denominator_reduction.solution_coefficient
    transformed_rhs_element = denominator_reduction.right_hand_side
    if (
        derivative_element is None
        or solution_element is None
        or transformed_rhs_element is None
    ):
        raise ValueError(
            "successful hyperexponential denominator reduction lacks an equation"
        )

    special_reduction = _hyperexponential_special_denominator(
        derivative_element,
        solution_element,
        transformed_rhs_element,
        derivation,
    )
    (
        degree_bound,
        degree_boundary_result,
        degree_exceptional_power,
    ) = _hyperexponential_rde_degree_bound(
        special_reduction.derivative_coefficient,
        special_reduction.solution_coefficient,
        special_reduction.right_hand_side,
        derivation,
    )
    spde_result = _solve_hyperexponential_spde(
        special_reduction.derivative_coefficient,
        special_reduction.solution_coefficient,
        special_reduction.right_hand_side,
        degree_bound,
        derivation,
    )
    if spde_result.solution is None:
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            special_reduction.solution_denominator_power,
            special_reduction.clearing_power,
            degree_bound,
            spde_result.laurent_result,
            spde_result.obstruction_polynomial,
            spde_result.obstruction_remainder,
            special_reduction.boundary_result,
            special_reduction.exceptional_power,
            degree_boundary_result,
            degree_exceptional_power,
        )

    solution = _coerce(
        field,
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_reduction.denominator_bound.as_expr()
            * extension.generator**special_reduction.solution_denominator_power
        ),
    )
    if not _field_elements_equal(
        derivation(solution) + coefficient_element * solution,
        rhs_element,
        field,
    ):
        raise ValueError("hyperexponential RDE reconstruction failed")
    return HyperexponentialRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        normalized_rhs,
        denominator_reduction.denominator_bound,
        special_reduction.solution_denominator_power,
        special_reduction.clearing_power,
        degree_bound,
        spde_result.laurent_result,
        special_boundary_result=special_reduction.boundary_result,
        special_exceptional_power=special_reduction.exceptional_power,
        degree_boundary_result=degree_boundary_result,
        degree_exceptional_power=degree_exceptional_power,
    )
