r"""Parametric Risch differential equations over hyperexponential ``K(t)``.

For ``Dt = eta*t``, Bronstein's normal-denominator, special-denominator,
degree-bound, and SPDE reductions are linear in the right-hand side.  This
module applies those reductions to an entire constant-vector space at once.
Whenever a scalar reduction would report a remainder, the parametric
algorithm instead restricts the active directions to the constant
combinations that cancel it.
"""

from __future__ import annotations

from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..monomial import solve_radical_logarithmic_derivative
from ..reduction.fractions import split_normal_special
from ..reduction.polynomials import polynomial_gcd, polynomial_inverse_modulo
from ..tower import Derivation, _coerce
from ._common import _field_element_is_zero
from ._hyperexponential_parametric_laurent import (
    _solve_hyperexponential_laurent_prde,
)
from ._parametric_linear import (
    ParametricRDESolution,
    canonicalize_directions,
)
from ._parametric_states import (
    _combine_field_states,
    _combine_polynomial_states,
    _FieldState,
    _identity_parameters,
    _polynomial_relation_kernel,
    _PolynomialSolution,
    _PolynomialState,
    _restrict_polynomial_states,
)
from ._tower_polynomials import (
    _as_primitive_fraction,
    _as_primitive_polynomial,
)
from .hyperexponential import (
    _hyperexponential_boundary_power,
    _hyperexponential_laurent_coefficients,
    _hyperexponential_laurent_rde_extension,
    _weakly_normalize_hyperexponential_rde_coefficient,
)
from .parametric_dispatch import (
    ParametricRischDifferentialEquationResult,
    _build_parametric_rde_result,
)


def _normal_denominator_equation(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> tuple[
    Poly,
    FieldElement,
    FieldElement,
    tuple[_FieldState, ...],
]:
    r"""Apply the normal-denominator theorem to all parameter directions.

    A common normal denominator gives a bound valid for every linear
    combination.  Directions for which the transformed right-hand side still
    has a normal pole are restricted by the exact polynomial remainder map.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    constants = derivation.tower.constants
    _, _, coefficient_denominator = _as_primitive_fraction(
        coefficient,
        extension,
    )
    coefficient_normal, _ = split_normal_special(
        coefficient_denominator,
        derivation,
    )

    rhs_fractions = tuple(
        _as_primitive_fraction(right_hand_side, extension)
        for right_hand_side in right_hand_sides
    )
    rhs_normal = Poly(1, extension.generator, domain=parent)
    for _, _, denominator in rhs_fractions:
        normal, _ = split_normal_special(denominator, derivation)
        rhs_normal = rhs_normal.lcm(normal).monic()

    shared_normal = polynomial_gcd(coefficient_normal, rhs_normal)
    denominator_bound = (
        polynomial_gcd(rhs_normal, rhs_normal.diff())
        .exquo(polynomial_gcd(shared_normal, shared_normal.diff()))
        .monic()
    )
    coefficient_normal_element = _coerce(field, coefficient_normal.as_expr())
    bound_element = _coerce(field, denominator_bound.as_expr())
    derivative_coefficient = coefficient_normal_element * bound_element
    solution_coefficient = (
        derivative_coefficient * coefficient
        - coefficient_normal_element * derivation(bound_element)
    )
    transformed = tuple(
        coefficient_normal_element * bound_element * bound_element * right_hand_side
        for right_hand_side in right_hand_sides
    )
    states = tuple(
        _FieldState(parameters, field.zero, right_hand_side)
        for parameters, right_hand_side in zip(
            _identity_parameters(len(transformed), constants),
            transformed,
            strict=True,
        )
    )

    transformed_fractions = tuple(
        _as_primitive_fraction(state.residual, extension) for state in states
    )
    common_denominator = Poly(1, extension.generator, domain=parent)
    for _, _, denominator in transformed_fractions:
        common_denominator = common_denominator.lcm(denominator).monic()
    common_normal, _ = split_normal_special(common_denominator, derivation)
    if common_normal.degree() > 0:
        remainders = tuple(
            (numerator * common_denominator.exquo(denominator)).rem(common_normal)
            for _, numerator, denominator in transformed_fractions
        )
        states = tuple(
            _combine_field_states(row, states, derivation)
            for row in _polynomial_relation_kernel(remainders, derivation)
        )

    for state in states:
        _hyperexponential_laurent_coefficients(
            state.residual,
            extension,
            "parametric normal-denominator right-hand side",
        )
    return (
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        states,
    )


def _special_denominator_equation(
    derivative_coefficient: FieldElement,
    solution_coefficient: FieldElement,
    states: tuple[_FieldState, ...],
    derivation: Derivation,
) -> tuple[
    Poly,
    Poly,
    tuple[_PolynomialState, ...],
    int,
]:
    """Apply the common hyperexponential special-denominator bound."""
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
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
        "parametric special-denominator solution coefficient",
    )
    rhs_terms = tuple(
        _hyperexponential_laurent_coefficients(
            state.residual,
            extension,
            "parametric special-denominator right-hand side",
        )[1]
        for state in states
    )
    nonzero_rhs_terms = tuple(terms for terms in rhs_terms if terms)
    if not nonzero_rhs_terms:
        raise ValueError(
            "special-denominator reduction requires a nonzero parameter direction"
        )

    solution_valuation = min(solution_terms) if solution_terms else None
    rhs_valuation = min(min(terms) for terms in nonzero_rhs_terms)
    bounded_solution_valuation = min(0, solution_valuation or 0)
    valuation_bound = min(
        0,
        rhs_valuation - bounded_solution_valuation,
    )
    if solution_valuation == 0:
        alpha = -solution_terms[0] / parent.convert(derivative_polynomial.nth(0))
        _, exceptional_power = _hyperexponential_boundary_power(
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
    t = extension.generator
    t_power = _coerce(field, t**clearing_power)
    eta = parent.convert(extension.generator_derivative.nth(1))
    eta_element = _coerce(field, parent.to_sympy(eta))
    derivative_element = _coerce(field, derivative_polynomial.as_expr())
    transformed_derivative = _as_primitive_polynomial(
        derivative_element * t_power,
        extension,
    )
    transformed_solution = _as_primitive_polynomial(
        (
            solution_element
            + field.convert(valuation_bound) * derivative_element * eta_element
        )
        * t_power,
        extension,
    )
    transformed_states = tuple(
        _PolynomialState(
            state.parameters,
            Poly(0, t, domain=parent),
            _as_primitive_polynomial(
                state.residual
                * _coerce(field, t ** (clearing_power - valuation_bound)),
                extension,
            ),
        )
        for state in states
    )
    return (
        transformed_derivative,
        transformed_solution,
        transformed_states,
        -valuation_bound,
    )


def _parametric_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    derivation: Derivation,
) -> int:
    """Return one infinity bound valid for every active direction."""
    nonzero = tuple(value for value in right_hand_sides if not value.is_zero)
    if not nonzero:
        return 0
    degree_a = derivative_coefficient.degree()
    degree_b = solution_coefficient.degree() if not solution_coefficient.is_zero else -1
    bound = max(
        0,
        max(value.degree() for value in nonzero) - max(degree_a, degree_b),
    )
    if degree_a == degree_b:
        parent = derivative_coefficient.domain
        alpha = parent.convert(-solution_coefficient.LC() / derivative_coefficient.LC())
        extension = _hyperexponential_laurent_rde_extension(derivation)
        _, exceptional_power = _hyperexponential_boundary_power(
            -alpha,
            extension,
            derivation.lower(),
        )
        if exceptional_power is not None:
            bound = max(bound, exceptional_power)
    return bound


def _solve_parametric_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    states: tuple[_PolynomialState, ...],
    degree_bound: int,
    derivation: Derivation,
) -> tuple[_PolynomialSolution, ...]:
    """Apply Rothstein SPDE while retaining the full active parameter space."""
    extension = _hyperexponential_laurent_rde_extension(derivation)
    parent = extension.parent_field
    alpha = derivative_coefficient.one
    current_bound = degree_bound

    while True:
        if current_bound < 0:
            states = _restrict_polynomial_states(
                states,
                tuple(state.residual for state in states),
                extension,
                derivation,
            )
            return tuple(
                _PolynomialSolution(state.parameters, state.partial_solution)
                for state in states
            )

        common_factor = polynomial_gcd(
            derivative_coefficient,
            solution_coefficient,
        )
        states = _restrict_polynomial_states(
            states,
            tuple(state.residual.rem(common_factor) for state in states),
            extension,
            derivation,
        )
        states = tuple(
            _PolynomialState(
                state.parameters,
                state.partial_solution,
                state.residual.exquo(common_factor),
            )
            for state in states
        )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        if not states:
            return ()

        if derivative_coefficient.degree() == 0:
            inverse = parent.one / parent.convert(derivative_coefficient.LC())
            tail_basis = _solve_hyperexponential_laurent_prde(
                _coerce(
                    extension.field,
                    solution_coefficient.mul_ground(inverse).as_expr(),
                ),
                tuple(
                    _coerce(
                        extension.field,
                        state.residual.mul_ground(inverse).as_expr(),
                    )
                    for state in states
                ),
                derivation,
                minimum_power=0,
                maximum_power=current_bound,
            )
            result = []
            for tail in tail_basis:
                combined = _combine_polynomial_states(
                    tail.parameters,
                    states,
                    extension,
                    derivation,
                )
                tail_polynomial = _as_primitive_polynomial(
                    tail.solution,
                    extension,
                )
                result.append(
                    _PolynomialSolution(
                        combined.parameters,
                        combined.partial_solution + alpha * tail_polynomial,
                    )
                )
            return tuple(result)

        inverse = polynomial_inverse_modulo(
            solution_coefficient,
            derivative_coefficient,
        )
        next_states = []
        for state in states:
            residue = (state.residual * inverse).rem(derivative_coefficient)
            quotient = (state.residual - solution_coefficient * residue).exquo(
                derivative_coefficient
            )
            next_states.append(
                _PolynomialState(
                    state.parameters,
                    state.partial_solution + alpha * residue,
                    quotient - derivation.polynomial(residue),
                )
            )
        states = tuple(next_states)
        solution_coefficient += derivation.polynomial(derivative_coefficient)
        current_bound -= derivative_coefficient.degree()
        alpha *= derivative_coefficient


def _homogeneous_direction(
    coefficient: FieldElement,
    derivation: Derivation,
    parameter_count: int,
) -> ParametricRDESolution | None:
    """Return the unique homogeneous direction when ``Dy + a*y = 0``."""
    result = solve_radical_logarithmic_derivative(-coefficient, derivation)
    if result.minimal_power != 1:
        return None
    if result.argument is None:
        raise ValueError("a solved homogeneous RDE lacks its argument")
    return ParametricRDESolution(
        tuple(derivation.tower.constants.zero for _ in range(parameter_count)),
        result.argument,
    )


def solve_hyperexponential_parametric_rde(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> ParametricRischDifferentialEquationResult:
    r"""Return the complete PRDE solution space over hyperexponential ``K(t)``."""
    if not right_hand_sides:
        raise ValueError("a hyperexponential parametric RDE needs a right-hand side")
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    coefficient_element = _coerce(field, coefficient)
    rhs_elements = tuple(
        _coerce(field, right_hand_side) for right_hand_side in right_hand_sides
    )
    homogeneous = _homogeneous_direction(
        coefficient_element,
        derivation,
        len(rhs_elements),
    )
    if all(_field_element_is_zero(value, field) for value in rhs_elements):
        directions = tuple(
            ParametricRDESolution(parameters, field.zero)
            for parameters in _identity_parameters(
                len(rhs_elements),
                derivation.tower.constants,
            )
        )
        if homogeneous is not None:
            directions += (homogeneous,)
        return _build_parametric_rde_result(
            coefficient_element,
            rhs_elements,
            derivation,
            canonicalize_directions(
                directions,
                field,
                derivation.tower.constants,
            ),
        )

    normalization_factor, normalized_coefficient = (
        _weakly_normalize_hyperexponential_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(field, normalization_factor.as_expr())
    normalized_rhs = tuple(
        normalization_element * right_hand_side for right_hand_side in rhs_elements
    )
    (
        normal_denominator,
        derivative_coefficient,
        solution_coefficient,
        field_states,
    ) = _normal_denominator_equation(
        normalized_coefficient,
        normalized_rhs,
        derivation,
    )

    directions: tuple[ParametricRDESolution, ...] = ()
    if field_states:
        if all(_field_element_is_zero(state.residual, field) for state in field_states):
            directions = tuple(
                ParametricRDESolution(
                    state.parameters,
                    field.zero,
                )
                for state in field_states
            )
        else:
            (
                derivative_polynomial,
                solution_polynomial,
                polynomial_states,
                special_denominator_power,
            ) = _special_denominator_equation(
                derivative_coefficient,
                solution_coefficient,
                field_states,
                derivation,
            )
            degree_bound = _parametric_degree_bound(
                derivative_polynomial,
                solution_polynomial,
                tuple(state.residual for state in polynomial_states),
                derivation,
            )
            numerator_basis = _solve_parametric_spde(
                derivative_polynomial,
                solution_polynomial,
                polynomial_states,
                degree_bound,
                derivation,
            )
            denominator = (
                normalization_element
                * _coerce(field, normal_denominator.as_expr())
                * _coerce(field, extension.generator**special_denominator_power)
            )
            directions = tuple(
                ParametricRDESolution(
                    vector.parameters,
                    _coerce(field, vector.solution.as_expr()) / denominator,
                )
                for vector in numerator_basis
            )

    if homogeneous is not None:
        directions += (homogeneous,)
    canonical = canonicalize_directions(
        directions,
        field,
        derivation.tower.constants,
    )
    result = _build_parametric_rde_result(
        coefficient_element,
        rhs_elements,
        derivation,
        canonical,
    )
    return result
