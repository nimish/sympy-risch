r"""Parametric Risch differential equations over primitive ``K(t)``."""

from __future__ import annotations

from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..reduction.polynomials import polynomial_gcd, polynomial_inverse_modulo
from ..tower import Derivation, _coerce
from ._parametric_linear import (
    ParametricRDESolution,
    canonicalize_directions,
)
from ._parametric_states import (
    _combine_parameters,
    _combine_polynomial_states,
    _combine_polynomials,
    _identity_parameters,
    _polynomial_relation_kernel,
    _PolynomialSolution,
    _PolynomialState,
    _restrict_polynomial_states,
)
from ._tower_polynomials import _as_primitive_fraction
from .parametric_dispatch import (
    ParametricRischDifferentialEquationResult,
    _build_parametric_rde_result,
    _solve_parametric_rde_basis,
)
from .primitive import (
    _primitive_parametric_rde_degree_bound,
    _primitive_rde_extension,
    _weakly_normalize_primitive_rde_coefficient,
)


def _solve_positive_degree_polynomial_prde(
    coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    derivation: Derivation,
) -> tuple[_PolynomialSolution, ...]:
    """Apply the deterministic no-cancellation recurrence simultaneously."""
    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    zero = Poly(0, extension.generator, domain=parent)
    partial_solutions = []
    residuals = []
    for right_hand_side in right_hand_sides:
        partial = zero
        residual = right_hand_side
        while not residual.is_zero:
            correction_degree = residual.degree() - coefficient.degree()
            if correction_degree < 0:
                break
            correction = Poly.from_dict(
                {
                    (correction_degree,): (
                        parent.convert(residual.LC()) / parent.convert(coefficient.LC())
                    ),
                },
                extension.generator,
                domain=parent,
            )
            partial += correction
            next_residual = (
                residual - derivation.polynomial(correction) - coefficient * correction
            )
            if (
                not next_residual.is_zero
                and next_residual.degree() >= residual.degree()
            ):
                raise ValueError(
                    "primitive parametric no-cancellation recurrence did not descend"
                )
            residual = next_residual
        partial_solutions.append(partial)
        residuals.append(residual)

    kernel = _polynomial_relation_kernel(tuple(residuals), derivation)
    parameters = _identity_parameters(
        len(right_hand_sides),
        derivation.tower.constants,
    )
    return tuple(
        _PolynomialSolution(
            _combine_parameters(
                row,
                parameters,
                derivation.tower.constants,
            ),
            _combine_polynomials(
                row,
                tuple(partial_solutions),
                parent,
                derivation.tower.constants,
                zero,
            ),
        )
        for row in kernel
    )


def _solve_lower_coefficient_polynomial_prde(
    coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    degree_bound: int,
    derivation: Derivation,
) -> tuple[_PolynomialSolution, ...]:
    r"""Descend through powers of ``t``, recursively solving lower PRDEs."""
    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    constants = derivation.tower.constants
    lower_derivation = derivation.lower()
    zero = Poly(0, extension.generator, domain=parent)
    coefficient_lower = parent.convert(coefficient.nth(0))
    states = tuple(
        _PolynomialState(parameters, zero, right_hand_side)
        for parameters, right_hand_side in zip(
            _identity_parameters(len(right_hand_sides), constants),
            right_hand_sides,
            strict=True,
        )
    )

    for degree in range(degree_bound, -1, -1):
        lower_right_hand_sides = tuple(
            parent.convert(state.residual.nth(degree)) for state in states
        )
        lower_basis = _solve_parametric_rde_basis(
            coefficient_lower,
            lower_right_hand_sides,
            lower_derivation,
        )
        next_states = []
        for lower_vector in lower_basis:
            combined = _combine_polynomial_states(
                lower_vector.parameters,
                states,
                extension,
                derivation,
            )
            correction = Poly.from_dict(
                {(degree,): parent.convert(lower_vector.solution)},
                extension.generator,
                domain=parent,
            )
            partial_solution = combined.partial_solution + correction
            residual = (
                combined.residual
                - derivation.polynomial(correction)
                - coefficient * correction
            )
            if not residual.is_zero and residual.degree() >= degree:
                raise ValueError(
                    "primitive parametric coefficient recurrence did not descend"
                )
            if (
                all(
                    constants.convert(value) == constants.zero
                    for value in combined.parameters
                )
                and partial_solution.is_zero
            ):
                continue
            next_states.append(
                _PolynomialState(
                    combined.parameters,
                    partial_solution,
                    residual,
                )
            )
        states = tuple(next_states)

    if any(not state.residual.is_zero for state in states):
        raise ValueError("primitive parametric polynomial recurrence left a residual")
    return tuple(
        _PolynomialSolution(state.parameters, state.partial_solution)
        for state in states
    )


def _solve_primitive_polynomial_prde(
    coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    degree_bound: int,
    derivation: Derivation,
) -> tuple[_PolynomialSolution, ...]:
    """Solve the terminal polynomial PRDE over a primitive extension."""
    if coefficient.is_zero or coefficient.degree() <= 0:
        return _solve_lower_coefficient_polynomial_prde(
            coefficient,
            right_hand_sides,
            degree_bound,
            derivation,
        )
    return _solve_positive_degree_polynomial_prde(
        coefficient,
        right_hand_sides,
        derivation,
    )


def _solve_primitive_parametric_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    degree_bound: int,
    derivation: Derivation,
) -> tuple[_PolynomialSolution, ...]:
    r"""Apply Rothstein SPDE while retaining every surviving parameter direction."""
    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    constants = derivation.tower.constants
    zero = Poly(0, extension.generator, domain=parent)
    states = tuple(
        _PolynomialState(parameters, zero, right_hand_side)
        for parameters, right_hand_side in zip(
            _identity_parameters(len(right_hand_sides), constants),
            right_hand_sides,
            strict=True,
        )
    )
    alpha = derivative_coefficient.one
    current_bound = degree_bound

    while True:
        if current_bound < 0:
            restricted = _restrict_polynomial_states(
                states,
                tuple(state.residual for state in states),
                extension,
                derivation,
            )
            return tuple(
                _PolynomialSolution(state.parameters, state.partial_solution)
                for state in restricted
            )

        common_factor = polynomial_gcd(
            derivative_coefficient,
            solution_coefficient,
        )
        remainders = tuple(state.residual.rem(common_factor) for state in states)
        states = _restrict_polynomial_states(
            states,
            remainders,
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

        if derivative_coefficient.degree() == 0:
            inverse = parent.one / parent.convert(derivative_coefficient.LC())
            polynomial_solutions = _solve_primitive_polynomial_prde(
                solution_coefficient.mul_ground(inverse),
                tuple(state.residual.mul_ground(inverse) for state in states),
                current_bound,
                derivation,
            )
            result = []
            for tail in polynomial_solutions:
                combined = _combine_polynomial_states(
                    tail.parameters,
                    states,
                    extension,
                    derivation,
                )
                result.append(
                    _PolynomialSolution(
                        combined.parameters,
                        combined.partial_solution + alpha * tail.solution,
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


def solve_primitive_parametric_rde(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> ParametricRischDifferentialEquationResult:
    r"""Return the complete PRDE solution space over primitive ``K(t)``."""
    if not right_hand_sides:
        raise ValueError("a primitive parametric RDE needs a right-hand side")
    extension = _primitive_rde_extension(derivation)
    field = extension.field
    coefficient_element = _coerce(field, coefficient)
    right_hand_side_elements = tuple(
        _coerce(field, right_hand_side) for right_hand_side in right_hand_sides
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_primitive_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(field, normalization_factor.as_expr())
    normalized_right_hand_sides = tuple(
        normalization_element * right_hand_side
        for right_hand_side in right_hand_side_elements
    )
    _, coefficient_numerator, coefficient_denominator = _as_primitive_fraction(
        normalized_coefficient,
        extension,
    )

    denominator_bound = Poly(1, extension.generator, domain=extension.parent_field)
    rhs_fractions = []
    for right_hand_side in normalized_right_hand_sides:
        fraction = _as_primitive_fraction(right_hand_side, extension)
        rhs_fractions.append(fraction)
        denominator_bound = denominator_bound.lcm(fraction[2]).monic()

    derivative_coefficient = coefficient_denominator * denominator_bound
    solution_coefficient = (
        denominator_bound * coefficient_numerator
        - coefficient_denominator * derivation.polynomial(denominator_bound)
    )
    multiplier = coefficient_denominator * denominator_bound * denominator_bound
    transformed_right_hand_sides = []
    for _, numerator, denominator in rhs_fractions:
        quotient, remainder = multiplier.div(denominator)
        if not remainder.is_zero:
            raise ValueError(
                "the primitive parametric denominator failed to clear a right-hand side"
            )
        transformed_right_hand_sides.append(quotient * numerator)
    transformed_tuple = tuple(transformed_right_hand_sides)

    degree_bound = _primitive_parametric_rde_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_tuple,
        derivation,
    )
    numerator_basis = _solve_primitive_parametric_spde(
        derivative_coefficient,
        solution_coefficient,
        transformed_tuple,
        degree_bound,
        derivation,
    )
    denominator = normalization_element * _coerce(
        field,
        denominator_bound.as_expr(),
    )
    directions = canonicalize_directions(
        tuple(
            ParametricRDESolution(
                vector.parameters,
                _coerce(field, vector.solution.as_expr()) / denominator,
            )
            for vector in numerator_basis
        ),
        field,
        derivation.tower.constants,
    )
    return _build_parametric_rde_result(
        coefficient_element,
        right_hand_side_elements,
        derivation,
        directions,
    )
