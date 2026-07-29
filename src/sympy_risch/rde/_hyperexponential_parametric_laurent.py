r"""Laurent recurrence for hyperexponential parametric RDEs."""

from __future__ import annotations

from .._typing import FieldElement
from ..tower import Derivation, MonomialExtension
from ._common import _field_element_is_zero
from ._parametric_linear import ParametricRDESolution, constant_linear_kernel
from ._parametric_states import (
    _combine_field_states,
    _combine_field_values,
    _FieldState,
    _identity_parameters,
)
from .hyperexponential import (
    _hyperexponential_laurent_coefficients,
    _hyperexponential_laurent_rde_extension,
    _hyperexponential_laurent_support_bounds,
    _hyperexponential_laurent_term,
)
from .parametric_dispatch import _solve_parametric_rde_basis


def _restrict_laurent_edge(
    states: tuple[_FieldState, ...],
    edge_coefficients: tuple[FieldElement, ...],
    derivation: Derivation,
) -> tuple[_FieldState, ...]:
    """Retain combinations cancelling one Laurent edge coefficient."""
    lower = derivation.lower()
    return tuple(
        _combine_field_states(row, states, derivation)
        for row in constant_linear_kernel(
            edge_coefficients,
            lower.field,
            derivation.tower.constants,
        )
    )


def _solve_diagonal_laurent_prde(
    coefficient_element: FieldElement,
    lower_coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    rhs_terms: tuple[dict[int, FieldElement], ...],
    derivation: Derivation,
    extension: MonomialExtension,
    minimum_power: int | None,
    maximum_power: int | None,
) -> tuple[ParametricRDESolution, ...]:
    """Solve ``a in K`` mode by mode, retaining all parameter directions."""
    field = extension.field
    parent = extension.parent_field
    constants = derivation.tower.constants
    eta = parent.convert(extension.generator_derivative.nth(1))
    lower_derivation = derivation.lower()
    states = tuple(
        _FieldState(parameters, field.zero, right_hand_side)
        for parameters, right_hand_side in zip(
            _identity_parameters(len(right_hand_sides), constants),
            right_hand_sides,
            strict=True,
        )
    )
    powers = sorted({power for terms in rhs_terms for power in terms})
    for power in powers:
        if not states:
            break
        original_coefficients = tuple(
            terms.get(power, parent.zero) for terms in rhs_terms
        )
        current_coefficients = tuple(
            _combine_field_values(
                state.parameters,
                original_coefficients,
                parent,
                constants,
            )
            for state in states
        )
        if (minimum_power is not None and power < minimum_power) or (
            maximum_power is not None and power > maximum_power
        ):
            states = _restrict_laurent_edge(
                states,
                current_coefficients,
                derivation,
            )
            continue

        lower_basis = _solve_parametric_rde_basis(
            lower_coefficient + parent.convert(power) * eta,
            current_coefficients,
            lower_derivation,
        )
        next_states = []
        for lower_vector in lower_basis:
            if all(
                constants.convert(value) == constants.zero
                for value in lower_vector.parameters
            ):
                continue
            combined = _combine_field_states(
                lower_vector.parameters,
                states,
                derivation,
            )
            correction = _hyperexponential_laurent_term(
                lower_vector.solution,
                power,
                extension,
            )
            next_states.append(
                _FieldState(
                    combined.parameters,
                    combined.partial_solution + correction,
                    (
                        combined.residual
                        - derivation(correction)
                        - coefficient_element * correction
                    ),
                )
            )
        states = tuple(next_states)

    if any(not _field_element_is_zero(state.residual, field) for state in states):
        raise ValueError("diagonal hyperexponential PRDE left a residual")
    return tuple(
        ParametricRDESolution(state.parameters, state.partial_solution)
        for state in states
    )


def _solve_nondiagonal_laurent_prde(
    coefficient_element: FieldElement,
    coefficient_terms: dict[int, FieldElement],
    right_hand_sides: tuple[FieldElement, ...],
    rhs_terms: tuple[dict[int, FieldElement], ...],
    derivation: Derivation,
    extension: MonomialExtension,
    minimum_power: int | None,
    maximum_power: int | None,
) -> tuple[ParametricRDESolution, ...]:
    """Apply the forced Laurent edge recurrence to a parameter space."""
    field = extension.field
    parent = extension.parent_field
    constants = derivation.tower.constants
    states = tuple(
        _FieldState(parameters, field.zero, right_hand_side)
        for parameters, right_hand_side in zip(
            _identity_parameters(len(right_hand_sides), constants),
            right_hand_sides,
            strict=True,
        )
    )
    nonzero_terms = tuple(terms for terms in rhs_terms if terms)
    if not nonzero_terms:
        return tuple(
            ParametricRDESolution(state.parameters, state.partial_solution)
            for state in states
        )

    envelope = {power: parent.one for terms in nonzero_terms for power in terms}
    support_minimum, support_maximum, _, _ = _hyperexponential_laurent_support_bounds(
        coefficient_terms,
        envelope,
        extension,
        derivation.lower(),
    )
    if minimum_power is not None:
        support_minimum = max(support_minimum, minimum_power)
    if maximum_power is not None:
        support_maximum = min(support_maximum, maximum_power)

    maximum_coefficient_power = max(coefficient_terms)
    minimum_coefficient_power = min(coefficient_terms)
    descending = maximum_coefficient_power > 0
    coefficient_edge_power = (
        maximum_coefficient_power if descending else minimum_coefficient_power
    )
    coefficient_edge = coefficient_terms[coefficient_edge_power]

    while states and any(
        not _field_element_is_zero(state.residual, field) for state in states
    ):
        state_terms = tuple(
            _hyperexponential_laurent_coefficients(
                state.residual,
                extension,
                "parametric Laurent recurrence residual",
            )[1]
            for state in states
        )
        nonempty = tuple(terms for terms in state_terms if terms)
        residual_edge_power = (
            max(max(terms) for terms in nonempty)
            if descending
            else min(min(terms) for terms in nonempty)
        )
        correction_power = residual_edge_power - coefficient_edge_power
        if not support_minimum <= correction_power <= support_maximum:
            edge_coefficients = tuple(
                terms.get(residual_edge_power, parent.zero) for terms in state_terms
            )
            states = _restrict_laurent_edge(
                states,
                edge_coefficients,
                derivation,
            )
            continue

        next_states = []
        for state, terms in zip(states, state_terms, strict=True):
            correction_coefficient = (
                terms.get(residual_edge_power, parent.zero) / coefficient_edge
            )
            correction = _hyperexponential_laurent_term(
                correction_coefficient,
                correction_power,
                extension,
            )
            residual = (
                state.residual
                - derivation(correction)
                - coefficient_element * correction
            )
            if not _field_element_is_zero(residual, field):
                _, terms_after = _hyperexponential_laurent_coefficients(
                    residual,
                    extension,
                    "parametric Laurent recurrence residual",
                )
                if descending and max(terms_after) >= residual_edge_power:
                    raise ValueError(
                        "parametric descending Laurent recurrence did not descend"
                    )
                if not descending and min(terms_after) <= residual_edge_power:
                    raise ValueError(
                        "parametric ascending Laurent recurrence did not ascend"
                    )
            next_states.append(
                _FieldState(
                    state.parameters,
                    state.partial_solution + correction,
                    residual,
                )
            )
        states = tuple(next_states)

    return tuple(
        ParametricRDESolution(state.parameters, state.partial_solution)
        for state in states
    )


def _solve_hyperexponential_laurent_prde(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
    *,
    minimum_power: int | None = None,
    maximum_power: int | None = None,
) -> tuple[ParametricRDESolution, ...]:
    """Return all parameter directions with Laurent solutions in given bounds."""
    extension = _hyperexponential_laurent_rde_extension(derivation)
    coefficient_element, coefficient_terms = _hyperexponential_laurent_coefficients(
        coefficient,
        extension,
        "parametric coefficient",
    )
    rhs_data = tuple(
        _hyperexponential_laurent_coefficients(
            right_hand_side,
            extension,
            "parametric right-hand side",
        )
        for right_hand_side in right_hand_sides
    )
    rhs_elements = tuple(item[0] for item in rhs_data)
    rhs_terms = tuple(item[1] for item in rhs_data)
    if not coefficient_terms or set(coefficient_terms) == {0}:
        return _solve_diagonal_laurent_prde(
            coefficient_element,
            coefficient_terms.get(0, extension.parent_field.zero),
            rhs_elements,
            rhs_terms,
            derivation,
            extension,
            minimum_power,
            maximum_power,
        )
    return _solve_nondiagonal_laurent_prde(
        coefficient_element,
        coefficient_terms,
        rhs_elements,
        rhs_terms,
        derivation,
        extension,
        minimum_power,
        maximum_power,
    )
