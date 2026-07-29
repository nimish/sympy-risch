"""Internal constant-linear state operations for recursive parametric RDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower import Derivation, MonomialExtension, _coerce
from ._parametric_linear import constant_linear_kernel


@dataclass(frozen=True, slots=True)
class _FieldState:
    """One active parameter direction in a tower fraction field."""

    parameters: tuple[FieldElement, ...]
    partial_solution: FieldElement
    residual: FieldElement


@dataclass(frozen=True, slots=True)
class _PolynomialState:
    """One active parameter direction during polynomial reduction."""

    parameters: tuple[FieldElement, ...]
    partial_solution: Poly
    residual: Poly


@dataclass(frozen=True, slots=True)
class _PolynomialSolution:
    """One parameter direction and its polynomial solution."""

    parameters: tuple[FieldElement, ...]
    solution: Poly


def _identity_parameters(
    count: int,
    constants: Domain,
) -> tuple[tuple[FieldElement, ...], ...]:
    """Return the standard basis of ``constants**count``."""
    return tuple(
        tuple(
            cast(
                FieldElement,
                constants.one if row == column else constants.zero,
            )
            for column in range(count)
        )
        for row in range(count)
    )


def _combine_parameters(
    coefficients: tuple[FieldElement, ...],
    rows: tuple[tuple[FieldElement, ...], ...],
    constants: Domain,
) -> tuple[FieldElement, ...]:
    """Combine parameter rows over the exact constant field."""
    if not rows:
        return ()
    return tuple(
        cast(
            FieldElement,
            sum(
                (
                    constants.convert(coefficient) * constants.convert(row[column])
                    for coefficient, row in zip(
                        coefficients,
                        rows,
                        strict=True,
                    )
                ),
                constants.zero,
            ),
        )
        for column in range(len(rows[0]))
    )


def _constant_in_field(
    value: FieldElement,
    field: Domain,
    constants: Domain,
) -> FieldElement:
    """Embed one exact constant into a tower field."""
    return cast(
        FieldElement,
        field.from_sympy(constants.to_sympy(constants.convert(value))),
    )


def _combine_field_values(
    coefficients: tuple[FieldElement, ...],
    values: tuple[FieldElement, ...],
    field: Domain,
    constants: Domain,
) -> FieldElement:
    """Form one constant-linear combination in ``field``."""
    return cast(
        FieldElement,
        sum(
            (
                _constant_in_field(coefficient, field, constants) * value
                for coefficient, value in zip(
                    coefficients,
                    values,
                    strict=True,
                )
            ),
            field.zero,
        ),
    )


def _combine_field_states(
    coefficients: tuple[FieldElement, ...],
    states: tuple[_FieldState, ...],
    derivation: Derivation,
) -> _FieldState:
    """Combine active fraction-field states by one constant relation."""
    field = derivation.field
    constants = derivation.tower.constants
    return _FieldState(
        _combine_parameters(
            coefficients,
            tuple(state.parameters for state in states),
            constants,
        ),
        _combine_field_values(
            coefficients,
            tuple(state.partial_solution for state in states),
            field,
            constants,
        ),
        _combine_field_values(
            coefficients,
            tuple(state.residual for state in states),
            field,
            constants,
        ),
    )


def _combine_polynomials(
    coefficients: tuple[FieldElement, ...],
    polynomials: tuple[Poly, ...],
    domain: Domain,
    constants: Domain,
    zero: Poly,
) -> Poly:
    """Form one constant-linear combination of top polynomials."""
    return sum(
        (
            polynomial.mul_ground(
                domain.from_sympy(
                    constants.to_sympy(constants.convert(coefficient)),
                ),
            )
            for coefficient, polynomial in zip(
                coefficients,
                polynomials,
                strict=True,
            )
        ),
        zero,
    )


def _combine_polynomial_states(
    coefficients: tuple[FieldElement, ...],
    states: tuple[_PolynomialState, ...],
    extension: MonomialExtension,
    derivation: Derivation,
) -> _PolynomialState:
    """Combine active polynomial states by one constant relation."""
    zero = Poly(0, extension.generator, domain=extension.parent_field)
    constants = derivation.tower.constants
    return _PolynomialState(
        _combine_parameters(
            coefficients,
            tuple(state.parameters for state in states),
            constants,
        ),
        _combine_polynomials(
            coefficients,
            tuple(state.partial_solution for state in states),
            extension.parent_field,
            constants,
            zero,
        ),
        _combine_polynomials(
            coefficients,
            tuple(state.residual for state in states),
            extension.parent_field,
            constants,
            zero,
        ),
    )


def _polynomial_relation_kernel(
    polynomials: tuple[Poly, ...],
    derivation: Derivation,
) -> tuple[tuple[FieldElement, ...], ...]:
    """Return every constant relation among top polynomials."""
    if not polynomials:
        return ()
    if all(polynomial.is_zero for polynomial in polynomials):
        return _identity_parameters(
            len(polynomials),
            derivation.tower.constants,
        )
    return constant_linear_kernel(
        tuple(
            _coerce(derivation.field, polynomial.as_expr())
            for polynomial in polynomials
        ),
        derivation.field,
        derivation.tower.constants,
    )


def _restrict_polynomial_states(
    states: tuple[_PolynomialState, ...],
    obstructions: tuple[Poly, ...],
    extension: MonomialExtension,
    derivation: Derivation,
) -> tuple[_PolynomialState, ...]:
    """Retain exactly the combinations annihilating polynomial obstructions."""
    return tuple(
        _combine_polynomial_states(row, states, extension, derivation)
        for row in _polynomial_relation_kernel(obstructions, derivation)
    )
