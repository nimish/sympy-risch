r"""Parametric RDE operations for one terminal diagonal simple radical."""

from __future__ import annotations

from sympy.polys.matrices import DomainMatrix

from .._typing import FieldElement
from ..algebraic.relative import (
    _diagonal_generator_logarithmic_derivative,
    _lower_scalar,
    recognize_relative_simple_radical,
)
from ..tower import AlgebraicExtension, Derivation
from ._parametric_linear import ParametricRDESolution
from ._parametric_states import _combine_field_values

type RelativeParametricMode = tuple[FieldElement, tuple[FieldElement, ...]]


def relative_simple_radical_parametric_modes(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> tuple[RelativeParametricMode, ...]:
    r"""Split one top PRDE into diagonal lower-field power-basis modes."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("relative simple-radical PRDE needs an algebraic top level")
    if recognize_relative_simple_radical(extension) is None:
        raise NotImplementedError(
            "parametric RDE descent needs a terminal pure simple radical"
        )
    lower_coefficient = _lower_scalar(extension, coefficient)
    if lower_coefficient is None:
        raise NotImplementedError(
            "a non-lower algebraic PRDE coefficient requires a coupled system"
        )
    eta = _diagonal_generator_logarithmic_derivative(extension, derivation)
    if eta is None:
        raise NotImplementedError(
            "the simple-radical generator derivative is not diagonal over the lower field"
        )

    coordinate_rows = tuple(
        extension.power_basis_coordinates(right_hand_side)
        for right_hand_side in right_hand_sides
    )
    return tuple(
        (
            lower_coefficient + extension.parent_field.convert(power) * eta,
            tuple(row[power] for row in coordinate_rows),
        )
        for power in range(extension.field.rank)
    )


def intersect_relative_parametric_modes(
    mode_bases: tuple[tuple[ParametricRDESolution, ...], ...],
    parameter_count: int,
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> tuple[ParametricRDESolution, ...]:
    r"""Intersect modewise PRDE spaces on their shared parameter vector.

    The unknowns are one common vector ``c`` followed by the coefficients of
    each mode's lower-field basis. For each mode, one exact constant-linear
    system equates ``c`` with that basis's parameter projection.
    """
    constants = derivation.tower.constants
    offsets = [parameter_count]
    for basis in mode_bases:
        offsets.append(offsets[-1] + len(basis))

    rows = []
    for mode, basis in enumerate(mode_bases):
        for parameter in range(parameter_count):
            row = [constants.zero] * offsets[-1]
            row[parameter] = constants.one
            for index, vector in enumerate(basis, start=offsets[mode]):
                row[index] = -constants.convert(vector.parameters[parameter])
            rows.append(row)

    constraints = DomainMatrix(
        rows,
        (len(rows), offsets[-1]),
        constants,
    )
    kernel, _ = constraints.nullspace(divide_last=True).rref()
    lower_field = extension.parent_field
    directions = []
    for row in kernel.to_list():
        parameters = tuple(constants.convert(value) for value in row[:parameter_count])
        coordinates = tuple(
            _combine_field_values(
                tuple(
                    constants.convert(value)
                    for value in row[offsets[mode] : offsets[mode + 1]]
                ),
                tuple(vector.solution for vector in basis),
                lower_field,
                constants,
            )
            for mode, basis in enumerate(mode_bases)
        )
        directions.append(
            ParametricRDESolution(
                parameters,
                extension.from_power_basis_coordinates(coordinates),
            )
        )
    return tuple(directions)
