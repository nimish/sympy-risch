r"""Exact constant-linear algebra shared by algebraic Risch algorithms.

Riemann--Roch spaces are vector spaces over the full constant field.  Once a
complete basis has been certified, exactness and bounded algebraic RDEs both
reduce to solving one finite linear system over that field.  This module keeps
that neutral operation separate from either theorem-specific caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement

from .._typing import FieldElement
from ..algebraic_geometry.function_field import AlgebraicFunctionField
from ..algebraic_geometry.riemann_roch import constant_coordinate_matrix
from ._field import _base_polynomial_ring, _extension_element


@dataclass(frozen=True, slots=True)
class ConstantLinearSolve:
    """The exact outcome of a constant-linear span-membership problem."""

    coordinates: tuple[FieldElement, ...] | None
    operator_rank: int
    augmented_rank: int

    @property
    def consistent(self) -> bool:
        """Whether the target belongs to the constant span of the columns."""
        return self.coordinates is not None


def constant_linear_combination(
    model: AlgebraicFunctionField,
    basis: tuple[ExtensionElement, ...],
    coordinates: tuple[FieldElement, ...],
) -> ExtensionElement:
    """Rebuild a constant-linear combination in the native function field."""
    if len(basis) != len(coordinates):
        raise ValueError("constant coordinates have the wrong basis dimension")
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    result = model.extension.field.zero
    for coordinate, element in zip(coordinates, basis, strict=True):
        coefficient = constants.convert(coordinate)
        if coefficient != constants.zero:
            result += (
                _extension_element(
                    model.extension,
                    constants.to_sympy(coefficient),
                )
                * element
            )
    return cast(ExtensionElement, result)


def solve_constant_linear_span(
    model: AlgebraicFunctionField,
    columns: tuple[ExtensionElement, ...],
    target: ExtensionElement,
) -> ConstantLinearSolve:
    r"""Solve ``sum(c_i*columns[i]) = target`` over the constant field.

    The coordinate matrix is constructed once.  A pivot in its final,
    augmented column is the exact inconsistency criterion.  When solutions
    are nonunique, free coordinates are deterministically set to zero.
    """
    augmented = constant_coordinate_matrix(model, (*columns, target))
    reduced, pivots = augmented.rref()
    right_column = len(columns)
    operator_rank = sum(pivot < right_column for pivot in pivots)
    augmented_rank = len(pivots)
    if right_column in pivots:
        return ConstantLinearSolve(None, operator_rank, augmented_rank)

    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    coordinates: list[FieldElement] = [constants.zero for _ in columns]
    reduced_rows = reduced.to_list()
    for row, pivot in enumerate(pivots):
        if pivot < right_column:
            coordinates[pivot] = constants.convert(reduced_rows[row][right_column])
    return ConstantLinearSolve(tuple(coordinates), operator_rank, augmented_rank)


__all__ = [
    "ConstantLinearSolve",
    "constant_linear_combination",
    "solve_constant_linear_span",
]
