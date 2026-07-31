r"""Affine spaces of differentials with prescribed residues.

For an effective pole bound ``E``, the second-kind construction represents
the residue map on ``Omega(-E)`` by an exact constant-field matrix ``M``.
Prescribing residues gives the affine system

``M c = r``.

If it is consistent, one particular differential together with the
second-kind kernel describes every solution.  If it is inconsistent, a row
in the left kernel of ``M`` whose pairing with ``r`` is nonzero is an exact
obstruction.  This includes the global residue theorem automatically; no
separate degree-one-place special case is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.matrices import DomainMatrix

from .._typing import FieldElement
from ..algebraic._field import _base_polynomial_ring
from .constant_fields import FunctionFieldConstantCertificate
from .constraints import (
    _linear_combination,
    _residue_coordinates,
    _UnsupportedValuationConstraint,
)
from .differential_constraints import (
    HessSecondKindDifferentialSpace,
    hess_second_kind_differential_space,
)
from .function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)


class HessPrescribedResidueStatus(Enum):
    """Whether the prescribed affine residue system has a solution."""

    SOLVABLE = "solvable"
    INCONSISTENT = "inconsistent"


def _target_column(
    space: HessSecondKindDifferentialSpace,
    prescribed_residues: Mapping[AlgebraicPlace, object],
) -> tuple[tuple[ExtensionElement, ...], tuple[FieldElement, ...], DomainMatrix]:
    """Normalize local residues and flatten them into one exact column."""
    unexpected = tuple(
        place for place in prescribed_residues if place not in space.residue_places
    )
    if unexpected:
        raise ValueError("a prescribed residue lies outside the pole-bound support")

    _parent, constants, _variable = _base_polynomial_ring(space.model.extension)
    residues: list[ExtensionElement] = []
    coordinates: list[FieldElement] = []
    for place in space.residue_places:
        raw_value = prescribed_residues.get(place, place.residue_field.zero)
        try:
            residue = cast(ExtensionElement, place.residue_field.convert(raw_value))
            local_coordinates = _residue_coordinates(place, residue, constants)
        except (TypeError, ValueError, _UnsupportedValuationConstraint) as exc:
            raise ValueError(
                "a prescribed residue does not belong to its place's residue field"
            ) from exc
        residues.append(residue)
        coordinates.extend(local_coordinates)

    target_coordinates = tuple(coordinates)
    column = DomainMatrix(
        [[coordinate] for coordinate in target_coordinates],
        (len(target_coordinates), 1),
        constants,
    )
    return tuple(residues), target_coordinates, column


def _solve_affine_system(
    matrix: DomainMatrix,
    target: DomainMatrix,
) -> tuple[FieldElement, ...] | None:
    """Return one exact solution, choosing every free coordinate to be zero."""
    reduced, pivots = matrix.hstack(target).rref()
    variable_count = matrix.shape[1]
    if variable_count in pivots:
        return None

    coordinates: list[FieldElement] = [
        matrix.domain.zero for _ in range(variable_count)
    ]
    rows = reduced.to_list()
    for row, pivot in enumerate(pivots):
        if pivot < variable_count:
            coordinates[pivot] = matrix.domain.convert(rows[row][variable_count])
    return tuple(coordinates)


def _left_kernel_obstruction(
    matrix: DomainMatrix,
    target: DomainMatrix,
) -> tuple[FieldElement, ...]:
    r"""Find ``u`` with ``u M = 0`` and ``u r != 0``."""
    target_entries = tuple(row[0] for row in target.to_list())
    for raw_row in matrix.transpose().nullspace().to_list():
        row = tuple(matrix.domain.convert(entry) for entry in raw_row)
        pairing = sum(
            (
                coefficient * value
                for coefficient, value in zip(row, target_entries, strict=True)
            ),
            matrix.domain.zero,
        )
        if pairing != matrix.domain.zero:
            return row
    raise ValueError("an inconsistent residue system has no left-kernel obstruction")


@dataclass(frozen=True, slots=True, init=False)
class HessPrescribedResidueSpace:
    r"""The affine solution set of one prescribed-residue problem.

    A solvable result is

    ``particular_differential + homogeneous_space``.

    An inconsistent result instead carries ``obstruction_coordinates``, an
    exact linear relation annihilating every realizable residue vector but
    not the requested target.
    """

    homogeneous_space: HessSecondKindDifferentialSpace
    prescribed_residues: tuple[ExtensionElement, ...]
    target_coordinates: tuple[FieldElement, ...]
    status: HessPrescribedResidueStatus
    particular_coordinates: tuple[FieldElement, ...] | None
    particular_differential: AlgebraicDifferential | None
    obstruction_coordinates: tuple[FieldElement, ...] | None

    def __init__(self) -> None:
        raise TypeError(
            "prescribed-residue spaces must be created by hess_prescribed_residue_space"
        )

    @classmethod
    def _create(
        cls,
        homogeneous_space: HessSecondKindDifferentialSpace,
        prescribed_residues: tuple[ExtensionElement, ...],
        target_coordinates: tuple[FieldElement, ...],
        status: HessPrescribedResidueStatus,
        particular_coordinates: tuple[FieldElement, ...] | None,
        particular_differential: AlgebraicDifferential | None,
        obstruction_coordinates: tuple[FieldElement, ...] | None,
    ) -> HessPrescribedResidueSpace:
        result = object.__new__(cls)
        object.__setattr__(result, "homogeneous_space", homogeneous_space)
        object.__setattr__(result, "prescribed_residues", prescribed_residues)
        object.__setattr__(result, "target_coordinates", target_coordinates)
        object.__setattr__(result, "status", status)
        object.__setattr__(
            result,
            "particular_coordinates",
            particular_coordinates,
        )
        object.__setattr__(
            result,
            "particular_differential",
            particular_differential,
        )
        object.__setattr__(
            result,
            "obstruction_coordinates",
            obstruction_coordinates,
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.homogeneous_space, HessSecondKindDifferentialSpace):
            raise TypeError("an affine residue space needs a second-kind kernel")
        if len(self.prescribed_residues) != len(self.residue_places):
            raise ValueError("the prescribed residues do not match the residue places")
        if len(self.target_coordinates) != self.residue_coordinate_dimension:
            raise ValueError("the prescribed residue vector has the wrong dimension")
        if not isinstance(self.status, HessPrescribedResidueStatus):
            raise TypeError("an affine residue space has an unknown status")

        if self.status is HessPrescribedResidueStatus.SOLVABLE:
            if (
                self.particular_coordinates is None
                or self.particular_differential is None
                or self.obstruction_coordinates is not None
            ):
                raise ValueError("a solvable residue space needs one particular point")
            if len(self.particular_coordinates) != len(self.ambient_basis):
                raise ValueError("the particular coordinates have the wrong dimension")
            return

        if (
            self.particular_coordinates is not None
            or self.particular_differential is not None
            or self.obstruction_coordinates is None
        ):
            raise ValueError("an inconsistent residue space needs one obstruction")
        if len(self.obstruction_coordinates) != self.residue_coordinate_dimension:
            raise ValueError("the residue obstruction has the wrong dimension")

    @property
    def model(self) -> AlgebraicFunctionField:
        """Underlying algebraic function field."""
        return self.homogeneous_space.model

    @property
    def pole_bound(self) -> AlgebraicDivisor:
        """Effective divisor bounding the allowed differential poles."""
        return self.homogeneous_space.pole_bound

    @property
    def residue_places(self) -> tuple[AlgebraicPlace, ...]:
        """Places, in coordinate order, at which residues were prescribed."""
        return self.homogeneous_space.residue_places

    @property
    def ambient_basis(self) -> tuple[AlgebraicDifferential, ...]:
        """Basis in which the particular coordinates are expressed."""
        return self.homogeneous_space.ambient_basis

    @property
    def residue_coordinate_dimension(self) -> int:
        """Dimension of the direct sum of the local residue fields."""
        return self.homogeneous_space.residue_coordinate_dimension

    @property
    def translation_dimension(self) -> int:
        """Dimension of the second-kind translation space."""
        return self.homogeneous_space.dimension

    @property
    def is_nonempty(self) -> bool:
        """Whether at least one differential realizes the prescribed residues."""
        return self.status is HessPrescribedResidueStatus.SOLVABLE

    def contains(self, differential: AlgebraicDifferential) -> bool:
        """Whether ``differential`` belongs to this affine solution set."""
        if not isinstance(differential, AlgebraicDifferential):
            raise TypeError("affine residue membership needs a differential")
        if not self.is_nonempty:
            return False
        if not self.homogeneous_space.ambient_space.contains(differential):
            return False
        return (
            self.homogeneous_space.residue_vector(differential)
            == self.target_coordinates
        )

    @property
    def reconstructs(self) -> bool:
        """Verify the particular point or the exact inconsistency obstruction."""
        if not self.homogeneous_space.reconstructs:
            return False
        matrix = self.homogeneous_space.residue_matrix
        target = DomainMatrix(
            [[coordinate] for coordinate in self.target_coordinates],
            (len(self.target_coordinates), 1),
            matrix.domain,
        )
        if self.is_nonempty:
            assert self.particular_coordinates is not None
            assert self.particular_differential is not None
            column = DomainMatrix(
                [[coordinate] for coordinate in self.particular_coordinates],
                (len(self.particular_coordinates), 1),
                matrix.domain,
            )
            if matrix * column != target:
                return False
            ambient_coefficients = tuple(
                cast(ExtensionElement, differential.coefficient)
                for differential in self.ambient_basis
            )
            reconstructed = self.model.differential(
                _linear_combination(
                    self.model,
                    ambient_coefficients,
                    self.particular_coordinates,
                )
            )
            return reconstructed == self.particular_differential and self.contains(
                reconstructed
            )

        assert self.obstruction_coordinates is not None
        obstruction = DomainMatrix(
            [list(self.obstruction_coordinates)],
            (1, len(self.obstruction_coordinates)),
            matrix.domain,
        )
        annihilates_image = all(
            entry == matrix.domain.zero
            for row in (obstruction * matrix).to_list()
            for entry in row
        )
        pairing = (obstruction * target).to_list()[0][0]
        return annihilates_image and pairing != matrix.domain.zero


def hess_prescribed_residue_space(
    model: AlgebraicFunctionField,
    pole_bound: AlgebraicDivisor,
    prescribed_residues: Mapping[AlgebraicPlace, object],
    constant_field_certificate: FunctionFieldConstantCertificate,
) -> HessPrescribedResidueSpace:
    r"""Solve for all differentials in ``Omega(-pole_bound)`` with given residues.

    Residues omitted from ``prescribed_residues`` are zero.  Every supplied
    place must occur in the pole-bound support.
    """
    if not isinstance(prescribed_residues, Mapping):
        raise TypeError("prescribed residues must be a mapping from places to values")
    homogeneous_space = hess_second_kind_differential_space(
        model,
        pole_bound,
        constant_field_certificate,
    )
    residues, target_coordinates, target = _target_column(
        homogeneous_space,
        prescribed_residues,
    )
    particular_coordinates = _solve_affine_system(
        homogeneous_space.residue_matrix,
        target,
    )
    if particular_coordinates is None:
        return HessPrescribedResidueSpace._create(
            homogeneous_space,
            residues,
            target_coordinates,
            HessPrescribedResidueStatus.INCONSISTENT,
            None,
            None,
            _left_kernel_obstruction(homogeneous_space.residue_matrix, target),
        )

    ambient_coefficients = tuple(
        cast(ExtensionElement, differential.coefficient)
        for differential in homogeneous_space.ambient_basis
    )
    particular_differential = model.differential(
        _linear_combination(
            model,
            ambient_coefficients,
            particular_coordinates,
        )
    )
    return HessPrescribedResidueSpace._create(
        homogeneous_space,
        residues,
        target_coordinates,
        HessPrescribedResidueStatus.SOLVABLE,
        particular_coordinates,
        particular_differential,
        None,
    )


__all__ = [
    "HessPrescribedResidueSpace",
    "HessPrescribedResidueStatus",
    "hess_prescribed_residue_space",
]
