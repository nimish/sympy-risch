r"""Exact residue constraints on finite-dimensional differential spaces.

Let ``E`` be an effective divisor.  The ambient space

``Omega(-E) = {omega : div(omega) + E >= 0}``

contains exactly the differentials whose poles are bounded by ``E``.  Only
places in the support of ``E`` can contribute a residue.  Flattening each
closed residue field into its native constant-field coordinates therefore
turns the residue map on ``Omega(-E)`` into one exact matrix.

The kernel is the complete space of second-kind differentials with that pole
bound.  Unlike the ambient Algorithm 24 representation, a residue constraint
has one matrix column per vector-space basis element, so this operation
expands the ambient basis deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.matrices import DomainMatrix

from .._typing import FieldElement
from ..algebraic._field import _base_polynomial_ring
from .constant_fields import (
    FunctionFieldConstantCertificate,
)
from .constraints import (
    _coordinate_rows,
    _linear_combination,
    _residue_coordinates,
    _UnsupportedValuationConstraint,
)
from .differential_spaces import (
    HessDifferentialSpace,
    hess_differential_space,
)
from .function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)


def _residue_vector(
    model: AlgebraicFunctionField,
    places: tuple[AlgebraicPlace, ...],
    differential: AlgebraicDifferential,
) -> tuple[FieldElement, ...]:
    """Flatten the differential's residues in the declared place order."""
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    if differential.coefficient == model.extension.field.zero:
        return tuple(
            constants.zero for place in places for _coordinate in range(place.degree)
        )
    return tuple(
        coordinate
        for place in places
        for coordinate in _residue_coordinates(
            place,
            differential.residue(place),
            constants,
        )
    )


def _residue_matrix(
    model: AlgebraicFunctionField,
    places: tuple[AlgebraicPlace, ...],
    basis: tuple[AlgebraicDifferential, ...],
) -> DomainMatrix:
    """Return the exact constant matrix of the residue map."""
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    height = sum(place.degree for place in places)
    columns = tuple(
        _residue_vector(model, places, differential) for differential in basis
    )
    rows = [
        [columns[column][row] for column in range(len(columns))]
        for row in range(height)
    ]
    return DomainMatrix(rows, (height, len(basis)), constants)


def _matrix_annihilates_coordinates(
    matrix: DomainMatrix,
    coordinates: tuple[tuple[object, ...], ...],
) -> bool:
    """Whether every coordinate row lies in the matrix kernel."""
    if any(len(row) != matrix.shape[1] for row in coordinates):
        return False
    coordinate_matrix = DomainMatrix(
        [list(row) for row in coordinates],
        (len(coordinates), matrix.shape[1]),
        matrix.domain,
    )
    image = matrix * coordinate_matrix.transpose()
    return all(entry == matrix.domain.zero for row in image.to_list() for entry in row)


@dataclass(frozen=True, slots=True, init=False)
class HessSecondKindDifferentialSpace:
    r"""The exact residue kernel inside one complete ``Omega(-E)``.

    ``residue_matrix`` maps coordinates in ``ambient_basis`` to the
    concatenated constant coordinates of residues at ``residue_places``.
    ``basis_coordinates`` is a row basis of its nullspace.
    """

    pole_bound: AlgebraicDivisor
    ambient_space: HessDifferentialSpace
    constant_field_certificate: FunctionFieldConstantCertificate
    residue_places: tuple[AlgebraicPlace, ...]
    ambient_basis: tuple[AlgebraicDifferential, ...]
    residue_matrix: DomainMatrix
    basis_coordinates: tuple[tuple[object, ...], ...]
    basis: tuple[AlgebraicDifferential, ...]

    def __init__(self) -> None:
        raise TypeError(
            "second-kind spaces must be created by hess_second_kind_differential_space"
        )

    @classmethod
    def _create(
        cls,
        pole_bound: AlgebraicDivisor,
        ambient_space: HessDifferentialSpace,
        constant_field_certificate: FunctionFieldConstantCertificate,
        residue_places: tuple[AlgebraicPlace, ...],
        ambient_basis: tuple[AlgebraicDifferential, ...],
        residue_matrix: DomainMatrix,
        basis_coordinates: tuple[tuple[object, ...], ...],
        basis: tuple[AlgebraicDifferential, ...],
    ) -> HessSecondKindDifferentialSpace:
        result = object.__new__(cls)
        object.__setattr__(result, "pole_bound", pole_bound)
        object.__setattr__(result, "ambient_space", ambient_space)
        object.__setattr__(
            result,
            "constant_field_certificate",
            constant_field_certificate,
        )
        object.__setattr__(result, "residue_places", residue_places)
        object.__setattr__(result, "ambient_basis", ambient_basis)
        object.__setattr__(result, "residue_matrix", residue_matrix)
        object.__setattr__(result, "basis_coordinates", basis_coordinates)
        object.__setattr__(result, "basis", basis)
        result._validate()
        return result

    def _validate(self) -> None:
        if not self.pole_bound.is_effective:
            raise ValueError("a differential pole bound must be effective")
        if self.ambient_space.model is not self.pole_bound.model:
            raise ValueError("the ambient differential space uses another model")
        if self.ambient_space.lower_bound != -self.pole_bound:
            raise ValueError("the ambient differential space has another pole bound")
        if self.constant_field_certificate.model is not self.pole_bound.model:
            raise ValueError("the constant-field certificate belongs to another model")
        if self.residue_places != self.pole_bound.support:
            raise ValueError("the residue places do not match the pole bound")
        expected_shape = (
            sum(place.degree for place in self.residue_places),
            len(self.ambient_basis),
        )
        if self.residue_matrix.shape != expected_shape:
            raise ValueError("the residue matrix has the wrong shape")
        if len(self.ambient_basis) != self.ambient_space.dimension:
            raise ValueError("the expanded ambient basis has the wrong dimension")
        if len(self.basis) != len(self.basis_coordinates):
            raise ValueError("second-kind basis coordinates have the wrong length")

    @property
    def model(self) -> AlgebraicFunctionField:
        """Underlying algebraic function field."""
        return self.pole_bound.model

    @property
    def dimension(self) -> int:
        """Dimension of the exact second-kind kernel."""
        return len(self.basis)

    @property
    def residue_coordinate_dimension(self) -> int:
        """Dimension of the displayed direct sum of residue fields."""
        return self.residue_matrix.shape[0]

    def residue_vector(
        self,
        differential: AlgebraicDifferential,
    ) -> tuple[FieldElement, ...]:
        """Apply the flattened residue map to one ambient differential."""
        if not isinstance(differential, AlgebraicDifferential):
            raise TypeError("the residue map needs an algebraic differential")
        if not self.ambient_space.contains(differential):
            raise ValueError("the differential lies outside the bounded ambient space")
        return _residue_vector(self.model, self.residue_places, differential)

    def contains(self, differential: AlgebraicDifferential) -> bool:
        """Whether a differential satisfies the pole bound and has zero residues."""
        if not isinstance(differential, AlgebraicDifferential):
            raise TypeError("second-kind membership needs an algebraic differential")
        if not self.ambient_space.contains(differential):
            return False
        _parent, constants, _variable = _base_polynomial_ring(self.model.extension)
        return all(
            coordinate == constants.zero
            for coordinate in _residue_vector(
                self.model,
                self.residue_places,
                differential,
            )
        )

    @property
    def reconstructs(self) -> bool:
        """Verify exact nullspace dimension, coordinates, and basis elements."""
        if (
            len(self.basis_coordinates)
            != len(self.ambient_basis) - self.residue_matrix.rank()
        ):
            return False
        if not _matrix_annihilates_coordinates(
            self.residue_matrix,
            self.basis_coordinates,
        ):
            return False
        coordinate_matrix = DomainMatrix(
            [list(row) for row in self.basis_coordinates],
            (len(self.basis_coordinates), len(self.ambient_basis)),
            self.residue_matrix.domain,
        )
        if coordinate_matrix.rank() != len(self.basis_coordinates):
            return False
        ambient_coefficients = tuple(
            cast(ExtensionElement, differential.coefficient)
            for differential in self.ambient_basis
        )
        reconstructed = tuple(
            self.model.differential(
                _linear_combination(
                    self.model,
                    ambient_coefficients,
                    coordinates,
                )
            )
            for coordinates in self.basis_coordinates
        )
        return reconstructed == self.basis


def hess_second_kind_differential_space(
    model: AlgebraicFunctionField,
    pole_bound: AlgebraicDivisor,
    constant_field_certificate: FunctionFieldConstantCertificate,
) -> HessSecondKindDifferentialSpace:
    r"""Compute the residue-zero kernel in ``Omega(-pole_bound)``."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a second-kind space needs an algebraic function field")
    if not isinstance(pole_bound, AlgebraicDivisor):
        raise TypeError("a second-kind space needs a pole-bound divisor")
    if pole_bound.model is not model:
        raise ValueError("the pole bound belongs to another global model")
    if not pole_bound.is_effective:
        raise ValueError("a differential pole bound must be effective")
    if not isinstance(
        constant_field_certificate,
        FunctionFieldConstantCertificate,
    ):
        raise TypeError("a second-kind space needs a constant-field certificate")
    if constant_field_certificate.model is not model:
        raise ValueError("the constant-field certificate belongs to another model")

    ambient_space = hess_differential_space(model, -pole_bound)
    ambient_basis = ambient_space.expand()
    residue_places = pole_bound.support
    try:
        residue_matrix = _residue_matrix(model, residue_places, ambient_basis)
    except _UnsupportedValuationConstraint as exc:
        raise NotImplementedError(
            f"second-kind residue coordinates are unavailable: {exc}"
        ) from exc
    basis_coordinates = _coordinate_rows(residue_matrix.nullspace())
    ambient_coefficients = tuple(
        cast(ExtensionElement, differential.coefficient)
        for differential in ambient_basis
    )
    basis = tuple(
        model.differential(
            _linear_combination(model, ambient_coefficients, coordinates)
        )
        for coordinates in basis_coordinates
    )
    return HessSecondKindDifferentialSpace._create(
        pole_bound,
        ambient_space,
        constant_field_certificate,
        residue_places,
        ambient_basis,
        residue_matrix,
        basis_coordinates,
        basis,
    )


__all__ = [
    "HessSecondKindDifferentialSpace",
    "hess_second_kind_differential_space",
]
