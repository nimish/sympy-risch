r"""Exact rational coordinates of displayed algebraic residues."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, lcm

from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed

from .._typing import FieldElement
from ..algebraic_geometry.function_field import AlgebraicDivisor
from ._field import _base_polynomial_ring
from .all_places import (
    AlgebraicAllPlacesCollected,
    AlgebraicLocalDifferentialResidue,
)
from .constant_analysis import AlgebraicConstantFieldStatus


def displayed_constant_residue(
    local_residue: AlgebraicLocalDifferentialResidue,
    constants: Domain,
) -> FieldElement | None:
    """Return a local residue when it lies in the displayed constant field."""
    coordinates = local_residue.constant_coordinates
    try:
        zero = constants.zero
    except AttributeError:
        return None
    if not coordinates or any(coordinate != zero for coordinate in coordinates[1:]):
        return None
    try:
        return constants.convert(coordinates[0])
    except (CoercionFailed, TypeError, ValueError):
        return None


def primitive_rational_scale(
    residues: tuple[FieldElement, ...],
) -> FieldElement:
    """Return the positive rational content of nonzero rational residues."""
    numerator_gcd = 0
    denominator_lcm = 1
    for residue in residues:
        rational = QQ.convert(residue)
        numerator_gcd = gcd(numerator_gcd, abs(int(QQ.numer(rational))))
        denominator_lcm = lcm(denominator_lcm, int(QQ.denom(rational)))
    if numerator_gcd == 0:
        raise ValueError("rational residue content needs a nonzero residue")
    return QQ.exquo(QQ.convert(numerator_gcd), QQ.convert(denominator_lcm))


def integer_multiple(residue: FieldElement, scale: FieldElement) -> int:
    """Return the exact integral quotient ``residue / scale``."""
    quotient = QQ.exquo(QQ.convert(residue), QQ.convert(scale))
    if QQ.denom(quotient) != 1:
        raise ValueError("the rational residue scale did not clear denominators")
    return int(QQ.numer(quotient))


def constant_coordinate_basis(
    constants: Domain,
) -> tuple[FieldElement, ...] | None:
    """Return the exact ``QQ`` power basis of supported displayed constants."""
    if constants == QQ:
        return (QQ.one,)
    if not isinstance(constants, AlgebraicField) or constants.dom != QQ:
        return None
    degree = constants.ext.minpoly.degree()
    return tuple(constants.unit**power for power in range(degree))


def rational_coordinates_of_constant(
    constants: Domain,
    value: FieldElement,
) -> tuple[FieldElement, ...] | None:
    """Express one displayed constant in its exact ``QQ`` power basis."""
    if constants == QQ:
        try:
            return (QQ.convert(value),)
        except (CoercionFailed, TypeError, ValueError):
            return None
    if not isinstance(constants, AlgebraicField) or constants.dom != QQ:
        return None
    try:
        converted = constants.convert(value)
        coefficients = tuple(QQ.convert(coefficient) for coefficient in converted.rep)
    except (AttributeError, CoercionFailed, TypeError, ValueError):
        return None
    degree = constants.ext.minpoly.degree()
    if len(coefficients) > degree:
        raise ValueError("an algebraic constant has an overlong power representation")
    return tuple(reversed((QQ.zero,) * (degree - len(coefficients)) + coefficients))


@dataclass(frozen=True, slots=True)
class AlgebraicResidueCoordinateCertificate:
    r"""Replayable evidence for one rational coordinate residue divisor.

    The coefficient and integral divisor are recomputed from the complete
    all-place residue collection.  Construction is therefore proof checking,
    rather than a provenance assertion about which caller created the object.
    """

    collection: AlgebraicAllPlacesCollected
    coordinate_index: int
    coefficient: FieldElement
    divisor: AlgebraicDivisor

    def __post_init__(self) -> None:
        collection = self.collection
        if not isinstance(collection, AlgebraicAllPlacesCollected):
            raise TypeError("a residue coordinate needs all-place residue data")
        if (
            collection.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError(
                "a residue coordinate needs complete residues and full constants"
            )
        if not isinstance(self.coordinate_index, int) or self.coordinate_index < 0:
            raise ValueError("a residue coordinate index must be nonnegative")

        _parent, constants, _variable = _base_polynomial_ring(
            collection.model.extension
        )
        basis = constant_coordinate_basis(constants)
        if basis is None or self.coordinate_index >= len(basis):
            raise ValueError("the displayed constants lack this rational coordinate")

        coordinate_records: list[
            tuple[AlgebraicLocalDifferentialResidue, FieldElement]
        ] = []
        for record in collection.residues:
            residue = displayed_constant_residue(record, constants)
            if residue is None:
                raise ValueError("a residue coordinate contains a nondisplayed value")
            coordinates = rational_coordinates_of_constant(constants, residue)
            if coordinates is None or len(coordinates) != len(basis):
                raise ValueError("a residue has no supported rational coordinates")
            coordinate = coordinates[self.coordinate_index]
            if coordinate != QQ.zero:
                coordinate_records.append((record, coordinate))
        if not coordinate_records:
            raise ValueError("a residue coordinate needs a nonzero local residue")

        scale = primitive_rational_scale(
            tuple(coordinate for _record, coordinate in coordinate_records)
        )
        expected_coefficient = constants.convert(basis[self.coordinate_index] * scale)
        expected_divisor = AlgebraicDivisor(
            collection.model,
            tuple(
                (record.place, integer_multiple(coordinate, scale))
                for record, coordinate in coordinate_records
            ),
        )
        if expected_divisor.degree != 0:
            raise ValueError("a residue coordinate violates the global residue theorem")
        if self.divisor != expected_divisor:
            raise ValueError(
                "the residue divisor does not match its rational coordinate"
            )
        try:
            coefficient = constants.convert(self.coefficient)
        except (CoercionFailed, TypeError, ValueError) as exc:
            raise ValueError(
                "a residue coordinate coefficient must lie in the constants"
            ) from exc
        if coefficient != expected_coefficient:
            raise ValueError("the residue coefficient does not match its coordinate")
        object.__setattr__(self, "coefficient", coefficient)


__all__ = ["AlgebraicResidueCoordinateCertificate"]
