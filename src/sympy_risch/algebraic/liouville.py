r"""Liouville obstructions for pure algebraic differentials.

This module owns the theorem boundary that turns a certified nontorsion
residue-coordinate divisor into a mathematical non-elementarity result.  It
is separate from logarithmic reconstruction so the algorithm does not also
own its proof-object validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .._typing import FieldElement
from ..algebraic_geometry.finite_torsion import (
    AlgebraicDivisorNontorsion,
    AlgebraicDivisorTorsionDecision,
)
from ..algebraic_geometry.function_field import AlgebraicDivisor
from ..algebraic_geometry.principal import (
    AlgebraicPrincipalityDecision,
    AlgebraicPrincipalityStatus,
)
from .all_places import (
    AlgebraicAllPlaceResidueResult,
    AlgebraicAllPlaceResidueStatus,
)
from .constant_analysis import AlgebraicConstantFieldStatus
from .exactness import (
    AlgebraicExactnessCertificate,
    AlgebraicExactnessResult,
    AlgebraicExactnessStatus,
)
from .residue_coordinates import AlgebraicResidueCoordinateCertificate


@dataclass(frozen=True, slots=True)
class AlgebraicLiouvilleObstruction:
    r"""A nontorsion residue divisor contradicting Liouville's theorem.

    In a Liouville representation, expand the finitely many logarithmic
    coefficients in a rational basis. Clearing denominators makes every
    coordinate residue divisor principal, hence the unscaled coordinate class
    must be torsion. A certified nontorsion coordinate therefore proves that
    no elementary antiderivative exists, even after algebraic constant
    extension.
    """

    coordinate: AlgebraicResidueCoordinateCertificate
    principality: AlgebraicPrincipalityDecision
    torsion: AlgebraicDivisorTorsionDecision
    detail: ClassVar[str] = (
        "a rational-coordinate residue divisor is certified nontorsion"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate, AlgebraicResidueCoordinateCertificate):
            raise TypeError("a Liouville obstruction needs a residue coordinate")
        model = self.coordinate.collection.model
        if (
            self.principality.model is not model
            or self.principality.divisor != self.residue_divisor
            or self.principality.status is not AlgebraicPrincipalityStatus.NONPRINCIPAL
        ):
            raise ValueError("the residue divisor lacks its nonprincipality proof")
        if (
            not isinstance(self.torsion, AlgebraicDivisorNontorsion)
            or self.torsion.model is not model
            or self.torsion.divisor != self.residue_divisor
        ):
            raise ValueError("the residue divisor lacks its nontorsion proof")

    @property
    def collection(self) -> AlgebraicAllPlaceResidueResult:
        """Complete all-place residues from which the coordinate was replayed."""
        return self.coordinate.collection

    @property
    def coefficient(self) -> FieldElement:
        """Displayed-constant coefficient of the rational coordinate."""
        return self.coordinate.coefficient

    @property
    def residue_divisor(self) -> AlgebraicDivisor:
        """Integral divisor reconstructed from the rational coordinate."""
        return self.coordinate.divisor


@dataclass(frozen=True, slots=True)
class AlgebraicExactnessLiouvilleObstruction:
    r"""A residue-free nonexact differential contradicting Liouville's theorem.

    Once every logarithmic residue has been removed, a logarithmic derivative
    with zero residues is zero: expand its constant coefficients in a rational
    basis, clear denominators, and each resulting product has zero principal
    divisor, hence is constant.  Elementary integrability therefore reduces to
    exactness.  A complete primitive-space decision proving nonexactness is a
    Liouville obstruction, including after finite algebraic extension by trace
    descent.
    """

    collection: AlgebraicAllPlaceResidueResult
    exactness: AlgebraicExactnessResult
    detail: ClassVar[str] = (
        "the complete residue-free primitive space contains no solution"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.collection, AlgebraicAllPlaceResidueResult):
            raise TypeError("a Liouville obstruction needs all-place residue data")
        if (
            self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
            or self.collection.model is None
            or self.collection.constant_field_analysis is None
            or self.collection.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError(
                "a Liouville obstruction needs complete residues and full constants"
            )
        if not isinstance(self.exactness, AlgebraicExactnessResult):
            raise TypeError("an exactness obstruction needs an exactness decision")
        if (
            self.exactness.model is not self.collection.model
            or not self.exactness.derivation.has_same_position(
                self.collection.derivation
            )
        ):
            raise ValueError("the exactness obstruction belongs to another problem")
        if (
            self.exactness.status is not AlgebraicExactnessStatus.NO_PRIMITIVE
            or self.exactness.certificate
            is not AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE
            or not self.exactness.complete_negative
        ):
            raise ValueError(
                "an exactness obstruction needs a complete residue-free decision"
            )


__all__ = [
    "AlgebraicExactnessLiouvilleObstruction",
    "AlgebraicLiouvilleObstruction",
]
