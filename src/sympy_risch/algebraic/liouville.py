r"""Liouville obstructions for pure algebraic differentials.

This module owns the theorem boundary that turns a certified nontorsion
residue-coordinate divisor into a mathematical non-elementarity result.  It
is separate from logarithmic reconstruction so the algorithm does not also
own its proof-object validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._typing import FieldElement
from ..algebraic_geometry.finite_torsion import (
    AlgebraicDivisorTorsionDecision,
    AlgebraicDivisorTorsionStatus,
)
from ..algebraic_geometry.function_field import AlgebraicDivisor
from ..algebraic_geometry.principal import (
    AlgebraicPrincipalityDecision,
    AlgebraicPrincipalityStatus,
)
from ._field import _base_polynomial_ring
from .all_places import (
    AlgebraicAllPlaceResidueResult,
    AlgebraicAllPlaceResidueStatus,
)
from .constant_analysis import AlgebraicConstantFieldStatus


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicLiouvilleObstruction:
    r"""A nontorsion residue divisor contradicting Liouville's theorem.

    In a Liouville representation, expand the finitely many logarithmic
    coefficients in a rational basis. Clearing denominators makes every
    coordinate residue divisor principal, hence the unscaled coordinate class
    must be torsion. A certified nontorsion coordinate therefore proves that
    no elementary antiderivative exists, even after algebraic constant
    extension.
    """

    collection: AlgebraicAllPlaceResidueResult
    coefficient: FieldElement
    residue_divisor: AlgebraicDivisor
    principality: AlgebraicPrincipalityDecision
    torsion: AlgebraicDivisorTorsionDecision
    detail: str

    def __init__(self) -> None:
        raise TypeError("Liouville obstructions must be created by the integrator")

    @classmethod
    def _create(
        cls,
        collection: AlgebraicAllPlaceResidueResult,
        coefficient: FieldElement,
        residue_divisor: AlgebraicDivisor,
        principality: AlgebraicPrincipalityDecision,
        torsion: AlgebraicDivisorTorsionDecision,
    ) -> AlgebraicLiouvilleObstruction:
        result = object.__new__(cls)
        object.__setattr__(result, "collection", collection)
        object.__setattr__(result, "coefficient", coefficient)
        object.__setattr__(result, "residue_divisor", residue_divisor)
        object.__setattr__(result, "principality", principality)
        object.__setattr__(result, "torsion", torsion)
        object.__setattr__(
            result,
            "detail",
            "a rational-coordinate residue divisor is certified nontorsion",
        )
        result._validate()
        return result

    def _validate(self) -> None:
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
        model = self.collection.model
        if self.residue_divisor.model is not model:
            raise ValueError("the residue obstruction belongs to another model")
        if (
            self.principality.model is not model
            or self.principality.divisor != self.residue_divisor
            or self.principality.status is not AlgebraicPrincipalityStatus.NONPRINCIPAL
        ):
            raise ValueError("the residue divisor lacks its nonprincipality proof")
        if (
            self.torsion.model is not model
            or self.torsion.divisor != self.residue_divisor
            or self.torsion.status is not AlgebraicDivisorTorsionStatus.NON_TORSION
            or not self.torsion.complete_negative
        ):
            raise ValueError("the residue divisor lacks its nontorsion proof")
        _parent, constants, _variable = _base_polynomial_ring(model.extension)
        coefficient = constants.convert(self.coefficient)
        if coefficient == constants.zero:
            raise ValueError("a residue obstruction needs a nonzero coefficient")
        object.__setattr__(self, "coefficient", coefficient)
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a Liouville obstruction needs a detail")


__all__ = ["AlgebraicLiouvilleObstruction"]
