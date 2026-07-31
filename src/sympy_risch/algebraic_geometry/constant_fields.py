r"""Geometric evidence for the full constant field of a function field.

If the full constant field of ``F/C`` were a proper finite extension ``C'``,
then ``C'`` would embed into every closed-point residue field. Consequently,
the degree of every place over ``C`` would be divisible by ``[C':C]``. A
degree-one place therefore proves ``C' = C``.

This certificate belongs to the reusable function-field layer. Risch's
derivation-specific constant analysis can be adapted explicitly, but the
geometry algorithms do not otherwise depend on that analyzer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)


class FunctionFieldConstantProof(Enum):
    """The theorem proving that the displayed constants are full."""

    DEGREE_ONE_PLACE = "degree_one_place"
    RISCH_ANALYSIS = "risch_analysis"


@dataclass(frozen=True, slots=True)
class FunctionFieldConstantCertificate(ABC):
    """Base class for proofs that a model's displayed constants are full."""

    model: AlgebraicFunctionField
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a constant-field certificate needs a global model")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a constant-field certificate needs a detail")

    @property
    @abstractmethod
    def proof(self) -> FunctionFieldConstantProof:
        """The theorem carried by this concrete certificate."""


@dataclass(frozen=True, slots=True)
class DegreeOnePlaceConstantCertificate(FunctionFieldConstantCertificate):
    """Full-constant proof supplied by one degree-one closed place."""

    degree_one_place: AlgebraicPlace

    def __post_init__(self) -> None:
        FunctionFieldConstantCertificate.__post_init__(self)
        # Divisor construction performs the complete model-ownership check.
        AlgebraicDivisor(self.model, ((self.degree_one_place, 1),))
        if self.degree_one_place.degree != 1:
            raise ValueError("the constant-field witness must have degree one")

    @property
    def proof(self) -> FunctionFieldConstantProof:
        """The degree-one-place theorem."""
        return FunctionFieldConstantProof.DEGREE_ONE_PLACE


def certify_function_field_constants(
    model: AlgebraicFunctionField,
    degree_one_place: AlgebraicPlace | None = None,
) -> FunctionFieldConstantCertificate:
    r"""Certify full constants from a degree-one place.

    If no place is supplied, the finite set of places above infinity is
    searched. Callers may supply any already-constructed finite or infinite
    degree-one place when infinity has no rational branch.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("constant-field certification needs a global model")

    witness = degree_one_place
    if witness is None:
        witness = next(
            (place for place in model.infinite_places() if place.degree == 1),
            None,
        )
    if witness is None:
        raise NotImplementedError(
            "no degree-one place was found above infinity; supply another "
            "degree-one place or adapt a preserved Risch analysis"
        )
    return DegreeOnePlaceConstantCertificate(
        model=model,
        detail="a degree-one closed place proves that the displayed constants are full",
        degree_one_place=witness,
    )


__all__ = [
    "FunctionFieldConstantCertificate",
    "FunctionFieldConstantProof",
    "certify_function_field_constants",
]
