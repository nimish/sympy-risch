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


@dataclass(frozen=True, slots=True, init=False)
class FunctionFieldConstantCertificate:
    """Proof that a global model's displayed constant field is exact."""

    model: AlgebraicFunctionField
    proof: FunctionFieldConstantProof
    degree_one_place: AlgebraicPlace | None
    source_analysis: object | None
    detail: str

    def __init__(self) -> None:
        raise TypeError(
            "constant-field certificates must be created by a certificate factory"
        )

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        proof: FunctionFieldConstantProof,
        degree_one_place: AlgebraicPlace | None,
        source_analysis: object | None,
        detail: str,
    ) -> FunctionFieldConstantCertificate:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "proof", proof)
        object.__setattr__(result, "degree_one_place", degree_one_place)
        object.__setattr__(result, "source_analysis", source_analysis)
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a constant-field certificate needs a global model")
        if not isinstance(self.proof, FunctionFieldConstantProof):
            raise TypeError("a constant-field certificate has an unknown proof")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a constant-field certificate needs a detail")

        if self.proof is FunctionFieldConstantProof.DEGREE_ONE_PLACE:
            if self.degree_one_place is None or self.source_analysis is not None:
                raise ValueError("a degree-one proof needs exactly one place")
            # Divisor construction performs the complete model-ownership check.
            AlgebraicDivisor(self.model, ((self.degree_one_place, 1),))
            if self.degree_one_place.degree != 1:
                raise ValueError("the constant-field witness must have degree one")
            return

        if self.proof is FunctionFieldConstantProof.RISCH_ANALYSIS:
            if self.degree_one_place is not None or self.source_analysis is None:
                raise ValueError("a Risch proof needs exactly one analysis")
            from ..algebraic.constant_analysis import AlgebraicConstantFieldAnalysis

            if not isinstance(
                self.source_analysis,
                AlgebraicConstantFieldAnalysis,
            ):
                raise TypeError("the Risch constant proof has the wrong type")
            if (
                self.source_analysis.extension is not self.model.extension
                or not self.source_analysis.preserved
            ):
                raise ValueError(
                    "the Risch analysis does not prove this model's constants"
                )
            return

        raise ValueError("a constant-field certificate has an unknown proof")


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
    return FunctionFieldConstantCertificate._create(
        model,
        FunctionFieldConstantProof.DEGREE_ONE_PLACE,
        witness,
        None,
        "a degree-one closed place proves that the displayed constants are full",
    )


__all__ = [
    "FunctionFieldConstantCertificate",
    "FunctionFieldConstantProof",
    "certify_function_field_constants",
]
