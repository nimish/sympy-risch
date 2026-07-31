"""Adapters from Risch proofs to reusable function-field evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .algebraic.constant_analysis import AlgebraicConstantFieldAnalysis
from .algebraic_geometry.constant_fields import (
    FunctionFieldConstantCertificate,
    FunctionFieldConstantProof,
)
from .algebraic_geometry.function_field import AlgebraicFunctionField


@dataclass(frozen=True, slots=True)
class RischConstantFieldCertificate(FunctionFieldConstantCertificate):
    """Geometric constant evidence adapted from a preserved Risch analysis."""

    source_analysis: AlgebraicConstantFieldAnalysis

    def __post_init__(self) -> None:
        FunctionFieldConstantCertificate.__post_init__(self)
        if not isinstance(self.source_analysis, AlgebraicConstantFieldAnalysis):
            raise TypeError("constant-field adaptation needs a Risch analysis")
        if self.source_analysis.extension is not self.model.extension:
            raise ValueError("the constant-field analysis belongs to another field")
        if not self.source_analysis.preserved:
            raise ValueError("the Risch analysis does not prove preserved constants")

    @property
    def proof(self) -> FunctionFieldConstantProof:
        """The preserved Risch-analysis theorem."""
        return FunctionFieldConstantProof.RISCH_ANALYSIS


def adapt_risch_constant_field_analysis(
    model: AlgebraicFunctionField,
    analysis: AlgebraicConstantFieldAnalysis,
) -> RischConstantFieldCertificate:
    """Adapt a preserved derivation-specific analysis to geometric evidence."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("constant-field adaptation needs a global model")
    return RischConstantFieldCertificate(
        model=model,
        detail="the preserved Risch analysis proves the displayed constants are full",
        source_analysis=analysis,
    )


__all__ = [
    "RischConstantFieldCertificate",
    "adapt_risch_constant_field_analysis",
]
