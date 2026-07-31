"""Adapters from Risch proofs to reusable function-field evidence."""

from __future__ import annotations

from .algebraic.constant_analysis import AlgebraicConstantFieldAnalysis
from .algebraic_geometry.constant_fields import (
    FunctionFieldConstantCertificate,
    FunctionFieldConstantProof,
)
from .algebraic_geometry.function_field import AlgebraicFunctionField


def adapt_risch_constant_field_analysis(
    model: AlgebraicFunctionField,
    analysis: AlgebraicConstantFieldAnalysis,
) -> FunctionFieldConstantCertificate:
    """Adapt a preserved derivation-specific analysis to geometric evidence."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("constant-field adaptation needs a global model")
    if not isinstance(analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("constant-field adaptation needs a Risch analysis")
    if analysis.extension is not model.extension:
        raise ValueError("the constant-field analysis belongs to another field")
    if not analysis.preserved:
        raise ValueError("the Risch analysis does not prove preserved constants")
    return FunctionFieldConstantCertificate._create(
        model,
        FunctionFieldConstantProof.RISCH_ANALYSIS,
        None,
        analysis,
        "the preserved Risch analysis proves the displayed constants are full",
    )


__all__ = ["adapt_risch_constant_field_analysis"]
