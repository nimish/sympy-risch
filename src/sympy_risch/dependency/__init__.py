r"""Dependency analysis for proposed differential-field extensions.

The public facade retains the original dependency-analysis API while its
implementation is organized by rational, primitive, mixed-tower, and
extension-certification theorems.
"""

from ._common import verify_parametric_logarithmic_derivative_witness
from .extensions import (
    analyze_hyperexponential_extension,
    analyze_hypertangent_extension,
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
    certify_primitive_extension,
)
from .models import (
    HyperexponentialDependency,
    HyperexponentialDependencyKind,
    HypertangentDependency,
    HypertangentGaussianParametricLogarithmicDerivativeCertificate,
    MonomialRelationSystemParametricCertificate,
    ParametricLogarithmicDerivativeHeuristicBranch,
    ParametricLogarithmicDerivativeHeuristicCertificate,
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
    PrimitiveExtensionAnalysis,
    PrimitiveParametricLogarithmicDerivativeCertificate,
    PrimitiveParametricLogarithmicDerivativeFactor,
    RationalParametricLogarithmicDerivativeCertificate,
    RationalParametricLogarithmicDerivativeResidue,
    TranscendentalExtensionAnalysis,
)
from .rational import solve_parametric_logarithmic_derivative_heuristic
from .transcendental import (
    solve_hypertangent_parametric_logarithmic_derivative_gaussian,
    solve_parametric_logarithmic_derivative,
)

__all__ = [
    "HyperexponentialDependency",
    "HyperexponentialDependencyKind",
    "HypertangentDependency",
    "HypertangentGaussianParametricLogarithmicDerivativeCertificate",
    "MonomialRelationSystemParametricCertificate",
    "ParametricLogarithmicDerivativeHeuristicBranch",
    "ParametricLogarithmicDerivativeHeuristicCertificate",
    "ParametricLogarithmicDerivativeOutcome",
    "ParametricLogarithmicDerivativeResult",
    "PrimitiveExtensionAnalysis",
    "PrimitiveParametricLogarithmicDerivativeCertificate",
    "PrimitiveParametricLogarithmicDerivativeFactor",
    "RationalParametricLogarithmicDerivativeCertificate",
    "RationalParametricLogarithmicDerivativeResidue",
    "TranscendentalExtensionAnalysis",
    "analyze_hyperexponential_extension",
    "analyze_hypertangent_extension",
    "certify_hyperexponential_extension",
    "certify_hypertangent_extension",
    "certify_primitive_extension",
    "solve_hypertangent_parametric_logarithmic_derivative_gaussian",
    "solve_parametric_logarithmic_derivative",
    "solve_parametric_logarithmic_derivative_heuristic",
    "verify_parametric_logarithmic_derivative_witness",
]
