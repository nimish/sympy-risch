r"""Reduction helpers for transcendental monomial extensions.

The facade keeps the established :mod:`sympy_risch.monomial` surface while
its implementation follows the primitive, hyperexponential, hypertangent,
normal-residue, and special-pole theorem families.
"""

# The radical dispatcher initializes first and completes its concrete result
# union through deliberate late imports; keep this order rather than sorting
# the facade imports alphabetically.
# ruff: noqa: I001

from ..reduction.fractions import split_normal_special
from .logarithmic import (
    RadicalLogarithmicDerivativeResult,
    solve_radical_logarithmic_derivative,
)
from .hermite import hermite_reduce_monomial
from .hyperexponential import (
    HyperexponentialLaurentReduction,
    HyperexponentialLaurentStep,
    HyperexponentialLogarithmicDerivativeFactor,
    HyperexponentialLogarithmicDerivativeSignature,
    HyperexponentialMonomialReduction,
    hyperexponential_logarithmic_derivative_signature,
    reduce_hyperexponential_laurent,
    reduce_hyperexponential_monomial,
)
from .hypertangent import (
    HypertangentCayleyStep,
    HypertangentGaussianLogarithmicDerivativeOracle,
    HypertangentLogarithmicDerivativeSignature,
    HypertangentMonomialReduction,
    HypertangentSpecialReduction,
    hypertangent_logarithmic_derivative_signature,
    hypertangent_logarithmic_derivative_signature_gaussian,
    reduce_hypertangent_monomial,
    reduce_hypertangent_special,
)
from .nonlinear import (
    NonlinearMonomialReduction,
    ReducedFunctionReduction,
    polynomial_reduce_monomial,
    reduce_monomial_reduced_function,
    reduce_nonlinear_monomial,
)
from .primitive import (
    PrimitiveExactDerivativeResult,
    PrimitiveLogarithmicDerivativeFactor,
    PrimitiveLogarithmicDerivativeSignature,
    PrimitiveMonomialReduction,
    primitive_logarithmic_derivative_signature,
    reduce_primitive_monomial,
    solve_primitive_exact_derivative,
)
from .primitive_polynomial import (
    PrimitivePolynomialReduction,
    PrimitivePolynomialStep,
    polynomial_reduce_primitive,
)
from .residues import (
    BronsteinResidueCertificate,
    BronsteinResidueFactor,
    BronsteinResidueReduction,
    bronstein_residue_criterion,
    residue_reduce_monomial,
)
from .special_poles import (
    LinearSpecialPoleReduction,
    LinearSpecialPoleReductionStatus,
    LinearSpecialPoleStep,
    reduce_certified_linear_special_poles,
)

__all__ = [
    "BronsteinResidueCertificate",
    "BronsteinResidueFactor",
    "BronsteinResidueReduction",
    "HyperexponentialLaurentReduction",
    "HyperexponentialLaurentStep",
    "HyperexponentialLogarithmicDerivativeFactor",
    "HyperexponentialLogarithmicDerivativeSignature",
    "HyperexponentialMonomialReduction",
    "HypertangentCayleyStep",
    "HypertangentGaussianLogarithmicDerivativeOracle",
    "HypertangentLogarithmicDerivativeSignature",
    "HypertangentMonomialReduction",
    "HypertangentSpecialReduction",
    "LinearSpecialPoleReduction",
    "LinearSpecialPoleReductionStatus",
    "LinearSpecialPoleStep",
    "NonlinearMonomialReduction",
    "PrimitiveExactDerivativeResult",
    "PrimitiveLogarithmicDerivativeFactor",
    "PrimitiveLogarithmicDerivativeSignature",
    "PrimitiveMonomialReduction",
    "PrimitivePolynomialReduction",
    "PrimitivePolynomialStep",
    "RadicalLogarithmicDerivativeResult",
    "ReducedFunctionReduction",
    "bronstein_residue_criterion",
    "hermite_reduce_monomial",
    "hyperexponential_logarithmic_derivative_signature",
    "hypertangent_logarithmic_derivative_signature",
    "hypertangent_logarithmic_derivative_signature_gaussian",
    "polynomial_reduce_monomial",
    "polynomial_reduce_primitive",
    "primitive_logarithmic_derivative_signature",
    "reduce_certified_linear_special_poles",
    "reduce_hyperexponential_laurent",
    "reduce_hyperexponential_monomial",
    "reduce_hypertangent_monomial",
    "reduce_hypertangent_special",
    "reduce_monomial_reduced_function",
    "reduce_nonlinear_monomial",
    "reduce_primitive_monomial",
    "residue_reduce_monomial",
    "solve_primitive_exact_derivative",
    "solve_radical_logarithmic_derivative",
    "split_normal_special",
]
