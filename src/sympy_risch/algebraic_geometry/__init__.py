r"""Exact algebraic-curve infrastructure over a rational function field."""

from .closure import IntegralClosureResult, integral_closure, polynomial_order
from .constant_extensions import (
    AlgebraicConstantFieldExtension,
    extend_algebraic_constants,
)
from .constant_fields import (
    FunctionFieldConstantCertificate,
    FunctionFieldConstantProof,
    certify_function_field_constants,
)
from .differential_constraints import (
    HessSecondKindDifferentialSpace,
    hess_second_kind_differential_space,
)
from .differential_spaces import (
    HessDifferentialSpace,
    hess_differential_space,
    holomorphic_differential_space,
)
from .divisor_classes import (
    AlgebraicTorsionSearch,
    AlgebraicTorsionStatus,
    ReducedAlgebraicDivisorClass,
    reduce_algebraic_divisor_class,
    search_algebraic_divisor_torsion,
)
from .divisor_reduction import (
    HessDivisorReduction,
    HessPowerProduct,
    HessReductionStrategy,
    reduce_hess_divisor,
)
from .finite_places import FiniteAlgebraicPlace, LocalLaurentExpansion
from .finite_reduction import (
    AlgebraicFunctionFieldReduction,
    BadAlgebraicReduction,
    reduce_algebraic_function_field,
)
from .finite_torsion import (
    AlgebraicDivisorTorsionDecision,
    AlgebraicDivisorTorsionStatus,
    FiniteDivisorClassOrder,
    decide_algebraic_divisor_torsion,
    finite_divisor_class_order,
)
from .function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from .hess import (
    HessDivisorIdeals,
    HessParametricBasis,
    hess_divisor_ideals,
    hess_parametric_basis,
)
from .infinity import InfiniteAlgebraicPlace
from .large_riemann_roch import (
    HessLargeRiemannRochSpace,
    HessPowerProductElement,
    hess_large_riemann_roch_space,
)
from .orders import AlgebraicFractionalIdeal, AlgebraicOrder
from .prescribed_residues import (
    HessPrescribedResidueSpace,
    HessPrescribedResidueStatus,
    hess_prescribed_residue_space,
)
from .principal import (
    AlgebraicPrincipalityDecision,
    AlgebraicPrincipalityStatus,
    decide_algebraic_principality,
)

__all__ = [
    "AlgebraicConstantFieldExtension",
    "AlgebraicDifferential",
    "AlgebraicDivisor",
    "AlgebraicDivisorTorsionDecision",
    "AlgebraicDivisorTorsionStatus",
    "AlgebraicFractionalIdeal",
    "AlgebraicFunctionField",
    "AlgebraicFunctionFieldReduction",
    "AlgebraicOrder",
    "AlgebraicPrincipalityDecision",
    "AlgebraicPrincipalityStatus",
    "AlgebraicTorsionSearch",
    "AlgebraicTorsionStatus",
    "BadAlgebraicReduction",
    "FiniteAlgebraicPlace",
    "FiniteDivisorClassOrder",
    "FunctionFieldConstantCertificate",
    "FunctionFieldConstantProof",
    "HessDifferentialSpace",
    "HessDivisorIdeals",
    "HessDivisorReduction",
    "HessLargeRiemannRochSpace",
    "HessParametricBasis",
    "HessPowerProduct",
    "HessPowerProductElement",
    "HessPrescribedResidueSpace",
    "HessPrescribedResidueStatus",
    "HessReductionStrategy",
    "HessSecondKindDifferentialSpace",
    "InfiniteAlgebraicPlace",
    "IntegralClosureResult",
    "LocalLaurentExpansion",
    "ReducedAlgebraicDivisorClass",
    "certify_function_field_constants",
    "decide_algebraic_divisor_torsion",
    "decide_algebraic_principality",
    "extend_algebraic_constants",
    "finite_divisor_class_order",
    "hess_differential_space",
    "hess_divisor_ideals",
    "hess_large_riemann_roch_space",
    "hess_parametric_basis",
    "hess_prescribed_residue_space",
    "hess_second_kind_differential_space",
    "holomorphic_differential_space",
    "integral_closure",
    "polynomial_order",
    "reduce_algebraic_divisor_class",
    "reduce_algebraic_function_field",
    "reduce_hess_divisor",
    "search_algebraic_divisor_torsion",
]
