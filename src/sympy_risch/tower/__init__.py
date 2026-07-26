r"""Public tower model and native-representation analysis.

The field and derivation kernel lives in :mod:`sympy_risch.tower.core`;
backend representation checks live in :mod:`sympy_risch.tower.representation`.
This facade is the supported advanced tower API.
"""

from . import core as _core
from .core import (
    AlgebraicExtension,
    CoefficientDerivation,
    ConstantFieldStatus,
    Derivation,
    DifferentialExtension,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    rebase_derivation_constants,
)
from .representation import (
    TowerRepresentationAnalysis,
    TowerRepresentationOutcome,
    TowerRepresentationProposal,
    TowerRepresentationReason,
    analyze_tower_representation,
)

# Private aliases remain temporarily available to the two legacy modules that
# cannot be rewired in the same atomic package split.
_coerce = _core._coerce
_rebase_monomial_derivation = _core._rebase_monomial_derivation

__all__ = [
    "AlgebraicExtension",
    "CoefficientDerivation",
    "ConstantFieldStatus",
    "Derivation",
    "DifferentialExtension",
    "DifferentialTower",
    "MonomialExtension",
    "MonomialFamily",
    "TowerRepresentationAnalysis",
    "TowerRepresentationOutcome",
    "TowerRepresentationProposal",
    "TowerRepresentationReason",
    "analyze_tower_representation",
    "rebase_derivation_constants",
]
