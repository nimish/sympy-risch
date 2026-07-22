"""Risch integration algorithms built directly on SymPy."""

from .algebraic import (
    AlgebraicFraction,
    IntegralBasis,
    hermite_reduce_algebraic,
)
from .lrt import lazard_rioboo_trager
from .polynomial import integrate_polynomial
from .rational import (
    hermite_reduce_rational,
    integrate_rational,
    integrate_rational_real,
    rothstein_trager_residue_polynomial,
)
from .tower import (
    AlgebraicExtension,
    Derivation,
    DifferentialExtension,
    DifferentialTower,
    MonomialExtension,
)

__version__ = "0.1.0"

__all__ = [
    "AlgebraicExtension",
    "AlgebraicFraction",
    "Derivation",
    "DifferentialExtension",
    "DifferentialTower",
    "hermite_reduce_rational",
    "hermite_reduce_algebraic",
    "integrate_polynomial",
    "integrate_rational",
    "integrate_rational_real",
    "IntegralBasis",
    "lazard_rioboo_trager",
    "MonomialExtension",
    "rothstein_trager_residue_polynomial",
]
