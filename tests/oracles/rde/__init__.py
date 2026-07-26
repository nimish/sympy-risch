"""Independent exact-linear-algebra oracles for Risch differential equations."""

from .hyperexponential import (
    solve_bounded_hyperexponential_laurent_rde_matrix,
)
from .models import (
    BoundedCoupledRDEMatrixResult,
    BoundedHyperexponentialLaurentRDEMatrixResult,
    BoundedPrimitiveCoupledRDEMatrixResult,
    BoundedPrimitiveRDEMatrixResult,
    BoundedRationalRDEMatrixResult,
)
from .polynomial import solve_polynomial_risch_differential_equation_matrix
from .primitive import (
    solve_bounded_primitive_coupled_rde_matrix,
    solve_bounded_primitive_rde_matrix,
)
from .rational import (
    solve_bounded_coupled_rde_matrix,
    solve_bounded_rational_rde_matrix,
)

__all__ = [
    "BoundedCoupledRDEMatrixResult",
    "BoundedHyperexponentialLaurentRDEMatrixResult",
    "BoundedPrimitiveCoupledRDEMatrixResult",
    "BoundedPrimitiveRDEMatrixResult",
    "BoundedRationalRDEMatrixResult",
    "solve_bounded_coupled_rde_matrix",
    "solve_bounded_hyperexponential_laurent_rde_matrix",
    "solve_bounded_primitive_coupled_rde_matrix",
    "solve_bounded_primitive_rde_matrix",
    "solve_bounded_rational_rde_matrix",
    "solve_polynomial_risch_differential_equation_matrix",
]
