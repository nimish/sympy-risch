r"""Risch differential equations over exact polynomial domains.

This package keeps the public API of the former :mod:`sympy_risch.rde` module
while organizing the implementation by base, primitive, hyperexponential,
and coupled theorem families.
"""

from .coupled import (
    solve_coupled_rational_rde,
    solve_coupled_risch_differential_equation,
    solve_hyperexponential_coupled_laurent_rde,
    solve_hypertangent_coupled_cancellation_rde,
    solve_hypertangent_coupled_polynomial_rde,
)
from .hyperexponential import (
    solve_hyperexponential_laurent_risch_differential_equation,
    solve_hyperexponential_risch_differential_equation,
)
from .models import (
    CoupledHyperexponentialLaurentRDEResult,
    CoupledHyperexponentialLaurentRDEStep,
    CoupledHypertangentPolynomialRDECorrection,
    CoupledHypertangentPolynomialRDEMode,
    CoupledHypertangentPolynomialRDEResult,
    CoupledRationalRDEResult,
    CoupledRDEObstruction,
    CoupledRischDifferentialEquationResult,
    HyperexponentialLaurentRDECorrection,
    HyperexponentialLaurentRDEDirection,
    HyperexponentialLaurentRDEObstruction,
    HyperexponentialLaurentRDEStep,
    HyperexponentialLaurentRischDifferentialEquationResult,
    HyperexponentialRDEObstruction,
    HyperexponentialRischDifferentialEquationResult,
    HypertangentGaussianRischDifferentialEquationResult,
    PolynomialRDEObstruction,
    PolynomialRischDifferentialEquationResult,
    PrimitivePolynomialRDEObstruction,
    PrimitivePolynomialRischDifferentialEquationResult,
    PrimitiveRDEObstruction,
    PrimitiveRischDifferentialEquationResult,
    RationalRDEObstruction,
    RationalRischDifferentialEquationResult,
    RischDifferentialEquationResult,
)
from .parametric import (
    RationalParametricRDESolution,
    RationalParametricRischDifferentialEquationResult,
    solve_rational_parametric_risch_differential_equation,
)
from .parametric_dispatch import (
    ParametricRDESolution,
    ParametricRischDifferentialEquationResult,
    solve_parametric_risch_differential_equation,
)
from .primitive import (
    solve_primitive_polynomial_risch_differential_equation,
    solve_primitive_risch_differential_equation,
)
from .rational import (
    solve_polynomial_risch_differential_equation,
    solve_rational_risch_differential_equation,
)
from .scalar import solve_risch_differential_equation

__all__ = [
    "CoupledHyperexponentialLaurentRDEResult",
    "CoupledHyperexponentialLaurentRDEStep",
    "CoupledHypertangentPolynomialRDECorrection",
    "CoupledHypertangentPolynomialRDEMode",
    "CoupledHypertangentPolynomialRDEResult",
    "CoupledRDEObstruction",
    "CoupledRationalRDEResult",
    "CoupledRischDifferentialEquationResult",
    "HyperexponentialLaurentRDECorrection",
    "HyperexponentialLaurentRDEDirection",
    "HyperexponentialLaurentRDEObstruction",
    "HyperexponentialLaurentRDEStep",
    "HyperexponentialLaurentRischDifferentialEquationResult",
    "HyperexponentialRDEObstruction",
    "HyperexponentialRischDifferentialEquationResult",
    "HypertangentGaussianRischDifferentialEquationResult",
    "ParametricRDESolution",
    "ParametricRischDifferentialEquationResult",
    "PolynomialRDEObstruction",
    "PolynomialRischDifferentialEquationResult",
    "PrimitivePolynomialRDEObstruction",
    "PrimitivePolynomialRischDifferentialEquationResult",
    "PrimitiveRDEObstruction",
    "PrimitiveRischDifferentialEquationResult",
    "RationalParametricRDESolution",
    "RationalParametricRischDifferentialEquationResult",
    "RationalRDEObstruction",
    "RationalRischDifferentialEquationResult",
    "RischDifferentialEquationResult",
    "solve_coupled_rational_rde",
    "solve_coupled_risch_differential_equation",
    "solve_hyperexponential_coupled_laurent_rde",
    "solve_hyperexponential_laurent_risch_differential_equation",
    "solve_hyperexponential_risch_differential_equation",
    "solve_hypertangent_coupled_cancellation_rde",
    "solve_hypertangent_coupled_polynomial_rde",
    "solve_parametric_risch_differential_equation",
    "solve_polynomial_risch_differential_equation",
    "solve_primitive_polynomial_risch_differential_equation",
    "solve_primitive_risch_differential_equation",
    "solve_rational_parametric_risch_differential_equation",
    "solve_rational_risch_differential_equation",
    "solve_risch_differential_equation",
]
