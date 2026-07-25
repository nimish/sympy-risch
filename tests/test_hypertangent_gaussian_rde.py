"""Scalar rational RDEs over real tangent towers via Cayley normal form."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_risch_differential_equation,
)
from sympy_risch.exact import (
    TranscendentalExactDerivativeCertificate,
    solve_exact_derivative,
)
from sympy_risch.rde import HypertangentGaussianRischDifferentialEquationResult

x, t, u = symbols("x t u")


def hypertangent_tower() -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def nested_hypertangent_tower() -> DifferentialTower:
    first = hypertangent_tower()
    return first.extend(
        MonomialExtension(
            Poly((t + 1)*(1 + u**2), u, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_gaussian_scalar_rde_solves_a_general_terminal_rational_case() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((t**2 + x)/(t + 1))
    coefficient = field.from_sympy(1/(t + 2))
    right_hand_side = derivation(solution) + coefficient*solution

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == solution
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianRischDifferentialEquationResult,
    )
    assert result.certificate.normal_form.verifies_derivation()
    assert result.certificate.target_result.reconstructs


def test_gaussian_scalar_rde_recurses_through_two_tangent_levels() -> None:
    tower = nested_hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((u**2 + t)/(u + 1))
    coefficient = field.from_sympy(1/(u + 2))
    right_hand_side = derivation(solution) + coefficient*solution

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == solution
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianRischDifferentialEquationResult,
    )
    assert result.certificate.normal_form.verifies_derivation()
    assert result.certificate.target_result.reconstructs


def test_exact_derivative_uses_direct_reduction_for_tangent_over_exponential() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    exponential = base.extend(
        MonomialExtension(
            Poly(x*t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = exponential.extend(
        MonomialExtension(
            Poly((t + 1)*(1 + u**2), u, domain=exponential.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((u**2 + t)/(u + 1))

    result = solve_exact_derivative(derivation(planted), derivation)

    assert result.has_solution
    assert derivation(result.exact_part) == derivation(planted)
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert result.certificate.reduction.special_reduction is not None
