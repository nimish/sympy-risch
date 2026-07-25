"""Scalar rational RDEs over real tangent towers via Cayley normal form."""

from sympy import I, Poly, cancel, symbols
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
from sympy_risch.gaussian import build_hypertangent_gaussian_normal_form
from sympy_risch.rde import (
    HypertangentGaussianRischDifferentialEquationResult,
    _as_primitive_fraction,
)

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
            Poly((t + 1) * (1 + u**2), u, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_gaussian_scalar_rde_solves_a_general_terminal_rational_case() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((t**2 + x) / (t + 1))
    coefficient = field.from_sympy(1 / (t + 2))
    right_hand_side = derivation(solution) + coefficient * solution

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


def test_gaussian_scalar_rde_solves_a_rational_exact_derivative() -> None:
    """A noncanonical Gaussian target fraction must not abort normalization."""
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((t * x + 1) / (t + 2))

    result = solve_risch_differential_equation(
        field.zero,
        derivation(solution),
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianRischDifferentialEquationResult,
    )
    assert result.certificate.target_result.reconstructs


def test_gaussian_scalar_rde_solves_a_diagonal_constant_case() -> None:
    """The diagonal Laurent certificate uses symbolic Gaussian equality."""
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy(1 / (t + 1))
    coefficient = field.one

    result = solve_risch_differential_equation(
        coefficient,
        derivation(solution) + coefficient * solution,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianRischDifferentialEquationResult,
    )
    assert result.certificate.target_result.reconstructs


def test_gaussian_target_fraction_normalization_accepts_equal_units() -> None:
    """A FLINT Gaussian fraction may be symbolically zero but not ``== 0``."""
    tower = hypertangent_tower()
    normal_form = build_hypertangent_gaussian_normal_form(
        tower.derivation(),
    )
    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    target_field = normal_form.target_field
    z = target_extension.generator
    value = target_field.from_sympy(
        (1 - 2 * I) * (z * (1 + 2 * I) - 1 + 2 * I) / 5,
    )

    _, numerator, denominator = _as_primitive_fraction(
        value,
        target_extension,
    )
    normalized = target_field.from_sympy(
        numerator.as_expr() / denominator.as_expr(),
    )

    assert cancel(target_field.to_sympy(normalized - value)) == 0


def test_gaussian_scalar_rde_recurses_through_two_tangent_levels() -> None:
    tower = nested_hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((u**2 + t) / (u + 1))
    coefficient = field.from_sympy(1 / (u + 2))
    right_hand_side = derivation(solution) + coefficient * solution

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
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = exponential.extend(
        MonomialExtension(
            Poly((t + 1) * (1 + u**2), u, domain=exponential.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((u**2 + t) / (u + 1))

    result = solve_exact_derivative(derivation(planted), derivation)

    assert result.has_solution
    assert derivation(result.exact_part) == derivation(planted)
    assert isinstance(
        result.certificate,
        TranscendentalExactDerivativeCertificate,
    )
    assert result.certificate.reduction.special_reduction is not None
