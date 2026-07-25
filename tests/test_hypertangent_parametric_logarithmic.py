"""Real parametric logarithmic derivatives through Cayley normal form."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
    analyze_hypertangent_extension,
    solve_parametric_logarithmic_derivative,
    solve_radical_logarithmic_derivative,
)
from sympy_risch.dependency import (
    HypertangentGaussianParametricLogarithmicDerivativeCertificate,
)
from sympy_risch.monomial import HypertangentLogarithmicDerivativeSignature

x, t, u, theta = symbols("x t u theta")


def hypertangent_tower() -> DifferentialTower:
    """Return ``QQ(x)(t)`` with ``Dt=1+t**2``."""
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
    """Return a real two-level tangent tower with a nonconstant coefficient."""
    first = hypertangent_tower()
    return first.extend(
        MonomialExtension(
            Poly((t + 1) * (1 + u**2), u, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_terminal_hypertangent_parametric_logarithmic_derivative_is_real() -> None:
    tower = hypertangent_tower()
    outer = MonomialExtension(
        Poly(x**2 * theta, theta, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = solve_parametric_logarithmic_derivative(
        x**2 + 2 * t,
        outer,
        tower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.normal_form.verifies_derivation()
    assert result.certificate.target_result.reconstructs


def test_terminal_hypertangent_radical_logarithmic_derivative_is_real() -> None:
    tower = hypertangent_tower()

    result = solve_radical_logarithmic_derivative(
        2 * t,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentLogarithmicDerivativeSignature,
    )
    assert result.certificate.normal_form.verifies_derivation()
    assert result.certificate.gaussian_result.reconstructs


def test_nested_hypertangent_parametric_logarithmic_derivative_rewrites_every_level() -> (
    None
):
    tower = nested_hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    outer = MonomialExtension(
        Poly(x**2 * theta, theta, domain=field),
        ConstantFieldStatus.PRESERVED,
    )
    argument = field.from_sympy(1 + u**2)
    integrand = field.from_sympy(x**2) + derivation(argument) / argument

    result = solve_parametric_logarithmic_derivative(
        integrand,
        outer,
        derivation,
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HypertangentGaussianParametricLogarithmicDerivativeCertificate,
    )
    assert result.certificate.normal_form.verifies_derivation()
    assert result.certificate.target_result.reconstructs


def test_hypertangent_dependency_analysis_normalizes_a_hypertangent_lower_field() -> (
    None
):
    tower = hypertangent_tower()
    proposal = MonomialExtension(
        Poly(1 + u**2, u, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = analyze_hypertangent_extension(
        proposal,
        tower.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.RATIONAL
    assert result.minimal_power == 1
    assert result.relation_argument is not None
