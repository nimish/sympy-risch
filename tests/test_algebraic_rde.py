"""The integrating-factor algebraic RDE cell."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
    analyze_algebraic_constant_field,
)
from sympy_risch.algebraic.log_derivative import (
    AlgebraicIntegratingFactorCertificate,
    AlgebraicIntegratingFactorStatus,
    find_algebraic_integrating_factor,
)
from sympy_risch.algebraic.rde import (
    AlgebraicIntegratingFactorRDEStatus,
    solve_algebraic_integrating_factor_rde,
)
from sympy_risch.algebraic_geometry.function_field import AlgebraicFunctionField

x, y = symbols("x y")


def algebraic_model():
    """Return the standard genus-zero field ``QQ(x, y)``, ``y**2 = x``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(y**2 - x, y, domain=rational.field()))
    tower = rational.extend(extension)
    derivation = tower.derivation()
    return (
        extension,
        derivation,
        AlgebraicFunctionField(extension),
        analyze_algebraic_constant_field(extension, derivation),
    )


def test_integrating_factor_rde_reduces_to_exactness() -> None:
    extension, derivation, model, analysis = algebraic_model()

    result = solve_algebraic_integrating_factor_rde(
        extension.field.from_sympy(1 / (2 * x)),
        extension.field.one / extension.field.generator,
        derivation,
        model,
        analysis,
        max_x_degree=1,
    )

    assert result.status is AlgebraicIntegratingFactorRDEStatus.SOLVED
    assert result.has_solution
    assert result.reconstructs
    assert result.solution == extension.field.generator
    assert result.integrating_factor is not None
    assert result.integrating_factor.factor == extension.field.generator
    assert result.exactness is not None and result.exactness.is_exact


def test_integrating_factor_rde_preserves_a_bounded_exactness_miss() -> None:
    extension, derivation, model, analysis = algebraic_model()

    result = solve_algebraic_integrating_factor_rde(
        extension.field.from_sympy(1 / (2 * x)),
        extension.field.one / extension.field.generator,
        derivation,
        model,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicIntegratingFactorRDEStatus.INCOMPLETE
    assert result.solution is None
    assert result.exactness is not None


def test_integrating_factor_rde_can_prove_no_solution_after_exactness() -> None:
    extension, derivation, model, analysis = algebraic_model()

    result = solve_algebraic_integrating_factor_rde(
        extension.field.from_sympy(1 / (2 * x)),
        extension.field.from_sympy(1 / (x - 1))
        * (extension.field.one / extension.field.generator),
        derivation,
        model,
        analysis,
        max_x_degree=1,
    )

    assert result.status is AlgebraicIntegratingFactorRDEStatus.NO_SOLUTION
    assert result.complete_negative
    assert result.exactness is not None and result.exactness.complete_negative


def test_nonintegral_logarithmic_residue_does_not_claim_an_rde_negative() -> None:
    extension, derivation, model, analysis = algebraic_model()
    coefficient = extension.field.from_sympy(1 / (6 * x))

    factor = find_algebraic_integrating_factor(
        coefficient,
        derivation,
        model,
        analysis,
    )
    result = solve_algebraic_integrating_factor_rde(
        coefficient,
        extension.field.one / extension.field.generator,
        derivation,
        model,
        analysis,
    )

    assert factor.status is AlgebraicIntegratingFactorStatus.NO_FACTOR
    assert (
        factor.certificate is AlgebraicIntegratingFactorCertificate.NONINTEGRAL_RESIDUE
    )
    assert result.status is AlgebraicIntegratingFactorRDEStatus.UNSUPPORTED
    assert not result.complete_negative


def test_homogeneous_algebraic_rde_keeps_the_zero_solution() -> None:
    extension, derivation, model, analysis = algebraic_model()

    result = solve_algebraic_integrating_factor_rde(
        extension.field.from_sympy(1 / (6 * x)),
        extension.field.zero,
        derivation,
        model,
        analysis,
    )

    assert result.status is AlgebraicIntegratingFactorRDEStatus.SOLVED
    assert result.solution == extension.field.zero
    assert result.integrating_factor is None
