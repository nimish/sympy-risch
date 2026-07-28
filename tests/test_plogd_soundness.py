"""Soundness regressions for nested hyperexponential PLogD descent."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

import sympy_risch.dependency.transcendental as dependency_module
from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
    ParametricLogarithmicDerivativeOutcome,
    RadicalLogarithmicDerivativeResult,
    solve_parametric_logarithmic_derivative,
    solve_radical_logarithmic_derivative,
)
from sympy_risch.dependency import (
    analyze_hyperexponential_extension,
    certify_hyperexponential_extension,
)
from sympy_risch.monomial import (
    HyperexponentialLogarithmicDerivativeSignature,
)

x, s, t, u = symbols("x s t u")


def _nested_hyperexponential_tower(
    lower_status: ConstantFieldStatus = ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    """Return ``QQ(x)(t)(u)`` with ``Dt=t`` and ``Du=x*u``."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            lower_status,
        ),
    )
    return lower.extend(
        MonomialExtension(
            Poly(x * u, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _adversarial_hyperexponential_tower() -> DifferentialTower:
    """Return an independent P/E prefix that forces a recursive relation query."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    first = base.extend(
        MonomialExtension(
            Poly(s, s, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    second = first.extend(
        MonomialExtension(
            Poly(x * t, t, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return second.extend(
        MonomialExtension(
            Poly(s * u, u, domain=second.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _hypertangent_lower_tower() -> DifferentialTower:
    """Return ``QQ(x)(t)(u)`` with ``Dt=1+t**2`` and ``Du=x*u``."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return lower.extend(
        MonomialExtension(
            Poly(x * u, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _unavailable_relation_system(*_args: object, **_kwargs: object) -> object:
    """Model a lower relation-system implementation that is unavailable."""
    raise NotImplementedError("test relation system is unavailable")


@pytest.mark.parametrize(
    "integrand_name, expected_argument",
    (
        ("zero", 1),
        ("generator_logarithmic_derivative", u),
        ("normal_factor_logarithmic_derivative", t + u),
    ),
)
def test_universal_nested_hyperexponential_witnesses_bypass_an_unavailable_relation_system(
    monkeypatch: pytest.MonkeyPatch,
    integrand_name: str,
    expected_argument: object,
) -> None:
    """Known witnesses do not require the lower structural fallback.

    The final case is deliberately a normal factor: its top-level quotient is
    removed first, leaving the universal lower query ``f=0``.
    """
    tower = _nested_hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + u)
    integrands = {
        "zero": field.zero,
        "generator_logarithmic_derivative": field.from_sympy(x),
        "normal_factor_logarithmic_derivative": derivation(factor) / factor,
    }
    monkeypatch.setattr(
        dependency_module,
        "solve_logarithmic_relation_system",
        _unavailable_relation_system,
    )

    result = solve_radical_logarithmic_derivative(
        integrands[integrand_name],
        derivation,
    )

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.argument == field.from_sympy(expected_argument)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )
    lower_result = result.certificate.lower_parametric_result
    assert lower_result is not None
    assert lower_result.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION


def test_universal_parametric_witnesses_precede_lower_tower_certification() -> None:
    """``f=0`` and ``f=eta`` remain exact even below an uncertified level."""
    tower = _nested_hyperexponential_tower(ConstantFieldStatus.UNKNOWN)
    lower_derivation = tower.derivation().lower()
    extension = tower.levels[-1]
    assert isinstance(extension, MonomialExtension)

    zero_result = solve_parametric_logarithmic_derivative(
        0,
        extension,
        lower_derivation,
    )
    eta_result = solve_parametric_logarithmic_derivative(
        x,
        extension,
        lower_derivation,
    )

    assert zero_result.has_solution
    assert (zero_result.n, zero_result.m, zero_result.argument) == (1, 0, 1)
    assert eta_result.has_solution
    assert (eta_result.n, eta_result.m, eta_result.argument) == (1, 1, 1)


@pytest.mark.parametrize(("integrand", "m"), [(0, 0), (x, 1)])
def test_universal_parametric_witnesses_precede_hypertangent_normalization(
    monkeypatch: pytest.MonkeyPatch,
    integrand: object,
    m: int,
) -> None:
    """The two identities do not require a Cayley-normal-form solver."""
    tower = _hypertangent_lower_tower()
    lower_derivation = tower.derivation().lower()
    extension = tower.levels[-1]
    assert isinstance(extension, MonomialExtension)
    monkeypatch.setattr(
        dependency_module,
        "_solve_monomial_relation_system_parametric_logarithmic_derivative",
        _unavailable_relation_system,
    )

    result = solve_parametric_logarithmic_derivative(
        integrand,
        extension,
        lower_derivation,
    )

    assert result.has_solution
    assert (result.n, result.m, result.argument) == (1, m, 1)
    assert result.reconstructs


def test_hyperexponential_dependency_is_transcendental_only_after_a_proved_negative() -> (
    None
):
    """A normal dependency classification retains its lower obstruction."""
    tower = _nested_hyperexponential_tower()
    lower_derivation = tower.derivation().lower()
    extension = tower.levels[-1]
    assert isinstance(extension, MonomialExtension)

    dependency = analyze_hyperexponential_extension(
        extension,
        lower_derivation,
    )

    assert dependency.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
    assert isinstance(
        dependency.logarithmic_result,
        RadicalLogarithmicDerivativeResult,
    )
    assert dependency.logarithmic_result.proves_no_solution
    assert isinstance(
        dependency.logarithmic_result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )
    lower_result = dependency.logarithmic_result.certificate.lower_parametric_result
    assert lower_result is not None
    assert lower_result.proves_no_solution


def test_unsupported_nested_plogd_is_not_a_transcendence_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing relation solver stays unsupported through dependency analysis."""
    tower = _adversarial_hyperexponential_tower()
    lower_derivation = tower.derivation().lower()
    extension = tower.levels[-1]
    assert isinstance(extension, MonomialExtension)
    monkeypatch.setattr(
        dependency_module,
        "solve_logarithmic_relation_system",
        _unavailable_relation_system,
    )

    parametric_result = solve_parametric_logarithmic_derivative(
        x**2,
        extension,
        lower_derivation,
    )
    assert parametric_result.is_unsupported
    assert not parametric_result.proves_no_solution

    with pytest.raises(
        NotImplementedError,
        match="test relation system is unavailable",
    ):
        analyze_hyperexponential_extension(extension, lower_derivation)

    analysis = certify_hyperexponential_extension(
        extension.generator_derivative,
        lower_derivation,
    )
    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.dependency is None
    assert analysis.detail is not None
    assert "test relation system is unavailable" in analysis.detail
