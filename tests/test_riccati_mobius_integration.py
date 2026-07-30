"""End-to-end integration through certified quadratic Möbius normal forms."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    RiccatiMobiusElementaryCertificate,
    RiccatiMobiusNormalForm,
    build_riccati_mobius_normal_form,
    certify_special_polynomial,
    integrate_elementary,
    integrate_riccati_mobius_normal_form,
)

x, t, u, s = symbols("x t u s")


def _riccati_tower() -> DifferentialTower:
    """Return ``Dt=t**2+(1/x-x)*t`` over the standard rational base."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(t**2 + (1 / x - x) * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _riccati_normal_form(tower: DifferentialTower) -> RiccatiMobiusNormalForm:
    """Build the checked coordinate ``u=(t-x)/t`` for ``tower``."""
    derivation = tower.derivation()
    parent = tower.field(0)
    numerator = certify_special_polynomial(
        Poly(t - x, t, domain=parent),
        derivation,
    )
    denominator = certify_special_polynomial(
        Poly(t, t, domain=parent),
        derivation,
    )
    return build_riccati_mobius_normal_form(
        derivation,
        u,
        numerator,
        denominator,
    )


def test_explicit_riccati_normal_form_integrates_a_planted_derivative() -> None:
    tower = _riccati_tower()
    derivation = tower.derivation()
    field = tower.field()
    normal_form = _riccati_normal_form(tower)
    planted = field.from_sympy((t**2 + x) / (t + 1))

    with pytest.raises(
        NotImplementedError,
        match="outside the complete elementary_integration cell",
    ):
        integrate_elementary(derivation(planted), derivation)

    result = integrate_riccati_mobius_normal_form(
        derivation(planted),
        normal_form,
    )

    assert result.is_elementary
    assert result.reconstructs
    assert result.obstruction is None
    assert isinstance(result.certificate, RiccatiMobiusElementaryCertificate)
    assert result.certificate.normal_form is normal_form
    assert result.certificate.target_result.is_elementary
    assert result.certificate.target_result.reconstructs


def test_explicit_riccati_normal_form_pulls_back_a_non_elementary_result() -> None:
    tower = _riccati_tower()
    field = tower.field()
    normal_form = _riccati_normal_form(tower)

    # This maps to 1/u, whose target integral is the non-elementary
    # hyperexponential integral exp(-x**2 / 2).
    result = integrate_riccati_mobius_normal_form(
        field.from_sympy(t / (t - x)),
        normal_form,
    )

    assert not result.is_elementary
    assert result.reconstructs
    assert result.obstruction is not None
    assert result.obstruction.level == 1
    assert result.obstruction.generator == t
    assert isinstance(result.certificate, RiccatiMobiusElementaryCertificate)
    assert result.certificate.target_result.obstruction is not None


def test_explicit_riccati_normal_form_pulls_back_logarithmic_terms() -> None:
    tower = _riccati_tower()
    field = tower.field()
    normal_form = _riccati_normal_form(tower)

    # In the target coordinate this is D(u + 1)/(u + 1), so it exercises
    # the logarithmic-term presentation rather than only native exact parts.
    result = integrate_riccati_mobius_normal_form(
        field.from_sympy(x * (t - x) / (2 * t - x)),
        normal_form,
    )

    assert result.is_elementary
    assert result.reconstructs
    assert len(result.all_logarithmic_terms) == 1
    assert result.antiderivative is not None
    assert t in result.antiderivative.free_symbols
    assert isinstance(result.certificate, RiccatiMobiusElementaryCertificate)
    assert len(result.certificate.target_result.all_logarithmic_terms) == 1


def test_riccati_normal_form_requires_a_replayable_target() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    unsupported_lower = base.extend(
        MonomialExtension(
            Poly(s**2, s, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = unsupported_lower.extend(
        MonomialExtension(
            Poly(t**2 - t, t, domain=unsupported_lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    derivation = tower.derivation()
    parent = tower.field(1)
    numerator = certify_special_polynomial(
        Poly(t, t, domain=parent),
        derivation,
    )
    denominator = certify_special_polynomial(
        Poly(t - 1, t, domain=parent),
        derivation,
    )
    with pytest.raises(
        NotImplementedError,
        match="independent constant-field replay is unavailable",
    ):
        build_riccati_mobius_normal_form(
            derivation,
            u,
            numerator,
            denominator,
        )
