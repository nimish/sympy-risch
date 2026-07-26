"""End-to-end elementary integration through the hyperbolic Möbius form."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
    integrate_elementary,
)
from sympy_risch.elementary import HyperbolicMobiusElementaryCertificate

x, t, e = symbols("x t e")


def hyperbolic_tower(
    eta,
    special_status: SpecialPolynomialStatus = SpecialPolynomialStatus.UNKNOWN,
) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    return base.extend(
        MonomialExtension(
            Poly(eta * (1 - t**2), t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
            special_status,
        )
    )


def test_hyperbolic_mobius_dispatch_recurses_to_the_rational_base() -> None:
    tower = hyperbolic_tower(1)
    field = tower.field()

    result = integrate_elementary(field.one, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.total_exact_part == field.from_sympy(x)
    assert result.antiderivative == x
    assert isinstance(result.certificate, HyperbolicMobiusElementaryCertificate)
    assert result.certificate.target_result.is_elementary
    assert result.certificate.normal_form.verifies_derivation()


def test_hyperbolic_mobius_dispatch_pulls_back_logarithmic_terms() -> None:
    tower = hyperbolic_tower(1)
    field = tower.field()

    result = integrate_elementary(field.from_sympy(t), tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative is not None
    assert len(result.all_logarithmic_terms) == 1
    assert t in result.antiderivative.free_symbols
    assert isinstance(result.certificate, HyperbolicMobiusElementaryCertificate)
    assert result.certificate.target_result.reconstructs


def test_hyperbolic_mobius_dispatch_preserves_a_non_elementary_obstruction() -> None:
    tower = hyperbolic_tower(x)
    field = tower.field()

    result = integrate_elementary(field.from_sympy(t), tower.derivation())

    assert not result.is_elementary
    assert result.reconstructs
    assert result.obstruction is not None
    assert result.obstruction.level == 1
    assert result.obstruction.generator == t
    assert result.obstruction.family is MonomialFamily.POLYNOMIAL
    assert isinstance(result.certificate, HyperbolicMobiusElementaryCertificate)
    assert result.certificate.target_result.obstruction is not None


def test_hyperbolic_mobius_dispatch_rejects_a_false_constant_field_claim() -> None:
    tower = hyperbolic_tower(1 / x)

    with pytest.raises(ValueError, match="algebraically dependent"):
        integrate_elementary(tower.field().one, tower.derivation())


def test_hyperbolic_mobius_dispatch_rejects_contradictory_absent_metadata() -> None:
    tower = hyperbolic_tower(1, SpecialPolynomialStatus.ABSENT)

    with pytest.raises(ValueError, match="contradicts"):
        integrate_elementary(tower.field().one, tower.derivation())


def test_hyperbolic_mobius_dispatch_composes_over_a_hyperexponential_lower_level() -> (
    None
):
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    lower = base.extend(
        MonomialExtension(
            Poly(e, e, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = lower.extend(
        MonomialExtension(
            Poly((e + 1) * (1 - t**2), t, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(t**2 / (t + 1))

    result = integrate_elementary(derivation(planted), derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.obstruction is None
