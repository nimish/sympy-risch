"""Tests for the narrow real hyperbolic Möbius normal form."""

import pytest
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
)
from sympy_risch.hyperbolic import build_hyperbolic_mobius_normal_form

x, t, u, v = symbols("x t u v")


def hyperbolic_tower(eta) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
            SpecialPolynomialStatus.ABSENT,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(eta * (1 - t**2), t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
            SpecialPolynomialStatus.PRESENT,
        ),
    )


def field_elements_equal(left, right, field) -> bool:
    return cancel(field.to_sympy(left - right)) == 0


def test_mobius_normal_form_builds_a_fresh_hyperexponential_tower() -> None:
    tower = hyperbolic_tower(x + 1)

    normal_form = build_hyperbolic_mobius_normal_form(tower.derivation(), u)

    target = normal_form.target_derivation.tower
    target_extension = target.levels[-1]
    assert target.constants == QQ
    assert target.levels[:-1] == tower.levels[:-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.generator == u
    assert target_extension.family is MonomialFamily.HYPEREXPONENTIAL
    assert target_extension.generator_derivative == Poly(
        2 * (x + 1) * u,
        u,
        domain=QQ.frac_field(x),
    )
    assert target_extension.constant_field_status is ConstantFieldStatus.PRESERVED
    assert target_extension.special_polynomial_status is SpecialPolynomialStatus.PRESENT
    assert normal_form.verifies_derivation()


def test_mobius_maps_are_exact_native_field_inverse_maps() -> None:
    tower = hyperbolic_tower(x**2 + 1)
    normal_form = build_hyperbolic_mobius_normal_form(tower.derivation(), u)
    source_field = normal_form.source_field
    target_field = normal_form.target_field

    for source_expression in (
        t,
        (t**2 + x) / (t + 1),
        (x * t + 1) / (1 - t**2),
    ):
        source_value = source_field.from_sympy(source_expression)
        assert field_elements_equal(
            normal_form.inverse(normal_form.forward(source_value)),
            source_value,
            source_field,
        )

    for target_expression in (
        u,
        (u**2 + x) / (u + 1),
        (x * u + 1) / (u**2 - 1),
    ):
        target_value = target_field.from_sympy(target_expression)
        assert field_elements_equal(
            normal_form.forward(normal_form.inverse(target_value)),
            target_value,
            target_field,
        )


def test_mobius_map_intertwines_derivations_on_rational_functions() -> None:
    tower = hyperbolic_tower(x + 1 / x)
    source_derivation = tower.derivation()
    normal_form = build_hyperbolic_mobius_normal_form(source_derivation, u)
    source_field = normal_form.source_field
    target_field = normal_form.target_field

    for source_expression in (
        t,
        (t**2 + x) / (t + 1),
        (x * t + 1) / (1 - t**2),
    ):
        source_value = source_field.from_sympy(source_expression)
        assert field_elements_equal(
            normal_form.target_derivation(normal_form.forward(source_value)),
            normal_form.forward(source_derivation(source_value)),
            target_field,
        )


@pytest.mark.parametrize(
    "derivative",
    (
        1 - t**2 + t,
        1 + t**2,
        t,
        1 - t**3,
    ),
)
def test_mobius_normal_form_rejects_other_polynomial_laws(derivative) -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(ValueError, match=r"Dt = eta\*\(1 - t\*\*2\)"):
        build_hyperbolic_mobius_normal_form(tower.derivation(), u)


def test_mobius_normal_form_requires_a_fresh_target_generator() -> None:
    tower = hyperbolic_tower(1)

    with pytest.raises(ValueError, match="fresh"):
        build_hyperbolic_mobius_normal_form(tower.derivation(), t)

    with pytest.raises(ValueError, match="fresh"):
        build_hyperbolic_mobius_normal_form(tower.derivation(), x)


def test_mobius_normal_form_keeps_unknown_constant_metadata() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(1 - t**2, t, domain=base.field()),
        ),
    )

    normal_form = build_hyperbolic_mobius_normal_form(tower.derivation(), u)

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.constant_field_status is ConstantFieldStatus.UNKNOWN
    assert normal_form.verifies_derivation()


def test_mobius_normal_form_can_normalize_a_top_level_of_a_longer_tower() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = lower.extend(
        MonomialExtension(
            Poly((t + 1) * (1 - v**2), v, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
            SpecialPolynomialStatus.PRESENT,
        ),
    )

    normal_form = build_hyperbolic_mobius_normal_form(tower.derivation(), u)

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.generator_derivative == Poly(
        2 * (t + 1) * u,
        u,
        domain=lower.field(),
    )
    assert normal_form.verifies_derivation()
