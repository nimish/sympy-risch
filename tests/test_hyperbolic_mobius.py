"""Tests for certified quadratic and exact hyperbolic Möbius normal forms."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
)
from sympy_risch.admission import assess_elementary_admission
from sympy_risch.hyperbolic import (
    build_hyperbolic_mobius_normal_form,
    build_riccati_mobius_normal_form,
)
from sympy_risch.special import certify_special_polynomial

x, t, u, v = symbols("x t u v")


def hyperbolic_tower(eta) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(eta * (1 - t**2), t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def riccati_tower(
    *,
    constant_status: ConstantFieldStatus = ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    """Return a non-hyperbolic Riccati level with two linear specials.

    Here ``Dt=t**2 + (1/x-x)*t`` and the ordered factors ``t-x``, ``t`` give
    ``u=(t-x)/t`` with ``Du=x*u``.  The first factor is intentionally hidden
    from the factorization of ``Dt`` so the test requires supplied, checked
    Darboux certificates rather than an opportunistic visible-factor pass.
    """
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(t**2 + (1 / x - x) * t, t, domain=base.field()),
            constant_status,
        ),
    )


def riccati_special_certificates(tower: DifferentialTower):
    """Return the ordered certificates yielding ``u=(t-x)/t``."""
    derivation = tower.derivation()
    parent = tower.field(0)
    return (
        certify_special_polynomial(Poly(t - x, t, domain=parent), derivation),
        certify_special_polynomial(Poly(t, t, domain=parent), derivation),
    )


def field_elements_equal(left, right, field) -> bool:
    return cancel(field.to_sympy(left - right)) == 0


def test_riccati_normal_form_builds_an_admitted_hyperexponential_target() -> None:
    tower = riccati_tower()
    numerator, denominator = riccati_special_certificates(tower)

    normal_form = build_riccati_mobius_normal_form(
        tower.derivation(),
        u,
        numerator,
        denominator,
    )

    target = normal_form.target_derivation.tower
    target_extension = target.levels[-1]
    assert target.levels[:-1] == tower.levels[:-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.family is MonomialFamily.HYPEREXPONENTIAL
    assert target_extension.generator_derivative == Poly(
        x * u,
        u,
        domain=QQ.frac_field(x),
    )
    assert target_extension.constant_field_status is ConstantFieldStatus.PRESERVED
    assert normal_form.numerator_certificate is numerator
    assert normal_form.denominator_certificate is denominator
    assert normal_form.numerator_factor == numerator.factor
    assert normal_form.denominator_factor == denominator.factor
    assert normal_form.inverse_generator_expression == cancel((t - x) / t)
    assert normal_form.forward_generator_expression == cancel(x / (1 - u))
    assert normal_form.verifies_isomorphism()
    assert normal_form.verifies_derivation()
    assert assess_elementary_admission(normal_form.target_derivation).accepted
    assert not assess_elementary_admission(tower.derivation()).accepted


def test_riccati_normal_form_maps_and_derivations_are_exact() -> None:
    tower = riccati_tower()
    numerator, denominator = riccati_special_certificates(tower)
    source_derivation = tower.derivation()
    normal_form = build_riccati_mobius_normal_form(
        source_derivation,
        u,
        numerator,
        denominator,
    )
    source_field = normal_form.source_field
    target_field = normal_form.target_field

    for source_expression in (
        t,
        (t**2 + x) / (t + 1),
        (x * t + 1) / (t - x),
    ):
        source_value = source_field.from_sympy(source_expression)
        assert field_elements_equal(
            normal_form.inverse(normal_form.forward(source_value)),
            source_value,
            source_field,
        )
        assert field_elements_equal(
            normal_form.target_derivation(normal_form.forward(source_value)),
            normal_form.forward(source_derivation(source_value)),
            target_field,
        )

    for target_expression in (
        u,
        (u**2 + x) / (u + 1),
        (x * u + 1) / (u - 1),
    ):
        target_value = target_field.from_sympy(target_expression)
        assert field_elements_equal(
            normal_form.forward(normal_form.inverse(target_value)),
            target_value,
            target_field,
        )


@given(
    a_slope=st.integers(-2, 2),
    a_intercept=st.integers(-2, 2),
    b_slope=st.integers(-2, 2),
    b_intercept=st.integers(-2, 2),
    leading=st.integers(-2, 2),
)
@settings(max_examples=20, deadline=None)
def test_generated_riccati_special_pairs_normalize_exactly(
    a_slope: int,
    a_intercept: int,
    b_slope: int,
    b_intercept: int,
    leading: int,
) -> None:
    """Generate quadratic laws with two prescribed moving special roots."""
    assume(leading != 0)
    assume((a_slope, a_intercept) != (b_slope, b_intercept))
    a = a_slope * x + a_intercept
    b = b_slope * x + b_intercept
    root_gap = a - b
    correction = a_slope * (t - b) / root_gap + b_slope * (t - a) / (-root_gap)
    derivative = leading * (t - a) * (t - b) + correction
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
    derivation = tower.derivation()
    numerator = certify_special_polynomial(
        Poly(t - a, t, domain=base.field()),
        derivation,
    )
    denominator = certify_special_polynomial(
        Poly(t - b, t, domain=base.field()),
        derivation,
    )

    normal_form = build_riccati_mobius_normal_form(
        derivation,
        u,
        numerator,
        denominator,
    )

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.generator_derivative == Poly(
        leading * root_gap * u,
        u,
        domain=QQ.frac_field(x),
    )
    assert normal_form.verifies_isomorphism()
    assert normal_form.verifies_derivation()


def test_riccati_normal_form_preserves_the_supplied_factor_orientation() -> None:
    tower = riccati_tower()
    numerator, denominator = riccati_special_certificates(tower)

    normal_form = build_riccati_mobius_normal_form(
        tower.derivation(),
        u,
        denominator,
        numerator,
    )

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.generator_derivative == Poly(
        -x * u,
        u,
        domain=QQ.frac_field(x),
    )
    assert normal_form.inverse_generator_expression == cancel(t / (t - x))
    assert normal_form.verifies_isomorphism()
    assert normal_form.verifies_derivation()


def test_riccati_normal_form_reuses_a_nontrivial_lower_monomial_prefix() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(x * v, v, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = lower.extend(
        MonomialExtension(
            Poly((x + 1) * t**2 / v - t, t, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    derivation = tower.derivation()
    numerator = certify_special_polynomial(
        Poly(t - v, t, domain=lower.field()),
        derivation,
    )
    denominator = certify_special_polynomial(
        Poly(t, t, domain=lower.field()),
        derivation,
    )

    normal_form = build_riccati_mobius_normal_form(
        derivation,
        u,
        numerator,
        denominator,
    )

    target = normal_form.target_derivation.tower
    target_extension = target.levels[-1]
    assert target.levels[:-1] == tower.levels[:-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.generator_derivative == Poly(
        (x + 1) * u,
        u,
        domain=lower.field(),
    )
    assert normal_form.verifies_isomorphism()
    assert normal_form.verifies_derivation()
    assert assess_elementary_admission(normal_form.target_derivation).accepted


def test_riccati_normal_form_replays_preservation_from_checked_coordinate() -> None:
    tower = riccati_tower(constant_status=ConstantFieldStatus.UNKNOWN)
    numerator, denominator = riccati_special_certificates(tower)

    normal_form = build_riccati_mobius_normal_form(
        tower.derivation(),
        u,
        numerator,
        denominator,
    )

    assert normal_form.verifies_derivation()


def test_riccati_normal_form_rejects_invalid_factor_pairs_and_targets() -> None:
    tower = riccati_tower()
    numerator, denominator = riccati_special_certificates(tower)
    product = certify_special_polynomial(
        numerator.factor * denominator.factor,
        tower.derivation(),
    )
    foreign_tower = hyperbolic_tower(1)
    foreign = certify_special_polynomial(
        Poly(t + 1, t, domain=foreign_tower.field(0)),
        foreign_tower.derivation(),
    )

    with pytest.raises(ValueError, match="distinct"):
        build_riccati_mobius_normal_form(
            tower.derivation(),
            u,
            numerator,
            numerator,
        )
    with pytest.raises(ValueError, match="linear"):
        build_riccati_mobius_normal_form(
            tower.derivation(),
            u,
            product,
            denominator,
        )
    with pytest.raises(ValueError, match="fresh"):
        build_riccati_mobius_normal_form(
            tower.derivation(),
            t,
            numerator,
            denominator,
        )
    with pytest.raises(ValueError, match="another derivation"):
        build_riccati_mobius_normal_form(
            tower.derivation(),
            u,
            foreign,
            denominator,
        )


def test_riccati_normal_form_rejects_certificates_from_an_equal_tower() -> None:
    first = riccati_tower()
    second = riccati_tower()
    numerator, denominator = riccati_special_certificates(first)

    assert first == second
    assert first is not second
    with pytest.raises(ValueError, match="another derivation"):
        build_riccati_mobius_normal_form(
            second.derivation(),
            u,
            numerator,
            denominator,
        )


def test_riccati_normal_form_rejects_a_nonquadratic_source() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    cubic_tower = base.extend(
        MonomialExtension(
            Poly(t * (t - 1) * (t - 2), t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    cubic_derivation = cubic_tower.derivation()
    cubic_numerator = certify_special_polynomial(
        Poly(t, t, domain=base.field()),
        cubic_derivation,
    )
    cubic_denominator = certify_special_polynomial(
        Poly(t - 1, t, domain=base.field()),
        cubic_derivation,
    )

    with pytest.raises(ValueError, match="quadratic monomial"):
        build_riccati_mobius_normal_form(
            cubic_derivation,
            u,
            cubic_numerator,
            cubic_denominator,
        )


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


def test_mobius_normal_form_replays_unknown_constant_metadata() -> None:
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
    assert target_extension.constant_field_status is ConstantFieldStatus.PRESERVED
    assert normal_form.verifies_derivation()


def test_hyperbolic_normal_form_constructs_its_checked_special_factors() -> None:
    tower = hyperbolic_tower(1)

    assert assess_elementary_admission(tower.derivation()).accepted
    normal_form = build_hyperbolic_mobius_normal_form(tower.derivation(), u)
    assert normal_form.verifies_derivation()


def test_riccati_normal_form_rejects_a_replayed_new_constant() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ), ConstantFieldStatus.PRESERVED)
    )
    tower = base.extend(
        MonomialExtension(
            Poly(t * (1 - t) / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    derivation = tower.derivation()
    numerator = certify_special_polynomial(Poly(t, t, domain=base.field()), derivation)
    denominator = certify_special_polynomial(
        Poly(t - 1, t, domain=base.field()), derivation
    )

    with pytest.raises(ValueError, match="constant-field replay is enlarged"):
        build_riccati_mobius_normal_form(derivation, u, numerator, denominator)


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
