"""Contract tests for certified elementary P/E/T tower admission."""

from dataclasses import FrozenInstanceError
from itertools import product

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    IntegrationObstructionKind,
    MonomialExtension,
    SpecialPolynomialStatus,
    TowerAdmissionOutcome,
    TowerAdmissionReason,
    assess_elementary_admission,
    assess_scalar_rde_admission,
    certify_primitive_extension,
    integrate_elementary,
    solve_risch_differential_equation,
)
from sympy_risch.dependency import (
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
)

x, a, b, t, y = symbols("x a b t y")

_FAMILIES = ("P", "E", "T")


def _rational_tower(
    status: ConstantFieldStatus = ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ), status),
    )


def _adjoin_certified_family(
    tower: DifferentialTower,
    family: str,
    generator,
    eta,
) -> DifferentialTower:
    """Adjoin a checked P/E/T proposal and require constant preservation."""
    parent = tower.field()
    if family == "P":
        analysis = certify_primitive_extension(
            Poly(eta, generator, domain=parent),
            tower.derivation(),
        )
    elif family == "E":
        analysis = certify_hyperexponential_extension(
            Poly(eta * generator, generator, domain=parent),
            tower.derivation(),
        )
    elif family == "T":
        analysis = certify_hypertangent_extension(
            Poly(eta * (1 + generator**2), generator, domain=parent),
            tower.derivation(),
        )
    else:
        raise ValueError(f"unknown P/E/T family {family}")

    assert analysis.status is ConstantFieldStatus.PRESERVED, analysis.detail
    return tower.extend(analysis.extension)


def _matrix_eta(family: str, index: int):
    """Choose independent laws, including on repeated-family arrows."""
    if family == "P":
        return 1 / (x**2 + index + 1)
    return x + index


def _certified_matrix_tower(order: tuple[str, ...]) -> DifferentialTower:
    tower = _rational_tower()
    for index, (family, generator) in enumerate(zip(order, (a, b), strict=True)):
        tower = _adjoin_certified_family(
            tower,
            family,
            generator,
            _matrix_eta(family, index),
        )
    return tower


def test_standard_rational_base_is_admitted_and_integrates() -> None:
    tower = _rational_tower()
    admission = assess_elementary_admission(tower.derivation())

    assert admission.outcome is TowerAdmissionOutcome.ACCEPTED
    assert admission.accepted
    assert not admission.unsupported
    assert admission.reason is None
    assert admission.tower is tower
    assert admission.level == 0

    result = integrate_elementary(tower.field().one, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.obstruction is None


@pytest.mark.parametrize(
    ("family", "eta", "obstruction", "expected_kind"),
    (
        ("P", 1 / x, 1 / t, IntegrationObstructionKind.NONCONSTANT_RESIDUE),
        (
            "E",
            1,
            t / x,
            IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
        ),
        (
            "T",
            1,
            2 * (1 - t**2) / (x * (1 + t**2)),
            IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
        ),
    ),
    ids=("primitive", "hyperexponential", "hypertangent"),
)
def test_each_admitted_top_family_has_exact_and_obstruction_cells(
    family: str,
    eta,
    obstruction,
    expected_kind: IntegrationObstructionKind,
) -> None:
    tower = _adjoin_certified_family(_rational_tower(), family, t, eta)
    field = tower.field()
    derivation = tower.derivation()
    admission = assess_elementary_admission(derivation)

    assert admission.accepted
    exact = field.from_sympy((t + 1) / (t - 1))
    exact_result = integrate_elementary(derivation(exact), derivation)

    assert exact_result.is_elementary
    assert exact_result.reconstructs
    assert exact_result.obstruction is None

    obstruction_result = integrate_elementary(
        field.from_sympy(obstruction),
        derivation,
    )

    assert not obstruction_result.is_elementary
    assert obstruction_result.reconstructs
    assert obstruction_result.obstruction is not None
    assert expected_kind in obstruction_result.obstruction.kinds
    assert obstruction_result.obstruction.certificate is obstruction_result.certificate


@pytest.mark.parametrize(
    ("lower_family", "top_family"),
    tuple(product(_FAMILIES, repeat=2)),
    ids=lambda value: value,
)
def test_every_two_level_pet_arrow_is_admitted_and_reconstructs(
    lower_family: str,
    top_family: str,
) -> None:
    """Cover all directed P/E/T pairs, including PP, EE, and TT."""
    tower = _certified_matrix_tower((lower_family, top_family))
    field = tower.field()
    derivation = tower.derivation()
    admission = assess_elementary_admission(derivation)

    assert admission.accepted
    planted = field.from_sympy((b**2 + a * b + 1) / (b + 1))
    result = integrate_elementary(derivation(planted), derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.obstruction is None


def test_exact_hyperbolic_mobius_normalization_is_admitted() -> None:
    base = _rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(1 - t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    admission = assess_elementary_admission(tower.derivation())

    assert admission.accepted
    assert "hyperbolic Möbius normalization" in admission.detail
    result = integrate_elementary(tower.field().one, tower.derivation())
    assert result.is_elementary
    assert result.reconstructs


def test_hyperbolic_mobius_normalization_rejects_contradictory_absence() -> None:
    base = _rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(1 - t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
            SpecialPolynomialStatus.ABSENT,
        ),
    )

    admission = assess_elementary_admission(tower.derivation())

    assert admission.unsupported
    assert admission.reason is TowerAdmissionReason.SPECIAL_POLYNOMIAL_STATUS


def test_hyperbolic_mobius_normalization_is_not_admitted_below_the_top() -> None:
    base = _rational_tower()
    hyperbolic = base.extend(
        MonomialExtension(
            Poly(1 - t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = hyperbolic.extend(
        MonomialExtension(
            Poly(a, a, domain=hyperbolic.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    admission = assess_elementary_admission(tower.derivation())

    assert admission.unsupported
    assert admission.reason is TowerAdmissionReason.HYPERBOLIC_LOWER_LEVEL


def _unknown_constant_tower() -> DifferentialTower:
    base = _rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        ),
    )


def _non_rational_base_field_tower() -> DifferentialTower:
    return DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _enlarged_constant_tower() -> DifferentialTower:
    base = _rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.ENLARGED,
        ),
    )


def _algebraic_tower() -> DifferentialTower:
    base = _rational_tower()
    return base.extend(AlgebraicExtension(Poly(y**2 - x, y, domain=base.field())))


def _linear_polynomial_tower() -> DifferentialTower:
    base = _rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(t + 1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _nonlinear_tower(
    special_status: SpecialPolynomialStatus,
) -> DifferentialTower:
    base = _rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(t**2 + x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
            special_status,
        ),
    )


@pytest.mark.parametrize(
    ("tower_factory", "reason"),
    (
        (
            lambda: DifferentialTower(QQ),
            TowerAdmissionReason.NO_RATIONAL_BASE,
        ),
        (
            lambda: DifferentialTower(QQ).extend(
                MonomialExtension(
                    Poly(2, x, domain=QQ),
                    ConstantFieldStatus.PRESERVED,
                ),
            ),
            TowerAdmissionReason.BASE_DERIVATIVE,
        ),
        (
            _non_rational_base_field_tower,
            TowerAdmissionReason.BASE_FIELD,
        ),
        (
            _unknown_constant_tower,
            TowerAdmissionReason.CONSTANT_FIELD_UNKNOWN,
        ),
        (
            _enlarged_constant_tower,
            TowerAdmissionReason.CONSTANT_FIELD_ENLARGED,
        ),
        (
            _algebraic_tower,
            TowerAdmissionReason.ALGEBRAIC_EXTENSION,
        ),
        (
            _linear_polynomial_tower,
            TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
        ),
        (
            lambda: _nonlinear_tower(SpecialPolynomialStatus.UNKNOWN),
            TowerAdmissionReason.SPECIAL_POLYNOMIAL_STATUS,
        ),
        (
            lambda: _nonlinear_tower(SpecialPolynomialStatus.PRESENT),
            TowerAdmissionReason.SPECIAL_POLYNOMIAL_STATUS,
        ),
        (
            lambda: _nonlinear_tower(SpecialPolynomialStatus.ABSENT),
            TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
        ),
    ),
    ids=(
        "constant-field-only",
        "nonstandard-base",
        "non-rational-base-field",
        "unknown-constants",
        "enlarged-constants",
        "algebraic",
        "linear-polynomial",
        "nonlinear-special-unknown",
        "nonlinear-special-present",
        "nonlinear-special-absent-is-outside-pet",
    ),
)
def test_rejected_towers_have_explicit_admission_reasons(
    tower_factory,
    reason: TowerAdmissionReason,
) -> None:
    tower = tower_factory()
    admission = assess_elementary_admission(tower.derivation())

    assert admission.outcome is TowerAdmissionOutcome.UNSUPPORTED
    assert admission.unsupported
    assert not admission.accepted
    assert admission.reason is reason


def test_admission_uses_only_the_active_tower_prefix() -> None:
    base = _rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(t + 1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    assert assess_elementary_admission(tower.derivation(0)).accepted
    assert assess_elementary_admission(tower.derivation()).reason is (
        TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY
    )


def test_base_metadata_is_not_a_nonbase_constant_field_requirement() -> None:
    tower = _rational_tower(ConstantFieldStatus.UNKNOWN)

    assert assess_elementary_admission(tower.derivation()).accepted


def test_admission_rechecks_a_claimed_constant_field_certificate() -> None:
    base = _rational_tower()
    forged = base.extend(
        MonomialExtension(
            Poly(1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    elementary = assess_elementary_admission(forged.derivation())
    scalar_rde = assess_scalar_rde_admission(forged.derivation())

    for admission in (elementary, scalar_rde):
        assert admission.unsupported
        assert admission.reason is TowerAdmissionReason.CONSTANT_FIELD_ENLARGED
        assert "independent certification proves" in admission.detail


def test_scalar_rde_admission_accepts_a_gaussian_rational_base() -> None:
    tower = DifferentialTower(QQ_I).extend(
        MonomialExtension(Poly(1, x, domain=QQ_I), ConstantFieldStatus.UNKNOWN),
    )

    admission = assess_scalar_rde_admission(tower.derivation())

    assert admission.accepted
    assert admission.problem == "scalar_rde"


def test_scalar_rde_admission_rejects_hypertangent_over_gaussian_constants() -> None:
    base = DifferentialTower(QQ_I).extend(
        MonomialExtension(Poly(1, x, domain=QQ_I), ConstantFieldStatus.UNKNOWN),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    admission = assess_scalar_rde_admission(tower.derivation())

    assert admission.unsupported
    assert admission.reason is TowerAdmissionReason.HYPERTANGENT_RDE_BASE
    assert "only over QQ(x)" in admission.detail


def test_scalar_rde_admission_checks_the_entire_active_prefix() -> None:
    base = _rational_tower()
    primitive = base.extend(
        MonomialExtension(
            Poly(1 / x, y, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        ),
    )
    hypertangent = primitive.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = hypertangent.extend(
        MonomialExtension(
            Poly(a, a, domain=hypertangent.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    derivation = tower.derivation()

    admission = assess_scalar_rde_admission(derivation)

    assert admission.unsupported
    assert admission.reason is TowerAdmissionReason.CONSTANT_FIELD_UNKNOWN
    assert "level 1" in admission.detail
    with pytest.raises(NotImplementedError, match="constant_field_unknown"):
        solve_risch_differential_equation(
            tower.field().zero,
            tower.field().zero,
            derivation,
        )


def test_admission_result_is_immutable_and_rejects_non_derivations() -> None:
    admission = assess_elementary_admission(_rational_tower().derivation())

    with pytest.raises(FrozenInstanceError):
        admission.detail = "mutated"
    with pytest.raises(TypeError, match="requires a Derivation"):
        assess_elementary_admission("not a derivation")  # type: ignore[arg-type]
