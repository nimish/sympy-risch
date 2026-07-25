"""Checked constant-field construction for exponential and tangent proposals."""

import pytest
from sympy import Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
)
from sympy_risch.dependency import (
    TranscendentalExtensionAnalysis,
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
)

x, t, u = symbols("x t u")


def _rational_base(
    status: ConstantFieldStatus = ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ), status),
    )


def test_hyperexponential_factory_certifies_independence() -> None:
    base = _rational_base()

    analysis = certify_hyperexponential_extension(
        Poly(t, t, domain=base.field()),
        base.derivation(),
    )

    assert isinstance(analysis, TranscendentalExtensionAnalysis)
    assert analysis.status is ConstantFieldStatus.PRESERVED
    assert not analysis.is_dependent
    assert analysis.dependency is not None
    assert analysis.dependency.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
    assert analysis.dependency.extension == analysis.extension
    assert analysis.detail is None
    assert analysis.lower_fundamental_signature is not None
    assert tuple(
        direction.level for direction in analysis.lower_fundamental_signature.directions
    ) == (0,)


@pytest.mark.parametrize(
    ("eta", "kind", "power"),
    (
        (1 / x, HyperexponentialDependencyKind.RATIONAL, 1),
        (1 / (2 * x), HyperexponentialDependencyKind.ALGEBRAIC, 2),
    ),
)
def test_hyperexponential_factory_records_the_new_constant_relation(
    eta,
    kind: HyperexponentialDependencyKind,
    power: int,
) -> None:
    base = _rational_base()

    analysis = certify_hyperexponential_extension(
        Poly(eta * t, t, domain=base.field()),
        base.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.ENLARGED
    assert analysis.is_dependent
    assert analysis.dependency is not None
    assert analysis.dependency.kind is kind
    assert analysis.dependency.extension == analysis.extension
    assert analysis.dependency.relation_argument == x
    assert analysis.lower_fundamental_signature is not None
    proposed = base.extend(analysis.extension)
    relation = proposed.field().from_sympy(
        t**power / analysis.dependency.relation_argument,
    )
    assert proposed.derivation()(relation) == proposed.field().zero


@pytest.mark.parametrize(
    ("eta", "status", "kind", "power"),
    (
        (
            1,
            ConstantFieldStatus.PRESERVED,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
        ),
        (
            1 / (x**2 + 1),
            ConstantFieldStatus.ENLARGED,
            HyperexponentialDependencyKind.RATIONAL,
            1,
        ),
        (
            1 / (2 * (x**2 + 1)),
            ConstantFieldStatus.ENLARGED,
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
        ),
    ),
)
def test_hypertangent_factory_certifies_independence_and_dependence(
    eta,
    status: ConstantFieldStatus,
    kind: HyperexponentialDependencyKind,
    power: int | None,
) -> None:
    base = _rational_base()

    analysis = certify_hypertangent_extension(
        Poly(eta * (1 + t**2), t, domain=base.field()),
        base.derivation(),
    )

    assert analysis.status is status
    assert analysis.is_dependent is (status is ConstantFieldStatus.ENLARGED)
    assert analysis.dependency is not None
    assert analysis.dependency.extension == analysis.extension
    assert analysis.dependency.kind is kind
    assert analysis.dependency.minimal_power == power
    assert analysis.lower_fundamental_signature is not None


def test_factories_reuse_the_mixed_hypertangent_relation_decision() -> None:
    base = _rational_base()
    tangent = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    analysis = certify_hyperexponential_extension(
        Poly(2 * t * u, u, domain=tangent.field()),
        tangent.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.ENLARGED
    assert analysis.dependency is not None
    assert analysis.dependency.kind is HyperexponentialDependencyKind.RATIONAL
    assert analysis.dependency.extension == analysis.extension
    assert cancel(analysis.dependency.relation_argument / (1 + t**2)) == Rational(1, 4)
    assert analysis.lower_fundamental_signature is not None


def test_hypertangent_factory_reuses_the_hyperexponential_lower_solver() -> None:
    base = _rational_base()
    exponential = base.extend(
        MonomialExtension(
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    analysis = certify_hypertangent_extension(
        Poly(1 + u**2, u, domain=exponential.field()),
        exponential.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.PRESERVED
    assert analysis.dependency is not None
    assert analysis.dependency.extension == analysis.extension
    assert analysis.dependency.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
    assert analysis.lower_fundamental_signature is not None


@pytest.mark.parametrize(
    "certify, derivative",
    (
        (certify_hyperexponential_extension, t),
        (certify_hypertangent_extension, 1 + t**2),
    ),
)
@pytest.mark.parametrize(
    "status",
    (ConstantFieldStatus.UNKNOWN, ConstantFieldStatus.ENLARGED),
)
def test_factories_leave_uncertified_lower_towers_unknown(
    certify,
    derivative,
    status: ConstantFieldStatus,
) -> None:
    base = _rational_base(status)

    analysis = certify(
        Poly(derivative, t, domain=base.field()),
        base.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.dependency is None
    assert analysis.detail is not None
    assert "without certified constant preservation" in analysis.detail
    assert analysis.lower_fundamental_signature is None


@pytest.mark.parametrize(
    "certify, derivative",
    (
        (certify_hyperexponential_extension, t),
        (certify_hypertangent_extension, 1 + t**2),
    ),
)
def test_factories_leave_a_constant_base_unknown(
    certify,
    derivative,
) -> None:
    constant_tower = DifferentialTower(QQ)

    analysis = certify(
        Poly(derivative, t, domain=QQ),
        constant_tower.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.dependency is None
    assert analysis.detail is not None
    assert "nonconstant lower differential field" in analysis.detail
    assert analysis.lower_fundamental_signature is None


def test_factory_leaves_an_unsupported_monomial_lower_family_unknown() -> None:
    base = _rational_base()
    unsupported = base.extend(
        MonomialExtension(
            Poly(1 + t**3, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    analysis = certify_hyperexponential_extension(
        Poly(u, u, domain=unsupported.field()),
        unsupported.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.dependency is None
    assert analysis.detail is not None
    assert "currently support" in analysis.detail
    assert analysis.lower_fundamental_signature is None


@pytest.mark.parametrize(
    "certify, derivative, message",
    (
        (
            certify_hyperexponential_extension,
            1,
            "Dtheta = eta\\*theta",
        ),
        (
            certify_hypertangent_extension,
            t,
            "requires Dt",
        ),
    ),
)
def test_factories_reject_the_wrong_derivative_family(
    certify,
    derivative,
    message: str,
) -> None:
    base = _rational_base()

    with pytest.raises(ValueError, match=message):
        certify(
            Poly(derivative, t, domain=base.field()),
            base.derivation(),
        )
