"""Adversarial coverage for replayed monomial constant-field evidence."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    certify_primitive_extension,
    rebase_derivation_constants,
)
from sympy_risch.constant_proofs import (
    ConstantFieldEvidence,
    replay_constant_field,
    require_preserved_constant_field,
)
from sympy_risch.dependency import PrimitiveExtensionAnalysis

x, t, u = symbols("x t u")


def rational_base(
    status: ConstantFieldStatus = ConstantFieldStatus.UNKNOWN,
) -> DifferentialTower:
    """Build the standard base with deliberately non-authoritative metadata."""
    return DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ), status)
    )


def test_replay_rejects_forged_preserved_primitive_level() -> None:
    base = rational_base()
    tower = base.extend(
        MonomialExtension(
            Poly(1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )

    evidence = replay_constant_field(tower.derivation())

    assert evidence.status is ConstantFieldStatus.ENLARGED
    with pytest.raises(
        ValueError, match="independent constant-field replay is enlarged"
    ):
        require_preserved_constant_field(tower.derivation(), "adversarial test")


def test_replay_accepts_independent_manual_unknown_level() -> None:
    base = rational_base()
    tower = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.UNKNOWN,
        )
    )

    evidence = replay_constant_field(tower.derivation())

    assert evidence.status is ConstantFieldStatus.PRESERVED
    require_preserved_constant_field(tower.derivation(), "adversarial test")


def test_replay_evidence_keeps_equal_towers_at_their_own_positions() -> None:
    first = rational_base()
    second = rational_base()

    first_evidence = replay_constant_field(first.derivation())
    second_evidence = replay_constant_field(second.derivation())

    assert first == second
    assert first is not second
    assert first_evidence is not second_evidence
    assert first_evidence.derivation.tower is first
    assert second_evidence.derivation.tower is second
    assert first_evidence.extension is first.levels[-1]
    assert second_evidence.extension is second.levels[-1]


def test_replay_descends_an_exact_gaussian_constant_rebase() -> None:
    """A QQ(i) coefficient rebase replays its QQ source instead of metadata."""
    base = rational_base()
    first = base.extend(MonomialExtension(Poly(1 / x, t, domain=base.field())))
    source = first.extend(MonomialExtension(Poly(1 / (x * t), u, domain=first.field())))

    rebased = rebase_derivation_constants(source.derivation(), QQ_I)

    assert all(
        level.constant_field_status is ConstantFieldStatus.UNKNOWN
        for level in rebased.tower.levels
        if isinstance(level, MonomialExtension)
    )
    evidence = replay_constant_field(rebased)
    assert evidence.status is ConstantFieldStatus.PRESERVED
    assert evidence.detail is not None
    assert "QQ(i)/QQ" in evidence.detail
    require_preserved_constant_field(rebased, "Gaussian rebase regression")


def test_factory_replays_a_forged_lower_status_before_recursing() -> None:
    base = rational_base()
    dependent = base.extend(
        MonomialExtension(
            Poly(1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )

    analysis = certify_primitive_extension(
        Poly(1, u, domain=dependent.field()),
        dependent.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.detail is not None
    assert "replayed constant preservation" in analysis.detail


def test_replay_fills_an_iterated_primitive_prefix_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deep replay must certify each prefix level once, not recursively retry it."""
    a1, a2, a3, a4, a5, a6 = symbols("a1:7")
    tower = rational_base()
    denominator = x
    for generator in (a1, a2, a3, a4, a5, a6):
        tower = tower.extend(
            MonomialExtension(
                Poly(1 / denominator, generator, domain=tower.field()),
            )
        )
        denominator *= generator

    import sympy_risch.constant_proofs as constant_proofs
    import sympy_risch.dependency as dependency

    calls = 0
    replay_calls = 0
    original = dependency.certify_primitive_extension
    original_replay = constant_proofs.replay_constant_field

    def counted_certification(
        generator_derivative: Poly,
        lower_derivation: Derivation,
    ) -> PrimitiveExtensionAnalysis:
        nonlocal calls
        calls += 1
        return original(generator_derivative, lower_derivation)

    def counted_replay(derivation: Derivation) -> ConstantFieldEvidence:
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(derivation)

    monkeypatch.setattr(
        dependency,
        "certify_primitive_extension",
        counted_certification,
    )
    monkeypatch.setattr(constant_proofs, "replay_constant_field", counted_replay)

    first = constant_proofs.replay_constant_field(tower.derivation())
    second = constant_proofs.replay_constant_field(tower.derivation())

    assert first is not second
    assert first.preserves_constants
    assert second.preserves_constants
    assert calls == len(tower.levels)
    assert replay_calls <= len(tower.levels) + 1
