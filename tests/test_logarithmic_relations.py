"""Independent-facing checks for the finite logarithmic relation compiler."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import I, Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    FundamentalSignature,
    FundamentalSignatureOutcome,
    FundamentalSignatureReason,
    HyperexponentialFundamentalDirection,
    HypertangentFundamentalDirection,
    MonomialExtension,
    PrimitiveFundamentalDirection,
    build_fundamental_signature,
)
from sympy_risch.relations import solve_logarithmic_relation_system

x, t, u, v, y = symbols("x t u v y")


def rational_tower(constants=QQ) -> DifferentialTower:
    """Return the standard rational differential field over ``constants``."""
    return DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def primitive_hyperexponential_hypertangent_tower() -> DifferentialTower:
    """Return one certified P/E/T tower with all native direction shapes."""
    base = rational_tower()
    primitive = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    hyperexponential = primitive.extend(
        MonomialExtension(
            Poly((t + 1) * u, u, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return hyperexponential.extend(
        MonomialExtension(
            Poly((u + x) * (1 + v**2), v, domain=hyperexponential.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def hyperexponential_primitive_tower() -> DifferentialTower:
    """Return the other small P/E ordering used by generated regressions."""
    base = rational_tower()
    hyperexponential = base.extend(
        MonomialExtension(
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return hyperexponential.extend(
        MonomialExtension(
            Poly(1 / (x**2 + 1), u, domain=hyperexponential.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def rejected_algebraic_tower() -> DifferentialTower:
    """Return a tower whose active top level is algebraic."""
    base = rational_tower()
    return base.extend(
        AlgebraicExtension(
            Poly(y**2 - x, y, domain=base.field()),
        ),
    )


def rejected_monomial_tower(
    derivative,
    status: ConstantFieldStatus = ConstantFieldStatus.UNKNOWN,
) -> DifferentialTower:
    """Return a one-level non-base tower for signature rejection checks."""
    base = rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(derivative, t, domain=base.field()),
            status,
        ),
    )


def test_fundamental_signature_replays_a_recursive_tangent_level() -> None:
    tower = primitive_hyperexponential_hypertangent_tower()
    signature = build_fundamental_signature(tower.derivation())

    assert signature.supported
    assert signature.reason is None
    assert signature.certified_constant_levels == tuple(range(len(tower.levels)))
    assert tuple(direction.level for direction in signature.directions) == tuple(
        range(len(tower.levels))
    )


def test_fundamental_signature_records_a_native_tangent_direction() -> None:
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )

    signature = build_fundamental_signature(tower.derivation())

    assert signature.supported
    assert tuple(type(direction) for direction in signature.directions) == (
        PrimitiveFundamentalDirection,
        HypertangentFundamentalDirection,
    )
    assert signature.hypertangent_directions[0].cayley_coefficient == (
        tower.levels[-1].parent_field.one
    )

    with pytest.raises(FrozenInstanceError):
        signature.detail = "mutated"  # type: ignore[misc]


def test_fundamental_signature_handles_the_other_pe_ordering() -> None:
    tower = hyperexponential_primitive_tower()

    signature = build_fundamental_signature(tower.derivation())

    assert signature.supported
    assert tuple(type(direction) for direction in signature.directions) == (
        PrimitiveFundamentalDirection,
        HyperexponentialFundamentalDirection,
        PrimitiveFundamentalDirection,
    )


def test_fundamental_signature_uses_only_the_active_tower_prefix() -> None:
    prefix = hyperexponential_primitive_tower()
    tower = prefix.extend(
        MonomialExtension(
            Poly(1 + v**3, v, domain=prefix.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    signature = build_fundamental_signature(tower.derivation(2))

    assert signature.supported
    assert tuple(direction.level for direction in signature.directions) == (0, 1, 2)


@pytest.mark.parametrize(
    ("tower_factory", "reason"),
    (
        (
            lambda: DifferentialTower(QQ),
            FundamentalSignatureReason.NO_RATIONAL_BASE,
        ),
        (
            lambda: DifferentialTower(QQ).extend(
                MonomialExtension(
                    Poly(2, x, domain=QQ),
                    ConstantFieldStatus.PRESERVED,
                ),
            ),
            FundamentalSignatureReason.BASE_DERIVATIVE,
        ),
        (
            rejected_algebraic_tower,
            FundamentalSignatureReason.ALGEBRAIC_EXTENSION,
        ),
        (
            lambda: rejected_monomial_tower(1),
            FundamentalSignatureReason.CONSTANT_FIELD_ENLARGED,
        ),
        (
            lambda: rejected_monomial_tower(1, ConstantFieldStatus.ENLARGED),
            FundamentalSignatureReason.CONSTANT_FIELD_ENLARGED,
        ),
        (
            lambda: rejected_monomial_tower(
                1 + t**3,
                ConstantFieldStatus.PRESERVED,
            ),
            FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
        ),
    ),
)
def test_fundamental_signature_rejects_outside_the_pet_contract(
    tower_factory: Callable[[], DifferentialTower],
    reason: FundamentalSignatureReason,
) -> None:
    tower = tower_factory()
    signature = build_fundamental_signature(tower.derivation())

    assert signature.unsupported
    assert signature.outcome is FundamentalSignatureOutcome.UNSUPPORTED
    assert signature.reason is reason
    assert signature.directions == ()
    assert signature.certified_constant_levels == ()


def test_fundamental_signature_rejects_inconsistent_manual_metadata() -> None:
    with pytest.raises(ValueError, match="rational base level"):
        FundamentalSignature(
            DifferentialTower(QQ).derivation(),
            FundamentalSignatureOutcome.SUPPORTED,
            None,
            "invalid test signature",
            (),
            (),
            (),
            (),
        )

    tower = rational_tower()
    foreign_base = MonomialExtension(
        Poly(1, x, domain=QQ),
        ConstantFieldStatus.PRESERVED,
    )
    with pytest.raises(ValueError, match="exact tower extension"):
        FundamentalSignature(
            tower.derivation(),
            FundamentalSignatureOutcome.SUPPORTED,
            None,
            "invalid test signature",
            (0,),
            (PrimitiveFundamentalDirection(0, foreign_base, QQ.one),),
            (),
            (),
        )


@given(
    order=st.sampled_from(
        ("primitive_then_hyperexponential", "hyperexponential_then_primitive")
    ),
    offset=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=12, deadline=None)
def test_signature_relation_system_recovers_generated_planted_logarithms(
    order: str,
    offset: int,
) -> None:
    tower = (
        primitive_hyperexponential_hypertangent_tower()
        if order == "primitive_then_hyperexponential"
        else hyperexponential_primitive_tower()
    )
    # This generated family only needs its first P/E prefix.
    derivation = tower.derivation(2)
    top_generator = tower.levels[2].generator
    factor = derivation.field.from_sympy(top_generator + offset)
    planted_term = derivation(factor) / factor

    signature = build_fundamental_signature(derivation)
    system = solve_logarithmic_relation_system((planted_term,), derivation)

    assert signature.supported
    assert system.fundamental_signature == signature
    assert system.reconstructs
    assert tuple(relation.source_coefficients for relation in system.relations) == (
        (1,),
    )


def test_relation_compiler_replays_constant_preservation_before_special_analysis() -> (
    None
):
    base = rational_tower()
    # Dt=t/x makes t/x a new constant.  A forged PRESERVED annotation must not
    # let the relation compiler reach its non-generator-special-factor path.
    tower = base.extend(
        MonomialExtension(
            Poly(t / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    signature = build_fundamental_signature(tower.derivation())

    assert not signature.supported
    assert signature.reason is FundamentalSignatureReason.CONSTANT_FIELD_ENLARGED
    with pytest.raises(ValueError, match="constant-field replay is enlarged"):
        solve_logarithmic_relation_system((1 / (t - x),), tower.derivation())


def test_rational_relation_system_removes_repeated_poles() -> None:
    tower = rational_tower()

    system = solve_logarithmic_relation_system(
        (1 / x, 1 / (x + 1), 1 / x**2),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 2
    assert system.fundamental_signature.supported
    assert system.fundamental_signature == build_fundamental_signature(
        tower.derivation(),
    )
    assert all(relation.source_coefficients[2] == 0 for relation in system.relations)
    assert {relation.source_coefficients[:2] for relation in system.relations} == {
        (1, 0),
        (0, 1),
    }


def test_gaussian_relation_system_keeps_valuations_rational() -> None:
    tower = rational_tower(QQ_I)

    system = solve_logarithmic_relation_system(
        (I / x, 1 / x),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 1
    assert system.fundamental_signature.supported
    relation = system.relations[0]
    assert relation.source_coefficients == (0, 1)
    assert relation.argument == tower.field().from_sympy(x)


def test_hyperexponential_relation_system_records_normal_and_special_witnesses() -> (
    None
):
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(x * t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    factor = field.from_sympy(t + x)

    system = solve_logarithmic_relation_system(
        (
            x,
            1 / x,
            tower.derivation()(factor) / factor,
        ),
        tower.derivation(),
    )

    assert system.reconstructs
    assert len(system.relations) == 3
    assert system.fundamental_signature.supported
    assert isinstance(
        system.fundamental_signature.direction_at(1),
        HyperexponentialFundamentalDirection,
    )
    assert {relation.source_coefficients for relation in system.relations} == {
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    }


@pytest.mark.parametrize("constants", (QQ, QQ_I))
def test_real_tangent_relation_uses_the_special_divisor(constants) -> None:
    base = rational_tower(constants)
    tower = base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    system = solve_logarithmic_relation_system(
        (tower.field().from_sympy(t),),
        tower.derivation(),
    )
    assert len(system.relations) == 1
    relation = system.relations[0]

    assert relation.source_coefficients == (2,)
    assert relation.argument == tower.field().from_sympy(1 + t**2)
    assert system.reconstructs
