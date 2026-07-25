"""End-to-end integration across mixed primitive, exponential, and tangent towers."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    IntegrationObstructionKind,
    MonomialExtension,
    integrate_elementary,
    reduce_hypertangent_monomial,
)

x, a, b, c = symbols("x a b c")


def _base_tower() -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _extend_family(
    tower: DifferentialTower,
    family: str,
    generator,
) -> DifferentialTower:
    """Adjoin an independent representative of one monomial family."""
    parent = tower.field()
    derivative_by_family = {
        "P": 1/(x**2 + 1),
        "E": x*generator,
        "T": 1 + generator**2,
    }
    return tower.extend(
        MonomialExtension(
            Poly(
                derivative_by_family[family],
                generator,
                domain=parent,
            ),
            ConstantFieldStatus.PRESERVED,
        ),
    )


@pytest.mark.parametrize("order", ("PET", "PTE", "EPT", "ETP", "TPE", "TEP"))
@pytest.mark.parametrize(
    "shape",
    ("linear_normal", "repeated_normal", "mixed_lower_coefficient"),
)
def test_mixed_family_orders_integrate_planted_rational_derivatives(
    order: str,
    shape: str,
) -> None:
    """Every ordering reaches its family reducer and recursively descends."""
    tower = _base_tower()
    generators = (a, b, c)
    for family, generator in zip(order, generators, strict=True):
        tower = _extend_family(tower, family, generator)

    field = tower.field()
    derivation = tower.derivation()
    top = generators[-1]
    lower = generators[-2]
    planted_by_shape = {
        "linear_normal": (top**2 + lower)/(top + 1),
        "repeated_normal": (top + lower)/(top**2 + top + 1)**2,
        "mixed_lower_coefficient": (
            top**2 + lower*top + 1
        )/(top + lower + 1),
    }
    planted = field.from_sympy(planted_by_shape[shape])

    result = integrate_elementary(derivation(planted), derivation)

    assert result.is_elementary
    assert result.reconstructs


def _hyperexponential_hypertangent_tower() -> DifferentialTower:
    base = _base_tower()
    exponential = base.extend(
        MonomialExtension(
            Poly(x*a, a, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return exponential.extend(
        MonomialExtension(
            Poly((a + 1)*(1 + b**2), b, domain=exponential.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def _hypertangent_hypertangent_tower() -> DifferentialTower:
    base = _base_tower()
    first = base.extend(
        MonomialExtension(
            Poly(1 + a**2, a, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return first.extend(
        MonomialExtension(
            Poly((a + 1)*(1 + b**2), b, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


@pytest.mark.parametrize(
    "tower_factory",
    (_hyperexponential_hypertangent_tower, _hypertangent_hypertangent_tower),
)
def test_mixed_hypertangent_special_modes_reduce_and_integrate(
    tower_factory,
) -> None:
    """Cayley special modes use the all-level lower coupled RDE path."""
    tower = tower_factory()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((b*(a + 2) + 1)/(1 + b**2))

    reduction = reduce_hypertangent_monomial(
        derivation(planted),
        derivation,
    )
    result = integrate_elementary(derivation(planted), derivation)

    assert reduction.special_reduction is not None
    assert not reduction.has_special_obstruction
    assert reduction.remainder == field.zero
    assert result.is_elementary
    assert result.reconstructs


def test_mixed_hypertangent_special_obstruction_is_a_certificate() -> None:
    """A failed lower coupled RDE is a negative decision, not a crash."""
    tower = _hyperexponential_hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(2*(1 - b**2)/(x*(1 + b**2)))

    reduction = reduce_hypertangent_monomial(integrand, derivation)
    result = integrate_elementary(integrand, derivation)

    assert reduction.has_special_obstruction
    assert not result.is_elementary
    assert result.obstruction is not None
    assert result.obstruction.kinds == (
        IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
    )


def test_hyperexponential_hypertangent_hypertangent_chain_uses_all_level_cayley() -> None:
    """Later tangent levels rewrite earlier ones before solving the lower RDE."""
    lower = _hyperexponential_hypertangent_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly((b + 1)*(1 + c**2), c, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((c*(b + 2) + 1)/(1 + c**2))

    reduction = reduce_hypertangent_monomial(
        derivation(planted),
        derivation,
    )
    result = integrate_elementary(derivation(planted), derivation)

    assert reduction.special_reduction is not None
    assert reduction.remainder == field.zero
    assert result.is_elementary
    assert result.reconstructs


@given(
    real_constants=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=3,
        max_size=3,
    ),
    tangent_constants=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=3,
        max_size=3,
    ),
    special_power=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_generated_mixed_special_modes_round_trip(
    real_constants,
    tangent_constants,
    special_power: int,
) -> None:
    """Generated Cayley special modes stay exact over an exponential lower field."""
    tower = _hyperexponential_hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    real_coefficient = (
        real_constants[0]
        + real_constants[1]*x
        + real_constants[2]*a
    )
    tangent_coefficient = (
        tangent_constants[0]
        + tangent_constants[1]*x
        + tangent_constants[2]*a
    )
    planted = field.from_sympy(
        (real_coefficient + tangent_coefficient*b)/(1 + b**2)**special_power,
    )

    result = integrate_elementary(derivation(planted), derivation)

    assert result.is_elementary
    assert result.reconstructs
