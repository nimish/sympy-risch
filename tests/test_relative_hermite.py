"""Relative normal-Hermite reduction for terminal pure radicals."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
)
from sympy_risch.algebraic.relative_hermite import (
    hermite_reduce_relative_simple_radical,
)

x, t, y = symbols("x t y")


def simple_radical_tower(*, derivative, degree=2) -> DifferentialTower:
    """Return ``QQ(x)(t)(y)`` with ``Dt=derivative`` and ``y**degree=t``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = rational.extend(
        MonomialExtension(
            Poly(derivative, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return lower.extend(
        AlgebraicExtension(Poly(y**degree - t, y, domain=lower.field()))
    )


def test_relative_hermite_reduces_a_primitive_normal_pole() -> None:
    tower = simple_radical_tower(derivative=1 / x)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y / t)

    result = hermite_reduce_relative_simple_radical(derivation(planted), derivation)

    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert result.reconstructs
    assert result.normal_denominators_are_square_free
    assert [mode.power for mode in result.modes] == [0, 1]


def test_relative_hermite_retains_hyperexponential_special_poles() -> None:
    tower = simple_radical_tower(derivative=t)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y / (t + 1))
    special = field.from_sympy(y / t**2)

    result = hermite_reduce_relative_simple_radical(
        derivation(planted) + special,
        derivation,
    )

    assert result.exact_part == planted
    assert result.remainder == special
    assert result.reconstructs
    assert result.normal_denominators_are_square_free


def test_relative_hermite_retains_hypertangent_special_poles() -> None:
    tower = simple_radical_tower(derivative=1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y / (t + 1))
    special = field.from_sympy(y / (t**2 + 1) ** 2)

    result = hermite_reduce_relative_simple_radical(
        derivation(planted) + special,
        derivation,
    )

    assert result.exact_part == planted
    assert result.remainder == special
    assert result.reconstructs
    assert result.normal_denominators_are_square_free


def test_relative_hermite_uses_each_cubic_power_basis_mode() -> None:
    tower = simple_radical_tower(derivative=1 / x, degree=3)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y**2 / t)

    result = hermite_reduce_relative_simple_radical(derivation(planted), derivation)

    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert result.reconstructs
    assert result.normal_denominators_are_square_free
    assert result.modes[2].exact_part == tower.field(1).from_sympy(1 / t)


def test_relative_hermite_uses_the_immediate_nested_lower_view() -> None:
    u = symbols("u")
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    first = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = first.extend(
        MonomialExtension(
            Poly(u, u, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = lower.extend(
        AlgebraicExtension(Poly(y**2 - u, y, domain=lower.field())),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y / (u + 1))

    result = hermite_reduce_relative_simple_radical(derivation(planted), derivation)

    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert result.reconstructs
    assert result.normal_denominators_are_square_free


def test_relative_hermite_keeps_a_resonant_pole_as_a_boundary() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = lower.extend(
        AlgebraicExtension(Poly(y**2 - t**2 * x, y, domain=lower.field())),
    )

    with pytest.raises(NotImplementedError, match="resonant normal pole"):
        hermite_reduce_relative_simple_radical(
            tower.field().from_sympy(y / t**2),
            tower.derivation(),
        )
