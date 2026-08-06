"""Parametric RDE descent through terminal relative simple radicals."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_parametric_risch_differential_equation,
)

x, t, y = symbols("x t y")


def _radical_tower(*, derivative=1, degree=2) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    if derivative == 1:
        lower = base
        radicand = x
    else:
        lower = base.extend(
            MonomialExtension(
                Poly(derivative, t, domain=base.field()),
                ConstantFieldStatus.PRESERVED,
            )
        )
        radicand = t
    return lower.extend(
        AlgebraicExtension(
            Poly(y**degree - radicand, y, domain=lower.field()),
        )
    )


def test_relative_parametric_modes_intersect_on_one_parameter_vector() -> None:
    tower = _radical_tower()
    field = tower.field()
    derivation = tower.derivation()
    first = derivation(field.from_sympy(x + y)) + field.from_sympy(1 / x)
    second = derivation(field.from_sympy(2 * y)) + field.from_sympy(1 / x)

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (first, second),
        derivation,
    )

    assert tuple(vector.parameters for vector in result.parameter_basis) == (
        (QQ.one, -QQ.one),
    )
    assert result.parameter_basis[0].solution == field.from_sympy(x - y)
    assert tuple(vector.solution for vector in result.homogeneous_basis) == (field.one,)


@pytest.mark.parametrize(
    "derivative, degree",
    [(1 / x, 2), (t, 3), (1 + t**2, 2)],
)
def test_relative_parametric_descent_reuses_each_supported_lower_family(
    derivative: object,
    degree: int,
) -> None:
    tower = _radical_tower(derivative=derivative, degree=degree)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x + 1) * y + (t + 1) * y ** (degree - 1))
    coefficient = field.from_sympy(t + 2)

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (derivation(planted) + coefficient * planted,),
        derivation,
    )

    assert len(result.parameter_basis) == 1
    assert result.parameter_basis[0].parameters == (QQ.one,)
    assert result.parameter_basis[0].solution == planted
    assert not result.homogeneous_basis


def test_relative_parametric_nonlower_coefficient_is_not_a_negative_result() -> None:
    tower = _radical_tower()
    field = tower.field()

    with pytest.raises(NotImplementedError, match="requires a coupled system"):
        solve_parametric_risch_differential_equation(
            field.generator,
            (field.one,),
            tower.derivation(),
        )


def test_relative_parametric_can_prove_the_shared_space_is_zero() -> None:
    tower = _radical_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.one,
        (field.from_sympy(1 / x),),
        tower.derivation(),
    )

    assert result.basis == ()
    assert not result.has_nontrivial_parameter_solution


def test_relative_parametric_rejects_a_new_algebraic_constant() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = base.extend(AlgebraicExtension(Poly(y**2 - 2, y, domain=base.field())))

    with pytest.raises(NotImplementedError, match="enlarges the constant field"):
        solve_parametric_risch_differential_equation(
            tower.field().zero,
            (tower.field().one,),
            tower.derivation(),
        )


def test_relative_parametric_detects_a_mixed_new_constant() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    lower = base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = lower.extend(
        AlgebraicExtension(Poly(y**2 - 2 * t**2, y, domain=lower.field()))
    )

    with pytest.raises(NotImplementedError, match="enlarges the constant field"):
        solve_parametric_risch_differential_equation(
            tower.field().zero,
            (tower.field().one,),
            tower.derivation(),
        )


def test_relative_parametric_batches_all_parameters_per_power_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy_risch.rde import parametric_dispatch

    tower = _radical_tower(degree=4)
    field = tower.field()
    derivation = tower.derivation()
    right_hand_sides = tuple(
        derivation(field.from_sympy((power + 1) * y)) for power in range(5)
    )
    original = parametric_dispatch._solve_nonempty_parametric_rde
    lower_batches: list[int] = []

    def counted(coefficient, right_hand_sides, active_derivation):
        if active_derivation.level == derivation.level - 1:
            lower_batches.append(len(right_hand_sides))
        return original(coefficient, right_hand_sides, active_derivation)

    monkeypatch.setattr(
        parametric_dispatch,
        "_solve_nonempty_parametric_rde",
        counted,
    )

    solve_parametric_risch_differential_equation(
        field.one,
        right_hand_sides,
        derivation,
    )

    assert lower_batches == [1] * 3 + [len(right_hand_sides)] * 4
