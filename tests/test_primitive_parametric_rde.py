"""Complete parametric RDEs over primitive tower levels."""

import pytest
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_parametric_risch_differential_equation,
    solve_risch_differential_equation,
)

x, t, u = symbols("x t u")


def logarithmic_tower(*, constants=QQ) -> DifferentialTower:
    """Return ``C(x, t)`` with ``Dx = 1`` and ``Dt = 1/x``."""
    base = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    return base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_generic_dispatch_retains_the_complete_rational_terminal_basis() -> None:
    tower = logarithmic_tower()
    field = tower.field(0)

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.from_sympy(1 / x + 1), field.from_sympy(1 / x)),
        tower.derivation(0),
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == field.from_sympy(x)
    assert result.homogeneous_basis[0].solution == field.one


def test_normal_pole_obstructions_can_cancel_parametrically() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(2 * t / x)
    planted = field.from_sympy((x + 1) / (t - 1) ** 2)
    obstruction = field.from_sympy(1 / (t + 2))
    right_hand_sides = (
        derivation(planted) + coefficient * planted + obstruction,
        obstruction,
    )

    assert all(
        not solve_risch_differential_equation(
            coefficient,
            right_hand_side,
            derivation,
        ).has_solution
        for right_hand_side in right_hand_sides
    )

    result = solve_parametric_risch_differential_equation(
        coefficient,
        right_hand_sides,
        derivation,
    )

    assert len(result.basis) == 1
    assert result.basis[0].parameters == (QQ.one, -QQ.one)
    assert result.basis[0].solution == planted


def test_positive_degree_coefficient_retains_only_residual_relations() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(t),
        (field.from_sympy(t + 1), field.one),
        tower.derivation(),
    )

    assert len(result.basis) == 1
    assert result.basis[0].parameters == (QQ.one, -QQ.one)
    assert result.basis[0].solution == field.one
    assert result.homogeneous_basis == ()


def test_unsolvable_single_direction_returns_the_empty_solution_space() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(t),
        (field.one,),
        tower.derivation(),
    )

    assert result.basis == ()


def test_lower_coefficient_recurrence_returns_parameter_and_homogeneous_bases() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(-1 / x),
        (field.one,),
        tower.derivation(),
    )

    assert result.parameter_basis[0].parameters == (QQ.one,)
    assert result.parameter_basis[0].solution == field.from_sympy(x * t)
    assert result.homogeneous_basis[0].parameters == (QQ.zero,)
    assert result.homogeneous_basis[0].solution == field.from_sympy(x)


def test_weak_normalization_transports_the_full_solution_basis() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = derivation(factor) / factor

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (field.one / factor,),
        derivation,
    )

    assert result.parameter_basis[0].solution == field.from_sympy(x) / factor
    assert result.homogeneous_basis[0].solution == field.one / factor


def test_primitive_degree_cancellation_integrates_the_generator_derivative() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.from_sympy(1 / x),),
        tower.derivation(),
    )

    assert result.parameter_basis[0].solution == field.from_sympy(t)
    assert result.homogeneous_basis[0].solution == field.one


def test_parametric_recurrence_descends_through_stacked_primitive_levels() -> None:
    lower = logarithmic_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(1 / t, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(x * t * u)
    obstruction = field.from_sympy(1 / (u - 1))

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (derivation(planted) + obstruction, obstruction),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted
    assert result.homogeneous_basis[0].solution == field.one


def test_primitive_parametric_rde_preserves_finite_algebraic_constants() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = logarithmic_tower(constants=constants)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(sqrt(2) * x * t)
    obstruction = field.from_sympy(1 / (t - 1))

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (derivation(planted) + obstruction, obstruction),
        derivation,
    )

    assert tuple(
        constants.to_sympy(constants.convert(parameter))
        for parameter in result.parameter_basis[0].parameters
    ) == (1, -1)
    assert result.parameter_basis[0].solution == planted


def test_recursive_parametric_dispatch_rejects_an_empty_family() -> None:
    tower = logarithmic_tower()

    with pytest.raises(ValueError, match="at least one right-hand side"):
        solve_parametric_risch_differential_equation(
            tower.field().zero,
            (),
            tower.derivation(),
        )
