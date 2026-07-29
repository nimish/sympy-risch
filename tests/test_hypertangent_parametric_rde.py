"""Complete parametric RDEs over real hypertangent tower levels."""

import pytest
from sympy import I, Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_parametric_risch_differential_equation,
    solve_risch_differential_equation,
)
from sympy_risch.gaussian import build_hypertangent_gaussian_normal_form
from sympy_risch.rde._parametric_linear import ParametricRDESolution
from sympy_risch.rde.hypertangent_parametric import (
    descend_hypertangent_parametric_basis,
)

x, t, u = symbols("x t u")


def hypertangent_tower(*, constants=QQ) -> DifferentialTower:
    base = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_real_basis_contains_parameter_and_homogeneous_directions() -> None:
    tower = hypertangent_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.one, field.from_sympy(1 + t**2)),
        tower.derivation(),
    )

    assert tuple((vector.parameters, vector.solution) for vector in result.basis) == (
        ((QQ.one, QQ.zero), field.from_sympy(x)),
        ((QQ.zero, QQ.one), field.from_sympy(t)),
        ((QQ.zero, QQ.zero), field.one),
    )


def test_real_descent_splits_a_nonreal_parameter_coordinate() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    normal_form = build_hypertangent_gaussian_normal_form(tower.derivation())
    target_constants = normal_form.target_derivation.tower.constants
    imaginary_unit = target_constants.from_sympy(I)
    target_imaginary_unit = normal_form.target_field.from_sympy(I)
    target_direction = ParametricRDESolution(
        (imaginary_unit,),
        target_imaginary_unit * normal_form.forward(field.from_sympy(t)),
    )

    basis = descend_hypertangent_parametric_basis(
        (target_direction,),
        normal_form,
    )

    assert basis == (ParametricRDESolution((QQ.one,), field.from_sympy(t)),)


def test_real_descent_retains_cancelling_parameter_directions() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((t**2 + x) / (t + 1))
    obstruction = field.from_sympy(1 / t)
    right_hand_sides = (
        derivation(planted) + obstruction,
        obstruction,
    )

    assert all(
        not solve_risch_differential_equation(
            field.zero,
            right_hand_side,
            derivation,
        ).has_solution
        for right_hand_side in right_hand_sides
    )

    result = solve_parametric_risch_differential_equation(
        field.zero,
        right_hand_sides,
        derivation,
    )

    assert result.parameter_basis == (result.basis[0],)
    assert result.basis[0].parameters == (QQ.one, -QQ.one)
    assert result.basis[0].solution == planted
    assert result.homogeneous_basis[0].solution == field.one


def test_unsolvable_parameter_direction_keeps_only_the_homogeneous_space() -> None:
    tower = hypertangent_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.from_sympy(1 / t),),
        tower.derivation(),
    )

    assert result.parameter_basis == ()
    assert result.homogeneous_basis[0].solution == field.one


@pytest.mark.parametrize(
    "lower_derivative",
    (
        1 / x,
        t,
        1 + t**2,
    ),
    ids=("primitive", "hyperexponential", "hypertangent"),
)
def test_hypertangent_prde_recurses_through_each_lower_family(
    lower_derivative,
) -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = base.extend(
        MonomialExtension(
            Poly(lower_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = lower.extend(
        MonomialExtension(
            Poly((t + 1) * (1 + u**2), u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((u**2 + t + x) / (u + 1))
    coefficient = field.from_sympy(1 / (u + 2))

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (derivation(planted) + coefficient * planted,),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one,)
    assert result.parameter_basis[0].solution == planted


def test_hypertangent_prde_preserves_real_algebraic_constants() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    alpha = constants.from_sympy(sqrt(2))
    base = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(
                constants.to_sympy(alpha) * (1 + t**2),
                t,
                domain=base.field(),
            ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((t**2 + x) / (t + 1))
    coefficient = field.from_sympy(1 / (t + 2))

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (derivation(planted) + coefficient * planted,),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (constants.one,)
    assert result.parameter_basis[0].solution == planted
