"""Complete parametric RDEs over hyperexponential tower levels."""

from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_parametric_risch_differential_equation,
    solve_risch_differential_equation,
)

x, s, t, u = symbols("x s t u")


def exponential_tower(*, constants=QQ) -> DifferentialTower:
    """Return ``C(x, t)`` with ``Dx = 1`` and ``Dt = t``."""
    base = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    return base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_diagonal_modes_retain_independent_and_homogeneous_directions() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.one, field.from_sympy(t)),
        tower.derivation(),
    )

    assert tuple((vector.parameters, vector.solution) for vector in result.basis) == (
        ((QQ.one, QQ.zero), field.from_sympy(x)),
        ((QQ.zero, QQ.one), field.from_sympy(t)),
        ((QQ.zero, QQ.zero), field.one),
    )


def test_normal_pole_obstructions_cancel_between_parameter_directions() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1 / (t + 1))
    planted = field.from_sympy((x + 1) / (t + 1) ** 2)
    obstruction = field.from_sympy(1 / (t + 1))
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


def test_normal_restriction_can_retain_a_two_dimensional_parameter_space() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.one
    first = field.from_sympy(x)
    second = field.from_sympy(t)
    obstruction = field.from_sympy(1 / (t + 1))

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(first) + first + obstruction,
            derivation(second) + second + 2 * obstruction,
            obstruction,
        ),
        derivation,
    )

    assert tuple(vector.parameters for vector in result.parameter_basis) == (
        (QQ.one, QQ.zero, -QQ.one),
        (QQ.zero, QQ.one, -2 * QQ.one),
    )
    assert tuple(vector.solution for vector in result.parameter_basis) == (
        first,
        second,
    )


def test_forced_laurent_edges_continue_after_parameter_cancellation() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    obstruction = field.from_sympy(1 / x)

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(t),
        (field.from_sympy(t) + obstruction, obstruction),
        derivation,
    )

    assert len(result.basis) == 1
    assert result.basis[0].parameters == (QQ.one, -QQ.one)
    assert result.basis[0].solution == field.one


def test_ascending_laurent_recurrence_retains_a_cancelling_direction() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1 / t)
    planted = field.from_sympy(t)
    obstruction = field.from_sympy(1 / x)

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(planted) + coefficient * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted


def test_hyperexponential_homogeneous_solution_is_retained() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(-1),
        (field.one,),
        tower.derivation(),
    )

    assert result.parameter_basis[0].solution == -field.one
    assert result.homogeneous_basis[0].solution == field.from_sympy(t)


def test_general_coefficient_over_algebraic_constants_reconstructs() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = exponential_tower(constants=constants)
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


def test_special_and_normal_denominators_share_one_parameter_space() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1 / t)
    planted = field.from_sympy((x + 1) / (t**2 * (t + 1) ** 2))
    obstruction = field.from_sympy(1 / (t + 2))

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(planted) + coefficient * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert len(result.basis) == 1
    assert result.basis[0].parameters == (QQ.one, -QQ.one)
    assert result.basis[0].solution == planted


def test_spde_restriction_preserves_a_cancelling_direction() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1 / (t + 1))
    planted = field.from_sympy(x)
    obstruction = field.one

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(planted) + coefficient * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted


def test_exceptional_infinity_power_survives_parameter_cancellation() -> None:
    tower = exponential_tower()
    field = tower.field()
    obstruction = field.from_sympy(1 / (t + 1))

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(1 / t - 1),
        (field.one + obstruction, obstruction),
        tower.derivation(),
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == field.from_sympy(t)


def test_diagonal_modes_recurse_over_a_primitive_lower_tower() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    primitive = rational.extend(
        MonomialExtension(
            Poly(1 / x, s, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = primitive.extend(
        MonomialExtension(
            Poly(s * t, t, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((s + x) * t**2 + x / t)
    obstruction = field.from_sympy(t / (x + 1))

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(3),
        (
            derivation(planted) + 3 * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted


def test_parametric_recurrence_descends_through_stacked_exponentials() -> None:
    lower = exponential_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(2 * x * u, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(((x + 1) * t**2 + x / t) * u)
    obstruction = field.from_sympy(u / (t + 1))

    result = solve_parametric_risch_differential_equation(
        field.from_sympy(2),
        (
            derivation(planted) + 2 * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted


def test_primitive_recurrence_can_descend_to_a_hyperexponential_prde() -> None:
    lower = exponential_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(1 / (x**2 + 1), u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t)
    planted = field.from_sympy(x * u + t)
    obstruction = field.from_sympy(1 / (u + 1))

    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(planted) + coefficient * planted + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == planted


def test_hyperexponential_parametric_rde_preserves_algebraic_constants() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = exponential_tower(constants=constants)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(sqrt(2) * x * t)
    obstruction = field.from_sympy(1 / (t + 1))

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (
            derivation(planted) + obstruction,
            obstruction,
        ),
        derivation,
    )

    assert tuple(
        constants.to_sympy(constants.convert(parameter))
        for parameter in result.parameter_basis[0].parameters
    ) == (1, -1)
    assert result.parameter_basis[0].solution == planted


def test_unsolvable_direction_leaves_only_the_homogeneous_basis() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_parametric_risch_differential_equation(
        field.zero,
        (field.from_sympy(t / x),),
        tower.derivation(),
    )

    assert result.parameter_basis == ()
    assert result.homogeneous_basis[0].solution == field.one


@given(
    first_coefficients=st.lists(
        st.integers(-3, 3),
        min_size=3,
        max_size=3,
    ),
    second_coefficients=st.lists(
        st.integers(-3, 3),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=20, deadline=None)
def test_generated_nondiagonal_parameter_directions_are_complete(
    first_coefficients,
    second_coefficients,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()

    def planted(coefficients):
        return field.from_sympy(
            sum(
                coefficient * t**power
                for coefficient, power in zip(
                    coefficients,
                    range(3),
                    strict=True,
                )
            )
        )

    first = planted(first_coefficients)
    second = planted(second_coefficients)
    coefficient = field.from_sympy(t)
    result = solve_parametric_risch_differential_equation(
        coefficient,
        (
            derivation(first) + coefficient * first,
            derivation(second) + coefficient * second,
        ),
        derivation,
    )

    assert len(result.parameter_basis) == 2
