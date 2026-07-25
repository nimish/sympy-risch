"""Risch differential equations in a hyperexponential Laurent branch."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    BoundedHyperexponentialLaurentRDEMatrixResult,
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialLaurentRDEDirection,
    HyperexponentialLaurentRDEObstruction,
    HyperexponentialLaurentRischDifferentialEquationResult,
    MonomialExtension,
    RationalRischDifferentialEquationResult,
    solve_bounded_hyperexponential_laurent_rde_matrix,
    solve_hyperexponential_laurent_risch_differential_equation,
    solve_risch_differential_equation,
)

x, t, u = symbols("x t u")


def exponential_tower(
    *,
    eta=1,
    constant_field_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    return base.extend(
        MonomialExtension(
            Poly(eta*t, t, domain=base.field()),
            constant_field_status,
        )
    )


def test_dispatch_solves_positive_zero_and_negative_laurent_modes() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(3)
    planted = field.from_sympy(
        (x + 1)/t**2 + (x - 1)/t + x**2 - 1 + (x + 2)*t**2
    )
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == planted
    assert isinstance(
        result.certificate,
        HyperexponentialLaurentRischDifferentialEquationResult,
    )
    assert [step.power for step in result.certificate.steps] == [-2, -1, 0, 2]
    assert all(step.has_solution for step in result.certificate.steps)
    assert all(
        isinstance(
            step.lower_field_result.certificate,
            RationalRischDifferentialEquationResult,
        )
        for step in result.certificate.steps
    )


def test_each_mode_uses_the_shifted_lower_field_coefficient() -> None:
    tower = exponential_tower(eta=2*x)
    field = tower.field()

    result = solve_hyperexponential_laurent_risch_differential_equation(
        field.from_sympy(x),
        field.from_sympy(t**3 + 1 + t**-2),
        tower.derivation(),
    )

    assert [step.power for step in result.steps] == [-2, 0, 3]
    assert [
        tower.field(0).to_sympy(step.shifted_coefficient)
        for step in result.steps
    ] == [-3*x, x, 7*x]


def test_failed_mode_retains_a_recursive_obstruction_and_residual() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()

    result = solve_risch_differential_equation(
        field.zero,
        field.from_sympy((1 + 2*x)*t**2 + t/x + 2*x),
        derivation,
    )

    assert not result.has_solution
    assert (
        result.obstruction
        is HyperexponentialLaurentRDEObstruction.LOWER_FIELD
    )
    certificate = result.certificate
    assert certificate.has_lower_field_obstruction
    assert certificate.partial_solution == field.from_sympy(x*t**2 + x**2)
    assert certificate.residual == field.from_sympy(t/x)
    assert [step.power for step in certificate.obstructed_steps] == [1]
    failed = certificate.obstructed_steps[0]
    assert not failed.lower_field_result.has_solution
    assert (
        derivation(certificate.partial_solution)
        + certificate.coefficient*certificate.partial_solution
        + certificate.residual
        == certificate.right_hand_side
    )


def test_zero_right_hand_side_has_the_canonical_zero_solution() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(x),
        field.zero,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.zero
    assert result.certificate.steps == ()
    assert result.reconstructs


def test_dispatch_recurses_through_two_hyperexponential_levels() -> None:
    lower = exponential_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(2*x*u, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(3*x)
    planted = field.from_sympy(
        ((x + 1)*t**2 + (x - 1)/t)/u + x*t*u**2
    )
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HyperexponentialLaurentRischDifferentialEquationResult,
    )
    assert all(
        isinstance(
            step.lower_field_result.certificate,
            HyperexponentialLaurentRischDifferentialEquationResult,
        )
        for step in result.certificate.steps
    )


def test_positive_laurent_coefficient_forces_descending_corrections() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()

    result = solve_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(t),
        derivation,
    )

    assert result.has_solution
    assert result.solution == field.one
    certificate = result.certificate
    assert (
        certificate.direction
        is HyperexponentialLaurentRDEDirection.DESCENDING
    )
    assert certificate.minimum_power_bound == 0
    assert certificate.maximum_power_bound == 0
    assert [step.power for step in certificate.corrections] == [0]
    assert all(
        step.residual_after
        == step.residual_before
        - derivation(step.correction)
        - certificate.coefficient*step.correction
        for step in certificate.corrections
    )


def test_negative_laurent_coefficient_forces_ascending_corrections() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(1/t),
        field.from_sympy(t + 1),
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(t)
    certificate = result.certificate
    assert (
        certificate.direction
        is HyperexponentialLaurentRDEDirection.ASCENDING
    )
    assert certificate.minimum_power_bound == 0
    assert certificate.maximum_power_bound == 1
    assert [step.power for step in certificate.corrections] == [1]


def test_two_sided_laurent_coefficient_uses_both_support_bounds() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t**2 + 2/t)
    planted = field.from_sympy((x + 1)/t**2 + x + (x - 1)*t)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.reconstructs
    assert result.certificate.minimum_power_bound == -2
    assert result.certificate.maximum_power_bound == 1


def test_high_degree_two_sided_recurrence_reconstructs_exactly() -> None:
    tower = exponential_tower(eta=x + 1)
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(2*t**7 + 3*t**2 + t**-4)
    planted = field.from_sympy(
        (x + 1)/t**6 + (x - 2)/t + (x**2 + 1)*t**3 + x*t**8
    )
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.reconstructs
    assert result.certificate.minimum_power_bound == -6
    assert result.certificate.maximum_power_bound == 8
    assert [step.power for step in result.certificate.corrections] == [
        8,
        3,
        -1,
        -6,
    ]


def test_support_bound_excludes_the_forced_correction() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(t),
        field.one,
        tower.derivation(),
    )

    assert not result.has_solution
    assert (
        result.obstruction
        is HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
    )
    assert result.certificate.has_nonzero_remainder_obstruction
    assert result.certificate.minimum_power_bound == 0
    assert result.certificate.maximum_power_bound == 0
    assert result.certificate.partial_solution == field.zero
    assert result.certificate.residual == field.one


def test_forced_recurrence_retains_an_uncancellable_remainder() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()

    result = solve_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(t + 1),
        derivation,
    )

    assert not result.has_solution
    certificate = result.certificate
    assert (
        result.obstruction
        is HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
    )
    assert certificate.has_nonzero_remainder_obstruction
    assert certificate.partial_solution == field.one
    assert certificate.residual == field.one
    assert [step.power for step in certificate.corrections] == [0]
    assert (
        derivation(certificate.partial_solution)
        + certificate.coefficient*certificate.partial_solution
        + certificate.residual
        == certificate.right_hand_side
    )


def test_lower_boundary_cancellation_uses_parametric_logarithmic_relation() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(t + 1),
        field.one,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(1/t)
    certificate = result.certificate
    assert certificate.minimum_power_bound == -1
    assert certificate.maximum_power_bound == 0
    assert certificate.exceptional_power == -1
    assert certificate.boundary_logarithmic_result.has_solution
    assert certificate.boundary_logarithmic_result.n == 1
    assert certificate.boundary_logarithmic_result.m == -1


def test_upper_boundary_cancellation_uses_parametric_logarithmic_relation() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(1/t - 1),
        field.one,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(t)
    certificate = result.certificate
    assert certificate.minimum_power_bound == 0
    assert certificate.maximum_power_bound == 1
    assert certificate.exceptional_power == 1
    assert certificate.boundary_logarithmic_result.has_solution
    assert certificate.boundary_logarithmic_result.n == 1
    assert certificate.boundary_logarithmic_result.m == 1


def test_radical_boundary_relation_is_not_an_ordinary_cancellation() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(t - 1/(2*x)),
        field.one,
        tower.derivation(),
    )

    assert not result.has_solution
    certificate = result.certificate
    assert certificate.has_nonzero_remainder_obstruction
    assert certificate.exceptional_power is None
    assert certificate.boundary_logarithmic_result.has_solution
    assert certificate.boundary_logarithmic_result.n == 2
    assert certificate.boundary_logarithmic_result.m == 0


def test_laurent_coefficient_recurs_through_an_outer_exponential_level() -> None:
    lower = exponential_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(2*x*u, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    coefficient = field.from_sympy(t)
    planted = field.from_sympy(x)
    right_hand_side = tower.derivation()(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.reconstructs
    lower_result = result.certificate.steps[0].lower_field_result
    assert (
        lower_result.certificate.direction
        is HyperexponentialLaurentRDEDirection.DESCENDING
    )


def test_solver_requires_no_new_constants_evidence() -> None:
    tower = exponential_tower(
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )

    with pytest.raises(ValueError, match="status is unknown"):
        solve_hyperexponential_laurent_risch_differential_equation(
            1,
            1,
            tower.derivation(),
        )


def test_matrix_oracle_agrees_on_a_mixed_laurent_solution() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(3)
    planted = field.from_sympy(
        (x + 1)/t**2 + (x - 1)/t + x**2 + (x + 2)*t**2
    )
    right_hand_side = derivation(planted) + coefficient*planted

    production = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        2,
        -2,
        2,
    )

    assert production.has_solution
    assert isinstance(
        matrix,
        BoundedHyperexponentialLaurentRDEMatrixResult,
    )
    assert matrix.has_solution
    assert matrix.solution == production.solution == planted
    assert matrix.coefficient_rank == matrix.augmented_rank


@pytest.mark.parametrize("power", [-4, -2, -1, 1, 2, 4])
def test_matrix_oracle_agrees_on_non_elementary_single_modes(power) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    right_hand_side = field.from_sympy(t**power/x)

    production = solve_risch_differential_equation(
        field.zero,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_hyperexponential_laurent_rde_matrix(
        field.zero,
        right_hand_side,
        derivation,
        x**2,
        5,
        power,
        power,
    )

    assert not production.has_solution
    assert not matrix.has_solution
    assert matrix.augmented_rank > matrix.coefficient_rank


@pytest.mark.parametrize(
    ("lower_degree", "minimum_power", "maximum_power", "message"),
    [
        (-1, 0, 0, "degree bound"),
        (True, 0, 0, "degree bound"),
        (1, False, 0, "minimum power"),
        (1, 0, True, "maximum power"),
        (1, 2, 1, "must not exceed"),
    ],
)
def test_matrix_oracle_rejects_invalid_bounds(
    lower_degree,
    minimum_power,
    maximum_power,
    message,
) -> None:
    tower = exponential_tower()

    with pytest.raises(ValueError, match=message):
        solve_bounded_hyperexponential_laurent_rde_matrix(
            1,
            1,
            tower.derivation(),
            1,
            lower_degree,
            minimum_power,
            maximum_power,
        )


def test_matrix_oracle_rejects_a_zero_lower_denominator() -> None:
    tower = exponential_tower()

    with pytest.raises(ValueError, match="denominator must be nonzero"):
        solve_bounded_hyperexponential_laurent_rde_matrix(
            1,
            1,
            tower.derivation(),
            0,
            1,
            0,
            0,
        )


@given(
    coefficient_values=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=5,
        max_size=5,
    ),
    rhs_values=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=5,
        max_size=5,
    ),
)
@settings(max_examples=80, deadline=None)
def test_general_laurent_recurrence_agrees_with_global_oracle(
    coefficient_values,
    rhs_values,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient_expression = sum(
        value*t**power
        for power, value in zip(
            range(-2, 3),
            coefficient_values,
            strict=True,
        )
    )
    rhs_expression = sum(
        value*t**power
        for power, value in zip(
            range(-2, 3),
            rhs_values,
            strict=True,
        )
    )
    assume(coefficient_expression != 0)
    assume(
        any(
            value
            for power, value in zip(
                range(-2, 3),
                coefficient_values,
                strict=True,
            )
            if power != 0
        )
    )
    assume(rhs_expression != 0)
    coefficient = field.from_sympy(coefficient_expression)
    right_hand_side = field.from_sympy(rhs_expression)

    production = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        0,
        -4,
        4,
    )

    assert production.has_solution is matrix.has_solution
    if production.has_solution:
        assert production.reconstructs
        assert (
            derivation(matrix.solution) + coefficient*matrix.solution
            == right_hand_side
        )
    else:
        assert (
            production.obstruction
            is HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
        )


@given(
    positive_leading=st.integers(min_value=-3, max_value=3).filter(bool),
    negative_leading=st.integers(min_value=-3, max_value=3).filter(bool),
    middle=st.integers(min_value=-3, max_value=3),
    mode_coefficients=st.lists(
        st.tuples(
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
        ),
        min_size=5,
        max_size=5,
    ),
)
@settings(max_examples=30, deadline=None)
def test_generated_two_sided_laurent_solutions_match_global_oracle(
    positive_leading,
    negative_leading,
    middle,
    mode_coefficients,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(
        positive_leading*t**2 + middle*t + negative_leading/t
    )
    planted_expression = sum(
        (constant + linear*x)*t**power
        for power, (constant, linear) in zip(
            range(-2, 3),
            mode_coefficients,
            strict=True,
        )
    )
    assume(planted_expression != 0)
    planted = field.from_sympy(planted_expression)
    right_hand_side = derivation(planted) + coefficient*planted

    production = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        -2,
        2,
    )

    assert production.has_solution
    assert production.solution == planted
    assert production.reconstructs
    assert matrix.has_solution
    assert matrix.solution == planted


@given(
    coefficient=st.integers(min_value=1, max_value=4),
    mode_coefficients=st.lists(
        st.tuples(
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
        ),
        min_size=5,
        max_size=5,
    ),
)
@settings(max_examples=35, deadline=None)
def test_generated_planted_solutions_agree_with_the_matrix_oracle(
    coefficient,
    mode_coefficients,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted_expression = sum(
        (constant + linear*x + quadratic*x**2)*t**power
        for power, (constant, linear, quadratic) in zip(
            range(-2, 3),
            mode_coefficients,
            strict=True,
        )
    )
    planted = field.from_sympy(planted_expression)
    coefficient_element = field.from_sympy(coefficient)
    right_hand_side = (
        derivation(planted) + coefficient_element*planted
    )

    production = solve_risch_differential_equation(
        coefficient_element,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient_element,
        right_hand_side,
        derivation,
        1,
        2,
        -2,
        2,
    )

    assert production.has_solution
    assert production.reconstructs
    assert matrix.has_solution
    assert derivation(matrix.solution) + coefficient_element*matrix.solution == (
        right_hand_side
    )
