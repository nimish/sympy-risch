"""Risch differential equations over a primitive lower tower."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    BoundedPrimitiveRDEMatrixResult,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    PrimitivePolynomialRDEObstruction,
    PrimitiveRDEObstruction,
    PrimitiveRischDifferentialEquationResult,
    RationalRischDifferentialEquationResult,
    solve_bounded_primitive_rde_matrix,
    solve_primitive_risch_differential_equation,
    solve_risch_differential_equation,
)


x, t, u = symbols("x t u")


def logarithmic_tower(
    *,
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
            Poly(1/x, t, domain=base.field()),
            constant_field_status,
        )
    )


def test_recursive_dispatch_preserves_the_rational_base_solver() -> None:
    tower = logarithmic_tower()
    base_derivation = tower.derivation(0)
    field = tower.field(0)

    result = solve_risch_differential_equation(
        field.from_sympy(1),
        field.from_sympy(x),
        base_derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == field.from_sympy(x - 1)
    assert isinstance(result.certificate, RationalRischDifferentialEquationResult)


def test_primitive_rde_solves_exp_log_squared_coefficient_equation() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()

    result = solve_risch_differential_equation(
        field.from_sympy(2*t/x),
        field.from_sympy(1 + 2*t),
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == field.from_sympy(x)
    assert isinstance(
        result.certificate,
        PrimitiveRischDifferentialEquationResult,
    )
    assert result.certificate.denominator_bound == Poly(
        1,
        t,
        domain=tower.field(0),
    )


def test_primitive_normal_denominator_recovers_a_planted_rational_solution() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(2*t/x)
    planted = field.from_sympy((x + 1)/(t - 1)**2)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.denominator_bound == Poly(
        (t - 1)**2,
        t,
        domain=tower.field(0),
    )
    assert derivation(result.solution) + coefficient*result.solution == (
        right_hand_side
    )


def test_simple_normal_pole_is_a_denominator_obstruction() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(1/(t - 1)),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.has_denominator_obstruction
    assert result.obstruction is PrimitiveRDEObstruction.DENOMINATOR
    assert result.obstruction_polynomial == Poly(
        t - 1,
        t,
        domain=tower.field(0),
    )
    assert not result.obstruction_remainder.is_zero


def test_repeated_pole_can_reach_an_spde_degree_obstruction() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(1/(t - 1)**2),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.has_spde_obstruction
    assert result.obstruction is PrimitiveRDEObstruction.SPDE_DEGREE
    assert result.denominator_bound == Poly(
        t - 1,
        t,
        domain=tower.field(0),
    )


def test_polynomial_recurrence_retains_the_uncancellable_residual() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(t),
        field.from_sympy(t + 1),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.has_polynomial_obstruction
    assert result.obstruction is PrimitiveRDEObstruction.POLYNOMIAL
    polynomial = result.polynomial_result
    assert polynomial is not None
    assert polynomial.obstruction is (
        PrimitivePolynomialRDEObstruction.NONZERO_REMAINDER
    )
    assert polynomial.partial_solution == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert polynomial.residual == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert (
        tower.derivation().polynomial(polynomial.partial_solution)
        + polynomial.coefficient*polynomial.partial_solution
        + polynomial.residual
        == polynomial.right_hand_side
    )


def test_zero_right_hand_side_uses_the_canonical_zero_solution() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(t),
        field.zero,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == field.zero


@pytest.mark.parametrize(
    "coefficient",
    [1, 1/(t - 1)],
)
def test_unsupported_primitive_coefficient_branches_are_explicit(
    coefficient,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()

    with pytest.raises(NotImplementedError, match="coefficient"):
        solve_risch_differential_equation(
            field.from_sympy(coefficient),
            field.from_sympy(1),
            tower.derivation(),
        )


def test_primitive_rde_requires_no_new_constants_evidence() -> None:
    tower = logarithmic_tower(
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )

    with pytest.raises(ValueError, match="status is unknown"):
        solve_primitive_risch_differential_equation(
            tower.field().from_sympy(t),
            tower.field().from_sympy(t),
            tower.derivation(),
        )


def test_bounded_matrix_oracle_agrees_on_success_and_failure() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(2*t/x)

    successful = solve_bounded_primitive_rde_matrix(
        coefficient,
        field.from_sympy(1 + 2*t),
        derivation,
        1,
        1,
        1,
    )
    impossible = solve_bounded_primitive_rde_matrix(
        coefficient,
        field.from_sympy(1),
        derivation,
        1,
        2,
        2,
    )

    assert isinstance(successful, BoundedPrimitiveRDEMatrixResult)
    assert successful.has_solution
    assert successful.solution == field.from_sympy(x)
    assert successful.coefficient_rank == successful.augmented_rank
    assert not impossible.has_solution
    assert impossible.augmented_rank == impossible.coefficient_rank + 1


@pytest.mark.parametrize(
    ("lower_bound", "top_bound"),
    [(-1, 0), (0, -1), (True, 0), (0, False)],
)
def test_bounded_matrix_oracle_rejects_invalid_bounds(
    lower_bound,
    top_bound,
) -> None:
    tower = logarithmic_tower()

    with pytest.raises(ValueError, match="degree bound"):
        solve_bounded_primitive_rde_matrix(
            tower.field().from_sympy(t),
            tower.field().from_sympy(1),
            tower.derivation(),
            1,
            lower_bound,
            top_bound,
        )


def test_high_degree_primitive_recurrence_reconstructs_exactly() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t**5/x + x*t + 1)
    planted = field.from_sympy(
        (x + 1)*t**8 + (x - 2)*t**5 + t**2/x
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


@given(
    coefficient_leading=st.integers(min_value=-3, max_value=3).filter(bool),
    coefficient_constant=st.integers(min_value=-3, max_value=3),
    coefficient_degree=st.integers(min_value=1, max_value=3),
    solution_data=st.lists(
        st.tuples(
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
        ),
        min_size=1,
        max_size=4,
    ),
    denominator_power=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=45, deadline=None)
def test_generated_planted_rational_solutions_are_recovered(
    coefficient_leading,
    coefficient_constant,
    coefficient_degree,
    solution_data,
    denominator_power,
) -> None:
    assume(any(pair != (0, 0) for pair in solution_data))
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(
        coefficient_leading*t**coefficient_degree/x
        + coefficient_constant
    )
    planted_expression = sum(
        (linear*x + constant)*t**degree
        for degree, (linear, constant) in enumerate(solution_data)
    )/(x + 1)**denominator_power
    planted = field.from_sympy(planted_expression)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == planted


@given(
    coefficient_leading=st.integers(min_value=-3, max_value=3).filter(bool),
    coefficient_degree=st.integers(min_value=1, max_value=4),
    solution_coefficient=st.integers(min_value=-3, max_value=3),
    solution_degree=st.integers(min_value=0, max_value=5),
    residual_coefficient=st.integers(min_value=-3, max_value=3).filter(bool),
    residual_degree=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_uncancellable_low_degree_residuals_are_certified(
    coefficient_leading,
    coefficient_degree,
    solution_coefficient,
    solution_degree,
    residual_coefficient,
    residual_degree,
) -> None:
    assume(residual_degree < coefficient_degree)
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(
        coefficient_leading*t**coefficient_degree/x
    )
    planted = field.from_sympy(solution_coefficient*t**solution_degree)
    residual = field.from_sympy(
        residual_coefficient*t**residual_degree/(x + 1)
    )
    right_hand_side = (
        derivation(planted) + coefficient*planted + residual
    )

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert not result.has_solution
    assert result.obstruction is PrimitiveRDEObstruction.POLYNOMIAL
    polynomial = result.certificate.polynomial_result
    assert polynomial is not None
    assert polynomial.partial_solution.as_expr() == planted.as_expr()
    assert polynomial.residual.as_expr() == residual.as_expr()


@given(
    coefficient_leading=st.integers(min_value=-2, max_value=2).filter(bool),
    coefficient_constant=st.integers(min_value=-2, max_value=2),
    solution_linear=st.integers(min_value=-2, max_value=2),
    solution_constant=st.integers(min_value=-2, max_value=2),
    top_degree=st.integers(min_value=0, max_value=2),
    denominator_power=st.integers(min_value=0, max_value=1),
)
@settings(max_examples=20, deadline=None)
def test_recurrence_agrees_with_independent_bounded_matrix_oracle(
    coefficient_leading,
    coefficient_constant,
    solution_linear,
    solution_constant,
    top_degree,
    denominator_power,
) -> None:
    assume(solution_linear != 0 or solution_constant != 0)
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(
        coefficient_leading*t/x + coefficient_constant
    )
    denominator = (x + 1)**denominator_power
    planted = field.from_sympy(
        (solution_linear*x + solution_constant)
        * t**top_degree
        / denominator
    )
    right_hand_side = derivation(planted) + coefficient*planted

    recurrence = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_primitive_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        denominator,
        1,
        top_degree,
    )

    assert recurrence.has_solution
    assert recurrence.solution == planted
    assert matrix.has_solution
    assert matrix.solution == planted
