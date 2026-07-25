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


def test_top_generator_logarithmic_pole_is_weakly_normalized() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = derivation(factor)/factor

    result = solve_risch_differential_equation(
        coefficient,
        field.one/factor,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == field.from_sympy(x/(t + x))
    certificate = result.certificate
    assert certificate.normalization_factor == Poly(
        t + x,
        t,
        domain=tower.field(0),
    )
    assert certificate.normalized_coefficient == field.zero
    assert certificate.normalized_right_hand_side == field.one
    assert certificate.exact_derivative_result.has_solution


def test_weak_normalization_removes_all_positive_integer_residues() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    first = field.from_sympy(t + x)
    second = field.from_sympy(t - 1)
    normalizer = first**2*second**3
    coefficient = derivation(normalizer)/normalizer
    planted = field.from_sympy(x)/normalizer
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        (t + x)**2*(t - 1)**3,
        t,
        domain=tower.field(0),
    )
    assert result.normalized_coefficient == field.zero
    assert result.normalized_right_hand_side == field.one


@pytest.mark.parametrize("residue", [QQ(1, 2), QQ(-2)])
def test_weak_normalization_leaves_nonpositive_integer_residues(
    residue,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = field.convert(residue)*derivation(factor)/factor

    result = solve_primitive_risch_differential_equation(
        coefficient,
        coefficient,
        derivation,
    )

    assert result.has_solution
    assert result.solution == field.one
    assert result.normalization_factor == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert result.normalized_coefficient == coefficient


def test_weak_normalization_preserves_a_higher_principal_part() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = derivation(factor)/factor + field.one/factor**2

    result = solve_primitive_risch_differential_equation(
        coefficient,
        coefficient,
        derivation,
    )

    assert result.has_solution
    assert result.solution == field.one
    assert result.normalization_factor == Poly(
        t + x,
        t,
        domain=tower.field(0),
    )
    assert result.normalized_coefficient == field.one/factor**2


def test_weak_normalization_handles_a_higher_degree_irreducible_factor() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t**2 + x)
    normalizer = factor**3
    coefficient = derivation(normalizer)/normalizer
    planted = field.from_sympy(x)/normalizer
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        (t**2 + x)**3,
        t,
        domain=tower.field(0),
    )
    assert result.normalized_coefficient == field.zero


def test_rational_primitive_coefficient_reaches_denominator_and_spde() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = derivation(factor)/(2*factor)
    planted = field.from_sympy((x + 1)/(t - 1)**2)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert result.denominator_bound == Poly(
        (t - 1)**2,
        t,
        domain=tower.field(0),
    )
    assert result.degree_bound == 1


def test_rational_primitive_coefficient_can_prove_nonsolution() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1/(t - 1))

    result = solve_primitive_risch_differential_equation(
        coefficient,
        field.one,
        derivation,
    )
    oracle = solve_bounded_primitive_rde_matrix(
        coefficient,
        field.one,
        derivation,
        1,
        3,
        result.degree_bound,
    )

    assert not result.has_solution
    assert result.obstruction is PrimitiveRDEObstruction.SPDE_DEGREE
    assert result.normalization_factor == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert not oracle.has_solution
    assert oracle.augmented_rank == oracle.coefficient_rank + 1


def test_lower_field_coefficient_uses_primitive_cancellation_recurrence() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1)
    planted = field.from_sympy((x + 1)*t**3 + (x - 2)*t + 4)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.degree_bound == 4
    assert result.polynomial_result.has_solution
    assert result.polynomial_result.lower_field_results
    assert result.polynomial_result.logarithmic_derivative_result is not None
    assert not (
        result.polynomial_result.logarithmic_derivative_result.has_solution
    )


def test_lower_field_logarithmic_coefficient_becomes_exact_derivative() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1/x)
    planted = field.from_sympy(t**5 + x*t)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.logarithmic_derivative_result.has_solution
    assert result.logarithmic_derivative_result.minimal_power == 1
    assert (
        result.logarithmic_derivative_result.argument
        == tower.field(0).from_sympy(x)
    )
    assert result.exact_derivative_result.has_solution


def test_zero_coefficient_dispatches_to_exact_derivative_solver() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.zero,
        field.from_sympy(1/x),
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(t)
    assert result.exact_derivative_result.has_solution


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side"),
    [
        (0, 1/t),
        (1/x, 1/t),
    ],
)
def test_exact_cancellation_obstruction_is_certified(
    coefficient,
    right_hand_side,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(coefficient),
        field.from_sympy(right_hand_side),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.has_exact_derivative_obstruction
    assert result.exact_derivative_result is not None
    assert not result.exact_derivative_result.has_solution


def test_nonintegral_radical_coefficient_does_not_trigger_exact_transform() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1/(2*x))
    planted = field.from_sympy((x + 1)*t**2)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.logarithmic_derivative_result.has_solution
    assert result.logarithmic_derivative_result.minimal_power == 2
    assert result.exact_derivative_result is None
    assert result.polynomial_result.lower_field_results


def test_lower_field_coefficient_with_rational_solution_denominator() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(1)
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


def test_lower_coefficient_failure_retains_recursive_obstruction() -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(1),
        field.from_sympy(t),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.has_polynomial_obstruction
    polynomial = result.polynomial_result
    assert polynomial.has_lower_field_obstruction
    assert polynomial.lower_field_results
    assert not polynomial.lower_field_results[-1].has_solution
    assert (
        tower.derivation().polynomial(polynomial.partial_solution)
        + polynomial.coefficient*polynomial.partial_solution
        + polynomial.residual
        == polynomial.right_hand_side
    )


def test_cancellation_recurrence_descends_through_nested_primitive_tower() -> None:
    lower = logarithmic_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(1/(x*t), u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t)
    planted = field.from_sympy((x + 1)*u**2 + x*u + 1)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.reconstructs
    lower_results = result.certificate.polynomial_result.lower_field_results
    assert lower_results
    assert all(lower_result.has_solution for lower_result in lower_results)
    assert all(
        isinstance(
            lower_result.certificate,
            PrimitiveRischDifferentialEquationResult,
        )
        for lower_result in lower_results
    )


def test_weak_normalization_descends_through_nested_primitive_tower() -> None:
    lower = logarithmic_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(1/(x*t), u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(u + t)
    coefficient = derivation(factor)/factor

    result = solve_risch_differential_equation(
        coefficient,
        field.one/factor,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.solution == field.from_sympy(x/(u + t))
    assert result.certificate.normalization_factor == Poly(
        u + t,
        u,
        domain=lower.field(),
    )
    assert result.certificate.normalized_coefficient == field.zero


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side"),
    [
        (-3/(x*t), 1 - 3/t),
        (-1/x - 3/(x*t), -3/t),
    ],
)
def test_primitive_degree_bound_includes_limited_integration_cancellation(
    coefficient,
    right_hand_side,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()

    result = solve_primitive_risch_differential_equation(
        field.from_sympy(coefficient),
        field.from_sympy(right_hand_side),
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(x)
    assert result.degree_bound == 3


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


def test_bounded_matrix_oracle_accepts_a_top_denominator() -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = derivation(factor)/factor + field.from_sympy(t/x)
    planted = field.from_sympy((x + 1)/(t + x))
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_bounded_primitive_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        0,
        t + x,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.top_denominator == Poly(
        t + x,
        t,
        domain=tower.field(0),
    )


def test_bounded_matrix_oracle_rejects_a_zero_top_denominator() -> None:
    tower = logarithmic_tower()

    with pytest.raises(ValueError, match="top ansatz denominator"):
        solve_bounded_primitive_rde_matrix(
            tower.field().from_sympy(t),
            tower.field().from_sympy(1),
            tower.derivation(),
            1,
            1,
            1,
            0,
        )


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


@given(
    coefficient=st.integers(min_value=-3, max_value=3).filter(bool),
    solution_data=st.lists(
        st.tuples(
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
        ),
        min_size=1,
        max_size=4,
    ),
    denominator_power=st.integers(min_value=0, max_value=1),
)
@settings(max_examples=35, deadline=None)
def test_generated_lower_coefficient_cancellation_matches_matrix_oracle(
    coefficient,
    solution_data,
    denominator_power,
) -> None:
    assume(any(pair != (0, 0) for pair in solution_data))
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    denominator = (x + 1)**denominator_power
    planted = field.from_sympy(
        sum(
            (linear*x + constant)*t**degree
            for degree, (linear, constant) in enumerate(solution_data)
        )/denominator
    )
    coefficient_element = field.from_sympy(coefficient)
    right_hand_side = (
        derivation(planted) + coefficient_element*planted
    )

    recurrence = solve_primitive_risch_differential_equation(
        coefficient_element,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_primitive_rde_matrix(
        coefficient_element,
        right_hand_side,
        derivation,
        denominator,
        1,
        len(solution_data) - 1,
    )

    assert recurrence.has_solution
    assert recurrence.solution == planted
    assert matrix.has_solution
    assert matrix.solution == planted


@given(
    first_power=st.integers(min_value=1, max_value=3),
    second_power=st.integers(min_value=1, max_value=3),
    solution_linear=st.integers(min_value=-3, max_value=3),
    solution_constant=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=25, deadline=None)
def test_generated_weak_normalization_matches_matrix_oracle(
    first_power,
    second_power,
    solution_linear,
    solution_constant,
) -> None:
    assume(solution_linear != 0 or solution_constant != 0)
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    normalizer_expression = (
        (t + x)**first_power*(t - 1)**second_power
    )
    normalizer = field.from_sympy(normalizer_expression)
    normalized_coefficient = field.from_sympy(t/x)
    coefficient = (
        derivation(normalizer)/normalizer + normalized_coefficient
    )
    planted = (
        field.from_sympy(solution_linear*x + solution_constant)
        / normalizer
    )
    right_hand_side = derivation(planted) + coefficient*planted

    recurrence = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_primitive_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        0,
        normalizer_expression,
    )

    assert recurrence.has_solution
    assert recurrence.solution == planted
    assert recurrence.normalization_factor == Poly(
        normalizer_expression,
        t,
        domain=tower.field(0),
    )
    assert matrix.has_solution
    assert matrix.solution == planted


@given(
    residue=st.sampled_from((QQ(-2), QQ(-1), QQ(1, 2), QQ(3, 2))),
    coefficient_leading=st.integers(min_value=-2, max_value=2).filter(bool),
    solution_linear=st.integers(min_value=-2, max_value=2),
    solution_constant=st.integers(min_value=-2, max_value=2),
    denominator_power=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=25, deadline=None)
def test_generated_weakly_normal_coefficients_match_matrix_oracle(
    residue,
    coefficient_leading,
    solution_linear,
    solution_constant,
    denominator_power,
) -> None:
    assume(solution_linear != 0 or solution_constant != 0)
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    coefficient = (
        field.convert(residue)*derivation(factor)/factor
        + field.from_sympy(coefficient_leading*t/x)
    )
    top_denominator = (t + x)**denominator_power
    planted = field.from_sympy(
        (solution_linear*x + solution_constant)/top_denominator
    )
    right_hand_side = derivation(planted) + coefficient*planted

    recurrence = solve_primitive_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_primitive_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        0,
        top_denominator,
    )

    assert recurrence.has_solution
    assert recurrence.solution == planted
    assert recurrence.normalization_factor == Poly(
        1,
        t,
        domain=tower.field(0),
    )
    assert matrix.has_solution
    assert matrix.solution == planted


@given(
    coefficient=st.integers(min_value=-3, max_value=3).filter(bool),
    rhs_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=9,
        max_size=9,
    ).map(tuple),
)
@settings(max_examples=35, deadline=None)
def test_lower_coefficient_decision_matches_global_matrix_oracle(
    coefficient,
    rhs_coefficients,
) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    right_hand_side = field.from_sympy(
        sum(
            rhs_coefficients[3*top_degree + lower_degree]
            * x**lower_degree
            * t**top_degree
            for top_degree in range(3)
            for lower_degree in range(3)
        )
    )
    coefficient_element = field.from_sympy(coefficient)

    recurrence = solve_primitive_risch_differential_equation(
        coefficient_element,
        right_hand_side,
        derivation,
    )
    matrix = solve_bounded_primitive_rde_matrix(
        coefficient_element,
        right_hand_side,
        derivation,
        1,
        2,
        2,
    )

    assert recurrence.has_solution is matrix.has_solution
    if recurrence.has_solution:
        assert recurrence.solution == matrix.solution
