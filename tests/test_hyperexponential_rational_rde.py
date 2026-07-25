"""Rational Risch differential equations in a hyperexponential field."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialRDEObstruction,
    HyperexponentialRischDifferentialEquationResult,
    MonomialExtension,
    solve_bounded_hyperexponential_laurent_rde_matrix,
    solve_hyperexponential_risch_differential_equation,
    solve_risch_differential_equation,
)

x, s, t = symbols("x s t")


def exponential_tower(
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
            Poly(t, t, domain=base.field()),
            constant_field_status,
        )
    )


def test_dispatch_accepts_a_normal_coefficient_and_denominator() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1/(t + 1))
    planted = field.from_sympy((x + 1)/(t + 1)**2)
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
        HyperexponentialRischDifferentialEquationResult,
    )
    assert result.certificate.normal_denominator_bound == Poly(
        (t + 1)**2,
        t,
        domain=tower.levels[-1].parent_field,
    )


def test_weak_normalization_removes_all_positive_integer_residues() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(
        2*t/(t + 1) + 3*t/(t + 2)
    )
    planted = field.from_sympy(x/((t + 1)**2*(t + 2)**3))
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        (t + 1)**2*(t + 2)**3,
        t,
        domain=tower.levels[-1].parent_field,
    )
    assert result.normalized_coefficient == field.zero


def test_weak_normalization_does_not_treat_a_repeated_pole_as_a_residue() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(2*t/(t + 1)**2)
    planted = field.from_sympy(x)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        1,
        t,
        domain=tower.levels[-1].parent_field,
    )


@pytest.mark.parametrize("residue", [-2, -1, QQ(1, 2)])
def test_weak_normalization_keeps_nonpositive_and_nonintegral_residues(
    residue,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(residue*t/(t + 1))
    planted = field.from_sympy(x)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        1,
        t,
        domain=tower.levels[-1].parent_field,
    )


def test_normal_denominator_obstruction_retains_failed_division() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_risch_differential_equation(
        field.one,
        field.from_sympy(1/(t + 1)),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.obstruction is HyperexponentialRDEObstruction.DENOMINATOR
    certificate = result.certificate
    assert certificate.has_denominator_obstruction
    assert certificate.obstruction_polynomial == Poly(
        t + 1,
        t,
        domain=tower.levels[-1].parent_field,
    )
    assert certificate.obstruction_remainder == Poly(
        1,
        t,
        domain=tower.levels[-1].parent_field,
    )


def test_special_denominator_reduction_handles_zero_and_normal_poles() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1/t)
    planted = field.from_sympy((x + 1)/(t**2*(t + 1)**2))
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normal_denominator_bound == Poly(
        (t + 1)**2,
        t,
        domain=tower.levels[-1].parent_field,
    )
    assert result.special_denominator_power == 2
    assert result.special_clearing_power == 1


def test_spde_divisibility_obstruction_is_certified() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_hyperexponential_risch_differential_equation(
        field.from_sympy(1/(t + 1)),
        field.one,
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.obstruction is (
        HyperexponentialRDEObstruction.SPDE_DIVISIBILITY
    )
    assert result.has_spde_obstruction
    assert result.obstruction_polynomial is not None
    assert result.obstruction_remainder is not None
    assert not result.obstruction_remainder.is_zero


def test_spde_degree_obstruction_is_certified() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_hyperexponential_risch_differential_equation(
        field.from_sympy(1/(t + 1)**2),
        field.one,
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.obstruction is HyperexponentialRDEObstruction.SPDE_DEGREE
    assert result.has_spde_obstruction
    assert result.degree_bound == 1


def test_degree_bound_retains_an_exceptional_infinity_cancellation() -> None:
    tower = exponential_tower()
    field = tower.field()

    result = solve_hyperexponential_risch_differential_equation(
        field.from_sympy(1/t - 1),
        field.one,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.solution == field.from_sympy(t)
    assert result.degree_bound == 1
    assert result.degree_exceptional_power == 1
    assert result.degree_boundary_result is not None
    assert result.degree_boundary_result.has_solution


def test_full_frontend_recurses_over_a_primitive_lower_field() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    primitive = rational.extend(
        MonomialExtension(
            Poly(1/x, s, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = primitive.extend(
        MonomialExtension(
            Poly(s*t, t, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1/(t + 1))
    planted = field.from_sympy((s + x)/(t*(t + s)**2))
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normal_denominator_bound == Poly(
        (t + s)**2,
        t,
        domain=primitive.field(),
    )
    assert result.special_denominator_power == 1


def test_weak_normalizer_works_over_a_primitive_lower_field() -> None:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    primitive = rational.extend(
        MonomialExtension(
            Poly(1/x, s, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = primitive.extend(
        MonomialExtension(
            Poly(s*t, t, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    normalizer = field.from_sympy((t + s)**2)
    coefficient = derivation(normalizer)/normalizer
    planted = field.from_sympy(x/(t + s)**2)
    right_hand_side = derivation(planted) + coefficient*planted

    result = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.has_solution
    assert result.solution == planted
    assert result.normalization_factor == Poly(
        (t + s)**2,
        t,
        domain=primitive.field(),
    )
    assert result.normalized_coefficient == field.zero


def test_matrix_oracle_accepts_an_independent_top_denominator() -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    coefficient = field.from_sympy(t + 1/(t + 1))
    planted = field.from_sympy(
        ((x + 1) + (x - 1)*t + x*t**2)/(t*(t + 1)**3)
    )
    right_hand_side = derivation(planted) + coefficient*planted

    production = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    oracle = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        -1,
        1,
        (t + 1)**3,
    )

    assert production.has_solution
    assert production.solution == planted
    assert oracle.has_solution
    assert oracle.solution == planted
    assert oracle.top_denominator == Poly(
        (t + 1)**3,
        t,
        domain=tower.levels[-1].parent_field,
    )


def test_matrix_oracle_rejects_a_zero_top_denominator() -> None:
    tower = exponential_tower()

    with pytest.raises(ValueError, match="top ansatz denominator"):
        solve_bounded_hyperexponential_laurent_rde_matrix(
            1,
            1,
            tower.derivation(),
            1,
            1,
            0,
            0,
            0,
        )


@pytest.mark.parametrize(
    ("coefficient_expression", "rhs_expression", "top_denominator"),
    [
        (1, 1/(t + 1), (t + 1)**3),
        (1/(t + 1), 1, (t + 1)**3),
        (1/(t + 1)**2, 1, (t + 1)**4),
    ],
)
def test_certified_obstructions_agree_with_broad_matrix_searches(
    coefficient_expression,
    rhs_expression,
    top_denominator,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    coefficient = field.from_sympy(coefficient_expression)
    right_hand_side = field.from_sympy(rhs_expression)

    production = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        tower.derivation(),
    )
    oracle = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        tower.derivation(),
        x**2 + 1,
        3,
        -3,
        5,
        top_denominator,
    )

    assert not production.has_solution
    assert not oracle.has_solution
    assert oracle.augmented_rank > oracle.coefficient_rank


def test_full_solver_requires_no_new_constants_evidence() -> None:
    tower = exponential_tower(
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )

    with pytest.raises(ValueError, match="status is unknown"):
        solve_hyperexponential_risch_differential_equation(
            1,
            1,
            tower.derivation(),
        )


@given(
    normal_offset=st.integers(min_value=1, max_value=3),
    multiplicity=st.integers(min_value=1, max_value=3),
    special_power=st.integers(min_value=0, max_value=2),
    numerator_coefficients=st.lists(
        st.tuples(
            st.integers(min_value=-2, max_value=2),
            st.integers(min_value=-2, max_value=2),
        ),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=30, deadline=None)
def test_generated_rational_solutions_match_the_matrix_oracle(
    normal_offset,
    multiplicity,
    special_power,
    numerator_coefficients,
) -> None:
    tower = exponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    numerator = sum(
        (constant + linear*x)*t**power
        for power, (constant, linear) in enumerate(numerator_coefficients)
    )
    assume(numerator != 0)
    top_denominator = (t + normal_offset)**multiplicity
    planted = field.from_sympy(
        numerator/(t**special_power*top_denominator)
    )
    coefficient = field.from_sympy(t + 1/(t + normal_offset))
    right_hand_side = derivation(planted) + coefficient*planted

    production = solve_hyperexponential_risch_differential_equation(
        coefficient,
        right_hand_side,
        derivation,
    )
    oracle = solve_bounded_hyperexponential_laurent_rde_matrix(
        coefficient,
        right_hand_side,
        derivation,
        1,
        1,
        -special_power,
        2 - special_power,
        top_denominator,
    )

    assert production.has_solution
    assert production.solution == planted
    assert oracle.has_solution
    assert oracle.solution == planted
