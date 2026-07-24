"""Polynomial-coefficient Risch differential equations."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    PolynomialRDEObstruction,
    solve_polynomial_risch_differential_equation,
)


x, t = symbols("x t")


@pytest.mark.parametrize(
    ("a", "b", "expected_solution", "expected_degree"),
    [
        (1, x, x - 1, 1),
        (x, x**2 + 1, x, 1),
        (2, 2*x**3 + 3*x**2 + 2*x + 1, x**3 + x, 3),
        (x**2 + 1, x**4 + x**2 + 2*x, x**2, 2),
        (-1, -x**2 + 2*x, x**2, 2),
    ],
)
def test_polynomial_rde_solves_basic_examples(
    a,
    b,
    expected_solution,
    expected_degree,
) -> None:
    result = solve_polynomial_risch_differential_equation(a, b, x)

    assert result.has_solution
    assert result.solution == Poly(expected_solution, x, domain=QQ)
    assert result.solution_expression == expected_solution
    assert result.degree_bound == expected_degree
    assert result.obstruction is None
    assert result.solution.diff() + result.a*result.solution == result.b
    assert result.coefficient_rank == expected_degree + 1
    assert result.augmented_rank == result.coefficient_rank


def test_polynomial_rde_integrates_x_times_exp_x_at_the_tower_level() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    exponential = MonomialExtension(
        Poly(t, t, domain=base.field()),
        ConstantFieldStatus.PRESERVED,
    )
    tower = base.extend(exponential)
    field = tower.field()
    integrand = field.from_sympy(x*t)

    result = solve_polynomial_risch_differential_equation(1, x, x)
    antiderivative = field.from_sympy(result.solution_expression*t)

    assert result.solution_expression == x - 1
    assert tower.derivation()(antiderivative) == integrand


def test_polynomial_rde_degree_obstruction_for_exp_x_squared() -> None:
    result = solve_polynomial_risch_differential_equation(2*x, 1, x)

    assert not result.has_solution
    assert result.solution is None
    assert result.solution_expression is None
    assert result.degree_bound == -1
    assert result.obstruction is PolynomialRDEObstruction.DEGREE
    assert result.has_degree_obstruction
    assert not result.has_inconsistent_system_obstruction
    assert result.coefficient_rank is None
    assert result.augmented_rank is None


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (x, 1),
        (x**2 + 1, x),
        (3*x**4 - 2, x**3 + 1),
    ],
)
def test_degree_smaller_than_a_proves_nonexistence(a, b) -> None:
    result = solve_polynomial_risch_differential_equation(a, b, x)

    assert not result.has_solution
    assert result.has_degree_obstruction
    assert result.degree_bound == -1


@pytest.mark.parametrize("candidate_degree", [1, 3, 5, 7])
def test_odd_monomial_targets_give_rank_inconsistency(
    candidate_degree,
) -> None:
    # For a=x, cancelling downward from the leading term of x**(d+1)
    # leaves a nonzero constant exactly when d is odd.
    b = x**(candidate_degree + 1)

    result = solve_polynomial_risch_differential_equation(x, b, x)

    assert not result.has_solution
    assert result.degree_bound == candidate_degree
    assert result.obstruction is PolynomialRDEObstruction.INCONSISTENT_SYSTEM
    assert result.has_inconsistent_system_obstruction
    assert not result.has_degree_obstruction
    assert result.coefficient_rank == candidate_degree + 1
    assert result.augmented_rank == result.coefficient_rank + 1


@pytest.mark.parametrize("candidate_degree", [0, 2, 4, 6])
def test_even_monomial_targets_have_polynomial_solutions(
    candidate_degree,
) -> None:
    b = x**(candidate_degree + 1)

    result = solve_polynomial_risch_differential_equation(x, b, x)

    assert result.has_solution
    assert result.degree_bound == candidate_degree
    assert result.solution.degree() == candidate_degree
    assert (
        result.solution.diff()
        + Poly(x, x, domain=QQ)*result.solution
        == Poly(b, x, domain=QQ)
    )


@pytest.mark.parametrize(
    ("b", "expected"),
    [
        (1, x),
        (x**2, x**3/3),
        (3*x**4 - 2*x + 7, 3*x**5/5 - x**2 + 7*x),
    ],
)
def test_zero_a_delegates_to_canonical_polynomial_integration(
    b,
    expected,
) -> None:
    result = solve_polynomial_risch_differential_equation(0, b, x)

    assert result.has_solution
    assert result.solution_expression == expected
    assert result.degree_bound == Poly(b, x).degree() + 1
    assert result.solution.diff() == result.b
    assert result.coefficient_rank is None
    assert result.augmented_rank is None


@pytest.mark.parametrize("a", [0, 1, x, x**4 - 3*x + 1])
def test_zero_right_hand_side_uses_the_zero_solution(a) -> None:
    result = solve_polynomial_risch_differential_equation(a, 0, x)

    assert result.has_solution
    assert result.solution.is_zero
    assert result.solution_expression == 0
    assert result.degree_bound == 0
    assert result.obstruction is None


@given(
    a_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=1,
        max_size=5,
    ),
    solution_coefficients=st.lists(
        st.integers(min_value=-4, max_value=4),
        min_size=1,
        max_size=6,
    ),
)
@settings(max_examples=55, deadline=None)
def test_generated_planted_polynomial_solutions_are_recovered_uniquely(
    a_coefficients,
    solution_coefficients,
) -> None:
    a = sum(
        coefficient*x**degree
        for degree, coefficient in enumerate(a_coefficients)
    )
    planted_solution = sum(
        coefficient*x**degree
        for degree, coefficient in enumerate(solution_coefficients)
    )
    assume(a != 0)
    b = Poly(planted_solution, x).diff().as_expr() + a*planted_solution

    result = solve_polynomial_risch_differential_equation(a, b, x)

    assert result.has_solution
    assert result.solution == Poly(planted_solution, x, domain=QQ)
    assert result.solution.diff() + result.a*result.solution == result.b
    if planted_solution != 0:
        assert result.degree_bound == Poly(planted_solution, x).degree()


def test_polynomial_rde_accepts_native_poly_inputs() -> None:
    a = Poly(x + 1, x, domain=QQ)
    solution = Poly(x**2 - 1, x, domain=QQ)
    b = solution.diff() + a*solution

    result = solve_polynomial_risch_differential_equation(a, b, x)

    assert result.solution == solution
    assert result.a == a
    assert result.b == b


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1/x, 1),
        (1, 1/x),
        (sqrt(2), 1),
        (1, sqrt(2)*x),
    ],
)
def test_polynomial_rde_rejects_inputs_outside_qq_x(a, b) -> None:
    with pytest.raises((CoercionFailed, PolynomialError)):
        solve_polynomial_risch_differential_equation(a, b, x)


@pytest.mark.parametrize("variable", [x + 1, 1, "x", None])
def test_polynomial_rde_rejects_nonsymbol_variables(variable) -> None:
    with pytest.raises(
        TypeError,
        match="differentiation variable must be a Symbol",
    ):
        solve_polynomial_risch_differential_equation(
            1,
            x,
            variable,
        )
