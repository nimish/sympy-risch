"""Real coupled rational Risch differential equations."""

from itertools import product

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import PolynomialError

from sympy_risch import (
    CoupledRDEObstruction,
    solve_bounded_coupled_rde_matrix,
    solve_coupled_rational_rde,
)


x = symbols("x")


@pytest.mark.parametrize(
    ("lambda_coefficient", "real_solution", "imaginary_solution"),
    [
        (2, x + 1, 1/x),
        (4*x, (x + 1)/x, x**2),
        (x**2 + 1, 1/(x - 1)**2, (x**2 + 1)/(x - 1)),
        (-3*x + 2, x**3 - 1/x**2, (2*x - 1)/(x + 1)**3),
    ],
)
def test_coupled_recurrence_recovers_planted_rational_solutions(
    lambda_coefficient,
    real_solution,
    imaginary_solution,
) -> None:
    real_rhs = cancel(
        real_solution.diff(x) - lambda_coefficient*imaginary_solution
    )
    imaginary_rhs = cancel(
        imaginary_solution.diff(x) + lambda_coefficient*real_solution
    )

    result = solve_coupled_rational_rde(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        x,
    )

    assert result.has_solution
    assert result.obstruction is None
    assert cancel(result.real_solution - real_solution) == 0
    assert cancel(result.imaginary_solution - imaginary_solution) == 0
    assert cancel(
        result.real_solution.diff(x)
        - lambda_coefficient*result.imaginary_solution
        - real_rhs
    ) == 0
    assert cancel(
        result.imaginary_solution.diff(x)
        + lambda_coefficient*result.real_solution
        - imaginary_rhs
    ) == 0


def test_zero_right_hand_side_has_the_unique_zero_solution() -> None:
    result = solve_coupled_rational_rde(x**2 + 1, 0, 0, x)

    assert result.has_solution
    assert result.real_solution == 0
    assert result.imaginary_solution == 0
    assert result.denominator_bound == Poly(1, x, domain=QQ)


def test_simple_pole_is_a_denominator_obstruction() -> None:
    result = solve_coupled_rational_rde(2, 1/x, 0, x)

    assert not result.has_solution
    assert result.obstruction is CoupledRDEObstruction.DENOMINATOR
    assert result.has_denominator_obstruction
    assert not result.has_degree_obstruction
    assert not result.has_nonzero_remainder_obstruction
    assert result.denominator_bound == Poly(1, x, domain=QQ)
    assert result.obstruction_polynomial == Poly(x, x, domain=QQ)
    assert result.obstruction_remainder == Poly(1, x, domain=QQ)


def test_leading_degree_can_prove_nonexistence() -> None:
    result = solve_coupled_rational_rde(x, 1, 0, x)

    assert not result.has_solution
    assert result.obstruction is CoupledRDEObstruction.DEGREE
    assert result.has_degree_obstruction
    assert result.degree_bound == -1
    assert result.real_residual == Poly(1, x, domain=QQ)
    assert result.imaginary_residual.is_zero


def test_recurrence_retains_an_uncancellable_polynomial_remainder() -> None:
    result = solve_coupled_rational_rde(x, 0, x**2, x)

    assert not result.has_solution
    assert result.obstruction is CoupledRDEObstruction.NONZERO_REMAINDER
    assert result.has_nonzero_remainder_obstruction
    assert result.degree_bound == 1
    assert result.real_residual == Poly(-1, x, domain=QQ)
    assert result.imaginary_residual.is_zero
    h = result.denominator_bound
    rotation = result.lambda_coefficient*h
    assert (
        h*result.partial_real_numerator.diff()
        - h.diff()*result.partial_real_numerator
        - rotation*result.partial_imaginary_numerator
        + result.real_residual
        == Poly(0, x, domain=QQ)
    )
    assert (
        h*result.partial_imaginary_numerator.diff()
        - h.diff()*result.partial_imaginary_numerator
        + rotation*result.partial_real_numerator
        + result.imaginary_residual
        == Poly(x**2, x, domain=QQ)
    )


def test_bounded_block_matrix_oracle_reports_exact_ranks() -> None:
    solution = solve_bounded_coupled_rde_matrix(
        2,
        1 - 2/x,
        2*x + 2 - 1/x**2,
        x,
        x,
        2,
    )
    obstruction = solve_bounded_coupled_rde_matrix(
        x,
        0,
        x**2,
        x,
        1,
        1,
    )

    assert solution.has_solution
    assert cancel(solution.real_solution - (x + 1)) == 0
    assert cancel(solution.imaginary_solution - 1/x) == 0
    assert solution.coefficient_rank == solution.augmented_rank
    assert not obstruction.has_solution
    assert obstruction.augmented_rank > obstruction.coefficient_rank


integer_coefficients = st.lists(
    st.integers(min_value=-3, max_value=3),
    min_size=1,
    max_size=4,
)


@given(
    lambda_coefficients=integer_coefficients,
    real_coefficients=integer_coefficients,
    imaginary_coefficients=integer_coefficients,
    denominator_power=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=55, deadline=None)
def test_generated_planted_solutions_are_recovered(
    lambda_coefficients,
    real_coefficients,
    imaginary_coefficients,
    denominator_power,
) -> None:
    lambda_coefficient = sum(
        coefficient*x**degree
        for degree, coefficient in enumerate(lambda_coefficients)
    )
    assume(lambda_coefficient != 0)
    denominator = (x**2 + 1)**denominator_power
    real_solution = cancel(
        sum(
            coefficient*x**degree
            for degree, coefficient in enumerate(real_coefficients)
        )
        / denominator
    )
    imaginary_solution = cancel(
        sum(
            coefficient*x**degree
            for degree, coefficient in enumerate(imaginary_coefficients)
        )
        / denominator
    )
    real_rhs = cancel(
        real_solution.diff(x) - lambda_coefficient*imaginary_solution
    )
    imaginary_rhs = cancel(
        imaginary_solution.diff(x) + lambda_coefficient*real_solution
    )

    result = solve_coupled_rational_rde(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        x,
    )

    assert result.has_solution
    assert cancel(result.real_solution - real_solution) == 0
    assert cancel(result.imaginary_solution - imaginary_solution) == 0


@pytest.mark.parametrize(
    ("lambda_coefficient", "real_rhs", "imaginary_rhs"),
    [
        (lambda_coefficient, real_rhs, imaginary_rhs)
        for lambda_coefficient, real_rhs, imaginary_rhs in product(
            (1, x, x + 1, x**2 + 1),
            (0, 1, x, x**2, 1/x, 1/x**2),
            (0, 1, x, x**2, 1/x, 1/x**2),
        )
        if real_rhs != 0 or imaginary_rhs != 0
    ],
)
def test_recurrence_decision_agrees_with_independent_block_matrix(
    lambda_coefficient,
    real_rhs,
    imaginary_rhs,
) -> None:
    result = solve_coupled_rational_rde(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        x,
    )
    if result.obstruction is CoupledRDEObstruction.DENOMINATOR:
        return
    matrix = solve_bounded_coupled_rde_matrix(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        x,
        result.denominator_bound,
        max(0, result.degree_bound),
    )

    assert result.has_solution == matrix.has_solution
    if result.has_solution:
        assert cancel(result.real_solution - matrix.real_solution) == 0
        assert cancel(result.imaginary_solution - matrix.imaginary_solution) == 0


@pytest.mark.parametrize("bad_variable", [1, x + 1])
def test_coupled_solver_requires_a_symbol(bad_variable) -> None:
    with pytest.raises(TypeError, match="variable must be a Symbol"):
        solve_coupled_rational_rde(1, 0, 0, bad_variable)


def test_coupled_solver_requires_nonzero_polynomial_rotation() -> None:
    with pytest.raises(ValueError, match="must be nonzero"):
        solve_coupled_rational_rde(0, 1, 0, x)
    with pytest.raises(PolynomialError):
        solve_coupled_rational_rde(1/x, 1, 0, x)


@pytest.mark.parametrize("bad_bound", [-1, True, 1.5])
def test_block_matrix_requires_a_nonnegative_integer_bound(bad_bound) -> None:
    with pytest.raises(ValueError, match="nonnegative int"):
        solve_bounded_coupled_rde_matrix(1, 0, 0, x, 1, bad_bound)


def test_block_matrix_requires_a_nonzero_denominator() -> None:
    with pytest.raises(ValueError, match="denominator must be nonzero"):
        solve_bounded_coupled_rde_matrix(1, 0, 0, x, 0, 0)
