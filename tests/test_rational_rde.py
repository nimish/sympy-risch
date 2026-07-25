"""Risch differential equations over the rational base field."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, sin, sqrt, symbols
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    PolynomialRDEObstruction,
    RationalRDEObstruction,
    solve_bounded_rational_rde_matrix,
    solve_rational_risch_differential_equation,
)

x, t = symbols("x t")


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side", "expected"),
    [
        (1, 1/x - 1/x**2, 1/x),
        (1/x, 1, x/2),
        (0, 1/x**2, -1/x),
        (
            2*x/(x**2 + 1),
            1,
            (x**3 + 3*x)/(3*(x**2 + 1)),
        ),
        (1, x, x - 1),
    ],
)
def test_rational_rde_solves_basic_examples(
    coefficient,
    right_hand_side,
    expected,
) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert result.has_solution
    assert cancel(result.solution - expected) == 0
    assert cancel(
        result.solution.diff(x)
        + coefficient*result.solution
        - right_hand_side
    ) == 0
    assert result.obstruction is None
    assert result.denominator_bound is not None


def test_rational_rde_reconstructs_exponential_tower_derivative() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(t, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    coefficient = 1/x - 1/x**2

    result = solve_rational_risch_differential_equation(
        1,
        coefficient,
        x,
    )
    antiderivative = field.from_sympy(result.solution*t)
    integrand = field.from_sympy(coefficient*t)

    assert result.solution == 1/x
    assert tower.derivation()(antiderivative) == integrand


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side"),
    [
        (1, 1/x),
        (0, 1/x),
        (1, 1/(x**2 + 1)),
    ],
)
def test_normal_denominator_obstruction_proves_nonexistence(
    coefficient,
    right_hand_side,
) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert not result.has_solution
    assert result.obstruction is RationalRDEObstruction.DENOMINATOR
    assert result.has_denominator_obstruction
    assert result.denominator_bound is not None
    assert result.obstruction_polynomial is not None
    assert result.obstruction_remainder is not None
    assert not result.obstruction_remainder.is_zero


@pytest.mark.parametrize(
    (
        "coefficient",
        "right_hand_side",
        "expected_obstruction",
    ),
    [
        (-2, -2/x**2, RationalRDEObstruction.SPDE_DEGREE),
        (
            -2,
            (x**2 + 1)/x**2,
            RationalRDEObstruction.SPDE_DIVISIBILITY,
        ),
    ],
)
def test_spde_obstructions_are_reported(
    coefficient,
    right_hand_side,
    expected_obstruction,
) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert not result.has_solution
    assert result.obstruction is expected_obstruction
    assert result.has_spde_obstruction
    assert result.denominator_bound == Poly(x, x, domain=QQ)
    if expected_obstruction is RationalRDEObstruction.SPDE_DIVISIBILITY:
        assert result.obstruction_polynomial == Poly(x, x, domain=QQ)
        assert result.obstruction_remainder == Poly(-2, x, domain=QQ)


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side", "polynomial_obstruction"),
    [
        (x, 1, PolynomialRDEObstruction.DEGREE),
        (x, x**2, PolynomialRDEObstruction.NONZERO_REMAINDER),
    ],
)
def test_final_polynomial_obstruction_is_retained(
    coefficient,
    right_hand_side,
    polynomial_obstruction,
) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert not result.has_solution
    assert result.obstruction is RationalRDEObstruction.POLYNOMIAL
    assert result.has_polynomial_obstruction
    assert result.polynomial_result is not None
    assert result.polynomial_result.obstruction is polynomial_obstruction


@pytest.mark.parametrize(
    (
        "coefficient",
        "right_hand_side",
        "expected_normalizer",
        "expected_normalized_coefficient",
    ),
    [
        (1/x, 1, x, 0),
        (2*x/(x**2 + 1), 1, x**2 + 1, 0),
        (-1/x, 1, 1, -1/x),
        (1/(x**2 + 1), 1, 1, 1/(x**2 + 1)),
    ],
)
def test_weak_normalization_removes_only_positive_integer_residues(
    coefficient,
    right_hand_side,
    expected_normalizer,
    expected_normalized_coefficient,
) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert result.normalization_factor == Poly(
        expected_normalizer,
        x,
        domain=QQ,
    )
    assert cancel(
        result.normalized_coefficient - expected_normalized_coefficient
    ) == 0
    assert cancel(
        result.normalized_coefficient
        + result.normalization_factor.diff().as_expr()
        / result.normalization_factor.as_expr()
        - coefficient
    ) == 0


@given(
    x_residue=st.integers(min_value=-3, max_value=4),
    linear_residue=st.integers(min_value=-3, max_value=4),
    quadratic_residue=st.integers(min_value=-3, max_value=4),
    repeated_pole=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=40, deadline=None)
def test_generated_weak_normalizers_match_positive_residue_factors(
    x_residue,
    linear_residue,
    quadratic_residue,
    repeated_pole,
) -> None:
    coefficient = (
        x_residue/x
        + linear_residue/(x - 1)
        + quadratic_residue*2*x/(x**2 + 1)
        + repeated_pole/(x + 2)**2
    )
    expected = (
        x**max(0, x_residue)
        * (x - 1)**max(0, linear_residue)
        * (x**2 + 1)**max(0, quadratic_residue)
    )

    result = solve_rational_risch_differential_equation(
        coefficient,
        0,
        x,
    )

    assert result.normalization_factor == Poly(expected, x, domain=QQ)
    assert cancel(
        coefficient
        - result.normalized_coefficient
        - result.normalization_factor.diff().as_expr()
        / result.normalization_factor.as_expr()
    ) == 0


def test_degree_bound_includes_resonance_at_infinity() -> None:
    result = solve_rational_risch_differential_equation(
        -3/x,
        x**2,
        x,
    )

    assert not result.has_solution
    assert result.degree_bound == 3
    assert result.obstruction is RationalRDEObstruction.SPDE_DIVISIBILITY


@pytest.mark.parametrize(
    "coefficient",
    [0, 1, 1/x, 3/x + 1/(x - 1)**2, x/(x**2 + 1)],
)
def test_zero_right_hand_side_uses_canonical_zero_solution(coefficient) -> None:
    result = solve_rational_risch_differential_equation(
        coefficient,
        0,
        x,
    )

    assert result.has_solution
    assert result.solution == 0
    assert result.obstruction is None


def test_bounded_matrix_oracle_finds_rational_solution() -> None:
    result = solve_bounded_rational_rde_matrix(
        1,
        1/x - 1/x**2,
        x,
        x,
        0,
    )

    assert result.has_solution
    assert result.solution == 1/x
    assert result.coefficient_rank == result.augmented_rank == 1


def test_bounded_matrix_oracle_proves_ansatz_inconsistent() -> None:
    result = solve_bounded_rational_rde_matrix(
        1,
        1/x,
        x,
        x,
        4,
    )

    assert not result.has_solution
    assert result.augmented_rank == result.coefficient_rank + 1


def test_bounded_matrix_oracle_normalizes_homogeneous_freedom_to_zero() -> None:
    result = solve_bounded_rational_rde_matrix(
        1/x,
        0,
        x,
        x,
        1,
    )

    assert result.has_solution
    assert result.solution == 0
    assert result.coefficient_rank < result.numerator_degree_bound + 1
    assert result.augmented_rank == result.coefficient_rank


def test_rational_rde_accepts_native_polynomial_inputs() -> None:
    coefficient = Poly(1, x, domain=QQ)
    right_hand_side = Poly(x, x, domain=QQ)

    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )
    matrix = solve_bounded_rational_rde_matrix(
        coefficient,
        right_hand_side,
        x,
        Poly(1, x, domain=QQ),
        1,
    )

    assert result.solution == x - 1
    assert matrix.solution == x - 1


@given(
    coefficient_coefficients=st.lists(
        st.integers(min_value=-3, max_value=3),
        min_size=1,
        max_size=4,
    ),
    solution_coefficients=st.lists(
        st.integers(min_value=-3, max_value=3),
        min_size=1,
        max_size=5,
    ),
    coefficient_poles=st.tuples(
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=1),
    ),
    solution_poles=st.tuples(
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
    ),
)
@settings(max_examples=80, deadline=None)
def test_generated_planted_rational_solutions_are_found(
    coefficient_coefficients,
    solution_coefficients,
    coefficient_poles,
    solution_poles,
) -> None:
    factors = (x, x - 1, x**2 + 1)
    coefficient_numerator = sum(
        value*x**degree
        for degree, value in enumerate(coefficient_coefficients)
    )
    solution_numerator = sum(
        value*x**degree
        for degree, value in enumerate(solution_coefficients)
    )
    coefficient_denominator = prod_powers(factors, coefficient_poles)
    solution_denominator = prod_powers(factors, solution_poles)
    coefficient = cancel(
        coefficient_numerator/coefficient_denominator
    )
    planted_solution = cancel(
        solution_numerator/solution_denominator
    )
    right_hand_side = cancel(
        planted_solution.diff(x) + coefficient*planted_solution
    )

    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    assert result.has_solution
    assert cancel(
        result.solution.diff(x)
        + coefficient*result.solution
        - right_hand_side
    ) == 0


@given(
    coefficient_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=1,
        max_size=4,
    ),
    rhs_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=1,
        max_size=4,
    ),
    coefficient_poles=st.tuples(
        st.integers(min_value=0, max_value=1),
        st.integers(min_value=0, max_value=1),
    ),
    rhs_poles=st.tuples(
        st.integers(min_value=0, max_value=2),
        st.integers(min_value=0, max_value=2),
    ),
)
@settings(max_examples=80, deadline=None)
def test_full_solver_agrees_with_bounded_matrix_after_denominator_reduction(
    coefficient_coefficients,
    rhs_coefficients,
    coefficient_poles,
    rhs_poles,
) -> None:
    factors = (x, x - 1)
    coefficient = cancel(
        sum(
            value*x**degree
            for degree, value in enumerate(coefficient_coefficients)
        )
        / prod_powers(factors, coefficient_poles)
    )
    right_hand_side = cancel(
        sum(
            value*x**degree
            for degree, value in enumerate(rhs_coefficients)
        )
        / prod_powers(factors, rhs_poles)
    )
    result = solve_rational_risch_differential_equation(
        coefficient,
        right_hand_side,
        x,
    )

    if result.has_denominator_obstruction:
        assert not result.obstruction_remainder.is_zero
        return

    ansatz_denominator = (
        result.normalization_factor*result.denominator_bound
    )
    matrix = solve_bounded_rational_rde_matrix(
        coefficient,
        right_hand_side,
        x,
        ansatz_denominator,
        result.degree_bound,
    )

    assert result.has_solution is matrix.has_solution
    if result.has_solution:
        assert cancel(
            matrix.solution.diff(x)
            + coefficient*matrix.solution
            - right_hand_side
        ) == 0
    else:
        assert matrix.augmented_rank > matrix.coefficient_rank


def prod_powers(factors, powers):
    """Return the product of corresponding polynomial powers."""
    result = 1
    for factor, power in zip(factors, powers, strict=True):
        result *= factor**power
    return result


@pytest.mark.parametrize(
    ("coefficient", "right_hand_side"),
    [
        (sin(x), 1),
        (1, sin(x)),
        (sqrt(2), 1),
        (1, sqrt(2)/x),
    ],
)
def test_rational_rde_rejects_inputs_outside_qq_x(
    coefficient,
    right_hand_side,
) -> None:
    with pytest.raises((CoercionFailed, PolynomialError)):
        solve_rational_risch_differential_equation(
            coefficient,
            right_hand_side,
            x,
        )


@pytest.mark.parametrize("variable", [x + 1, 1, "x", None])
def test_rational_rde_rejects_nonsymbol_variables(variable) -> None:
    with pytest.raises(
        TypeError,
        match="differentiation variable must be a Symbol",
    ):
        solve_rational_risch_differential_equation(1, x, variable)


@pytest.mark.parametrize(
    ("denominator", "degree_bound", "exception", "message"),
    [
        (0, 1, ValueError, "denominator must be nonzero"),
        (x, -1, ValueError, "degree bound must be a nonnegative int"),
        (x, True, ValueError, "degree bound must be a nonnegative int"),
        (sqrt(2)*x, 1, CoercionFailed, None),
    ],
)
def test_bounded_matrix_oracle_rejects_invalid_ansatz(
    denominator,
    degree_bound,
    exception,
    message,
) -> None:
    context = pytest.raises(exception)
    if message is not None:
        context = pytest.raises(exception, match=message)
    with context:
        solve_bounded_rational_rde_matrix(
            1,
            x,
            x,
            denominator,
            degree_bound,
        )
