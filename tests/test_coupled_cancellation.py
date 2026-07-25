"""Direct Chapter 8 coupled-cancellation recurrences."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, linear_eq_to_matrix, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    CoupledHyperexponentialLaurentRDEResult,
    CoupledHypertangentPolynomialRDEMode,
    CoupledRDEObstruction,
    DifferentialTower,
    MonomialExtension,
    solve_coupled_risch_differential_equation,
    solve_hyperexponential_coupled_laurent_rde,
    solve_hypertangent_coupled_cancellation_rde,
    solve_hypertangent_coupled_polynomial_rde,
)

x, e, t = symbols("x e t")


def rational_tower() -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )


def hyperexponential_tower() -> DifferentialTower:
    lower = rational_tower()
    return lower.extend(
        MonomialExtension(
            Poly(e, e, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def hypertangent_tower() -> DifferentialTower:
    lower = rational_tower()
    return lower.extend(
        MonomialExtension(
            Poly(1 + t**2, t, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def bounded_polynomial_pair_has_solution(
    tower: DifferentialTower,
    diagonal,
    rotation,
    real_rhs,
    imaginary_rhs,
    *,
    lower_degree: int,
    top_degree: int,
) -> bool:
    """Independent global real block-matrix oracle for one monomial level."""
    extension = tower.levels[-1]
    top_generator = extension.generator
    generator_derivative = extension.generator_derivative.as_expr()
    basis = [
        x**lower_power*top_generator**top_power
        for top_power in range(top_degree + 1)
        for lower_power in range(lower_degree + 1)
    ]
    unknowns = symbols(f"c0:{2*len(basis)}")
    split = len(basis)
    real_candidate = sum(
        coefficient*monomial
        for coefficient, monomial in zip(
            unknowns[:split],
            basis,
            strict=True,
        )
    )
    imaginary_candidate = sum(
        coefficient*monomial
        for coefficient, monomial in zip(
            unknowns[split:],
            basis,
            strict=True,
        )
    )
    def differentiate(expression):
        return (
            expression.diff(x)
            + generator_derivative*expression.diff(top_generator)
        )

    real_residual = (
        differentiate(real_candidate)
        + diagonal*real_candidate
        - rotation*imaginary_candidate
        - real_rhs
    )
    imaginary_residual = (
        differentiate(imaginary_candidate)
        + rotation*real_candidate
        + diagonal*imaginary_candidate
        - imaginary_rhs
    )
    equations = []
    for residual in (real_residual, imaginary_residual):
        numerator = cancel(residual).as_numer_denom()[0]
        equations.extend(Poly(numerator, x, top_generator).coeffs())
    if not equations:
        return True
    coefficient_matrix, right_hand_side = linear_eq_to_matrix(
        equations,
        unknowns,
    )
    return (
        coefficient_matrix.rank()
        == coefficient_matrix.row_join(right_hand_side).rank()
    )


def test_bronstein_example_8_4_1() -> None:
    """Reproduce the worked cancellation in Chapter 8, pages 261-263."""
    tower = hypertangent_tower()

    result = solve_hypertangent_coupled_cancellation_rde(
        0,
        4*x,
        -t**2 + 2*t - 8*x**2 + 1,
        2*(1 - 2*x),
        tower.derivation(),
        2,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.mode is CoupledHypertangentPolynomialRDEMode.CANCELLATION
    assert result.real_solution == Poly(t - 1, t, domain=tower.field(0))
    assert result.imaginary_solution == Poly(2*x, t, domain=tower.field(0))
    assert result.boundary_result is not None
    assert result.recursive_result is not None
    assert bounded_polynomial_pair_has_solution(
        tower,
        -2*t,
        4*x,
        -t**2 + 2*t - 8*x**2 + 1,
        2*(1 - 2*x),
        lower_degree=2,
        top_degree=2,
    )


def test_hypertangent_no_cancellation_recovers_a_planted_pair() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    diagonal = field.from_sympy(x)
    rotation = field.from_sympy(x + 1)
    planted_real = field.from_sympy((x + 1)*t**3 + t)
    planted_imaginary = field.from_sympy(x*t**2 - 2)
    real_rhs = (
        derivation(planted_real)
        + diagonal*planted_real
        - rotation*planted_imaginary
    )
    imaginary_rhs = (
        derivation(planted_imaginary)
        + rotation*planted_real
        + diagonal*planted_imaginary
    )

    result = solve_hypertangent_coupled_polynomial_rde(
        diagonal,
        rotation,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert (
        result.mode
        is CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION
    )
    assert result.real_solution is not None
    assert result.imaginary_solution is not None
    assert field.from_sympy(result.real_solution.as_expr()) == planted_real
    assert (
        field.from_sympy(result.imaginary_solution.as_expr())
        == planted_imaginary
    )
    assert result.corrections


def test_hypertangent_cancellation_rejects_rhs_above_the_bound() -> None:
    tower = hypertangent_tower()

    result = solve_hypertangent_coupled_cancellation_rde(
        0,
        1,
        t**3,
        0,
        tower.derivation(),
        2,
    )

    assert not result.has_solution
    assert result.reconstructs
    assert result.obstruction is CoupledRDEObstruction.DEGREE


@pytest.mark.parametrize("degree_bound", [-1, True, 1.5])
def test_hypertangent_cancellation_rejects_invalid_degree_bound(
    degree_bound,
) -> None:
    tower = hypertangent_tower()

    with pytest.raises(ValueError, match="nonnegative int"):
        solve_hypertangent_coupled_cancellation_rde(
            0,
            1,
            0,
            0,
            tower.derivation(),
            degree_bound,
        )


def test_tangent_cancellation_handles_gaussian_unit_normalization() -> None:
    """Regression for algebraically equal, noncanonically stored QQ_I units."""
    tower = hypertangent_tower()

    result = solve_hypertangent_coupled_cancellation_rde(
        1,
        2,
        0,
        t**2,
        tower.derivation(),
        2,
    )

    assert result.reconstructs
    assert result.has_solution is bounded_polynomial_pair_has_solution(
        tower,
        1 - 2*t,
        2,
        0,
        t**2,
        lower_degree=0,
        top_degree=2,
    )


def test_hypertangent_polynomial_solver_rejects_nonlower_rotation() -> None:
    tower = hypertangent_tower()

    with pytest.raises(NotImplementedError, match="rotation coefficient"):
        solve_hypertangent_coupled_polynomial_rde(
            0,
            t,
            1,
            0,
            tower.derivation(),
        )


def test_hypertangent_polynomial_solver_rejects_top_generator_poles() -> None:
    tower = hypertangent_tower()

    with pytest.raises(NotImplementedError, match="top-generator poles"):
        solve_hypertangent_coupled_polynomial_rde(
            0,
            1,
            1/(t + 1),
            0,
            tower.derivation(),
        )


def test_hyperexponential_laurent_recurrence_recovers_mixed_modes() -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    diagonal = field.from_sympy(x)
    rotation = field.from_sympy(x + 1)
    planted_real = field.from_sympy((x + 1)*e**2 + (x - 1)/e)
    planted_imaginary = field.from_sympy(x*e**2 + 2/e)
    real_rhs = (
        derivation(planted_real)
        + diagonal*planted_real
        - rotation*planted_imaginary
    )
    imaginary_rhs = (
        derivation(planted_imaginary)
        + rotation*planted_real
        + diagonal*planted_imaginary
    )

    result = solve_hyperexponential_coupled_laurent_rde(
        diagonal,
        rotation,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary
    assert [step.power for step in result.steps] == [-1, 2]
    assert all(step.has_solution for step in result.steps)


def test_hyperexponential_laurent_recurrence_retains_lower_obstruction() -> None:
    tower = hyperexponential_tower()

    result = solve_hyperexponential_coupled_laurent_rde(
        0,
        1,
        1/(x + 1),
        0,
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.reconstructs
    assert result.obstruction is CoupledRDEObstruction.LOWER_FIELD
    assert len(result.steps) == 1
    assert not result.steps[0].has_solution


def test_hyperexponential_direct_recurrence_rejects_nonlower_matrix_entry() -> None:
    tower = hyperexponential_tower()

    with pytest.raises(NotImplementedError, match="rotation coefficient"):
        solve_hyperexponential_coupled_laurent_rde(
            0,
            e,
            1,
            0,
            tower.derivation(),
        )


def test_generic_dispatch_uses_direct_hyperexponential_recurrence() -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted_real = field.from_sympy((x + 1)*e**2)
    planted_imaginary = field.from_sympy((x - 1)/e)
    rotation = field.from_sympy(x + 1)
    real_rhs = derivation(planted_real) - rotation*planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + rotation*planted_real

    result = solve_coupled_risch_differential_equation(
        rotation,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.scalar_result is None
    assert isinstance(
        result.direct_result,
        CoupledHyperexponentialLaurentRDEResult,
    )


def test_non_laurent_hyperexponential_case_reports_missing_lower_decision() -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    rotation = field.from_sympy((e + 1)/(e - 1))
    planted_real = field.from_sympy((x + e)/(e - 1)**2)
    planted_imaginary = field.from_sympy((x*e + 1)/(e - 1)**2)
    real_rhs = derivation(planted_real) - rotation*planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + rotation*planted_real

    with pytest.raises(
        NotImplementedError,
        match="parametric logarithmic-derivative decision",
    ):
        solve_coupled_risch_differential_equation(
            rotation,
            real_rhs,
            imaginary_rhs,
            derivation,
        )


def test_generic_dispatch_uses_direct_hypertangent_polynomial_solver() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    derivation = tower.derivation()
    rotation = field.from_sympy(2)
    planted_real = field.from_sympy(t**2 + x)
    planted_imaginary = field.from_sympy(t**3 - 1)
    real_rhs = derivation(planted_real) - rotation*planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + rotation*planted_real

    result = solve_coupled_risch_differential_equation(
        rotation,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.direct_result is not None
    assert result.scalar_result is None


@given(
    real_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=4,
        max_size=4,
    ),
    imaginary_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=4,
        max_size=4,
    ),
)
@settings(max_examples=30, deadline=None)
def test_generated_hypertangent_decisions_match_global_matrix_oracle(
    real_coefficients,
    imaginary_coefficients,
) -> None:
    tower = hypertangent_tower()
    real_rhs = sum(
        coefficient*t**degree
        for degree, coefficient in enumerate(real_coefficients)
    )
    imaginary_rhs = sum(
        coefficient*t**degree
        for degree, coefficient in enumerate(imaginary_coefficients)
    )
    assume(real_rhs != 0 or imaginary_rhs != 0)

    production = solve_hypertangent_coupled_polynomial_rde(
        1,
        2,
        real_rhs,
        imaginary_rhs,
        tower.derivation(),
    )
    oracle_has_solution = bounded_polynomial_pair_has_solution(
        tower,
        1,
        2,
        real_rhs,
        imaginary_rhs,
        lower_degree=0,
        top_degree=3,
    )

    assert production.has_solution is oracle_has_solution
    assert production.reconstructs


@given(
    real_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=4,
        max_size=4,
    ),
    imaginary_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=4,
        max_size=4,
    ),
)
@settings(max_examples=30, deadline=None)
def test_generated_hyperexponential_modes_match_global_matrix_oracle(
    real_coefficients,
    imaginary_coefficients,
) -> None:
    tower = hyperexponential_tower()
    real_rhs = sum(
        coefficient*e**degree
        for degree, coefficient in enumerate(real_coefficients)
    )
    imaginary_rhs = sum(
        coefficient*e**degree
        for degree, coefficient in enumerate(imaginary_coefficients)
    )
    assume(real_rhs != 0 or imaginary_rhs != 0)

    production = solve_hyperexponential_coupled_laurent_rde(
        1,
        2,
        real_rhs,
        imaginary_rhs,
        tower.derivation(),
    )
    oracle_has_solution = bounded_polynomial_pair_has_solution(
        tower,
        1,
        2,
        real_rhs,
        imaginary_rhs,
        lower_degree=0,
        top_degree=3,
    )

    assert production.has_solution is oracle_has_solution
    assert production.reconstructs


@given(
    real_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=3,
        max_size=3,
    ),
    imaginary_coefficients=st.lists(
        st.integers(min_value=-2, max_value=2),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=25, deadline=None)
def test_generated_tangent_cancellation_matches_global_matrix_oracle(
    real_coefficients,
    imaginary_coefficients,
) -> None:
    tower = hypertangent_tower()
    real_rhs = sum(
        coefficient*t**degree
        for degree, coefficient in enumerate(real_coefficients)
    )
    imaginary_rhs = sum(
        coefficient*t**degree
        for degree, coefficient in enumerate(imaginary_coefficients)
    )
    assume(real_rhs != 0 or imaginary_rhs != 0)

    production = solve_hypertangent_coupled_cancellation_rde(
        1,
        2,
        real_rhs,
        imaginary_rhs,
        tower.derivation(),
        2,
    )
    oracle_has_solution = bounded_polynomial_pair_has_solution(
        tower,
        1 - 2*t,
        2,
        real_rhs,
        imaginary_rhs,
        lower_degree=0,
        top_degree=2,
    )

    assert production.has_solution is oracle_has_solution
    assert production.reconstructs
