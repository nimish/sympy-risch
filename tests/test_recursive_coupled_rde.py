"""Coupled Risch differential equations over primitive lower towers."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import I, Poly, sqrt, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    CoupledRDEObstruction,
    CoupledRischDifferentialEquationResult,
    DifferentialTower,
    MonomialExtension,
    solve_coupled_rational_rde,
    solve_coupled_risch_differential_equation,
    solve_hyperexponential_coupled_laurent_rde,
)
from tests.oracles.rde import (
    BoundedPrimitiveCoupledRDEMatrixResult,
    solve_bounded_primitive_coupled_rde_matrix,
)

x, s, u = symbols("x s u")


def primitive_tower() -> DifferentialTower:
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    return rational.extend(
        MonomialExtension(
            Poly(1 / x, s, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_generic_solver_agrees_with_the_specialized_rational_solver() -> None:
    tower = primitive_tower()
    rational_derivation = tower.derivation(0)
    field = rational_derivation.field
    lambda_coefficient = field.from_sympy((x + 1) / (x**2 + 1))
    planted_real = field.from_sympy((x + 2) / (x + 1) ** 2)
    planted_imaginary = field.from_sympy((x - 1) / (x + 1) ** 2)
    real_rhs = (
        rational_derivation(planted_real) - lambda_coefficient * planted_imaginary
    )
    imaginary_rhs = (
        rational_derivation(planted_imaginary) + lambda_coefficient * planted_real
    )

    generic = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        rational_derivation,
    )
    specialized = solve_coupled_rational_rde(
        field.to_sympy(lambda_coefficient),
        field.to_sympy(real_rhs),
        field.to_sympy(imaginary_rhs),
        x,
    )

    assert generic.has_solution
    assert generic.reconstructs
    assert generic.real_solution == field.from_sympy(specialized.real_solution)
    assert generic.imaginary_solution == field.from_sympy(
        specialized.imaginary_solution
    )


def test_recursive_solver_handles_rational_primitive_coefficients() -> None:
    tower = primitive_tower()
    field = tower.field()
    derivation = tower.derivation()
    lambda_coefficient = field.from_sympy(2 * s)
    planted_real = field.from_sympy((x + s) / (s + 1) ** 2)
    planted_imaginary = field.from_sympy((x * s + 1) / (s + 1) ** 2)
    real_rhs = derivation(planted_real) - lambda_coefficient * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + lambda_coefficient * planted_real

    result = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert isinstance(result, CoupledRischDifferentialEquationResult)
    assert result.has_solution
    assert result.reconstructs
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary
    assert result.scalar_result is not None
    assert result.scalar_result.reconstructs


def test_recursive_solver_handles_two_primitive_levels() -> None:
    lower = primitive_tower()
    tower = lower.extend(
        MonomialExtension(
            Poly(1 / (x * s), u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    lambda_coefficient = field.from_sympy(2 * (s + u))
    planted_real = field.from_sympy((x + s * u) / (u + 1) ** 2)
    planted_imaginary = field.from_sympy((s + u) / (u + 1) ** 2)
    real_rhs = derivation(planted_real) - lambda_coefficient * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + lambda_coefficient * planted_real

    result = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary


def test_recursive_denominator_obstruction_is_retained() -> None:
    tower = primitive_tower()
    field = tower.field()

    result = solve_coupled_risch_differential_equation(
        field.from_sympy(2 * s),
        field.from_sympy(1 / (s + 1)),
        field.zero,
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.obstruction is CoupledRDEObstruction.DENOMINATOR
    assert result.scalar_result is not None
    assert result.scalar_result.obstruction is not None


def test_direct_real_matrix_oracle_recovers_the_same_pair() -> None:
    tower = primitive_tower()
    field = tower.field()
    derivation = tower.derivation()
    lambda_coefficient = field.from_sympy(2 * s)
    planted_real = field.from_sympy((x + s) / (s + 1) ** 2)
    planted_imaginary = field.from_sympy((x * s + 1) / (s + 1) ** 2)
    real_rhs = derivation(planted_real) - lambda_coefficient * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + lambda_coefficient * planted_real

    production = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
    )
    oracle = solve_bounded_primitive_coupled_rde_matrix(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
        1,
        1,
        1,
        (s + 1) ** 2,
    )

    assert production.has_solution
    assert isinstance(oracle, BoundedPrimitiveCoupledRDEMatrixResult)
    assert oracle.has_solution
    assert oracle.real_solution == production.real_solution == planted_real
    assert (
        oracle.imaginary_solution == production.imaginary_solution == planted_imaginary
    )
    assert oracle.coefficient_rank == oracle.augmented_rank


def test_direct_real_matrix_oracle_confirms_a_denominator_failure() -> None:
    tower = primitive_tower()
    field = tower.field()
    lambda_coefficient = field.from_sympy(2 * s)
    real_rhs = field.from_sympy(1 / (s + 1))

    production = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        field.zero,
        tower.derivation(),
    )
    oracle = solve_bounded_primitive_coupled_rde_matrix(
        lambda_coefficient,
        real_rhs,
        field.zero,
        tower.derivation(),
        x**2 + 1,
        3,
        4,
        (s + 1) ** 3,
    )

    assert not production.has_solution
    assert not oracle.has_solution
    assert oracle.augmented_rank > oracle.coefficient_rank


@pytest.mark.parametrize(
    ("lower_bound", "top_bound", "message"),
    [
        (-1, 1, "lower numerator"),
        (True, 1, "lower numerator"),
        (1, -1, "top"),
        (1, False, "top"),
    ],
)
def test_direct_oracle_rejects_invalid_bounds(
    lower_bound,
    top_bound,
    message,
) -> None:
    tower = primitive_tower()

    with pytest.raises(ValueError, match=message):
        solve_bounded_primitive_coupled_rde_matrix(
            1,
            0,
            0,
            tower.derivation(),
            1,
            lower_bound,
            top_bound,
        )


def test_recursive_solver_handles_a_hyperexponential_level() -> None:
    primitive = primitive_tower()
    hyperexponential = primitive.extend(
        MonomialExtension(
            Poly(s * u, u, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = hyperexponential.field()
    derivation = hyperexponential.derivation()
    planted_real = field.from_sympy((x + s) * u**2)
    planted_imaginary = field.from_sympy((x * s + 1) / u)
    lambda_coefficient = field.from_sympy(2 * s)
    real_rhs = derivation(planted_real) - lambda_coefficient * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + lambda_coefficient * planted_real

    result = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary
    assert result.direct_result is not None


@pytest.mark.parametrize(
    "top_derivative",
    (
        None,
        1 / x,
        s,
        1 + s**2,
    ),
    ids=("rational", "primitive", "hyperexponential", "hypertangent"),
)
def test_recursive_solver_uses_the_exact_gaussian_constant_compositum(
    top_derivative,
) -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    if top_derivative is not None:
        tower = tower.extend(
            MonomialExtension(
                Poly(top_derivative, s, domain=tower.field()),
                ConstantFieldStatus.PRESERVED,
            )
        )

    field = tower.field()
    derivation = tower.derivation()
    if top_derivative is None:
        planted_real = field.from_sympy(x)
        planted_imaginary = field.zero
        rotation = field.from_sympy(sqrt(2) / (x + 1))
    else:
        planted_real = field.from_sympy(s)
        planted_imaginary = field.zero
        # A top-generator rotation bypasses the direct lower-coefficient
        # recurrences and exercises the C(i) scalar fallback in every family.
        rotation = field.from_sympy(s)
    real_rhs = derivation(planted_real) - rotation * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + rotation * planted_real

    result = solve_coupled_risch_differential_equation(
        rotation,
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary


def test_finite_constant_recursive_system_keeps_the_diagonal_coefficient() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    base = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = base.extend(
        MonomialExtension(
            Poly(x * s, s, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    diagonal = field.from_sympy(sqrt(2) / (x + 1))
    rotation = field.from_sympy(x / (x + 1))
    planted_real = field.from_sympy(s)
    planted_imaginary = field.from_sympy(2 * s)
    real_rhs = (
        derivation(planted_real)
        + diagonal * planted_real
        - rotation * planted_imaginary
    )
    imaginary_rhs = (
        derivation(planted_imaginary)
        + rotation * planted_real
        + diagonal * planted_imaginary
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


def test_real_coupled_solver_rejects_a_nonreal_source_constant_field() -> None:
    tower = DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()

    with pytest.raises(NotImplementedError, match="provably real"):
        solve_coupled_risch_differential_equation(
            field.one,
            field.from_sympy(I),
            field.from_sympy(I * x),
            tower.derivation(),
        )


@given(
    real_coefficients=st.lists(
        st.tuples(
            st.integers(min_value=-2, max_value=2),
            st.integers(min_value=-2, max_value=2),
        ),
        min_size=2,
        max_size=2,
    ),
    imaginary_coefficients=st.lists(
        st.tuples(
            st.integers(min_value=-2, max_value=2),
            st.integers(min_value=-2, max_value=2),
        ),
        min_size=2,
        max_size=2,
    ),
    denominator_power=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=30, deadline=None)
def test_generated_recursive_solutions_match_the_real_matrix_oracle(
    real_coefficients,
    imaginary_coefficients,
    denominator_power,
) -> None:
    tower = primitive_tower()
    field = tower.field()
    derivation = tower.derivation()
    real_numerator = sum(
        (constant + linear * x) * s**degree
        for degree, (constant, linear) in enumerate(real_coefficients)
    )
    imaginary_numerator = sum(
        (constant + linear * x) * s**degree
        for degree, (constant, linear) in enumerate(imaginary_coefficients)
    )
    assume(real_numerator != 0 or imaginary_numerator != 0)
    top_denominator = (s + 1) ** denominator_power
    planted_real = field.from_sympy(real_numerator / top_denominator)
    planted_imaginary = field.from_sympy(imaginary_numerator / top_denominator)
    lambda_coefficient = field.from_sympy(2 * s)
    real_rhs = derivation(planted_real) - lambda_coefficient * planted_imaginary
    imaginary_rhs = derivation(planted_imaginary) + lambda_coefficient * planted_real

    production = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
    )
    oracle = solve_bounded_primitive_coupled_rde_matrix(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        derivation,
        1,
        1,
        1,
        top_denominator,
    )

    assert production.has_solution
    assert production.real_solution == planted_real
    assert production.imaginary_solution == planted_imaginary
    assert production.reconstructs
    assert oracle.has_solution
    assert oracle.real_solution == planted_real
    assert oracle.imaginary_solution == planted_imaginary


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
@settings(max_examples=35, deadline=None)
def test_generated_arbitrary_polynomial_pairs_match_the_oracle(
    real_coefficients,
    imaginary_coefficients,
) -> None:
    tower = primitive_tower()
    field = tower.field()
    real_rhs_expression = sum(
        coefficient * s**degree for degree, coefficient in enumerate(real_coefficients)
    )
    imaginary_rhs_expression = sum(
        coefficient * s**degree
        for degree, coefficient in enumerate(imaginary_coefficients)
    )
    assume(real_rhs_expression != 0 or imaginary_rhs_expression != 0)
    lambda_coefficient = field.from_sympy(2 * s)
    real_rhs = field.from_sympy(real_rhs_expression)
    imaginary_rhs = field.from_sympy(imaginary_rhs_expression)

    production = solve_coupled_risch_differential_equation(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        tower.derivation(),
    )
    oracle = solve_bounded_primitive_coupled_rde_matrix(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        tower.derivation(),
        1,
        0,
        2,
    )

    assert production.has_solution is oracle.has_solution
    if production.has_solution:
        assert production.reconstructs
        assert oracle.has_solution
    else:
        assert oracle.augmented_rank > oracle.coefficient_rank
