"""Complete rational-base parametric Risch differential equations."""

import pytest
from sympy import sqrt, symbols
from sympy.polys.domains import GF, QQ

from sympy_risch import solve_rational_parametric_risch_differential_equation

x, a = symbols("x a")


def test_combination_can_succeed_when_each_rhs_fails_separately() -> None:
    result = solve_rational_parametric_risch_differential_equation(
        0,
        (1 / x + 1, 1 / x),
        x,
    )

    assert result.has_nontrivial_parameter_solution
    assert len(result.parameter_basis) == 1
    assert result.parameter_basis[0].parameters == (QQ.one, -QQ.one)
    assert result.parameter_basis[0].solution == x
    assert len(result.homogeneous_basis) == 1
    assert result.homogeneous_basis[0].solution == 1


def test_nonzero_residue_proves_no_nontrivial_parameter_combination() -> None:
    result = solve_rational_parametric_risch_differential_equation(
        0,
        (1 / x,),
        x,
    )

    assert not result.has_nontrivial_parameter_solution
    assert result.parameter_basis == ()
    assert len(result.homogeneous_basis) == 1
    assert result.homogeneous_basis[0].solution == 1


def test_homogeneous_cancellation_degree_is_included_in_complete_bound() -> None:
    result = solve_rational_parametric_risch_differential_equation(
        -1 / x,
        (1,),
        x,
    )

    assert result.degree_bound == 1
    assert not result.has_nontrivial_parameter_solution
    assert len(result.homogeneous_basis) == 1
    assert result.homogeneous_basis[0].solution == x


def test_weak_normalization_transports_parameter_and_homogeneous_solutions() -> None:
    result = solve_rational_parametric_risch_differential_equation(
        1 / x,
        (2,),
        x,
    )

    assert result.normalization_factor.as_expr() == x
    assert result.parameter_basis[0].parameters == (QQ.one,)
    assert result.parameter_basis[0].solution == x
    assert result.homogeneous_basis[0].solution == 1 / x


def test_parametric_rde_uses_native_finite_algebraic_constants() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    result = solve_rational_parametric_risch_differential_equation(
        0,
        (1 / x + sqrt(2), 1 / x),
        x,
        domain=constants,
    )

    parameter_vector = result.parameter_basis[0]
    assert tuple(
        constants.to_sympy(parameter) for parameter in parameter_vector.parameters
    ) == (1, -1)
    assert parameter_vector.solution == sqrt(2) * x


def test_parametric_rde_accepts_native_fraction_field_inputs() -> None:
    field = QQ.frac_field(x)
    result = solve_rational_parametric_risch_differential_equation(
        field.zero,
        (field.from_sympy(1 / x),),
        x,
    )

    assert not result.has_nontrivial_parameter_solution


@pytest.mark.parametrize(
    "constants",
    (QQ.frac_field(a), QQ.algebraic_field(sqrt(2)).frac_field(a)),
)
def test_parametric_rde_over_rational_function_constants(constants) -> None:
    result = solve_rational_parametric_risch_differential_equation(
        0,
        (1 / x + a, 1 / x),
        x,
        domain=constants,
    )

    direction = result.parameter_basis[0]
    assert direction.parameters == (constants.one, -constants.one)
    assert direction.solution == a * x


def test_parametric_rde_rejects_an_empty_family_or_unsupported_constants() -> None:
    with pytest.raises(ValueError, match="at least one right-hand side"):
        solve_rational_parametric_risch_differential_equation(0, (), x)
    with pytest.raises(ValueError, match="exact rational constant field"):
        solve_rational_parametric_risch_differential_equation(
            0,
            (1,),
            x,
            domain=GF(5),
        )
