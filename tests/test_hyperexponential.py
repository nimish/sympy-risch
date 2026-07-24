"""Reduction in one hyperexponential monomial extension."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    RationalRDEObstruction,
    reduce_hyperexponential_laurent,
    reduce_hyperexponential_monomial,
    solve_bounded_rational_rde_matrix,
)


x, t = symbols("x t")


def hyperexponential_tower(
    eta,
    *,
    constant_field_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return base.extend(
        MonomialExtension(
            Poly(eta*t, t, domain=base.field()),
            constant_field_status,
        ),
    )


def test_laurent_reduction_handles_positive_and_negative_modes() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(
        (2*x + 3)*t**2
        + (-1/x - 1/x**2)/t
        + 1/(x + 1)
    )

    result = reduce_hyperexponential_laurent(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.exact_part == field.from_sympy(
        (x + 1)*t**2 + 1/(x*t)
    )
    assert result.remainder == field.from_sympy(1/(x + 1))
    assert result.lower_remainder == tower.field(0).from_sympy(1/(x + 1))
    assert tuple(step.power for step in result.steps) == (-1, 2)
    assert all(step.has_solution for step in result.steps)
    assert not result.has_obstruction
    assert result.reduced_to_lower_field


def test_laurent_reduction_retains_unsolvable_exponential_mode() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    integrand = field.from_sympy(t/x)

    result = reduce_hyperexponential_laurent(
        integrand,
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.remainder == integrand
    assert result.lower_remainder == tower.field(0).zero
    assert result.has_obstruction
    assert not result.reduced_to_lower_field
    assert len(result.obstructed_steps) == 1
    step = result.obstructed_steps[0]
    assert step.power == 1
    assert step.exact_coefficient is None
    assert step.rde_result.obstruction is RationalRDEObstruction.DENOMINATOR


def test_laurent_reduction_returns_a_pure_lower_field_input() -> None:
    tower = hyperexponential_tower(2*x)
    field = tower.field()
    lower = tower.field(0).from_sympy((x + 1)/(x**2 + 1))
    integrand = field.convert(lower)

    result = reduce_hyperexponential_laurent(
        integrand,
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.remainder == integrand
    assert result.lower_remainder == lower
    assert result.steps == ()
    assert result.reduced_to_lower_field


def test_full_reduction_composes_normal_residues_and_laurent_modes() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(
        t/(t - 1)
        + (1/x - 1/x**2)*t
        + 1/(x + 1)
    )

    result = reduce_hyperexponential_monomial(
        integrand,
        derivation,
    )

    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )
    assert result.exact_part == field.from_sympy(t/x)
    assert result.logarithmic_derivative == field.from_sympy(t/(t - 1))
    assert result.remainder == field.from_sympy(1/(x + 1))
    assert result.lower_remainder == tower.field(0).from_sympy(1/(x + 1))
    assert result.laurent_reduction is not None
    assert not result.has_finite_residue_obstruction
    assert not result.has_laurent_obstruction
    assert not result.is_non_elementary
    assert result.reduced_to_lower_field


def test_full_reduction_handles_a_negative_special_power() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(t*(t - 1)))

    result = reduce_hyperexponential_monomial(
        integrand,
        derivation,
    )

    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )
    assert result.exact_part == field.from_sympy(1/t)
    assert result.logarithmic_derivative == field.from_sympy(t/(t - 1))
    assert result.remainder == field.from_sympy(-1)
    assert result.lower_remainder == tower.field(0).convert(-1)
    assert result.reduced_to_lower_field


def test_full_reduction_extracts_repeated_normal_pole_derivative() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact_part = field.from_sympy(1/(t*(t - 1)**2))
    integrand = derivation(planted_exact_part)

    result = reduce_hyperexponential_monomial(
        integrand,
        derivation,
    )

    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )
    assert result.logarithmic_derivative == field.zero
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert not result.is_non_elementary
    assert result.reduced_to_lower_field


def test_full_reduction_preserves_finite_residue_obstruction() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1/(x*(t - 1)))

    result = reduce_hyperexponential_monomial(
        integrand,
        derivation,
    )

    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )
    assert result.has_finite_residue_obstruction
    assert result.laurent_reduction is None
    assert result.lower_remainder is None
    assert result.is_non_elementary
    assert not result.reduced_to_lower_field


def test_full_reduction_proves_exp_x_over_x_mode_non_elementary() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    integrand = field.from_sympy(t/x)

    result = reduce_hyperexponential_monomial(
        integrand,
        tower.derivation(),
    )

    assert not result.has_finite_residue_obstruction
    assert result.has_laurent_obstruction
    assert result.is_non_elementary
    assert result.remainder == integrand
    assert result.lower_remainder is None


def test_each_laurent_step_agrees_with_bounded_matrix_oracle() -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    result = reduce_hyperexponential_laurent(
        field.from_sympy(
            (2*x + 3)*t**2 + (-1/x - 1/x**2)/t
        ),
        tower.derivation(),
    )

    for step in result.steps:
        certificate = step.rde_result
        ansatz_denominator = (
            certificate.normalization_factor
            * certificate.denominator_bound
        )
        matrix = solve_bounded_rational_rde_matrix(
            certificate.coefficient,
            certificate.right_hand_side,
            x,
            ansatz_denominator,
            certificate.degree_bound,
        )

        assert matrix.has_solution is step.has_solution
        assert cancel(
            matrix.solution - tower.field(0).to_sympy(step.exact_coefficient)
        ) == 0


@given(
    normal_shift=st.integers(min_value=1, max_value=4),
    logarithmic_coefficient=st.integers(min_value=-4, max_value=4),
    exact_linear=st.integers(min_value=-3, max_value=3),
    exact_constant=st.integers(min_value=-3, max_value=3),
    lower_numerator=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=40, deadline=None)
def test_generated_full_reductions_compose_logs_exact_modes_and_lower_terms(
    normal_shift,
    logarithmic_coefficient,
    exact_linear,
    exact_constant,
    lower_numerator,
) -> None:
    tower = hyperexponential_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(t - normal_shift)
    exact_part = field.from_sympy(
        (exact_linear*x + exact_constant)*t**2
    )
    lower_remainder = tower.field(0).from_sympy(
        lower_numerator/(x + 1)
    )
    logarithmic_derivative = (
        logarithmic_coefficient
        * derivation(normal_factor)
        / normal_factor
    )
    integrand = (
        derivation(exact_part)
        + logarithmic_derivative
        + field.convert(lower_remainder)
    )

    result = reduce_hyperexponential_monomial(
        integrand,
        derivation,
    )

    assert (
        derivation(result.exact_part)
        + result.logarithmic_derivative
        + result.remainder
        == integrand
    )
    assert result.logarithmic_derivative == logarithmic_derivative
    assert result.remainder == field.convert(lower_remainder)
    assert result.lower_remainder == lower_remainder
    assert not result.is_non_elementary
    assert result.reduced_to_lower_field


@given(
    eta_coefficient=st.integers(min_value=-3, max_value=3).filter(bool),
    eta_degree=st.integers(min_value=0, max_value=3),
    solution_data=st.lists(
        st.tuples(
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=-3, max_value=3),
            st.integers(min_value=0, max_value=2),
        ),
        min_size=4,
        max_size=4,
    ),
    lower_numerator=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=55, deadline=None)
def test_generated_planted_laurent_derivatives_reduce_exactly(
    eta_coefficient,
    eta_degree,
    solution_data,
    lower_numerator,
) -> None:
    eta = eta_coefficient*x**eta_degree
    tower = hyperexponential_tower(eta)
    field = tower.field()
    derivation = tower.derivation()
    powers = (-2, -1, 1, 2)
    exact_expression = 0
    for power, (linear, constant, pole_order) in zip(
        powers,
        solution_data,
        strict=True,
    ):
        exact_expression += (
            (linear*x + constant)/x**pole_order
        )*t**power
    exact_part = field.from_sympy(exact_expression)
    lower_remainder = tower.field(0).from_sympy(
        lower_numerator/(x + 1)
    )
    integrand = derivation(exact_part) + field.convert(lower_remainder)

    result = reduce_hyperexponential_laurent(
        integrand,
        derivation,
    )

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.remainder == field.convert(lower_remainder)
    assert result.lower_remainder == lower_remainder
    assert not result.has_obstruction
    assert result.reduced_to_lower_field


@pytest.mark.parametrize(
    "generator_derivative",
    [
        1,
        1 + t**2,
        t**2 + x,
    ],
)
def test_hyperexponential_reduction_rejects_other_families(
    generator_derivative,
) -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(ValueError, match=r"Dt = eta\*t"):
        reduce_hyperexponential_laurent(
            tower.field().from_sympy(t),
            tower.derivation(),
        )


@pytest.mark.parametrize(
    "status",
    [ConstantFieldStatus.UNKNOWN, ConstantFieldStatus.ENLARGED],
)
def test_hyperexponential_reduction_requires_preserved_constants(
    status,
) -> None:
    tower = hyperexponential_tower(
        1,
        constant_field_status=status,
    )

    with pytest.raises(ValueError, match=r"Const\(K\(t\)\)"):
        reduce_hyperexponential_laurent(
            tower.field().from_sympy(t),
            tower.derivation(),
        )


@pytest.mark.parametrize("eta", [1/x, 1/(2*x)])
def test_hyperexponential_reduction_rejects_dependent_generators(eta) -> None:
    tower = hyperexponential_tower(eta)

    with pytest.raises(ValueError, match="algebraically dependent"):
        reduce_hyperexponential_laurent(
            tower.field().from_sympy(t),
            tower.derivation(),
        )


def test_laurent_reduction_supports_a_rational_logarithmic_derivative() -> None:
    eta = 1/(x**2 + 1)
    tower = hyperexponential_tower(eta)
    field = tower.field()
    derivation = tower.derivation()
    planted_exact_part = field.from_sympy(x*t)
    integrand = derivation(planted_exact_part)

    result = reduce_hyperexponential_laurent(
        integrand,
        derivation,
    )

    assert derivation(result.exact_part) == integrand
    assert result.remainder == field.zero
    assert not result.has_obstruction


def test_laurent_reduction_rejects_a_normal_denominator() -> None:
    tower = hyperexponential_tower(1)

    with pytest.raises(ValueError, match="power of the generator"):
        reduce_hyperexponential_laurent(
            tower.field().from_sympy(1/(t - 1)),
            tower.derivation(),
        )


def test_hyperexponential_reduction_requires_a_qq_x_parent() -> None:
    tower = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(t, t, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(TypeError, match=r"over QQ\(x\)"):
        reduce_hyperexponential_laurent(
            tower.field().from_sympy(t),
            tower.derivation(),
        )
