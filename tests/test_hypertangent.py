"""Reduction in one real hypertangent monomial extension."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import I, Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    CoupledRDEObstruction,
    DifferentialTower,
    MonomialExtension,
    reduce_hypertangent_monomial,
    reduce_hypertangent_special,
)
from tests.oracles.rde import solve_bounded_coupled_rde_matrix

x, t = symbols("x t")


def hypertangent_tower(
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
            Poly(eta * (1 + t**2), t, domain=base.field()),
            constant_field_status,
        ),
    )


def test_special_reduction_of_inverse_special_polynomial_stays_real() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    integrand = field.from_sympy(1 / (1 + t**2))

    result = reduce_hypertangent_special(
        integrand,
        tower.derivation(),
    )

    assert result.exact_part == field.from_sympy(t / (2 * (1 + t**2)))
    assert result.remainder == field.from_sympy(1 / 2)
    assert result.lower_remainder == tower.field(0).convert(QQ(1, 2))
    assert len(result.steps) == 1
    assert result.steps[0].power == 1
    assert result.steps[0].real_coefficient == tower.field(0).convert(QQ(1, 4))
    assert result.steps[0].imaginary_coefficient == tower.field(0).zero
    assert result.steps[0].has_solution
    assert not result.has_obstruction
    assert result.reduced_to_lower_field
    assert not field.to_sympy(result.exact_part).has(I)


def test_special_reduction_pairs_a_purely_imaginary_cayley_mode() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(t / (1 + t**2))

    result = reduce_hypertangent_special(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.exact_part == field.from_sympy((t**2 - 1) / (4 * (1 + t**2)))
    assert result.remainder == field.zero
    step = result.steps[0]
    assert step.real_coefficient == tower.field(0).zero
    assert step.imaginary_coefficient == tower.field(0).convert(QQ(-1, 4))
    assert step.has_solution
    assert not field.to_sympy(result.exact_part).has(I)


def test_higher_cayley_mode_is_reduced() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy((1 + t**4) / (1 + t**2) ** 2)

    result = reduce_hypertangent_special(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.remainder == field.from_sympy(QQ(3, 4))
    assert tuple(step.power for step in result.steps) == (2,)
    assert result.steps[0].has_solution
    assert result.reduced_to_lower_field


def test_pure_lower_field_input_has_no_cayley_steps() -> None:
    tower = hypertangent_tower(x**3 + 1)
    field = tower.field()
    lower = tower.field(0).from_sympy((x + 1) / (x**2 + 1))

    result = reduce_hypertangent_special(
        field.convert(lower),
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.remainder == field.convert(lower)
    assert result.lower_remainder == lower
    assert result.steps == ()
    assert result.reduced_to_lower_field


def test_special_mode_denominator_obstruction_is_retained() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    integrand = field.from_sympy(2 * (1 - t**2) / (x * (1 + t**2)))

    result = reduce_hypertangent_special(
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
    assert step.real_coefficient == tower.field(0).from_sympy(1 / x)
    assert step.imaginary_coefficient == tower.field(0).zero
    assert step.rde_result.obstruction is CoupledRDEObstruction.DENOMINATOR


@pytest.mark.parametrize(
    ("integrand", "expected_obstruction"),
    [
        (
            2 * (1 - t**2) / (1 + t**2),
            CoupledRDEObstruction.DEGREE,
        ),
        (
            -4 * x**2 * t / (1 + t**2),
            CoupledRDEObstruction.NONZERO_REMAINDER,
        ),
    ],
)
def test_special_reduction_exposes_other_coupled_obstructions(
    integrand,
    expected_obstruction,
) -> None:
    tower = hypertangent_tower(x / 2)
    field = tower.field()
    element = field.from_sympy(integrand)

    result = reduce_hypertangent_special(
        element,
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.remainder == element
    assert result.has_obstruction
    assert result.obstructed_steps[0].rde_result.obstruction is (expected_obstruction)


def test_each_special_step_agrees_with_block_matrix_oracle() -> None:
    tower = hypertangent_tower(x + 1)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x * t + 2) / (1 + t**2) + (x**2 - t) / (1 + t**2) ** 2)
    integrand = derivation(planted) + field.from_sympy(1 / (x + 2))

    result = reduce_hypertangent_special(integrand, derivation)

    assert result.reduced_to_lower_field
    assert result.remainder == field.convert(result.lower_remainder)
    assert tuple(step.power for step in result.steps) == (1, 2)
    for step in result.steps:
        rde = step.rde_result
        matrix = solve_bounded_coupled_rde_matrix(
            rde.lambda_coefficient,
            rde.real_right_hand_side,
            rde.imaginary_right_hand_side,
            x,
            rde.denominator_bound,
            max(0, rde.degree_bound),
        )
        assert matrix.has_solution == step.has_solution
        assert cancel(matrix.real_solution - rde.real_solution) == 0
        assert cancel(matrix.imaginary_solution - rde.imaginary_solution) == 0


coefficient_lists = st.lists(
    st.integers(min_value=-2, max_value=2),
    min_size=1,
    max_size=3,
)


@given(
    eta_coefficients=coefficient_lists,
    real_coefficients=coefficient_lists,
    tangent_coefficients=coefficient_lists,
    special_power=st.integers(min_value=1, max_value=3),
    lower_numerator=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_special_exact_derivatives_reduce_to_the_lower_field(
    eta_coefficients,
    real_coefficients,
    tangent_coefficients,
    special_power,
    lower_numerator,
) -> None:
    eta = sum(
        coefficient * x**degree for degree, coefficient in enumerate(eta_coefficients)
    )
    assume(eta != 0)
    tower = hypertangent_tower(eta)
    field = tower.field()
    derivation = tower.derivation()
    real_numerator = sum(
        coefficient * x**degree for degree, coefficient in enumerate(real_coefficients)
    )
    tangent_numerator = sum(
        coefficient * x**degree
        for degree, coefficient in enumerate(tangent_coefficients)
    )
    planted = field.from_sympy(
        (real_numerator + tangent_numerator * t) / (1 + t**2) ** special_power
    )
    lower = tower.field(0).from_sympy(lower_numerator / (x**2 + 1))
    integrand = derivation(planted) + field.convert(lower)

    result = reduce_hypertangent_special(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.remainder == field.convert(result.lower_remainder)
    assert result.reduced_to_lower_field
    assert not field.to_sympy(result.exact_part).has(I)


def test_high_degree_eta_and_high_cayley_power_reconstruct_exactly() -> None:
    tower = hypertangent_tower(x**6 - 3 * x**2 + 1)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x**5 + (x**3 - 2) * t) / (1 + t**2) ** 4)
    lower = tower.field(0).from_sympy(1 / (x**4 + 1))
    integrand = derivation(planted) + field.convert(lower)

    result = reduce_hypertangent_special(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.remainder == field.convert(result.lower_remainder)
    assert tuple(step.power for step in result.steps) == (1, 2, 3, 4)
    assert result.reduced_to_lower_field
    assert not field.to_sympy(result.exact_part).has(I)


def test_full_reduction_composes_normal_log_special_exact_and_lower_part() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    planted_special = field.from_sympy(t / (1 + t**2))
    integrand = derivation(planted_special) + field.from_sympy(
        t / (t - 1) + 1 / (x + 1)
    )

    result = reduce_hypertangent_monomial(integrand, derivation)

    assert (
        derivation(result.exact_part) + result.logarithmic_derivative + result.remainder
        == integrand
    )
    assert result.exact_part == planted_special
    assert result.logarithmic_derivative == field.from_sympy((t + 1) / (2 * (t - 1)))
    assert result.remainder == field.from_sympy((x + 3) / (2 * (x + 1)))
    assert result.lower_remainder == tower.field(0).from_sympy((x + 3) / (2 * (x + 1)))
    assert result.special_reduction is not None
    assert not result.is_non_elementary
    assert result.reduced_to_lower_field


def test_full_reduction_extracts_the_special_logarithm_at_infinity() -> None:
    tower = hypertangent_tower(x + 1)
    field = tower.field()

    result = reduce_hypertangent_monomial(
        field.from_sympy(2 * (x + 1) * t),
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.logarithmic_derivative == field.from_sympy(2 * (x + 1) * t)
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.reduced_to_lower_field


def test_full_reduction_handles_a_repeated_normal_pole() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(1 / (t - 1) ** 2)
    integrand = derivation(planted)

    result = reduce_hypertangent_monomial(integrand, derivation)

    assert derivation(result.exact_part) == integrand
    assert result.logarithmic_derivative == field.zero
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.reduced_to_lower_field


def test_full_reduction_preserves_a_normal_residue_obstruction() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    integrand = field.from_sympy(1 / (x * t))

    result = reduce_hypertangent_monomial(
        integrand,
        tower.derivation(),
    )

    assert result.has_common_obstruction
    assert not result.has_special_obstruction
    assert result.special_reduction is None
    assert result.remainder == integrand
    assert result.lower_remainder is None
    assert result.is_non_elementary
    assert not result.reduced_to_lower_field


def test_full_reduction_preserves_a_coupled_rde_obstruction() -> None:
    tower = hypertangent_tower(1)
    field = tower.field()
    integrand = field.from_sympy(2 * (1 - t**2) / (x * (1 + t**2)))

    result = reduce_hypertangent_monomial(
        integrand,
        tower.derivation(),
    )

    assert not result.has_common_obstruction
    assert result.has_special_obstruction
    assert result.special_reduction is not None
    assert result.remainder == integrand
    assert result.lower_remainder is None
    assert result.is_non_elementary
    assert not result.reduced_to_lower_field


def test_special_reduction_rejects_an_unreduced_normal_denominator() -> None:
    tower = hypertangent_tower(1)

    with pytest.raises(ValueError, match="requires a Laurent polynomial"):
        reduce_hypertangent_special(
            1 / (t - 1),
            tower.derivation(),
        )


def test_hypertangent_reduction_requires_the_right_family() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    primitive = base.extend(
        MonomialExtension(
            Poly(1, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    with pytest.raises(ValueError, match="requires Dt = eta"):
        reduce_hypertangent_special(1, primitive.derivation())


def test_hypertangent_reduction_requires_preserved_constants() -> None:
    tower = hypertangent_tower(
        1,
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )

    with pytest.raises(ValueError, match=r"Const\(K\(t\)\)"):
        reduce_hypertangent_special(1, tower.derivation())


def test_hypertangent_reduction_rejects_a_dependent_cayley_generator() -> None:
    tower = hypertangent_tower(1 / (x**2 + 1))

    with pytest.raises(
        ValueError,
        match="Cayley generator is algebraically dependent",
    ):
        reduce_hypertangent_special(1, tower.derivation())


def test_tan_log_x_level_extracts_the_special_logarithm() -> None:
    tower = hypertangent_tower(1 / x)
    field = tower.field()

    result = reduce_hypertangent_monomial(
        field.from_sympy(t / x),
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.logarithmic_derivative == field.from_sympy(t / x)
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.reduced_to_lower_field


def test_tan_log_x_special_pair_is_an_exact_derivative() -> None:
    tower = hypertangent_tower(1 / x)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(2 * (1 - t**2) / (x * (1 + t**2)))

    result = reduce_hypertangent_special(integrand, derivation)

    assert result.exact_part == field.from_sympy(2 * t / (1 + t**2))
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(0).zero
    assert result.steps[0].rde_result.gaussian_normalization_factor is not None
    assert result.reduced_to_lower_field


def test_rational_eta_special_mode_retains_a_degree_obstruction() -> None:
    tower = hypertangent_tower(1 / (2 * x**2))
    field = tower.field()
    integrand = field.from_sympy(2 * (1 - t**2) / (1 + t**2))

    result = reduce_hypertangent_special(
        integrand,
        tower.derivation(),
    )

    assert result.exact_part == field.zero
    assert result.remainder == integrand
    assert result.has_obstruction
    assert result.obstructed_steps[0].rde_result.obstruction is (
        CoupledRDEObstruction.DEGREE
    )


@given(
    eta_numerator_coefficients=coefficient_lists,
    real_coefficients=coefficient_lists,
    tangent_coefficients=coefficient_lists,
    special_power=st.integers(min_value=1, max_value=2),
)
@settings(max_examples=12, deadline=None)
def test_generated_rational_eta_exact_derivatives_reconstruct(
    eta_numerator_coefficients,
    real_coefficients,
    tangent_coefficients,
    special_power,
) -> None:
    eta_numerator = sum(
        coefficient * x**degree
        for degree, coefficient in enumerate(eta_numerator_coefficients)
    )
    assume(eta_numerator != 0)
    eta = eta_numerator + 1 / (x**2 + 1) ** 2
    tower = hypertangent_tower(eta)
    field = tower.field()
    derivation = tower.derivation()
    real_numerator = sum(
        coefficient * x**degree for degree, coefficient in enumerate(real_coefficients)
    )
    tangent_numerator = sum(
        coefficient * x**degree
        for degree, coefficient in enumerate(tangent_coefficients)
    )
    planted = field.from_sympy(
        (real_numerator + tangent_numerator * t) / (1 + t**2) ** special_power
    )
    integrand = derivation(planted)

    result = reduce_hypertangent_special(integrand, derivation)

    assert derivation(result.exact_part) + result.remainder == integrand
    assert result.remainder == field.convert(result.lower_remainder)
    assert result.reduced_to_lower_field
    assert not field.to_sympy(result.exact_part).has(I)
