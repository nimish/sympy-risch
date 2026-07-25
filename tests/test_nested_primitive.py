"""Recursive primitive reduction over a primitive lower tower."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, diff, log, symbols, together
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    IntegrationObstructionKind,
    MonomialExtension,
    MonomialFamily,
    integrate_elementary,
    polynomial_reduce_primitive,
    solve_exact_derivative,
)

x, t, u, v = symbols("x t u v")


def rational_tower() -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )


def logarithmic_tower(
    *,
    constant_field_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    base = rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(1/x, t, domain=base.field()),
            constant_field_status,
        )
    )


def stacked_logarithmic_tower(
    *,
    lower_constant_field_status=ConstantFieldStatus.PRESERVED,
    top_eta=None,
) -> DifferentialTower:
    lower = logarithmic_tower(
        constant_field_status=lower_constant_field_status,
    )
    eta = 1/(x*t) if top_eta is None else top_eta
    return lower.extend(
        MonomialExtension(
            Poly(eta, u, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def triply_logarithmic_tower() -> DifferentialTower:
    lower = stacked_logarithmic_tower()
    return lower.extend(
        MonomialExtension(
            Poly(1/(x*t*u), v, domain=lower.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_recursive_polynomial_reduction_recognizes_the_eta_direction() -> None:
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    eta = parent.from_sympy(1/(x*t))
    polynomial = Poly.from_dict(
        {(4,): parent.convert(QQ(3, 2))*eta},
        u,
        domain=parent,
    )

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part == Poly(u**5*QQ(3, 10), u, domain=parent)
    assert result.remainder.is_zero
    assert len(result.steps) == 1
    assert result.steps[0].eta_multiple == QQ(3, 2)
    assert result.steps[0].obstruction == parent.zero
    assert tower.derivation().polynomial(result.exact_part) == polynomial


def test_eta_with_an_exact_lower_part_is_normalized_before_comparison() -> None:
    eta_expression = 1/(x*t) - 1/(x*(t - 2)**2)
    tower = stacked_logarithmic_tower(top_eta=eta_expression)
    parent = tower.field(1)
    eta = parent.from_sympy(eta_expression)
    polynomial = Poly.from_dict({(2,): eta}, u, domain=parent)

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.exact_part == Poly(u**3/3, u, domain=parent)
    assert result.remainder.is_zero
    assert result.steps[0].exact_coefficient == parent.zero
    assert result.steps[0].eta_multiple == QQ.one
    assert tower.derivation().polynomial(result.exact_part) == polynomial


def test_recursive_reduction_uses_an_exact_part_from_the_lower_tower() -> None:
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    derivation = tower.derivation()
    lower_exact = parent.from_sympy((x + 1)/(t - 2)**2 + t**3)
    coefficient = derivation.lower()(lower_exact)
    polynomial = Poly.from_dict({(0,): coefficient}, u, domain=parent)

    result = polynomial_reduce_primitive(polynomial, derivation)

    assert result.remainder.is_zero
    assert derivation.polynomial(result.exact_part) == polynomial
    assert result.steps[0].exact_coefficient == lower_exact


def test_nonproportional_lower_obstruction_is_retained() -> None:
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    derivation = tower.derivation()
    coefficient = parent.from_sympy(1/(t - 1))
    polynomial = Poly.from_dict({(2,): coefficient}, u, domain=parent)

    result = polynomial_reduce_primitive(polynomial, derivation)

    assert result.exact_part.is_zero
    assert result.remainder == polynomial
    assert result.steps[0].eta_multiple == QQ.zero
    assert result.steps[0].obstruction == coefficient
    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial

    second = polynomial_reduce_primitive(result.remainder, derivation)
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


def test_mixed_eta_and_independent_class_is_projected_canonically() -> None:
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    derivation = tower.derivation()
    eta = parent.from_sympy(1/(x*t))
    independent = parent.from_sympy(1/(t - 1))
    coefficient = 2*eta + independent
    polynomial = Poly.from_dict({(1,): coefficient}, u, domain=parent)

    result = polynomial_reduce_primitive(polynomial, derivation)
    step = result.steps[0]

    assert step.eta_multiple == QQ(2)
    assert step.obstruction == independent
    assert result.exact_part == Poly(u**2, u, domain=parent)
    assert result.remainder == Poly.from_dict(
        {(1,): independent},
        u,
        domain=parent,
    )
    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial

    second = polynomial_reduce_primitive(result.remainder, derivation)
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


def test_recursive_exact_solver_recovers_a_planted_derivative() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(
        (x + u)/(u - 2)**3 + u**5 + t*u**2
    )
    integrand = derivation(planted)

    result = solve_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == integrand


def test_recursive_exact_solver_preserves_a_polynomial_obstruction() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(u/(t - 1))

    result = solve_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.exact_part == field.zero
    assert result.obstruction == integrand
    assert derivation(result.exact_part) + result.obstruction == integrand

    second = solve_exact_derivative(result.obstruction, derivation)
    assert second.exact_part == field.zero
    assert second.obstruction == result.obstruction


def test_recursive_reduction_rejects_an_exact_top_generator_derivative() -> None:
    tower = stacked_logarithmic_tower(top_eta=2*t/x)
    parent = tower.field(1)

    with pytest.raises(
        ValueError,
        match="Dt is exact in the lower field",
    ):
        polynomial_reduce_primitive(
            Poly(0, u, domain=parent),
            tower.derivation(),
        )


def test_recursive_reduction_checks_lower_constant_field_evidence() -> None:
    tower = stacked_logarithmic_tower(
        lower_constant_field_status=ConstantFieldStatus.UNKNOWN,
    )
    parent = tower.field(1)

    with pytest.raises(ValueError, match="status is unknown"):
        polynomial_reduce_primitive(
            Poly(u, u, domain=parent),
            tower.derivation(),
        )


def test_elementary_dispatch_integrates_a_full_stacked_log_example() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy((x + u)/(u - 2)**2 + u**4)
    top_logarithm = 2*derivation(field.from_sympy(u))/field.from_sympy(u)
    lower = field.from_sympy(1/(x*t) + 1/(x + 1))
    integrand = derivation(exact) + top_logarithm + lower

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.obstruction is None
    assert result.lower_result is not None
    assert result.lower_result.lower_result is not None
    # The lower term 1/(x*t) is Du, so the top reduction absorbs it into the
    # exact part instead of redundantly presenting it as log(t).
    assert tuple(term.level for term in result.all_logarithmic_terms) == (2, 0)
    assert result.total_exact_part - exact == field.from_sympy(u)
    assert result.antiderivative is not None

    substitutions = {t: log(x), u: log(log(x))}
    concrete_antiderivative = result.antiderivative.subs(
        substitutions,
        simultaneous=True,
    )
    concrete_integrand = integrand.as_expr().subs(
        substitutions,
        simultaneous=True,
    )
    assert cancel(
        together(diff(concrete_antiderivative, x) - concrete_integrand)
    ) == 0


def test_logarithmic_derivative_of_log_log_x_is_constructive() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(1/(x*t*u))

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative == log(u)
    assert tuple(term.level for term in result.all_logarithmic_terms) == (2,)


@pytest.mark.parametrize(
    ("integrand_expression", "expected_antiderivative"),
    [
        (v/(x*t*u), v**2/2),
        (1/(x*t*u*v), log(v)),
    ],
)
def test_primitive_recursion_continues_through_four_levels(
    integrand_expression,
    expected_antiderivative,
) -> None:
    tower = triply_logarithmic_tower()

    result = integrate_elementary(
        tower.field().from_sympy(integrand_expression),
        tower.derivation(),
    )

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative == expected_antiderivative


@given(
    eta_multiple=st.integers(min_value=-4, max_value=4),
    independent_multiple=st.integers(min_value=-4, max_value=4),
    power=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=20, deadline=None)
def test_generated_four_level_projections_use_the_iterated_coordinate(
    eta_multiple,
    independent_multiple,
    power,
) -> None:
    assume(eta_multiple != 0 or independent_multiple != 0)
    tower = triply_logarithmic_tower()
    parent = tower.field(2)
    derivation = tower.derivation()
    eta = parent.from_sympy(1/(x*t*u))
    independent = parent.from_sympy(independent_multiple/(u - 1))
    polynomial = Poly.from_dict(
        {
            (power,): (
                parent.convert(eta_multiple)*eta + independent
            )
        },
        v,
        domain=parent,
    )

    result = polynomial_reduce_primitive(polynomial, derivation)
    leading_step = result.steps[0]

    assert leading_step.eta_multiple == QQ(eta_multiple)
    assert leading_step.obstruction == independent
    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial

    second = polynomial_reduce_primitive(result.remainder, derivation)
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


def test_one_over_log_log_x_has_a_top_level_residue_obstruction() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(1/u)

    result = integrate_elementary(integrand, tower.derivation())

    assert not result.is_elementary
    assert result.reconstructs
    assert result.obstruction is not None
    assert result.obstruction.level == 2
    assert result.obstruction.generator == u
    assert result.obstruction.family is MonomialFamily.PRIMITIVE
    assert result.obstruction.kinds == (
        IntegrationObstructionKind.NONCONSTANT_RESIDUE,
    )


def test_lower_primitive_obstruction_propagates_through_the_top_level() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(1/t)

    result = integrate_elementary(integrand, tower.derivation())

    assert not result.is_elementary
    assert result.reconstructs
    assert result.lower_result is not None
    assert result.obstruction is result.lower_result.obstruction
    assert result.obstruction is not None
    assert result.obstruction.level == 1
    assert result.obstruction.generator == t


def test_top_polynomial_obstruction_stops_before_lower_recursion() -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    integrand = field.from_sympy(u/(t - 1))

    result = integrate_elementary(integrand, tower.derivation())

    assert not result.is_elementary
    assert result.reconstructs
    assert result.lower_result is None
    assert result.obstruction is not None
    assert result.obstruction.level == 2
    assert result.obstruction.kinds == (
        IntegrationObstructionKind.REDUCED_POLYNOMIAL,
    )


@given(
    coefficient=st.integers(min_value=-5, max_value=5),
    power=st.integers(min_value=0, max_value=7),
)
@settings(max_examples=30, deadline=None)
def test_generated_eta_modes_are_exact_derivatives(
    coefficient,
    power,
) -> None:
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    eta = parent.from_sympy(1/(x*t))
    polynomial = Poly.from_dict(
        {(power,): parent.convert(coefficient)*eta},
        u,
        domain=parent,
    )

    result = polynomial_reduce_primitive(polynomial, tower.derivation())

    assert result.remainder.is_zero
    assert tower.derivation().polynomial(result.exact_part) == polynomial


@given(
    eta_multiple=st.integers(min_value=-5, max_value=5),
    independent_multiple=st.integers(min_value=-5, max_value=5),
    power=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=30, deadline=None)
def test_generated_mixed_classes_remove_the_eta_coordinate(
    eta_multiple,
    independent_multiple,
    power,
) -> None:
    assume(eta_multiple != 0 or independent_multiple != 0)
    tower = stacked_logarithmic_tower()
    parent = tower.field(1)
    derivation = tower.derivation()
    eta = parent.from_sympy(1/(x*t))
    independent = parent.from_sympy(independent_multiple/(t - 1))
    coefficient = parent.convert(eta_multiple)*eta + independent
    polynomial = Poly.from_dict(
        {(power,): coefficient},
        u,
        domain=parent,
    )

    result = polynomial_reduce_primitive(polynomial, derivation)
    leading_step = result.steps[0]

    assert leading_step.eta_multiple == QQ(eta_multiple)
    assert leading_step.obstruction == independent
    assert derivation.polynomial(result.exact_part) + result.remainder == polynomial

    second = polynomial_reduce_primitive(result.remainder, derivation)
    assert second.exact_part.is_zero
    assert second.remainder == result.remainder


@given(
    proper_coefficient=st.integers(min_value=-3, max_value=3),
    proper_shift=st.integers(min_value=2, max_value=4),
    proper_power=st.integers(min_value=1, max_value=3),
    mixed_coefficient=st.integers(min_value=-3, max_value=3),
    mixed_power=st.integers(min_value=0, max_value=5),
    polynomial_coefficient=st.integers(min_value=-3, max_value=3),
    polynomial_power=st.integers(min_value=0, max_value=6),
    lower_coefficient=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_derivatives_are_solved_across_all_three_levels(
    proper_coefficient,
    proper_shift,
    proper_power,
    mixed_coefficient,
    mixed_power,
    polynomial_coefficient,
    polynomial_power,
    lower_coefficient,
) -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(
        proper_coefficient/(u - proper_shift)**proper_power
        + mixed_coefficient*t*u**mixed_power
        + polynomial_coefficient*u**polynomial_power
        + lower_coefficient/(t - 1)**2
    )
    integrand = derivation(planted)

    result = solve_exact_derivative(integrand, derivation)

    assert result.has_solution
    assert result.obstruction == field.zero
    assert derivation(result.exact_part) == integrand


@given(
    exact_coefficient=st.integers(min_value=-3, max_value=3),
    exact_power=st.integers(min_value=1, max_value=4),
    logarithmic_residue=st.integers(min_value=-3, max_value=3),
    lower_logarithmic_residue=st.integers(min_value=-3, max_value=3),
    rational_residue=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=35, deadline=None)
def test_generated_stacked_log_compositions_reconstruct(
    exact_coefficient,
    exact_power,
    logarithmic_residue,
    lower_logarithmic_residue,
    rational_residue,
) -> None:
    tower = stacked_logarithmic_tower()
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy(
        exact_coefficient*(x + t)/(u - 2)**exact_power
    )
    top_factor = field.from_sympy(u - 1)
    lower = field.from_sympy(
        lower_logarithmic_residue/(x*t)
        + rational_residue/(x + 1)
    )
    integrand = (
        derivation(exact)
        + logarithmic_residue*derivation(top_factor)/top_factor
        + lower
    )

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.unintegrated_remainder == field.zero
