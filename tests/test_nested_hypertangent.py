"""Hypertangent reduction over recursively primitive lower towers."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import I, Poly, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    CoupledRDEObstruction,
    CoupledRischDifferentialEquationResult,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
    analyze_hypertangent_extension,
    integrate_elementary,
    reduce_hypertangent_monomial,
    reduce_hypertangent_special,
)

x, s, u, t = symbols("x s u t")


def primitive_lower_tower() -> DifferentialTower:
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


def tangent_over_primitive(
    *,
    eta=s,
    constant_field_status=ConstantFieldStatus.PRESERVED,
) -> DifferentialTower:
    lower = primitive_lower_tower()
    return lower.extend(
        MonomialExtension(
            Poly(
                eta * (1 + t**2),
                t,
                domain=lower.field(),
            ),
            constant_field_status,
        )
    )


def cayley_pair(real_coefficient, imaginary_coefficient, power=1):
    cayley = (1 + I * t) / (1 - I * t)
    return cancel(
        (real_coefficient + I * imaginary_coefficient) * cayley**power
        + (real_coefficient - I * imaginary_coefficient) * cayley ** (-power)
    )


def test_dependency_analysis_descends_through_a_primitive_level() -> None:
    tower = tangent_over_primitive()

    result = analyze_hypertangent_extension(
        tower.levels[-1],
        tower.derivation(1),
    )

    assert result.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
    assert result.minimal_power is None
    assert result.relation_argument is None


def test_dependency_analysis_detects_a_nested_cayley_relation() -> None:
    eta = 1 / (x * (s**2 + 1))
    tower = tangent_over_primitive(eta=eta)

    result = analyze_hypertangent_extension(
        tower.levels[-1],
        tower.derivation(1),
    )

    assert result.kind is HyperexponentialDependencyKind.RATIONAL
    assert result.minimal_power == 1
    assert cancel(result.relation_argument - (s - I) / (s + I)) == 0


def test_special_reduction_uses_the_recursive_coupled_solver() -> None:
    tower = tangent_over_primitive()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(cayley_pair(x + s, x * s + 1))
    integrand = derivation(planted)

    result = reduce_hypertangent_special(integrand, derivation)

    assert result.reduced_to_lower_field
    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.has_solution
    assert isinstance(
        step.rde_result,
        CoupledRischDifferentialEquationResult,
    )
    assert step.rde_result.reconstructs


def test_special_reduction_retains_a_recursive_rde_obstruction() -> None:
    tower = tangent_over_primitive()
    field = tower.field()
    integrand = field.from_sympy(cayley_pair(1 / (s + 1), 0))

    result = reduce_hypertangent_special(
        integrand,
        tower.derivation(),
    )

    assert result.has_obstruction
    assert len(result.obstructed_steps) == 1
    step = result.obstructed_steps[0]
    assert isinstance(
        step.rde_result,
        CoupledRischDifferentialEquationResult,
    )
    assert step.rde_result.obstruction is CoupledRDEObstruction.DENOMINATOR
    assert result.remainder == integrand


def test_full_monomial_reduction_handles_nested_tangent_modes() -> None:
    tower = tangent_over_primitive()
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(
        cayley_pair(x + s, x * s + 1) + cayley_pair(s, x, power=2)
    )

    result = reduce_hypertangent_monomial(
        derivation(planted),
        derivation,
    )

    assert not result.is_non_elementary
    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert result.lower_remainder == tower.field(1).zero
    assert result.special_reduction is not None
    assert [step.power for step in result.special_reduction.steps] == [1, 2]


def test_elementary_dispatch_integrates_nested_tangent_and_lower_parts() -> None:
    tower = tangent_over_primitive()
    field = tower.field()
    derivation = tower.derivation()
    tangent_part = field.from_sympy(cayley_pair(x + s, x * s + 1))
    expected = tangent_part + field.from_sympy(s)
    integrand = derivation(expected)

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert field.from_sympy(result.antiderivative) == expected
    assert result.lower_result is not None
    assert result.lower_result.is_elementary


def test_nested_hypertangent_reduction_rejects_a_dependent_generator() -> None:
    tower = tangent_over_primitive(eta=1 / (x * (s**2 + 1)))

    with pytest.raises(ValueError, match="constant-field replay is enlarged"):
        reduce_hypertangent_special(1, tower.derivation())


def test_nested_hypertangent_reduction_replays_advisory_constant_metadata() -> None:
    tower = tangent_over_primitive(
        constant_field_status=ConstantFieldStatus.UNKNOWN,
    )

    result = reduce_hypertangent_special(1, tower.derivation())

    assert tower.derivation()(result.exact_part) + result.remainder == tower.field().one


def test_nested_hypertangent_reduces_over_two_primitive_lower_levels() -> None:
    primitive = primitive_lower_tower()
    lower = primitive.extend(
        MonomialExtension(
            Poly(1 / (x * s), u, domain=primitive.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = lower.extend(
        MonomialExtension(
            Poly(
                (s + u) * (1 + t**2),
                t,
                domain=lower.field(),
            ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    derivation = tower.derivation()

    planted = tower.field().from_sympy(t * (s + u))
    reduction = reduce_hypertangent_monomial(derivation(planted), derivation)

    assert reduction.remainder == tower.field().zero
    assert reduction.exact_part == planted
    assert derivation(reduction.exact_part) + reduction.remainder == derivation(planted)


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
    power=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_nested_cayley_modes_reduce_exactly(
    real_coefficients,
    imaginary_coefficients,
    power,
) -> None:
    tower = tangent_over_primitive()
    field = tower.field()
    derivation = tower.derivation()
    real_coefficient = sum(
        (constant + linear * x) * s**degree
        for degree, (constant, linear) in enumerate(real_coefficients)
    )
    imaginary_coefficient = sum(
        (constant + linear * x) * s**degree
        for degree, (constant, linear) in enumerate(imaginary_coefficients)
    )
    assume(real_coefficient != 0 or imaginary_coefficient != 0)
    planted = field.from_sympy(
        cayley_pair(real_coefficient, imaginary_coefficient, power)
    )

    result = reduce_hypertangent_special(
        derivation(planted),
        derivation,
    )

    assert result.reduced_to_lower_field
    assert result.exact_part == planted
    assert result.remainder == field.zero
    assert len(result.steps) == 1
    assert result.steps[0].power == power
