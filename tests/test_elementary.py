"""End-to-end tests for recursive elementary integration."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, diff, exp, log, sqrt, symbols, tan, trigsimp
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    ElementaryIntegralResult,
    IntegrationObstructionKind,
    MonomialExtension,
    MonomialFamily,
    RationalElementaryReduction,
    integrate_elementary,
)

x, t, u, y = symbols("x t u y")


def rational_tower() -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )


def monomial_tower(
    generator_derivative,
) -> DifferentialTower:
    base = rational_tower()
    return base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_rational_base_returns_structured_exact_and_logarithmic_parts() -> None:
    tower = rational_tower()
    field = tower.field()
    integrand = field.from_sympy(-1 / (x - 1) ** 2 + 3 / x)

    result = integrate_elementary(integrand, tower.derivation())

    assert isinstance(result, ElementaryIntegralResult)
    assert isinstance(result.certificate, RationalElementaryReduction)
    assert result.is_elementary
    assert result.reconstructs
    assert result.exact_part == field.from_sympy(1 / (x - 1))
    assert result.logarithmic_derivative == field.from_sympy(3 / x)
    assert result.unintegrated_remainder == field.zero
    assert result.lower_result is None
    assert result.obstruction is None
    assert len(result.logarithmic_terms) == 1
    assert result.antiderivative is not None
    assert cancel(diff(result.antiderivative, x) - integrand.as_expr()) == 0


def test_rational_base_encodes_irreducible_quadratic_residues() -> None:
    tower = rational_tower()
    integrand = tower.field().from_sympy(1 / (x**2 + 1))

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative is not None
    assert cancel(diff(result.antiderivative, x) - integrand.as_expr()) == 0


def test_rational_base_over_real_algebraic_constants() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    integrand = field.from_sympy(-sqrt(2) / (x - 1) ** 2 + sqrt(2) / (x + 1))

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative is not None
    assert cancel(diff(result.antiderivative, x) - integrand.as_expr()) == 0


def test_primitive_level_composes_exact_logarithmic_and_lower_parts() -> None:
    tower = monomial_tower(1 / x)
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy(t**2)
    integrand = (
        derivation(exact)
        + field.from_sympy(1 / (x * t))
        + field.from_sympy(1 / (x - 3))
    )

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.lower_result is not None
    assert result.lower_result.derivation.level == 0
    assert result.obstruction is None
    assert tuple(term.level for term in result.all_logarithmic_terms) == (1, 0)
    assert result.antiderivative is not None
    concrete_antiderivative = result.antiderivative.subs(t, log(x))
    concrete_integrand = integrand.as_expr().subs(t, log(x))
    assert cancel(diff(concrete_antiderivative, x) - concrete_integrand) == 0


def test_hyperexponential_level_produces_a_complete_antiderivative() -> None:
    tower = monomial_tower(t)
    field = tower.field()
    integrand = field.from_sympy(t / (t - 1) + (1 / x - 1 / x**2) * t + 1 / (x + 1))

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative is not None
    concrete_antiderivative = result.antiderivative.subs(t, exp(x))
    concrete_integrand = integrand.as_expr().subs(t, exp(x))
    assert cancel(diff(concrete_antiderivative, x) - concrete_integrand) == 0


def test_hyperexponential_over_log_x_recurses_to_the_rational_base() -> None:
    base = rational_tower()
    logarithmic = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = logarithmic.extend(
        MonomialExtension(
            Poly(2 * t * u / x, u, domain=logarithmic.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    integrand = field.from_sympy((1 + 2 * t) * u + 1 / (x + 1))

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.total_exact_part == field.from_sympy(x * u)
    assert result.lower_result is not None
    assert result.lower_result.lower_result is not None
    assert result.antiderivative is not None
    substitutions = {
        t: log(x),
        u: exp(log(x) ** 2),
    }
    concrete_antiderivative = result.antiderivative.subs(
        substitutions,
        simultaneous=True,
    )
    concrete_integrand = integrand.as_expr().subs(
        substitutions,
        simultaneous=True,
    )
    assert cancel(diff(concrete_antiderivative, x) - concrete_integrand) == 0


def test_hyperexponential_over_log_x_propagates_the_rde_obstruction() -> None:
    base = rational_tower()
    logarithmic = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = logarithmic.extend(
        MonomialExtension(
            Poly(2 * t * u / x, u, domain=logarithmic.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )

    result = integrate_elementary(
        tower.field().from_sympy(u),
        tower.derivation(),
    )

    assert not result.is_elementary
    assert result.reconstructs
    assert result.obstruction is not None
    assert result.obstruction.level == 2
    assert result.obstruction.family is MonomialFamily.HYPEREXPONENTIAL
    assert result.obstruction.kinds == (
        IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
    )


def test_hyperexponential_reduction_accepts_a_stacked_primitive_parent() -> None:
    base = rational_tower()
    logarithmic = base.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    double_logarithmic = logarithmic.extend(
        MonomialExtension(
            Poly(1 / (x * t), u, domain=logarithmic.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    tower = double_logarithmic.extend(
        MonomialExtension(
            Poly(
                2 * u * y / (x * t),
                y,
                domain=double_logarithmic.field(),
            ),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    integrand = field.from_sympy((1 + 2 * u) * y / x)

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative == t * y


def test_hypertangent_level_encodes_the_special_logarithm() -> None:
    tower = monomial_tower(1 + t**2)
    field = tower.field()
    integrand = field.from_sympy(2 * t)

    result = integrate_elementary(integrand, tower.derivation())

    assert result.is_elementary
    assert result.reconstructs
    assert result.antiderivative == log(t**2 + 1)
    concrete_antiderivative = result.antiderivative.subs(t, tan(x))
    concrete_integrand = integrand.as_expr().subs(t, tan(x))
    assert trigsimp(diff(concrete_antiderivative, x) - concrete_integrand) == 0


def test_direct_tan_log_x_field_reaches_the_rational_base() -> None:
    tower = monomial_tower((1 + t**2) / x)
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy(2 * t / (1 + t**2))
    lower = tower.field(0).from_sympy(1 / (x + 1))
    integrand = derivation(exact) + field.convert(lower)

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.lower_result is not None
    assert result.lower_result.integrand == lower


def test_general_polynomial_level_is_outside_the_complete_operation_cell() -> None:
    tower = monomial_tower(t**2 + x)

    with pytest.raises(
        NotImplementedError,
        match="outside the complete elementary_integration cell",
    ):
        integrate_elementary(tower.field().from_sympy(t), tower.derivation())


@pytest.mark.parametrize(
    ("tower", "integrand_expression", "expected_kind"),
    [
        (
            monomial_tower(1 / x),
            1 / t,
            IntegrationObstructionKind.NONCONSTANT_RESIDUE,
        ),
        (
            monomial_tower(t),
            t / x,
            IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
        ),
        (
            monomial_tower(1 + t**2),
            2 * (1 - t**2) / (x * (1 + t**2)),
            IntegrationObstructionKind.UNSOLVABLE_RISCH_DIFFERENTIAL_EQUATION,
        ),
    ],
)
def test_each_obstruction_family_retains_its_level_certificate(
    tower,
    integrand_expression,
    expected_kind,
) -> None:
    field = tower.field()

    result = integrate_elementary(
        field.from_sympy(integrand_expression),
        tower.derivation(),
    )

    assert not result.is_elementary
    assert result.reconstructs
    assert result.obstruction is not None
    assert result.obstruction.level == tower.top
    assert expected_kind in result.obstruction.kinds
    assert result.obstruction.certificate is result.certificate
    assert result.antiderivative is None


def test_simultaneous_primitive_obstructions_are_both_reported() -> None:
    tower = monomial_tower(1 / x)
    field = tower.field()
    derivation = tower.derivation()
    normal_factor = field.from_sympy(t - 1)
    integrand = field.from_sympy(x) * derivation(
        normal_factor
    ) / normal_factor + field.from_sympy(t / (x - 2))

    result = integrate_elementary(integrand, derivation)

    assert result.obstruction is not None
    assert result.obstruction.kinds == (
        IntegrationObstructionKind.NONCONSTANT_RESIDUE,
        IntegrationObstructionKind.REDUCED_POLYNOMIAL,
    )
    assert result.reconstructs


def test_constant_field_call_distinguishes_zero_from_nonzero() -> None:
    tower = DifferentialTower(QQ)

    zero = integrate_elementary(0, tower.derivation())
    nonzero = integrate_elementary(2, tower.derivation())

    assert zero.is_elementary
    assert zero.reconstructs
    assert zero.antiderivative == 0
    assert not nonzero.is_elementary
    assert nonzero.reconstructs
    assert nonzero.obstruction is not None
    assert nonzero.obstruction.kinds == (IntegrationObstructionKind.CONSTANT_FIELD,)


def test_general_algebraic_top_level_remains_outside_simple_radical_scope() -> None:
    base = rational_tower()
    tower = base.extend(AlgebraicExtension(Poly(y**3 + y - x, y, domain=base.field())))

    with pytest.raises(
        NotImplementedError,
        match="terminal pure simple-radical extension",
    ):
        integrate_elementary(tower.field().one, tower.derivation())


def test_general_polynomial_level_refuses_even_a_trivial_integrand() -> None:
    tower = monomial_tower(t**2 + x)

    with pytest.raises(
        NotImplementedError,
        match="outside the complete elementary_integration cell",
    ):
        integrate_elementary(tower.field().one, tower.derivation())


def test_nonstandard_base_is_rejected_before_family_dispatch() -> None:
    tower = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(2, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="rational base must have derivative Dx = 1",
    ):
        integrate_elementary(tower.field().one, tower.derivation())


@given(
    family=st.sampled_from(("primitive", "hyperexponential", "hypertangent")),
    exact_coefficient=st.integers(min_value=-3, max_value=3),
    logarithmic_residue=st.integers(min_value=-3, max_value=3),
    lower_linear=st.integers(min_value=-3, max_value=3),
    lower_constant=st.integers(min_value=-3, max_value=3),
)
@settings(max_examples=40, deadline=None)
def test_generated_level_compositions_reconstruct_recursively(
    family,
    exact_coefficient,
    logarithmic_residue,
    lower_linear,
    lower_constant,
) -> None:
    if family == "primitive":
        tower = monomial_tower(1 / x)
    elif family == "hyperexponential":
        tower = monomial_tower(t)
    elif family == "hypertangent":
        tower = monomial_tower(1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    exact = field.from_sympy(exact_coefficient * (x + 1) / (t - 2) ** 2)
    normal_factor = field.from_sympy(t - 1)
    lower = tower.field(0).from_sympy((lower_linear * x + lower_constant) / (x + 1))
    integrand = (
        derivation(exact)
        + logarithmic_residue * derivation(normal_factor) / normal_factor
        + field.convert(lower)
    )

    result = integrate_elementary(integrand, derivation)

    assert result.is_elementary
    assert result.reconstructs
    assert result.lower_result is not None
    assert field.convert(result.lower_result.integrand) == result.remainder
    assert result.unintegrated_remainder == field.zero
