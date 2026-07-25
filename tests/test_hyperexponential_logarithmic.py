"""Radical logarithmic derivatives at a hyperexponential level."""

from math import gcd, lcm

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import I, Poly, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    ParametricLogarithmicDerivativeOutcome,
    solve_parametric_logarithmic_derivative,
    solve_radical_logarithmic_derivative,
)
from sympy_risch.dependency import (
    MonomialRelationSystemParametricCertificate,
)
from sympy_risch.monomial import (
    HyperexponentialLogarithmicDerivativeSignature,
    hyperexponential_logarithmic_derivative_signature,
)

x, s, t, u = symbols("x s t u")


def hyperexponential_tower() -> DifferentialTower:
    """Return ``QQ(x)(t)`` with ``Dx=1`` and ``Dt=x*t``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return rational.extend(
        MonomialExtension(
            Poly(x * t, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def stacked_hyperexponential_tower() -> DifferentialTower:
    """Return ``QQ(x)(s)(t)`` with ``Ds=x*s`` and ``Dt=s*t``."""
    base = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    first = base.extend(
        MonomialExtension(
            Poly(x * s, s, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    return first.extend(
        MonomialExtension(
            Poly(s * t, t, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_normal_factor_removes_its_lower_logarithmic_quotient() -> None:
    """``D(t+x)/(t+x)`` must not count its lower ``x`` term twice."""
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    integrand = derivation(factor) / factor

    result = solve_radical_logarithmic_derivative(integrand, derivation)

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.argument == factor
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )
    assert result.certificate.lower_component == tower.levels[0].field.zero
    assert result.certificate.reconstructs_decomposition
    assert result.certificate.reconstructs_relation


def test_hyperexponential_signature_combines_normal_and_special_exponents() -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    factor = field.from_sympy(t + x)
    integrand = (derivation(factor) / factor + 2 * field.from_sympy(x)) / 3

    result = solve_radical_logarithmic_derivative(integrand, derivation)

    assert result.has_solution
    assert result.minimal_power == 3
    assert result.argument == field.from_sympy(t**2 * (t + x))
    assert result.reconstructs


def test_hyperexponential_signature_descends_through_the_special_generator() -> None:
    tower = hyperexponential_tower()
    derivation = tower.derivation()

    result = solve_radical_logarithmic_derivative(
        1 / (5 * x) + 3 * x / 5,
        derivation,
    )

    assert result.has_solution
    assert result.minimal_power == 5
    assert result.argument == tower.field().from_sympy(x * t**3)
    assert result.reconstructs


def test_hyperexponential_parametric_solver_uses_free_special_direction() -> None:
    tower = hyperexponential_tower()
    outer = MonomialExtension(
        Poly(x**2 * u, u, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = solve_parametric_logarithmic_derivative(
        x**2 + x,
        outer,
        tower.derivation(),
    )

    assert result.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION
    assert (result.n, result.m) == (1, 1)
    assert result.argument == tower.field().from_sympy(t)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        MonomialRelationSystemParametricCertificate,
    )
    assert result.certificate.system.reconstructs


def test_hyperexponential_parametric_solver_handles_stacked_special_directions() -> (
    None
):
    tower = stacked_hyperexponential_tower()
    outer = MonomialExtension(
        Poly(x**2 * u, u, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = solve_parametric_logarithmic_derivative(
        x**2 + x + s,
        outer,
        tower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 1)
    assert result.argument == tower.field().from_sympy(s * t)
    assert result.reconstructs


def test_hyperexponential_radical_solver_uses_stacked_special_directions() -> None:
    tower = stacked_hyperexponential_tower()

    result = solve_radical_logarithmic_derivative(
        s + x,
        tower.derivation(),
    )

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.argument == tower.field().from_sympy(s * t)
    assert result.reconstructs


def test_hyperexponential_parametric_solver_cancels_special_poles() -> None:
    tower = hyperexponential_tower()
    outer = MonomialExtension(
        Poly(u / t, u, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = solve_parametric_logarithmic_derivative(
        2 / t,
        outer,
        tower.derivation(),
    )

    assert result.has_solution
    assert (result.n, result.m) == (1, 2)
    assert result.argument == tower.field().one
    assert result.reconstructs


def test_hyperexponential_relation_system_proves_a_top_polynomial_obstruction() -> None:
    tower = hyperexponential_tower()
    outer = MonomialExtension(
        Poly(x**2 * u, u, domain=tower.field()),
        ConstantFieldStatus.PRESERVED,
    )

    result = solve_parametric_logarithmic_derivative(
        t,
        outer,
        tower.derivation(),
    )

    assert result.proves_no_solution
    assert isinstance(
        result.certificate,
        MonomialRelationSystemParametricCertificate,
    )


@pytest.mark.parametrize(
    "integrand",
    [
        1 / t,
        1 / (t + x) ** 2,
        1 / (t + x),
        t,
    ],
)
def test_hyperexponential_signature_rejects_each_local_obstruction(
    integrand,
) -> None:
    tower = hyperexponential_tower()

    result = solve_radical_logarithmic_derivative(
        integrand,
        tower.derivation(),
    )

    assert not result.has_solution
    assert isinstance(
        result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )
    assert result.certificate.reconstructs_decomposition


def test_normal_quotient_can_be_compensated_by_the_special_exponent() -> None:
    tower = hyperexponential_tower()
    field = tower.field()

    result = solve_radical_logarithmic_derivative(
        (1 - x**2) / (t + x),
        tower.derivation(),
    )

    assert result.has_solution
    assert result.minimal_power == 1
    assert result.argument == field.from_sympy((t + x) / t)
    assert result.reconstructs
    assert isinstance(
        result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )
    assert result.certificate.lower_parametric_result is not None
    assert result.certificate.lower_parametric_result.m == -1


def test_hyperexponential_signature_keeps_normal_and_special_denominators_separate() -> (
    None
):
    tower = hyperexponential_tower()
    field = tower.field()
    factor = field.from_sympy(t + x)
    integrand = tower.derivation()(factor) / factor + field.from_sympy(1 / t)

    signature = hyperexponential_logarithmic_derivative_signature(
        integrand,
        tower.derivation(),
    )

    assert signature.normal_denominator.as_expr() == t + x
    assert signature.special_denominator.as_expr() == t
    assert signature.has_special_pole_obstruction
    assert not signature.has_solution


def test_hyperexponential_signature_handles_degree_two_normal_quotients() -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    factor = field.from_sympy(t**2 + x * t + 1)

    result = solve_radical_logarithmic_derivative(
        tower.derivation()(factor) / (2 * factor),
        tower.derivation(),
    )

    assert result.has_solution
    assert result.minimal_power == 2
    assert result.argument == factor
    assert result.reconstructs


def test_gaussian_hyperexponential_signature_rejects_nonrational_valuation() -> None:
    rational = DifferentialTower(QQ_I).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ_I),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    tower = rational.extend(
        MonomialExtension(
            Poly(x * t, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    field = tower.field()
    factor = field.from_sympy(t + x)

    result = solve_radical_logarithmic_derivative(
        I * tower.derivation()(factor) / factor,
        tower.derivation(),
    )

    assert not result.has_solution
    assert isinstance(
        result.certificate,
        HyperexponentialLogarithmicDerivativeSignature,
    )


@settings(max_examples=45, deadline=None)
@given(
    power=st.integers(min_value=1, max_value=6),
    multiplier=st.integers(min_value=-4, max_value=4),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_x_power=st.integers(min_value=-3, max_value=3),
    t_power=st.integers(min_value=-3, max_value=3),
    normal_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_hyperexponential_parametric_witnesses(
    power: int,
    multiplier: int,
    x_power: int,
    shifted_x_power: int,
    t_power: int,
    normal_power: int,
) -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    outer = MonomialExtension(
        Poly(x**2 * u, u, domain=field),
        ConstantFieldStatus.PRESERVED,
    )
    argument = field.from_sympy(
        x**x_power * (x + 1) ** shifted_x_power * t**t_power * (t + x) ** normal_power
    )
    integrand = (
        derivation(argument) / argument + multiplier * field.from_sympy(x**2)
    ) / power

    result = solve_parametric_logarithmic_derivative(
        integrand,
        outer,
        derivation,
    )

    assert result.has_solution
    assert QQ(result.m, result.n) == QQ(multiplier, power)
    assert result.reconstructs


@settings(max_examples=45, deadline=None)
@given(
    power=st.integers(min_value=1, max_value=6),
    x_power=st.integers(min_value=-3, max_value=3),
    shifted_x_power=st.integers(min_value=-3, max_value=3),
    t_power=st.integers(min_value=-3, max_value=3),
    normal_power=st.integers(min_value=-3, max_value=3),
)
def test_generated_hyperexponential_radical_logarithmic_witnesses(
    power: int,
    x_power: int,
    shifted_x_power: int,
    t_power: int,
    normal_power: int,
) -> None:
    tower = hyperexponential_tower()
    field = tower.field()
    derivation = tower.derivation()
    argument = field.from_sympy(
        x**x_power * (x + 1) ** shifted_x_power * t**t_power * (t + x) ** normal_power
    )
    integrand = derivation(argument) / (power * argument)

    result = solve_radical_logarithmic_derivative(integrand, derivation)

    expected_power = 1
    for exponent in (
        x_power,
        shifted_x_power,
        t_power,
        normal_power,
    ):
        expected_power = lcm(
            expected_power,
            power // gcd(abs(exponent), power),
        )
    assert result.has_solution
    assert result.minimal_power == expected_power
    assert result.reconstructs
