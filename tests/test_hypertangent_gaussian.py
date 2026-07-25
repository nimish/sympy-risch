"""Tests for the immutable Cayley Gaussian normal form."""

import pytest
from sympy import I, Poly, cancel, symbols
from sympy.polys.domains import QQ, QQ_I

from sympy_risch import (
    ConstantFieldStatus,
    CoupledRDEObstruction,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    solve_coupled_risch_differential_equation,
)
from sympy_risch.gaussian import build_hypertangent_gaussian_normal_form

x, t, u = symbols("x t u")


def hypertangent_tower() -> DifferentialTower:
    return (
        DifferentialTower(QQ)
        .extend(
            MonomialExtension(
                Poly(1, x, domain=QQ),
                ConstantFieldStatus.PRESERVED,
            ),
        )
        .extend(
            MonomialExtension(
                Poly(1 + t**2, t, domain=QQ.frac_field(x)),
                ConstantFieldStatus.PRESERVED,
            ),
        )
    )


def nested_hypertangent_tower() -> DifferentialTower:
    first = hypertangent_tower()
    return first.extend(
        MonomialExtension(
            Poly((t + 1) * (1 + u**2), u, domain=first.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def test_cayley_normal_form_turns_a_hypertangent_level_hyperexponential() -> None:
    tower = hypertangent_tower()

    normal_form = build_hypertangent_gaussian_normal_form(
        tower.derivation(),
    )

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert normal_form.target_derivation.tower.constants == QQ_I
    assert target_extension.family is MonomialFamily.HYPEREXPONENTIAL
    assert target_extension.generator_derivative == Poly(
        2 * I * t,
        t,
        domain=QQ_I.frac_field(x),
    )
    assert normal_form.verifies_derivation()


def test_cayley_normal_form_intertwines_rational_field_elements() -> None:
    tower = hypertangent_tower()
    derivation = tower.derivation()
    field = tower.field()
    normal_form = build_hypertangent_gaussian_normal_form(derivation)

    for expression in (
        t,
        (t**2 + x) / (t + 1),
        (x * t + 1) / (t**2 + 1),
    ):
        source_value = field.from_sympy(expression)
        target_difference = normal_form.target_derivation(
            normal_form.forward(source_value)
        ) - normal_form.forward(derivation(source_value))
        assert cancel(normal_form.target_field.to_sympy(target_difference)) == 0


def test_cayley_normal_form_recovers_exact_real_coordinates() -> None:
    tower = hypertangent_tower()
    field = tower.field()
    normal_form = build_hypertangent_gaussian_normal_form(tower.derivation())
    real_part = field.from_sympy((t**2 + x) / (t + 1))
    imaginary_part = field.from_sympy((x * t - 1) / (t**2 + 1))
    target_value = normal_form.forward(real_part) + normal_form.target_field.from_sympy(
        I
    ) * normal_form.forward(
        imaginary_part,
    )

    recovered_real, recovered_imaginary = normal_form.split_inverse(
        target_value,
    )

    assert recovered_real == real_part
    assert recovered_imaginary == imaginary_part


def test_cayley_normal_form_rewrites_later_hypertangent_coefficients() -> None:
    tower = nested_hypertangent_tower()

    normal_form = build_hypertangent_gaussian_normal_form(
        tower.derivation(),
    )

    target_extension = normal_form.target_derivation.tower.levels[-1]
    assert isinstance(target_extension, MonomialExtension)
    assert target_extension.family is MonomialFamily.HYPEREXPONENTIAL
    expected = cancel(2 * I * (-I * (t - 1) / (t + 1) + 1) * u)
    assert target_extension.generator_derivative == Poly(
        expected,
        u,
        domain=QQ_I.frac_field(x, t),
    )
    assert normal_form.verifies_derivation()


def test_cayley_normal_form_requires_certified_constants() -> None:
    tower = (
        DifferentialTower(QQ)
        .extend(
            MonomialExtension(
                Poly(1, x, domain=QQ),
                ConstantFieldStatus.PRESERVED,
            ),
        )
        .extend(
            MonomialExtension(
                Poly(1 + t**2, t, domain=QQ.frac_field(x)),
            ),
        )
    )

    with pytest.raises(ValueError, match=r"Const\(K\(t\)\)"):
        build_hypertangent_gaussian_normal_form(tower.derivation())


def test_coupled_rational_hypertangent_frontend_uses_cayley_normal_form() -> None:
    tower = hypertangent_tower()
    derivation = tower.derivation()
    field = tower.field()
    rotation = field.from_sympy(x + 1 / (t + 2))
    planted_real = field.from_sympy((t**2 + x) / (t + 1))
    planted_imaginary = field.from_sympy((t**3 - 1) / (t**2 + 1))
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
    assert result.scalar_result is not None
    assert result.gaussian_normal_form is not None
    assert result.real_solution == planted_real
    assert result.imaginary_solution == planted_imaginary


def test_coupled_rational_hypertangent_frontend_propagates_obstruction() -> None:
    tower = hypertangent_tower()
    derivation = tower.derivation()
    field = tower.field()
    normal_form = build_hypertangent_gaussian_normal_form(derivation)
    target_rhs = normal_form.target_field.from_sympy(1 / (t - 1))
    real_rhs, imaginary_rhs = normal_form.split_inverse(target_rhs)

    result = solve_coupled_risch_differential_equation(
        field.from_sympy(1),
        real_rhs,
        imaginary_rhs,
        derivation,
    )

    assert not result.has_solution
    assert result.obstruction is CoupledRDEObstruction.DENOMINATOR
    assert result.scalar_result is not None
    assert result.gaussian_normal_form is not None
