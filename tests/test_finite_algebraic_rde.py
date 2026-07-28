"""Recursive scalar RDEs over finite algebraic constant fields."""

import pytest
from sympy import I, Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    DifferentialTower,
    MonomialExtension,
    assess_scalar_rde_admission,
    solve_risch_differential_equation,
)
from sympy_risch.reduction.polynomials import (
    canonicalize_polynomial,
    exact_polynomial_equal,
    polynomial_inverse_modulo,
)

x, t, u = symbols("x t u")


def _finite_constant_tower(family: str) -> DifferentialTower:
    constants = QQ.algebraic_field(sqrt(2))
    base = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    derivative = {
        "P": 1 / x,
        "E": t,
        "T": 1 + t**2,
    }[family]
    return base.extend(
        MonomialExtension(Poly(derivative, t, domain=base.field())),
    )


@pytest.mark.parametrize("family", ("P", "E", "T"))
def test_recursive_finite_constant_rde_reconstructs_planted_solution(
    family: str,
) -> None:
    tower = _finite_constant_tower(family)
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((t + x) / (t + 1))
    coefficient = field.from_sympy(sqrt(2))

    result = solve_risch_differential_equation(
        coefficient,
        derivation(solution) + coefficient * solution,
        derivation,
    )

    assert assess_scalar_rde_admission(derivation).accepted
    assert result.has_solution
    assert result.reconstructs


@pytest.mark.parametrize(
    ("family", "coefficient", "right_hand_side"),
    (
        ("P", x, 1 / t),
        ("E", 0, 1 / x),
        ("T", 0, 1 / x),
    ),
)
def test_recursive_finite_constant_rde_returns_proved_obstruction(
    family: str,
    coefficient: object,
    right_hand_side: object,
) -> None:
    tower = _finite_constant_tower(family)
    field = tower.field()

    result = solve_risch_differential_equation(
        field.from_sympy(coefficient),
        field.from_sympy(right_hand_side),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.obstruction is not None
    assert not result.certificate.has_solution


def test_real_tangent_rde_recurses_through_a_primitive_native_constant_level() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    base = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    primitive = base.extend(
        MonomialExtension(Poly(1 / x, t, domain=base.field())),
    )
    tower = primitive.extend(
        MonomialExtension(
            Poly((t + 1) * (1 + u**2), u, domain=primitive.field()),
        ),
    )
    field = tower.field()
    derivation = tower.derivation()
    solution = field.from_sympy((u + t + x) / (u + 1))
    coefficient = field.from_sympy(sqrt(2) / (t + 1))

    result = solve_risch_differential_equation(
        coefficient,
        derivation(solution) + coefficient * solution,
        derivation,
    )

    assert result.has_solution
    assert result.reconstructs


def test_algebraic_fraction_polynomial_inverse_uses_canonical_units() -> None:
    constants = QQ.algebraic_field(sqrt(2), I)
    coefficient_field = constants.frac_field(x)
    modulus = Poly(t + 1, t, domain=coefficient_field)
    value = Poly(-2 * I * t, t, domain=coefficient_field)

    inverse = polynomial_inverse_modulo(value, modulus)
    residue = canonicalize_polynomial((value * inverse).rem(modulus))

    assert exact_polynomial_equal(residue, modulus.one)
    assert residue == modulus.one
