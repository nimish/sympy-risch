"""Relative simple-radical descent over monomial lower towers."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_exact_derivative,
)
from sympy_risch.algebraic.relative import (
    RelativeSimpleRadicalRDEResult,
    RelativeSimpleRadicalRDEStatus,
    solve_relative_simple_radical_rde,
)

x, t, y = symbols("x t y")


def simple_radical_tower(*, derivative, degree=2) -> DifferentialTower:
    """Return ``QQ(x)(t)(y)`` with ``Dt=derivative`` and ``y**degree=t``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(1, x, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    lower = rational.extend(
        MonomialExtension(
            Poly(derivative, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    extension = AlgebraicExtension(
        Poly(y**degree - t, y, domain=lower.field()),
    )
    return lower.extend(extension)


def test_generic_exact_dispatch_descends_through_a_primitive_radical() -> None:
    tower = simple_radical_tower(derivative=1 / x)
    extension = tower.levels[-1]
    assert isinstance(extension, AlgebraicExtension)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(y)

    result = solve_exact_derivative(derivation(planted), derivation)

    assert result.has_solution
    assert result.antiderivative == planted
    assert isinstance(result.certificate, RelativeSimpleRadicalRDEResult)
    assert result.certificate.logarithmic_derivative == tower.field(1).from_sympy(
        1 / (2 * x * t),
    )
    assert [component.power for component in result.certificate.components] == [0, 1]
    assert all(component.has_solution for component in result.certificate.components)

    coordinates = extension.power_basis_coordinates(field.from_sympy(x + y))
    assert coordinates == (tower.field(1).from_sympy(x), tower.field(1).one)
    assert extension.from_power_basis_coordinates(coordinates) == field.from_sympy(
        x + y
    )


def test_relative_radical_rde_handles_a_nonzero_primitive_coefficient() -> None:
    """A primitive degree-bound cancellation remains native-field arithmetic."""
    tower = simple_radical_tower(derivative=1 / x)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x + 1) + (t + 1) * y)
    coefficient = field.from_sympy(1 / x)
    right_hand_side = derivation(planted) + coefficient * planted

    result = solve_relative_simple_radical_rde(
        coefficient,
        right_hand_side,
        derivation,
    )

    assert result.status is RelativeSimpleRadicalRDEStatus.SOLVED
    assert result.solution == planted
    assert result.reconstructs
    assert all(component.has_solution for component in result.components)
    assert [component.lower_result.solution for component in result.components] == [
        tower.field(1).from_sympy(x + 1),
        tower.field(1).from_sympy(t + 1),
    ]


def test_high_degree_relative_radical_uses_every_power_coordinate() -> None:
    tower = simple_radical_tower(derivative=t, degree=3)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((x + 1) + (t + x) * y + (x**2 + t) * y**2)
    coefficient = field.from_sympy(2)

    result = solve_relative_simple_radical_rde(
        coefficient,
        derivation(planted) + coefficient * planted,
        derivation,
    )

    assert result.status is RelativeSimpleRadicalRDEStatus.SOLVED
    assert result.reconstructs
    assert result.solution == planted
    assert [component.power for component in result.components] == [0, 1, 2]
    assert [component.coefficient for component in result.components] == [
        tower.field(1).from_sympy(2),
        tower.field(1).from_sympy(7 / 3),
        tower.field(1).from_sympy(8 / 3),
    ]


def test_relative_radical_exact_descent_works_over_a_hypertangent_lower() -> None:
    tower = simple_radical_tower(derivative=1 + t**2)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(x + y)

    result = solve_relative_simple_radical_rde(0, derivation(planted), derivation)

    assert result.status is RelativeSimpleRadicalRDEStatus.SOLVED
    assert result.solution == planted
    assert result.reconstructs
    assert result.logarithmic_derivative == tower.field(1).from_sympy(
        (1 + t**2) / (2 * t)
    )


def test_relative_radical_no_solution_lifts_a_lower_exactness_obstruction() -> None:
    tower = simple_radical_tower(derivative=1 / x)
    field = tower.field()
    derivation = tower.derivation()
    integrand = field.from_sympy(1 / (x * t))

    result = solve_exact_derivative(integrand, derivation)

    assert not result.has_solution
    assert result.obstruction == integrand
    assert isinstance(result.certificate, RelativeSimpleRadicalRDEResult)
    assert result.certificate.status is RelativeSimpleRadicalRDEStatus.NO_SOLUTION
    assert result.certificate.complete_negative


def test_nonlower_coefficient_is_a_coupled_system_boundary() -> None:
    tower = simple_radical_tower(derivative=1 / x)
    field = tower.field()

    result = solve_relative_simple_radical_rde(
        field.generator,
        field.one,
        tower.derivation(),
    )

    assert result.status is RelativeSimpleRadicalRDEStatus.UNSUPPORTED
    assert not result.complete_negative
    assert result.solution is None


def test_unsupported_lower_scalar_rde_stays_an_implementation_boundary() -> None:
    tower = simple_radical_tower(derivative=t**3)
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy(x + y)

    result = solve_relative_simple_radical_rde(0, derivation(planted), derivation)

    assert result.status is RelativeSimpleRadicalRDEStatus.UNSUPPORTED
    assert not result.complete_negative
    with pytest.raises(NotImplementedError, match="lower scalar RDE"):
        solve_exact_derivative(derivation(planted), derivation)
