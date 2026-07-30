"""Limited integration over the complete transcendental PRDE cell."""

import pytest
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    solve_limited_integration,
)

x, t = symbols("x t")


def rational_tower(*, constants=QQ) -> DifferentialTower:
    """Return ``C(x)`` with ``Dx = 1``."""
    return DifferentialTower(constants).extend(
        MonomialExtension(
            Poly(1, x, domain=constants),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_rational_obstructions_cancel_in_the_affine_problem() -> None:
    tower = rational_tower()
    field = tower.field()

    result = solve_limited_integration(
        field.from_sympy(1 + 1 / x),
        (field.from_sympy(1 / x),),
        tower.derivation(),
    )

    assert result.has_solution
    assert not result.proves_no_solution
    assert result.particular is not None
    assert result.particular.coefficients == (-QQ.one,)
    assert result.particular.antiderivative == field.from_sympy(x)
    assert result.direction_basis[0].coefficients == (QQ.zero,)
    assert result.direction_basis[0].antiderivative == field.one


def test_multiple_corrections_return_the_complete_affine_family() -> None:
    tower = rational_tower()
    field = tower.field()

    result = solve_limited_integration(
        field.from_sympy(1 + 1 / x),
        (field.from_sympy(1 / x), field.from_sympy(2 / x)),
        tower.derivation(),
    )

    assert result.particular is not None
    assert result.particular.coefficients == (QQ.zero, QQ(-1, 2))
    assert result.particular.antiderivative == field.from_sympy(x)
    assert tuple(
        (direction.coefficients, direction.antiderivative)
        for direction in result.direction_basis
    ) == (
        ((QQ.one, QQ(-1, 2)), field.zero),
        ((QQ.zero, QQ.zero), field.one),
    )


def test_failed_affine_section_is_a_complete_in_field_obstruction() -> None:
    tower = rational_tower()
    field = tower.field()

    result = solve_limited_integration(
        field.from_sympy(1 / x),
        (),
        tower.derivation(),
    )

    assert not result.has_solution
    assert result.proves_no_solution
    assert result.particular is None
    assert result.direction_basis == ()
    assert result.parametric_result.homogeneous_basis[0].solution == field.one


@pytest.mark.parametrize(
    "top_derivative",
    (
        1 / x,
        t,
        1 + t**2,
    ),
    ids=("primitive", "hyperexponential", "hypertangent"),
)
def test_limited_integration_covers_each_monomial_family(top_derivative) -> None:
    base = rational_tower()
    tower = base.extend(
        MonomialExtension(
            Poly(top_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    field = tower.field()
    derivation = tower.derivation()
    planted = field.from_sympy((t**2 + x) / (t + 1))
    obstruction = field.from_sympy(1 / (t + 2))

    result = solve_limited_integration(
        derivation(planted) + obstruction,
        (obstruction,),
        derivation,
    )

    assert result.particular is not None
    assert result.particular.coefficients == (-QQ.one,)
    assert result.particular.antiderivative == planted


def test_affine_normalization_uses_algebraic_constant_arithmetic() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower = rational_tower(constants=constants)
    field = tower.field()

    result = solve_limited_integration(
        field.from_sympy(1 + sqrt(2) / x),
        (field.from_sympy(1 / x),),
        tower.derivation(),
    )

    assert result.particular is not None
    assert constants.to_sympy(result.particular.coefficients[0]) == -sqrt(2)
    assert result.particular.antiderivative == field.from_sympy(x)
