"""Complete principality decisions on exact algebraic function fields."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicPrincipalityStatus,
    DifferentialTower,
    MonomialExtension,
    decide_algebraic_principality,
)
from sympy_risch.algebraic_geometry.function_field import AlgebraicDivisor
from sympy_risch.algebraic_geometry.riemann_roch import RiemannRochSpaceStatus

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return a ``QQ(x)(y)`` algebraic function-field model."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension), extension


def test_principality_finds_a_verified_genus_zero_witness() -> None:
    model, extension = algebraic_model(y**2 - x)
    divisor = model.principal_divisor(extension.field.generator)

    result = decide_algebraic_principality(model, divisor)

    assert result.status is AlgebraicPrincipalityStatus.PRINCIPAL
    assert result.principal
    assert result.witness is not None
    assert model.principal_divisor(result.witness) == divisor
    assert result.riemann_roch_space is not None
    assert result.riemann_roch_space.status is RiemannRochSpaceStatus.CERTIFIED


def test_principality_finds_a_witness_without_a_degree_bound() -> None:
    model, extension = algebraic_model(y**2 - x**3 + x)
    divisor = model.principal_divisor(extension.field.from_sympy(x + y))

    result = decide_algebraic_principality(model, divisor)

    assert result.status is AlgebraicPrincipalityStatus.PRINCIPAL
    assert result.witness is not None
    assert model.principal_divisor(result.witness) == divisor


def test_degree_zero_empty_riemann_roch_space_proves_nonprincipality() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x)
    finite_place = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity_place = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinity_place, -1)))

    result = decide_algebraic_principality(model, divisor)

    assert result.status is AlgebraicPrincipalityStatus.NONPRINCIPAL
    assert result.complete_negative
    assert result.witness is None
    assert result.riemann_roch_space is not None
    assert result.riemann_roch_space.basis == ()


def test_nonzero_degree_is_a_complete_nonprincipality_certificate() -> None:
    model, extension = algebraic_model(y**2 - x**3 + x)
    positive_part = model.principal_divisor(extension.field.from_sympy(x)).positive_part

    result = decide_algebraic_principality(model, positive_part)

    assert result.status is AlgebraicPrincipalityStatus.NONPRINCIPAL
    assert result.complete_negative
    assert result.witness is None
    assert result.riemann_roch_space is None

    with pytest.raises(ValueError, match="certified empty"):
        replace(result, divisor=model.principal_divisor(extension.field.one))
