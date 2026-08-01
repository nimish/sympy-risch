"""Complete principality decisions on exact algebraic function fields."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicNonprincipal,
    AlgebraicPrincipal,
    DifferentialTower,
    MonomialExtension,
    decide_algebraic_principality,
)
from sympy_risch.algebraic_geometry.function_field import AlgebraicDivisor

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

    assert isinstance(result, AlgebraicPrincipal)
    assert not hasattr(result, "status")
    assert model.principal_divisor(result.witness) == divisor
    assert result.divisor_class_search is not None
    assert result.divisor_class_search.order == 1
    with pytest.raises(TypeError):
        replace(result, status="nonprincipal")


def test_principality_finds_a_witness_without_a_degree_bound() -> None:
    model, extension = algebraic_model(y**2 - x**3 + x)
    divisor = model.principal_divisor(extension.field.from_sympy(x + y))

    result = decide_algebraic_principality(model, divisor)

    assert isinstance(result, AlgebraicPrincipal)
    assert model.principal_divisor(result.witness) == divisor


def test_degree_zero_reduced_class_proves_nonprincipality() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x)
    finite_place = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity_place = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinity_place, -1)))

    result = decide_algebraic_principality(model, divisor)

    assert isinstance(result, AlgebraicNonprincipal)
    assert not hasattr(result, "witness")
    assert result.riemann_roch_space is None
    assert result.divisor_class_search is not None
    assert result.divisor_class_search.last_representative is not None
    assert not result.divisor_class_search.last_representative.is_zero


def test_nonzero_degree_is_a_complete_nonprincipality_certificate() -> None:
    model, extension = algebraic_model(y**2 - x**3 + x)
    positive_part = model.principal_divisor(extension.field.from_sympy(x)).positive_part

    result = decide_algebraic_principality(model, positive_part)

    assert isinstance(result, AlgebraicNonprincipal)
    assert not hasattr(result, "witness")
    assert result.riemann_roch_space is None

    with pytest.raises(ValueError, match="complete class proof"):
        replace(result, divisor=model.principal_divisor(extension.field.one))
