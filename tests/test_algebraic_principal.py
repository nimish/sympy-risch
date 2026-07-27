"""Bounded principal-divisor searches on exact algebraic function fields."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicPrincipalDivisorStatus,
    DifferentialTower,
    MonomialExtension,
    analyze_algebraic_constant_field,
    search_algebraic_principal_divisor,
)

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return an admitted ``QQ(x)(y)`` model and constant-field analysis."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return (
        AlgebraicFunctionField(extension),
        extension,
        analyze_algebraic_constant_field(extension, tower.derivation()),
    )


def test_principal_search_finds_a_verified_genus_zero_witness() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    divisor = model.principal_divisor(extension.field.generator)

    result = search_algebraic_principal_divisor(
        model,
        divisor,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicPrincipalDivisorStatus.PRINCIPAL
    assert result.principal
    assert not result.incomplete
    assert result.witness is not None
    assert model.principal_divisor(result.witness) == divisor
    assert result.searched_dimension == 1


def test_principal_search_is_bounded_not_a_false_degree_zero_negative() -> None:
    model, extension, analysis = algebraic_model(y**2 - x**3 + x)
    divisor = model.principal_divisor(extension.field.from_sympy(x))

    incomplete = search_algebraic_principal_divisor(
        model,
        divisor,
        analysis,
        max_x_degree=0,
    )
    principal = search_algebraic_principal_divisor(
        model,
        divisor,
        analysis,
        max_x_degree=2,
    )

    assert incomplete.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE
    assert incomplete.incomplete
    assert incomplete.witness is None
    assert incomplete.searched_dimension == 0
    assert principal.status is AlgebraicPrincipalDivisorStatus.PRINCIPAL
    assert principal.witness is not None
    assert model.principal_divisor(principal.witness) == divisor


def test_nonzero_degree_is_a_complete_nonprincipality_certificate() -> None:
    model, extension, analysis = algebraic_model(y**2 - x**3 + x)
    positive_part = model.principal_divisor(extension.field.from_sympy(x)).positive_part

    result = search_algebraic_principal_divisor(
        model,
        positive_part,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicPrincipalDivisorStatus.NONPRINCIPAL
    assert result.complete_negative
    assert result.witness is None
    assert result.searched_dimension is None

    with pytest.raises(ValueError, match="degree-zero"):
        replace(result, status=AlgebraicPrincipalDivisorStatus.INCOMPLETE)


def test_degree_obstruction_does_not_depend_on_constant_preservation() -> None:
    model, extension, analysis = algebraic_model(y**4 - 2 * x**2)
    divisor = model.principal_divisor(extension.field.generator).positive_part

    result = search_algebraic_principal_divisor(
        model,
        divisor,
        analysis,
        max_x_degree=0,
    )

    assert result.status is AlgebraicPrincipalDivisorStatus.NONPRINCIPAL
    assert result.complete_negative


def test_principal_search_requires_a_nonnegative_finite_bound() -> None:
    model, _extension, analysis = algebraic_model(y**2 - x)

    with pytest.raises(ValueError, match="nonnegative"):
        search_algebraic_principal_divisor(
            model,
            model.principal_divisor(_extension.field.generator),
            analysis,
            max_x_degree=-1,
        )
