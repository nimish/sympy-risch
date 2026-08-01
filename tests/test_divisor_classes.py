"""General reduced divisor-class arithmetic and positive torsion."""

from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicIntegrationStatus,
    AlgebraicNonprincipal,
    AlgebraicPrincipal,
    AlgebraicTorsionBoundExhausted,
    AlgebraicTorsionRelation,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    decide_algebraic_principality,
    integrate_algebraic,
    reduce_algebraic_divisor_class,
    search_algebraic_divisor_torsion,
)

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension), extension


def test_reduction_returns_a_bounded_equivalent_class() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x)
    point = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((point, 1), (infinity, -1)))

    reduction = reduce_algebraic_divisor_class(model, divisor)

    assert reduction.representative.degree == 0
    assert reduction.anchor_multiple <= model.genus
    assert reduction.representative + reduction.correction_divisor == reduction.divisor


def test_general_torsion_search_finds_two_torsion_on_an_elliptic_curve() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x)
    point = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((point, 1), (infinity, -1)))
    anchor = AlgebraicDivisor(model, ((infinity, 2),))

    result = search_algebraic_divisor_torsion(
        model,
        divisor,
        max_order=3,
        anchor=anchor,
    )

    assert isinstance(result, AlgebraicTorsionRelation)
    assert not hasattr(result, "status")
    assert result.order == 2
    assert result.witness is not None
    assert model.principal_divisor(result.witness) == divisor.scale(2)
    with pytest.raises(TypeError):
        replace(result, status="inconclusive")

    with pytest.raises(TypeError, match="torsion search"):
        replace(
            result,
            witness=model.extension.field.one,
            reductions=(*result.reductions, result.reductions[0]),
        )


def test_exhausted_positive_search_is_explicitly_inconclusive() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x - 1)
    places = model.finite_places(Poly(x, x, domain=QQ))
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((places[0], 1), (infinity, -1)))

    result = search_algebraic_divisor_torsion(
        model,
        divisor,
        max_order=2,
    )

    assert isinstance(result, AlgebraicTorsionBoundExhausted)
    assert not hasattr(result, "order")
    assert not hasattr(result, "witness")
    assert result.last_representative is not None
    assert not result.last_representative.is_zero


def test_principality_uses_the_primitive_class_order() -> None:
    model, _extension = algebraic_model(y**2 - x**3 + x)
    point = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity = model.infinite_places()[0]
    primitive = AlgebraicDivisor(model, ((point, 1), (infinity, -1)))

    principal = decide_algebraic_principality(model, primitive.scale(2))
    nonprincipal = decide_algebraic_principality(model, primitive.scale(3))

    assert isinstance(principal, AlgebraicPrincipal)
    assert principal.divisor_class_search is not None
    assert principal.divisor_class_search.order == 2
    assert model.principal_divisor(principal.witness) == primitive.scale(2)

    assert isinstance(nonprincipal, AlgebraicNonprincipal)
    assert nonprincipal.divisor_class_search is not None
    assert nonprincipal.divisor_class_search.order == 2


def test_degree_29_genus_two_logarithm_is_found_by_divisor_classes() -> None:
    polynomial = x**6 + 4 * x**5 + 6 * x**4 - 12 * x**3 + 33 * x**2 - 16 * x
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(
        Poly(y**2 - polynomial, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    integrand = extension.field.from_sympy(
        (29 * x**2 + 18 * x - 3) * y / polynomial,
    )

    result = integrate_algebraic(
        integrand,
        tower.derivation(),
        IntegralBasis.power_basis(extension),
    )

    assert result.status is AlgebraicIntegrationStatus.ELEMENTARY
    assert result.reconstructs
    assert result.remainder == extension.field.zero
    assert len(result.principality_decisions) == 1
    decision = result.principality_decisions[0]
    assert decision.divisor_class_search is not None
    assert decision.divisor_class_search.order == 29
    assert result.logarithmic_argument is not None
    assert result.logarithmic_coefficient == QQ.one
