"""Hess Algorithm 21 divisor reduction and compact witnesses."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessDivisorReduction,
    HessReductionStrategy,
    MonomialExtension,
    compute_riemann_roch_space,
    reduce_hess_divisor,
)

x, y = symbols("x y")


def algebraic_model(modulus) -> AlgebraicFunctionField:
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension)


def test_reduction_results_require_complete_evidence() -> None:
    with pytest.raises(TypeError):
        HessDivisorReduction()


def test_degree_reduction_returns_exact_bounded_power_product() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 3), (infinity, -2)))

    reduction = reduce_hess_divisor(model, divisor)

    assert reduction.reduced_divisor.is_effective
    assert reduction.reduced_divisor.degree < model.genus + reduction.anchor.degree
    assert reduction.divisor == (
        reduction.reduced_divisor
        + reduction.anchor.scale(reduction.anchor_multiple)
        - reduction.witness.principal_divisor
    )
    assert reduction.reconstructs


def test_maximal_reduction_stops_at_the_last_nonzero_space() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 2), (infinity, -2)))
    anchor = AlgebraicDivisor(model, ((infinity, 1),))

    reduction = reduce_hess_divisor(
        model,
        divisor,
        anchor=anchor,
        strategy=HessReductionStrategy.MAXIMAL,
    )

    next_space = compute_riemann_roch_space(
        model,
        reduction.reduced_divisor - anchor,
    )
    assert next_space.complete
    assert not next_space.basis
    assert reduction.reconstructs


def test_default_anchor_maximal_reduction_uses_parametric_family() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 3), (infinity, -2)))

    reduction = reduce_hess_divisor(
        model,
        divisor,
        strategy=HessReductionStrategy.MAXIMAL,
    )

    next_space = compute_riemann_roch_space(
        model,
        reduction.reduced_divisor - reduction.anchor,
    )
    assert next_space.complete
    assert not next_space.basis
    assert reduction.reconstructs


def test_custom_anchor_degree_reduction_uses_complete_spaces() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 3), (infinity, -2)))
    anchor = AlgebraicDivisor(model, ((infinity, 1),))

    reduction = reduce_hess_divisor(model, divisor, anchor=anchor)

    assert reduction.anchor == anchor
    assert reduction.reduced_divisor.is_effective
    assert reduction.reduced_divisor.degree < model.genus + anchor.degree
    assert reduction.reconstructs


def test_default_anchor_uses_every_place_above_infinity() -> None:
    model = algebraic_model(y**2 - x**2 + 1)
    infinity_places = model.infinite_places()
    assert len(infinity_places) == 2
    divisor = AlgebraicDivisor(
        model,
        ((infinity_places[0], 3), (infinity_places[1], -2)),
    )

    reduction = reduce_hess_divisor(model, divisor)

    assert reduction.anchor == model.base_parameter_pole_divisor()
    assert reduction.anchor.support == infinity_places
    assert reduction.reconstructs


def test_binary_reduction_work_grows_with_coefficient_bit_length() -> None:
    model = algebraic_model(y**2 - x**2 + 1)
    infinity_places = model.infinite_places()
    coefficient = 257
    divisor = AlgebraicDivisor(
        model,
        (
            (infinity_places[0], coefficient),
            (infinity_places[1], -coefficient),
        ),
    )

    reduction = reduce_hess_divisor(model, divisor)

    assert reduction.binary_layers == coefficient.bit_length()
    assert reduction.support_reductions == 4
    assert reduction.composition_reductions == coefficient.bit_length() - 1
    assert len(reduction.witness.factors) <= (
        reduction.support_reductions + reduction.composition_reductions
    )
    assert reduction.divisor == (
        reduction.reduced_divisor
        + reduction.anchor.scale(reduction.anchor_multiple)
        - reduction.witness.principal_divisor
    )


def test_small_signed_divisor_family_reconstructs_independently() -> None:
    """Exercise both coefficient signs and unequal binary expansions."""
    model = algebraic_model(y**2 - x**2 + 1)
    places = model.infinite_places()

    for left in (-3, -1, 0, 2):
        for right in (-2, 0, 1):
            divisor = AlgebraicDivisor(
                model,
                ((places[0], left), (places[1], right)),
            )
            reduction = reduce_hess_divisor(model, divisor)

            assert reduction.reduced_divisor.is_effective
            assert reduction.reconstructs


def test_zero_divisor_has_the_empty_reduction() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    divisor = AlgebraicDivisor(model)

    reduction = reduce_hess_divisor(model, divisor)

    assert reduction.reduced_divisor.is_zero
    assert reduction.anchor_multiple == 0
    assert not reduction.witness.factors
    assert reduction.binary_layers == 0
    assert reduction.reconstructs
