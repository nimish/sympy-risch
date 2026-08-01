"""Hess Algorithm 24 short large-divisor Riemann--Roch spaces."""

from unittest.mock import patch

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessLargeRiemannRochSpace,
    MonomialExtension,
    compute_riemann_roch_space,
    hess_large_riemann_roch_space,
)
from sympy_risch.algebraic_geometry.hess import HessParametricBasis
from sympy_risch.algebraic_geometry.riemann_roch import (
    constant_coordinate_matrix,
)

x, y = symbols("x y")


def algebraic_model(modulus) -> AlgebraicFunctionField:
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension)


def test_large_space_results_require_complete_evidence() -> None:
    with pytest.raises(TypeError):
        HessLargeRiemannRochSpace()


def test_short_space_expands_to_the_direct_hess_oracle() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 3), (infinity, 2)))

    short = hess_large_riemann_roch_space(model, divisor)
    direct = compute_riemann_roch_space(model, divisor)
    expanded = short.expand()

    assert short.reconstructs
    assert short.verify_expansion()
    assert direct.complete
    assert len(expanded) == short.dimension == direct.dimension
    assert all(direct.contains(element) for element in expanded)
    assert (
        constant_coordinate_matrix(model, direct.basis + expanded).rank()
        == direct.dimension
    )


def test_rank_three_nonradical_space_matches_the_direct_oracle() -> None:
    model = algebraic_model(y**3 + y + x)
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((infinity, 5),))

    short = hess_large_riemann_roch_space(model, divisor)
    direct = compute_riemann_roch_space(model, divisor)
    expanded = short.expand()

    assert model.extension.field.rank == 3
    assert short.reconstructs
    assert short.verify_expansion()
    assert len(expanded) == short.dimension == direct.dimension
    assert (
        constant_coordinate_matrix(model, direct.basis + expanded).rank()
        == direct.dimension
    )


def test_short_generators_follow_algorithm_24_degree_shift() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((point, 3), (infinity, -2)))

    short = hess_large_riemann_roch_space(model, divisor)
    witness = short.reduction.witness.evaluate()

    assert short.invariant_degrees == tuple(
        degree + short.reduction.anchor_multiple
        for degree in short.reduced_basis.invariant_degrees
    )
    assert all(
        generator.evaluate() == witness * reduced_generator
        for generator, reduced_generator in zip(
            short.generators,
            short.reduced_basis.generators,
            strict=True,
        )
    )


def test_negative_degree_space_remains_short_and_empty() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    divisor = AlgebraicDivisor(model, ((point, -1),))

    short = hess_large_riemann_roch_space(model, divisor)

    assert short.dimension == 0
    assert short.expand() == ()
    assert short.reconstructs
    assert short.verify_expansion()


def test_high_height_and_dimension_do_not_expand_the_long_basis() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    coefficient = 257
    divisor = AlgebraicDivisor(
        model,
        ((point, coefficient), (infinity, coefficient)),
    )

    original_expand = HessParametricBasis.expand
    intermediate_dimensions: list[int] = []

    def tracked_expand(
        basis: HessParametricBasis,
        shift: int = 0,
    ):
        expanded = original_expand(basis, shift)
        intermediate_dimensions.append(len(expanded))
        return expanded

    with patch.object(HessParametricBasis, "expand", tracked_expand):
        short = hess_large_riemann_roch_space(model, divisor)

    assert short.reduction.binary_layers == coefficient.bit_length()
    assert len(short.generators) == model.extension.field.rank
    assert short.dimension == divisor.degree + 1 - model.genus
    assert short.dimension > 500
    assert intermediate_dimensions
    assert max(intermediate_dimensions) <= model.genus + short.reduction.anchor.degree
    assert short.reconstructs


def test_split_infinity_space_uses_the_full_parameter_pole_divisor() -> None:
    model = algebraic_model(y**2 - x**2 + 1)
    infinity_places = model.infinite_places()
    divisor = AlgebraicDivisor(
        model,
        ((infinity_places[0], 3), (infinity_places[1], 2)),
    )

    short = hess_large_riemann_roch_space(model, divisor)

    assert short.reduction.anchor == model.base_parameter_pole_divisor()
    assert short.verify_expansion()
