"""Differential spaces obtained from Hess Riemann--Roch spaces."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessDifferentialSpace,
    MonomialExtension,
    hess_differential_space,
    hess_large_riemann_roch_space,
    holomorphic_differential_space,
)
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


def test_differential_space_results_require_complete_evidence() -> None:
    with pytest.raises(TypeError):
        HessDifferentialSpace()


@pytest.mark.parametrize(
    ("modulus", "genus"),
    [
        (y**2 - x, 0),
        (y**2 - x**3 + x, 1),
        (y**2 - x**5 + x, 2),
    ],
)
def test_holomorphic_differential_dimension_equals_genus(
    modulus,
    genus: int,
) -> None:
    model = algebraic_model(modulus)

    space = holomorphic_differential_space(model)
    expanded = space.expand()

    assert model.genus == genus
    assert space.dimension == genus
    assert len(expanded) == genus
    assert all(space.contains(differential) for differential in expanded)
    assert space.reconstructs
    assert space.verify_expansion()


def test_genus_two_holomorphic_basis_matches_the_classical_forms() -> None:
    model = algebraic_model(y**2 - x**5 + x)
    field = model.extension.field
    generator = field.generator
    known_coefficients = (
        generator**-1,
        field.from_sympy(x) / generator,
    )

    space = holomorphic_differential_space(model)
    computed_coefficients = tuple(
        differential.coefficient for differential in space.expand()
    )

    assert len(computed_coefficients) == 2
    assert space.verify_expansion()
    assert (
        constant_coordinate_matrix(
            model,
            known_coefficients + computed_coefficients,
        ).rank()
        == 2
    )


def test_differential_spaces_satisfy_riemann_roch_duality() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    divisors = (
        AlgebraicDivisor(model),
        AlgebraicDivisor(model, ((point, 2),)),
        AlgebraicDivisor(model, ((point, 3), (infinity, -2))),
        AlgebraicDivisor(model, ((point, -4), (infinity, -3))),
    )

    for divisor in divisors:
        function_space = hess_large_riemann_roch_space(model, divisor)
        differential_space = hess_differential_space(model, divisor)

        assert function_space.dimension == (
            divisor.degree + 1 - model.genus + differential_space.dimension
        )
        assert differential_space.reconstructs


def test_zero_differential_belongs_to_every_space_but_models_do_not_mix() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    other_model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    space = hess_differential_space(
        model,
        AlgebraicDivisor(model, ((point, 3),)),
    )

    assert space.contains(model.differential(0))
    with pytest.raises(ValueError, match="another global model"):
        space.contains(other_model.differential(0))


def test_large_differential_space_retains_rank_sized_generators() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    (point,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    lower_bound = AlgebraicDivisor(
        model,
        ((point, -257), (infinity, -257)),
    )

    space = hess_differential_space(model, lower_bound)

    assert space.dimension == 514
    assert len(space.generators) == model.extension.field.rank
    assert space.reconstructs
