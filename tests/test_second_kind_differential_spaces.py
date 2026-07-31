"""Residue maps and bounded second-kind differential spaces."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessSecondKindDifferentialSpace,
    MonomialExtension,
    analyze_algebraic_constant_field,
    hess_second_kind_differential_space,
)

x, y = symbols("x y")


def analyzed_model(modulus):
    """Return a direct model and its preserved-constant analysis."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return (
        AlgebraicFunctionField(extension),
        analyze_algebraic_constant_field(extension, tower.derivation()),
    )


def test_second_kind_results_are_factory_owned() -> None:
    with pytest.raises(
        TypeError,
        match="created by hess_second_kind_differential_space",
    ):
        HessSecondKindDifferentialSpace()


def test_simple_poles_on_a_rational_curve_have_no_second_kind_form() -> None:
    model, constants = analyzed_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1), (infinity, 1)))

    space = hess_second_kind_differential_space(model, pole_bound, constants)
    (third_kind_form,) = space.ambient_basis

    assert space.ambient_space.dimension == 1
    assert space.residue_matrix.shape == (2, 1)
    assert space.residue_matrix.rank() == 1
    assert space.dimension == 0
    assert space.basis == ()
    assert not space.contains(third_kind_form)
    residues = space.residue_vector(third_kind_form)
    assert all(coordinate != QQ.zero for coordinate in residues)
    assert sum(residues, QQ.zero) == QQ.zero
    assert space.reconstructs


def test_double_pole_bound_has_an_exact_two_dimensional_residue_kernel() -> None:
    model, constants = analyzed_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 2), (infinity, 2)))

    space = hess_second_kind_differential_space(model, pole_bound, constants)

    assert space.ambient_space.dimension == 3
    assert space.residue_matrix.rank() == 1
    assert space.dimension == 2
    assert all(space.contains(differential) for differential in space.basis)
    assert all(
        all(coordinate == QQ.zero for coordinate in space.residue_vector(differential))
        for differential in space.basis
    )
    assert space.reconstructs


def test_holomorphic_differentials_are_automatically_second_kind() -> None:
    model, constants = analyzed_model(y**2 - x**3 + x)
    pole_bound = AlgebraicDivisor(model)

    space = hess_second_kind_differential_space(model, pole_bound, constants)

    assert space.ambient_space.dimension == model.genus == 1
    assert space.residue_places == ()
    assert space.residue_matrix.shape == (0, 1)
    assert space.dimension == 1
    assert space.reconstructs


def test_closed_degree_two_place_contributes_two_residue_coordinates() -> None:
    model, constants = analyzed_model(y**2 - x**3 + x)
    places = model.finite_places(Poly(x**2 + 1, x, domain=QQ))
    place = places[0]
    pole_bound = AlgebraicDivisor(model, ((place, 1),))

    space = hess_second_kind_differential_space(model, pole_bound, constants)

    assert place.degree == 2
    assert space.residue_coordinate_dimension == 2
    assert space.residue_matrix.shape == (2, 2)
    assert space.residue_matrix.rank() == 1
    assert space.dimension == 1
    assert space.reconstructs
