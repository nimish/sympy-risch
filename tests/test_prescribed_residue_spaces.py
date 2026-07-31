"""Affine differential spaces with prescribed local residues."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessPrescribedResidueSpace,
    HessPrescribedResidueStatus,
    MonomialExtension,
    certify_function_field_constants,
    hess_prescribed_residue_space,
)

x, y = symbols("x y")


def certified_model(modulus):
    """Return a direct model with geometric constant-field evidence."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    model = AlgebraicFunctionField(extension)
    return model, certify_function_field_constants(model)


def test_prescribed_residue_results_are_factory_owned() -> None:
    with pytest.raises(TypeError, match="created by hess_prescribed_residue_space"):
        HessPrescribedResidueSpace()


def test_opposite_simple_residues_determine_one_rational_differential() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1), (infinity, 1)))

    space = hess_prescribed_residue_space(
        model,
        pole_bound,
        {zero: zero.residue_field.one, infinity: -infinity.residue_field.one},
        constants,
    )

    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.is_nonempty
    assert space.translation_dimension == 0
    assert space.target_coordinates == (QQ.one, -QQ.one)
    assert space.particular_differential is not None
    assert space.contains(space.particular_differential)
    assert space.reconstructs


def test_global_residue_obstruction_is_returned_exactly() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1), (infinity, 1)))

    space = hess_prescribed_residue_space(
        model,
        pole_bound,
        {zero: zero.residue_field.one},
        constants,
    )

    assert space.status is HessPrescribedResidueStatus.INCONSISTENT
    assert not space.is_nonempty
    assert space.particular_differential is None
    assert space.obstruction_coordinates is not None
    assert space.reconstructs


def test_empty_residue_problem_is_the_holomorphic_space() -> None:
    model, constants = certified_model(y**2 - x**3 + x)

    space = hess_prescribed_residue_space(
        model,
        AlgebraicDivisor(model),
        {},
        constants,
    )

    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.target_coordinates == ()
    assert space.translation_dimension == model.genus == 1
    assert space.particular_differential is not None
    assert space.particular_differential.coefficient == model.extension.field.zero
    assert space.reconstructs


def test_every_solution_is_a_particular_form_plus_the_second_kind_kernel() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 2), (infinity, 2)))

    space = hess_prescribed_residue_space(
        model,
        pole_bound,
        {zero: zero.residue_field.one, infinity: -infinity.residue_field.one},
        constants,
    )

    assert space.translation_dimension == 2
    assert space.particular_differential is not None
    translated = model.differential(
        space.particular_differential.coefficient
        + space.homogeneous_space.basis[0].coefficient
    )
    assert space.contains(translated)
    assert space.reconstructs


def test_closed_place_residue_is_solved_in_its_native_coordinate_field() -> None:
    model, constants = certified_model(y**2 - x**3 + x)
    place = model.finite_places(Poly(x**2 + 1, x, domain=QQ))[0]
    pole_bound = AlgebraicDivisor(model, ((place, 1),))

    zero_target = hess_prescribed_residue_space(
        model,
        pole_bound,
        {},
        constants,
    )
    seed = next(
        differential
        for differential in zero_target.homogeneous_space.ambient_basis
        if differential.residue(place) != place.residue_field.zero
    )
    residue = seed.residue(place)
    space = hess_prescribed_residue_space(
        model,
        pole_bound,
        {place: residue},
        constants,
    )

    assert place.degree == 2
    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.target_coordinates == place.constant_coordinates(residue)
    assert space.particular_differential is not None
    assert space.contains(space.particular_differential)
    assert space.reconstructs


def test_residue_outside_the_pole_bound_is_rejected() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1),))

    with pytest.raises(ValueError, match="outside the pole-bound support"):
        hess_prescribed_residue_space(
            model,
            pole_bound,
            {infinity: infinity.residue_field.one},
            constants,
        )
