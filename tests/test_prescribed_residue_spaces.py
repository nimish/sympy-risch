"""Affine differential spaces with prescribed local residues."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessPrescribedResidueInconsistency,
    HessPrescribedResidueSpace,
    HessPrescribedResidueStatus,
    MonomialExtension,
    certify_function_field_constants,
    hess_prescribed_residues,
    hess_second_kind_differential_space,
    solve_hess_prescribed_residues,
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


def test_one_residue_operator_solves_consistent_and_inconsistent_targets() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1), (infinity, 1)))
    operator = hess_second_kind_differential_space(model, pole_bound, constants)

    space = solve_hess_prescribed_residues(
        operator,
        {zero: zero.residue_field.one, infinity: -infinity.residue_field.one},
    )
    inconsistency = solve_hess_prescribed_residues(
        operator,
        {zero: zero.residue_field.one},
    )

    assert isinstance(space, HessPrescribedResidueSpace)
    assert space.homogeneous_space is operator
    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.is_nonempty
    assert space.translation_dimension == 0
    assert space.target_coordinates == (QQ.one, -QQ.one)
    assert space.contains(space.particular_differential)
    assert space.reconstructs
    assert isinstance(inconsistency, HessPrescribedResidueInconsistency)
    assert inconsistency.homogeneous_space is operator
    assert inconsistency.status is HessPrescribedResidueStatus.INCONSISTENT
    assert not inconsistency.is_nonempty
    assert inconsistency.reconstructs


def test_empty_residue_problem_is_the_holomorphic_space() -> None:
    model, constants = certified_model(y**2 - x**3 + x)

    result = hess_prescribed_residues(
        model,
        AlgebraicDivisor(model),
        {},
        constants,
    )

    assert isinstance(result, HessPrescribedResidueSpace)
    space = result
    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.target_coordinates == ()
    assert space.translation_dimension == model.genus == 1
    assert space.particular_differential.coefficient == model.extension.field.zero
    assert space.reconstructs


def test_every_solution_is_a_particular_form_plus_the_second_kind_kernel() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 2), (infinity, 2)))

    result = hess_prescribed_residues(
        model,
        pole_bound,
        {zero: zero.residue_field.one, infinity: -infinity.residue_field.one},
        constants,
    )

    assert isinstance(result, HessPrescribedResidueSpace)
    space = result
    assert space.translation_dimension == 2
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

    operator = hess_second_kind_differential_space(model, pole_bound, constants)
    seed = next(
        differential
        for differential in operator.ambient_basis
        if differential.residue(place) != place.residue_field.zero
    )
    residue = seed.residue(place)
    result = solve_hess_prescribed_residues(
        operator,
        {place: residue},
    )

    assert isinstance(result, HessPrescribedResidueSpace)
    space = result
    assert space.homogeneous_space is operator
    assert place.degree == 2
    assert space.status is HessPrescribedResidueStatus.SOLVABLE
    assert space.target_coordinates == place.constant_coordinates(residue)
    assert space.contains(space.particular_differential)
    assert space.reconstructs


def test_residue_outside_the_pole_bound_is_rejected() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinity,) = model.infinite_places()
    pole_bound = AlgebraicDivisor(model, ((zero, 1),))

    with pytest.raises(ValueError, match="outside the pole-bound support"):
        solve_hess_prescribed_residues(
            hess_second_kind_differential_space(model, pole_bound, constants),
            {infinity: infinity.residue_field.one},
        )


def test_invalid_residue_value_has_a_public_value_error() -> None:
    model, constants = certified_model(y**2 - x)
    (zero,) = model.finite_places(Poly(x, x, domain=QQ))
    pole_bound = AlgebraicDivisor(model, ((zero, 1),))
    operator = hess_second_kind_differential_space(model, pole_bound, constants)

    with pytest.raises(
        ValueError,
        match="does not belong to its place's residue field",
    ):
        solve_hess_prescribed_residues(operator, {zero: object()})
