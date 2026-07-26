"""Exact finite-dimensional valuation kernels for algebraic function fields."""

from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicNonspecialLatticeSearch,
    AlgebraicNonspecialSearchStatus,
    AlgebraicValuationKernelStatus,
    DifferentialTower,
    MonomialExtension,
    RiemannRochCertificate,
    RiemannRochSpace,
    analyze_algebraic_constant_field,
    finite_lattice_slice,
    search_nonspecial_lattice_slice,
    solve_algebraic_valuation_constraints,
)
from sympy_risch.algebraic_geometry import InfiniteAlgebraicPlace

x, y = symbols("x y")
distinct_constants = st.tuples(
    st.integers(min_value=-4, max_value=4),
    st.integers(min_value=-4, max_value=4),
).filter(lambda values: values[0] != values[1])


def algebraic_model(modulus, constants=QQ):
    """Return a direct ``QQ(x)`` function field with certified constants."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return (
        AlgebraicFunctionField(extension),
        extension,
        analyze_algebraic_constant_field(extension, tower.derivation()),
    )


def test_kernel_solves_cancellation_at_a_ramified_infinite_place() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.generator, field.generator + field.one),
    )

    # Neither ambient vector is regular at infinity, but their difference is.
    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.ambient_basis == (field.generator, field.generator + field.one)
    assert kernel.coordinates == ((-1, 1),)
    assert kernel.basis == (field.one,)
    assert kernel.dimension == 1
    assert all(kernel.contains(element) for element in kernel.basis)
    assert model.principal_divisor(kernel.basis[0]).is_zero
    assert any(
        isinstance(place, InfiniteAlgebraicPlace) for place in kernel.constraint_places
    )
    assert any(place.ramification_index == 2 for place in kernel.constraint_places)


def test_kernel_enforces_finite_and_infinite_ramified_bounds() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field
    parameter = field.generator
    divisor = -model.principal_divisor(parameter)

    kernel = solve_algebraic_valuation_constraints(
        model,
        divisor,
        analysis,
        (field.one, parameter),
    )

    # D = (parameter)_infinity - (parameter)_zero.  The finite zero condition
    # removes 1, while the ramified infinite pole allowance admits parameter.
    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.basis == (parameter,)
    assert kernel.coordinates == ((0, 1),)
    assert divisor.degree == 0
    assert all(kernel.contains(element) for element in kernel.basis)
    assert any(
        not isinstance(place, InfiniteAlgebraicPlace) and place.ramification_index == 2
        for place in kernel.constraint_places
    )
    assert any(
        isinstance(place, InfiniteAlgebraicPlace) for place in kernel.constraint_places
    )


def test_kernel_canonicalizes_zero_and_dependent_ambient_inputs() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (
            field.zero,
            field.generator,
            2 * field.generator,
            field.generator + field.one,
        ),
    )

    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.ambient_basis == (field.generator, field.generator + field.one)
    assert kernel.basis == (field.one,)


def test_kernel_uses_the_certified_exact_algebraic_constant_field() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    model, extension, analysis = algebraic_model(y**2 - x, constants)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.generator, field.generator + field.from_sympy(sqrt(2))),
    )

    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.dimension == 1
    assert model.principal_divisor(kernel.basis[0]).is_zero


def test_kernel_refuses_a_required_nonrational_residue_coordinate() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.from_sympy(1 / (x**2 + 1)),),
    )

    assert kernel.status is AlgebraicValuationKernelStatus.UNSUPPORTED
    assert kernel.ambient_basis == ()
    assert kernel.basis == ()
    assert kernel.coordinates == ()
    assert "residue field equal to the constant field" in kernel.detail


def test_kernel_ignores_a_nonrational_place_that_needs_no_equation() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.one, field.from_sympy(x**2 + 1)),
    )

    # The zeros above x**2 + 1 are not rational over QQ, but both ambient
    # vectors are already regular there.  Only the rational infinite place
    # contributes an equation, so rejecting the closed finite fiber would be
    # an unnecessary loss of a sound result.
    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.basis == (field.one,)


def test_kernel_never_treats_rebased_constants_as_preserved() -> None:
    model, extension, analysis = algebraic_model(y**4 - 2 * x**2)

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (extension.field.one,),
    )

    assert kernel.status is AlgebraicValuationKernelStatus.UNSUPPORTED
    assert kernel.dimension is None
    assert kernel.basis == ()
    assert "preserved constant field" in kernel.detail


def test_kernel_rejects_forged_coordinates_and_partial_unsupported_data() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field
    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.generator, field.generator + field.one),
    )

    with pytest.raises(ValueError, match="exact local nullspace"):
        replace(kernel, coordinates=((1, 0),))
    with pytest.raises(ValueError, match="partial data"):
        replace(kernel, status=AlgebraicValuationKernelStatus.UNSUPPORTED)
    with pytest.raises(TypeError, match="immutable tuples"):
        replace(kernel, basis=list(kernel.basis))


@given(distinct_constants)
@settings(max_examples=8, deadline=None)
def test_generated_planted_constant_cancellations(values) -> None:
    first, second = values
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field

    kernel = solve_algebraic_valuation_constraints(
        model,
        AlgebraicDivisor(model),
        analysis,
        (field.generator + first, field.generator + second),
    )

    assert kernel.status is AlgebraicValuationKernelStatus.CERTIFIED
    assert kernel.dimension == 1
    assert kernel.basis[0] != field.zero
    assert model.principal_divisor(kernel.basis[0]).is_zero


def test_nonspecial_search_uses_existing_global_certificate_after_exhaustion() -> None:
    model, extension, analysis = algebraic_model(y**2 - x**3 + x)
    field = extension.field
    divisor = model.principal_divisor(field.from_sympy(x)).negative_part

    incomplete = search_nonspecial_lattice_slice(
        model,
        divisor,
        analysis,
        max_x_degree=0,
    )
    certified = search_nonspecial_lattice_slice(
        model,
        divisor,
        analysis,
        max_x_degree=1,
    )

    assert model.genus == 1
    assert incomplete.status is AlgebraicNonspecialSearchStatus.INCOMPLETE
    assert incomplete.basis == ()
    assert incomplete.space is None
    assert incomplete.searched_dimension == 1
    assert certified.status is AlgebraicNonspecialSearchStatus.CERTIFIED
    assert isinstance(certified, AlgebraicNonspecialLatticeSearch)
    assert isinstance(certified.space, RiemannRochSpace)
    assert certified.space.certificate is RiemannRochCertificate.NONSPECIAL
    assert certified.basis == (field.one, field.from_sympy(x))
    assert certified.dimension == certified.expected_dimension == 2


def test_finite_lattice_slice_encodes_finite_conditions_before_infinity() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field
    parameter = field.generator
    divisor = -model.principal_divisor(parameter)

    slice_basis = finite_lattice_slice(model, divisor, 0)
    result = search_nonspecial_lattice_slice(
        model,
        divisor,
        analysis,
        max_x_degree=0,
    )

    assert parameter in slice_basis
    assert result.status is AlgebraicNonspecialSearchStatus.CERTIFIED
    assert result.basis == (parameter,)


def test_nonspecial_search_rejects_forged_space_and_nonnegative_partial_data() -> None:
    model, extension, analysis = algebraic_model(y**2 - x)
    field = extension.field
    divisor = model.principal_divisor(field.from_sympy(x)).negative_part
    result = search_nonspecial_lattice_slice(
        model,
        divisor,
        analysis,
        max_x_degree=1,
    )

    assert result.status is AlgebraicNonspecialSearchStatus.CERTIFIED
    with pytest.raises(ValueError, match="does not reach the theorem dimension"):
        replace(result, space=None)
    with pytest.raises(ValueError, match="partial data"):
        replace(result, status=AlgebraicNonspecialSearchStatus.UNSUPPORTED)
