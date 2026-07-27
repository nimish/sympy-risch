"""Closed-place residue profiles for algebraic Hermite remainders."""

import pytest
from sympy import Poly, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicResidueProfileStatus,
    DifferentialTower,
    InfiniteAlgebraicPlace,
    IntegralBasis,
    MonomialExtension,
    build_algebraic_residue_profile,
    collect_algebraic_all_place_residues,
    trager_algebraic_residue_polynomial,
)

x, y, z = symbols("x y z")


def algebraic_tower(modulus, constants=QQ):
    """Return the standard ``C(x)(y)`` tower and its power integral basis."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return rational.extend(extension), extension, IntegralBasis.power_basis(extension)


def test_profile_includes_ramified_branch_and_infinity_residues() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    collection = collect_algebraic_all_place_residues(1 / x, tower.derivation(), basis)

    profile = build_algebraic_residue_profile(collection, z)

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert profile.residue_polynomial == Poly(z**2 - 4, z, domain=QQ)
    assert {factor.characteristic_polynomial for factor in profile.factors} == {
        Poly(z - 2, z, domain=QQ),
        Poly(z + 2, z, domain=QQ),
    }
    assert profile.finite_unramified_simple_pole_polynomial == Poly(
        1,
        z,
        domain=QQ,
    )

    with pytest.raises(NotImplementedError, match="branch"):
        trager_algebraic_residue_polynomial(1 / x, tower.derivation(), basis, z)


def test_profile_unramified_slice_agrees_with_trager_resultant() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    integrand = y / (x - 1)
    collection = collect_algebraic_all_place_residues(
        integrand,
        tower.derivation(),
        basis,
    )

    profile = build_algebraic_residue_profile(collection, z)
    trager = trager_algebraic_residue_polynomial(
        integrand,
        tower.derivation(),
        basis,
        z,
    )

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert profile.residue_polynomial == Poly(z**3 - z, z, domain=QQ)
    assert profile.finite_unramified_simple_pole_polynomial == (
        trager.residue_polynomial
    )


def test_profile_keeps_distinct_higher_degree_closed_place_factors() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    collection = collect_algebraic_all_place_residues(
        y / ((x - 2) * (x - 8)),
        tower.derivation(),
        basis,
    )

    profile = build_algebraic_residue_profile(collection, z)
    expected_factors = {
        Poly(z**2 - QQ(1, 18), z, domain=QQ),
        Poly(z**2 - QQ(2, 9), z, domain=QQ),
    }

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert {
        factor.characteristic_polynomial for factor in profile.factors
    } == expected_factors
    assert profile.residue_polynomial == Poly(
        (z**2 - QQ(1, 18)) * (z**2 - QQ(2, 9)),
        z,
        domain=QQ,
    )
    assert profile.finite_unramified_simple_pole_polynomial == (
        profile.residue_polynomial
    )


def test_profile_handles_a_degree_four_closed_place() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    collection = collect_algebraic_all_place_residues(
        y / (x**2 + 1),
        tower.derivation(),
        basis,
    )

    profile = build_algebraic_residue_profile(collection, z)

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert len(profile.factors) == 1
    assert profile.factors[0].local_residue.place.degree == 4
    assert profile.factors[0].characteristic_polynomial == Poly(
        z**4 + QQ(1, 16),
        z,
        domain=QQ,
    )
    assert profile.residue_polynomial == profile.factors[0].characteristic_polynomial


def test_unit_profile_does_not_claim_second_kind_exactness() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x**3 + x)
    collection = collect_algebraic_all_place_residues(
        extension.field.one / extension.field.generator,
        tower.derivation(),
        basis,
    )

    profile = build_algebraic_residue_profile(collection, z)

    assert not collection.exact
    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert profile.factors == ()
    assert profile.residue_polynomial == Poly(1, z, domain=QQ)


def test_profile_uses_the_full_degree_of_an_infinite_closed_place() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - 2 * x**2 - 1)
    collection = collect_algebraic_all_place_residues(1 / x, tower.derivation(), basis)

    profile = build_algebraic_residue_profile(collection, z)
    infinity_factor = next(
        factor
        for factor in profile.factors
        if isinstance(factor.local_residue.place, InfiniteAlgebraicPlace)
    )

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert infinity_factor.local_residue.place.degree == 2
    assert infinity_factor.characteristic_polynomial == Poly(
        (z + 1) ** 2,
        z,
        domain=QQ,
    )
    assert profile.residue_polynomial == Poly((z**2 - 1) ** 2, z, domain=QQ)


def test_profile_preserves_an_algebraic_constant_domain() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    tower, _extension, basis = algebraic_tower(y**2 - x, constants)
    collection = collect_algebraic_all_place_residues(1 / x, tower.derivation(), basis)

    profile = build_algebraic_residue_profile(collection, z)

    assert profile.status is AlgebraicResidueProfileStatus.PROFILED
    assert profile.residue_polynomial == Poly(z**2 - 4, z, domain=constants)
    assert all(
        factor.characteristic_polynomial.domain == constants
        for factor in profile.factors
    )


def test_profile_preserves_exact_and_unsupported_collection_outcomes() -> None:
    exact_tower, exact_extension, exact_basis = algebraic_tower(y**3 - y - x)
    exact_part = exact_extension.field.from_sympy(1 / x)
    exact_collection = collect_algebraic_all_place_residues(
        exact_tower.derivation()(exact_part),
        exact_tower.derivation(),
        exact_basis,
    )
    exact_profile = build_algebraic_residue_profile(exact_collection, z)

    unsupported_tower, _unsupported_extension, unsupported_basis = algebraic_tower(
        y**4 - 2 * x**2,
    )
    unsupported_collection = collect_algebraic_all_place_residues(
        1,
        unsupported_tower.derivation(),
        unsupported_basis,
    )
    unsupported_profile = build_algebraic_residue_profile(unsupported_collection, z)

    assert exact_profile.status is AlgebraicResidueProfileStatus.EXACT
    assert exact_profile.residue_polynomial == Poly(1, z, domain=QQ)
    assert exact_profile.factors == ()
    assert unsupported_profile.status is AlgebraicResidueProfileStatus.UNSUPPORTED
    assert unsupported_profile.residue_polynomial is None
    assert unsupported_profile.factors == ()
