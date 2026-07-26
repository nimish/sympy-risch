"""Unramified finite places and exact algebraic residues."""

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

import sympy_risch
from sympy_risch import (
    AlgebraicExtension,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    trager_algebraic_place_residues,
    trager_algebraic_residue_polynomial,
)
from sympy_risch.algebraic.trager import unramified_finite_places

x, y, z = symbols("x y z")


def test_only_canonical_places_are_exported_from_the_package_root() -> None:
    assert {
        "FiniteAlgebraicPlace",
        "InfiniteAlgebraicPlace",
    } <= set(sympy_risch.__all__)
    assert {
        "AlgebraicPlace",
        "IntegralClosureStep",
        "OrderMaximalityCertificate",
        "ReciprocalFunctionField",
        "UnramifiedFinitePlace",
        "unramified_finite_places",
    }.isdisjoint(sympy_risch.__all__)


def algebraic_tower(modulus):
    """Return ``QQ(x)(y)``, its extension, and its power basis."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(
        Poly(modulus, y, domain=rational.field()),
    )
    tower = rational.extend(extension)
    return tower, extension, IntegralBasis.power_basis(extension)


def residue_satisfies_trager_polynomial(record, result) -> bool:
    """Test a place residue in its own exact residue field."""
    residue_field = record.place.residue_field
    residue_expression = residue_field.to_sympy(record.value)
    polynomial_value = result.residue_polynomial.as_expr().subs(
        result.residue_symbol,
        residue_expression,
    )
    return residue_field.from_sympy(polynomial_value) == residue_field.zero


def test_unramified_places_split_a_rational_fiber() -> None:
    _, extension, _ = algebraic_tower(y**2 - x)

    places = unramified_finite_places(
        extension,
        Poly(x - 1, x, domain=QQ),
    )

    assert len(places) == 2
    assert {place.fiber_factor.as_expr() for place in places} == {y - 1, y + 1}
    for place in places:
        assert place.base_degree == 1
        assert place.relative_degree == 1
        assert place.degree == 1
        assert place.ramification_index == 1
        assert place.reduce_polynomial(place.defining_polynomial) == (
            place.residue_field.zero
        )
        assert place.reduce_polynomial(place.base_prime) == (place.residue_field.zero)
        assert place.uniformizer == extension.field.from_sympy(x - 1)


def test_place_residues_refine_the_trager_polynomial() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)
    result = trager_algebraic_residue_polynomial(
        y / (x - 1),
        tower.derivation(),
        basis,
        z,
    )

    records = result.place_residues()

    assert len(records) == 2
    assert records == trager_algebraic_place_residues(result)
    assert {
        record.place.residue_field.to_sympy(record.value) for record in records
    } == {-1, 1}
    assert all(
        residue_satisfies_trager_polynomial(record, result) for record in records
    )


def test_closed_place_residue_can_have_nontrivial_relative_degree() -> None:
    tower, _, basis = algebraic_tower(y**3 + y - x)
    result = trager_algebraic_residue_polynomial(
        y / x,
        tower.derivation(),
        basis,
        z,
    )

    records = trager_algebraic_place_residues(result)

    assert sorted(record.place.relative_degree for record in records) == [1, 2]
    assert sum(record.place.degree for record in records) == 3
    assert all(
        residue_satisfies_trager_polynomial(record, result) for record in records
    )


def test_irreducible_base_prime_builds_a_nested_residue_field() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)
    result = trager_algebraic_residue_polynomial(
        y / (x**2 + 1),
        tower.derivation(),
        basis,
        z,
    )

    records = trager_algebraic_place_residues(result)

    assert len(records) == 1
    record = records[0]
    assert record.place.base_degree == 2
    assert record.place.relative_degree == 2
    assert record.place.degree == 4
    assert residue_satisfies_trager_polynomial(record, result)


def test_place_residues_are_empty_without_finite_poles() -> None:
    tower, _, basis = algebraic_tower(y**2 - x)
    result = trager_algebraic_residue_polynomial(
        x + y,
        tower.derivation(),
        basis,
        z,
    )

    assert trager_algebraic_place_residues(result) == ()


def test_branch_fiber_is_explicitly_rejected() -> None:
    _, extension, _ = algebraic_tower(y**2 - x)

    with pytest.raises(
        NotImplementedError,
        match="branch locus x",
    ):
        unramified_finite_places(
            extension,
            Poly(x, x, domain=QQ),
        )


def test_finite_place_contract_rejects_invalid_base_primes() -> None:
    _, extension, _ = algebraic_tower(y**2 - x)

    with pytest.raises(TypeError, match="SymPy Poly"):
        unramified_finite_places(extension, x - 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match"):
        unramified_finite_places(
            extension,
            Poly(z - 1, z, domain=QQ),
        )
    with pytest.raises(ValueError, match="nonconstant"):
        unramified_finite_places(
            extension,
            Poly(1, x, domain=QQ),
        )
    with pytest.raises(ValueError, match="irreducible"):
        unramified_finite_places(
            extension,
            Poly(x**2 - 1, x, domain=QQ),
        )


def test_place_residue_rejects_a_nonpole() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    result = trager_algebraic_residue_polynomial(
        y / (x - 1),
        tower.derivation(),
        basis,
        z,
    )
    place = unramified_finite_places(
        extension,
        Poly(x + 1, x, domain=QQ),
    )[0]

    with pytest.raises(ValueError, match="does not lie over a pole"):
        place.residue(result)
