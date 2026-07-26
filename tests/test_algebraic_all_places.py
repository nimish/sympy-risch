"""All-place algebraic residue collection after Hermite reduction."""

from collections import Counter
from dataclasses import replace

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

import sympy_risch.algebraic as algebraic_module
from sympy_risch import (
    AlgebraicAllPlaceResidueStatus,
    AlgebraicConstantFieldStatus,
    AlgebraicExtension,
    AlgebraicFunctionField,
    AlgebraicLocalDifferentialResidue,
    DifferentialTower,
    FiniteAlgebraicPlace,
    InfiniteAlgebraicPlace,
    IntegralBasis,
    MonomialExtension,
    collect_algebraic_all_place_residues,
    simple_radical_integral_basis,
    trager_algebraic_residue_polynomial,
)

x, y, z = symbols("x y z")


def algebraic_tower(modulus):
    """Return the standard ``QQ(x)(y)`` tower and a power basis."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    tower = rational.extend(extension)
    return tower, extension, IntegralBasis.power_basis(extension)


def test_all_place_collector_uses_a_branch_uniformizer_and_infinity() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    field = extension.field

    result = collect_algebraic_all_place_residues(
        1 / x,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert (
        result.constant_field_analysis.status is AlgebraicConstantFieldStatus.PRESERVED
    )
    assert result.reconstructs_input
    assert result.differential_divisor is not None
    assert len(result.residues) == 2

    finite = next(
        record
        for record in result.residues
        if isinstance(record.place, FiniteAlgebraicPlace)
    )
    infinity = next(
        record
        for record in result.residues
        if isinstance(record.place, InfiniteAlgebraicPlace)
    )
    assert finite.place.is_ramified
    assert finite.pole_order == 1
    assert finite.uniformizer == field.generator
    assert finite.value == finite.place.residue_field.from_sympy(2)
    assert infinity.place.is_ramified
    assert infinity.pole_order == 1
    assert infinity.value == infinity.place.residue_field.from_sympy(-2)

    with pytest.raises(ValueError, match="do not cover every pole place"):
        replace(result, residues=(finite,))

    incorrect_finite = replace(finite, value=finite.place.residue_field.zero)
    with pytest.raises(ValueError, match="local residue is inconsistent"):
        replace(result, residues=(incorrect_finite, infinity))


def test_local_residue_record_rejects_a_value_from_another_field() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x)
    result = collect_algebraic_all_place_residues(1 / x, tower.derivation(), basis)
    finite = next(
        record
        for record in result.residues
        if isinstance(record.place, FiniteAlgebraicPlace)
    )

    with pytest.raises(ValueError, match="place residue field"):
        AlgebraicLocalDifferentialResidue(
            finite.place,
            finite.pole_order,
            finite.uniformizer,
            extension.field.one,
        )


def test_all_place_collector_reuses_verified_global_differential_data(
    monkeypatch,
) -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    calls: Counter[str] = Counter()
    principal_divisor = AlgebraicFunctionField.principal_divisor
    finite_residue = FiniteAlgebraicPlace.differential_residue
    infinity_residue = InfiniteAlgebraicPlace.differential_residue

    def counted_principal_divisor(self, coefficient):
        calls["principal_divisor"] += 1
        return principal_divisor(self, coefficient)

    def counted_finite_residue(self, coefficient):
        calls["finite_residue"] += 1
        return finite_residue(self, coefficient)

    def counted_infinity_residue(self, coefficient):
        calls["infinity_residue"] += 1
        return infinity_residue(self, coefficient)

    monkeypatch.setattr(
        AlgebraicFunctionField,
        "principal_divisor",
        counted_principal_divisor,
    )
    monkeypatch.setattr(
        FiniteAlgebraicPlace,
        "differential_residue",
        counted_finite_residue,
    )
    monkeypatch.setattr(
        InfiniteAlgebraicPlace,
        "differential_residue",
        counted_infinity_residue,
    )

    result = collect_algebraic_all_place_residues(1 / x, tower.derivation(), basis)

    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert calls == Counter(
        principal_divisor=1,
        # The infinity computation uses its reciprocal finite-place chart.
        finite_residue=2,
        infinity_residue=1,
    )


def test_all_place_collector_reports_exactness_without_global_analysis(
    monkeypatch,
) -> None:
    tower, extension, basis = algebraic_tower(y**3 - y - x)
    exact_part = extension.field.from_sympy(1 / x)
    integrand = tower.derivation()(exact_part)

    def unexpected_analysis(*_args, **_kwargs):
        raise AssertionError("exact Hermite reduction must not analyze constants")

    monkeypatch.setattr(
        algebraic_module,
        "analyze_algebraic_constant_field",
        unexpected_analysis,
    )
    result = collect_algebraic_all_place_residues(
        integrand,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.EXACT
    assert result.exact
    assert result.collected
    assert result.constant_field_analysis is None
    assert result.model is None
    assert result.differential_divisor is None
    assert result.residues == ()
    assert result.reconstructs_input


def test_all_place_collector_cross_checks_unramified_finite_trager_residues() -> None:
    tower, _extension, basis = algebraic_tower(y**2 - x)
    integrand = y / (x - 1)

    collected = collect_algebraic_all_place_residues(
        integrand,
        tower.derivation(),
        basis,
    )
    trager = trager_algebraic_residue_polynomial(
        integrand,
        tower.derivation(),
        basis,
        z,
    )

    finite_values = Counter(
        str(record.value)
        for record in collected.residues
        if isinstance(record.place, FiniteAlgebraicPlace)
    )
    trager_values = Counter(str(record.value) for record in trager.place_residues())
    infinity = next(
        record
        for record in collected.residues
        if isinstance(record.place, InfiniteAlgebraicPlace)
    )

    assert collected.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert finite_values == trager_values == Counter({"-1": 1, "1": 1})
    assert infinity.pole_order == 2
    assert infinity.value == infinity.place.residue_field.zero


def test_all_place_collector_normalizes_a_singular_displayed_model() -> None:
    tower, extension, _power_basis = algebraic_tower(y**2 - x**2 * (x + 1))

    result = collect_algebraic_all_place_residues(
        1 / x,
        tower.derivation(),
        simple_radical_integral_basis(extension),
    )

    finite = tuple(
        record
        for record in result.residues
        if isinstance(record.place, FiniteAlgebraicPlace)
    )
    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert len(finite) == 2
    assert all(not record.place.is_ramified for record in finite)
    assert all(record.value == record.place.residue_field.one for record in finite)


def test_all_place_collector_keeps_a_lazily_refined_integral_basis() -> None:
    tower, extension, power_basis = algebraic_tower(y**2 - x**2 * (x + 1))
    field = extension.field

    result = collect_algebraic_all_place_residues(
        field.from_sympy(y / x**3),
        tower.derivation(),
        power_basis,
        lazy=True,
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert result.reconstructs_input
    assert result.basis != power_basis
    assert result.basis.fraction(field.from_sympy(y / x)).denominator.degree() == 0


def test_all_place_collector_retains_a_zero_residue_genus_one_differential() -> None:
    tower, extension, basis = algebraic_tower(y**2 - x**3 + x)
    field = extension.field

    result = collect_algebraic_all_place_residues(
        field.one / field.generator,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert result.model is not None
    assert result.model.genus == 1
    assert result.exact_part == field.zero
    assert result.remainder != field.zero
    assert result.differential_divisor is not None
    assert result.differential_divisor.is_zero
    assert result.residues == ()


def test_all_place_collector_rejects_an_enlarged_constant_field() -> None:
    tower, _extension, basis = algebraic_tower(y**4 - 2 * x**2)

    result = collect_algebraic_all_place_residues(
        1,
        tower.derivation(),
        basis,
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.UNSUPPORTED
    assert result.constant_field_analysis.status is (
        AlgebraicConstantFieldStatus.REBASING_REQUIRED
    )
    assert result.model is None
    assert result.differential_divisor is None
    assert result.residues == ()
    assert result.reconstructs_input
