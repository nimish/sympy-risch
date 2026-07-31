"""Conservative algebraic-constant transport for closed finite places."""

import pytest
from sympy import Poly, Rational, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicAllPlaceResidueStatus,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    FiniteAlgebraicPlace,
    IntegralBasis,
    MonomialExtension,
    collect_algebraic_all_place_residues,
)
from sympy_risch.algebraic_geometry.base_change import (
    AlgebraicConstantBaseChangeStatus,
    AlgebraicPlaceSplitStatus,
    build_algebraic_constant_base_change,
    build_algebraic_constant_compositum_base_change,
    split_algebraic_place,
    split_finite_algebraic_place,
)
from sympy_risch.algebraic_geometry.constant_extensions import (
    extend_algebraic_constant_compositum,
)

x, y, z = symbols("x y z")


def _source_model():
    """Return the standard geometrically integral QQ(x, y), y**2 = x model."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(y**2 - x, y, domain=rational.field()))
    tower = rational.extend(extension)
    return tower, extension, AlgebraicFunctionField(extension)


def _quadratic_factor() -> Poly:
    """A quadratic whose two roots lie in its exact root field."""
    return Poly(z**2 - Rational(1, 18), z, domain=QQ)


def _algebraic_source_model():
    """Return QQ(sqrt(2))(x, y), y**2 = x with its standard derivation."""
    constants = QQ.algebraic_field(sqrt(2))
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    extension = AlgebraicExtension(Poly(y**2 - x, y, domain=rational.field()))
    tower = rational.extend(extension)
    return constants, tower, extension, AlgebraicFunctionField(extension)


def test_base_change_rebuilds_a_preserved_target_and_commutes_with_d() -> None:
    tower, extension, model = _source_model()
    source_derivation = tower.derivation()

    change = build_algebraic_constant_base_change(
        model,
        source_derivation,
        _quadratic_factor(),
    )

    assert change.status is AlgebraicConstantBaseChangeStatus.CERTIFIED
    assert change.certified
    assert change.target_extension is not None
    assert change.target_derivation is not None
    assert change.target_model is not None
    assert change.target_constant_field_analysis is not None

    source_values = (
        extension.field.from_sympy(x),
        extension.field.generator,
        extension.field.from_sympy(y / (x - 2)),
    )
    for source_value in source_values:
        mapped_derivative = change.embed(source_derivation(source_value))
        target_derivative = change.target_derivation(change.embed(source_value))
        assert (
            mapped_derivative - target_derivative == change.target_extension.field.zero
        )


def test_base_change_embeds_an_existing_algebraic_constant_field() -> None:
    constants, tower, extension, model = _algebraic_source_model()
    derivation = tower.derivation()

    change = build_algebraic_constant_base_change(
        model,
        derivation,
        Poly(z**2 - 3, z, domain=QQ),
    )

    assert change.certified
    assert change.constant_compositum is not None
    assert change.target_constants is change.constant_compositum.target
    assert change.target_constants.ext.minpoly.degree() == 4
    assert change.target_extension is not None
    assert change.target_derivation is not None

    source_constant = constants.from_sympy(sqrt(2))
    mapped_constant = change.constant_compositum.embed(source_constant)
    assert change.target_constants.to_sympy(mapped_constant) == sqrt(2)

    value = extension.field.from_sympy((sqrt(2) * y + 1) / (x - 2))
    assert change.embed(derivation(value)) == change.target_derivation(
        change.embed(value)
    )


def test_two_successive_base_changes_form_an_exact_compositum_chain() -> None:
    tower, extension, model = _source_model()
    first = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        Poly(z**2 - 2, z, domain=QQ),
    )
    assert first.certified
    assert first.target_model is not None
    assert first.target_derivation is not None

    second = build_algebraic_constant_base_change(
        first.target_model,
        first.target_derivation,
        Poly(z**2 - 3, z, domain=QQ),
    )

    assert second.certified
    assert second.target_constants is not None
    assert second.target_constants.ext.minpoly.degree() == 4
    assert second.target_derivation is not None
    value = extension.field.from_sympy((y + 1) / (x - 5))
    embedded = second.embed(first.embed(value))
    embedded_derivative = second.embed(first.embed(tower.derivation()(value)))
    assert second.target_derivation(embedded) == embedded_derivative


def test_common_constant_compositum_splits_independent_factors() -> None:
    """The reusable number-field layer retains one exact source embedding."""
    first = Poly(z**2 - 2, z, domain=QQ)
    second = Poly(z**2 - 3, z, domain=QQ)

    compositum = extend_algebraic_constant_compositum(QQ, (first, second))

    assert not compositum.is_identity
    assert len(compositum.steps) == 2
    assert compositum.target.ext.minpoly.degree() == 4
    assert compositum.embed(QQ(5, 7)) == compositum.target.convert(QQ(5, 7))
    for factor in (first, second):
        _unit, factors = Poly(
            factor.as_expr(),
            z,
            domain=compositum.target,
        ).factor_list()
        assert len(factors) == 2
        assert all(component.degree() == 1 for component, _multiplicity in factors)


def test_common_compositum_is_independent_of_factor_order() -> None:
    """Support and rejection do not depend on caller ordering."""
    first = Poly(z**2 - 2, z, domain=QQ)
    second = Poly(z**2 - 3, z, domain=QQ)

    forward = extend_algebraic_constant_compositum(QQ, (first, second))
    reverse = extend_algebraic_constant_compositum(QQ, (second, first))

    assert forward.target.ext.minpoly.degree() == 4
    assert reverse.target.ext.minpoly.degree() == 4

    cubic = Poly(z**3 - 2, z, domain=QQ)
    discriminant = Poly(z**2 + 3, z, domain=QQ)
    for factors in ((cubic, discriminant), (discriminant, cubic)):
        with pytest.raises(ValueError, match="general splitting fields"):
            extend_algebraic_constant_compositum(QQ, factors)


def test_common_compositum_skips_a_factor_already_in_the_source() -> None:
    """An absorbed residue field does not create a redundant extension step."""
    constants = QQ.algebraic_field(sqrt(2))
    first = Poly(z**2 - 2, z, domain=QQ)
    second = Poly(z**2 - 3, z, domain=QQ)

    compositum = extend_algebraic_constant_compositum(
        constants,
        (first, second),
    )

    assert len(compositum.steps) == 1
    assert compositum.steps[0].factor == second
    assert compositum.target.ext.minpoly.degree() == 4
    source_generator = constants.from_sympy(sqrt(2))
    embedded = compositum.embed(source_generator)
    assert compositum.target.to_sympy(embedded) == sqrt(2)


def test_curve_is_rebuilt_once_over_a_common_constant_compositum() -> None:
    """Independent residue fields share one derivative-commuting target map."""
    tower, extension, model = _source_model()
    factors = (
        Poly(z**2 - 2, z, domain=QQ),
        Poly(z**2 - 3, z, domain=QQ),
    )

    change = build_algebraic_constant_compositum_base_change(
        model,
        tower.derivation(),
        factors,
    )

    assert change.certified
    assert change.factors == factors
    assert change.target_constants is not None
    assert change.target_constants.ext.minpoly.degree() == 4
    assert change.target_derivation is not None
    value = extension.field.from_sympy((y + 1) / (x - 5))
    assert change.target_derivation(change.embed(value)) == change.embed(
        tower.derivation()(value)
    )


def test_finite_closed_places_split_and_transport_local_residues() -> None:
    tower, extension, model = _source_model()
    change = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        _quadratic_factor(),
    )
    assert change.certified
    assert change.target_constants is not None

    alpha = change.target_constants.ext.root
    field = extension.field
    first = model.finite_places(Poly(x - 2, x, domain=QQ))[0]
    second = model.finite_places(Poly(x - 8, x, domain=QQ))[0]
    first_split = split_finite_algebraic_place(change, first)
    second_split = split_algebraic_place(change, second)

    for source_place, split in ((first, first_split), (second, second_split)):
        assert split.status is AlgebraicPlaceSplitStatus.SPLIT
        assert split.split
        assert len(split.target_places) == 2
        assert split.ramification_indices == (1, 1)
        assert sum(place.degree for place in split.target_places) == source_place.degree

    # These are the local residues of y/((x - 2)(x - 8))*dx at x = 2, 8.
    first_residue = first.reduce_local(field.from_sympy(y / (x - 8)))
    second_residue = second.reduce_local(field.from_sympy(y / (x - 2)))
    first_values = first_split.transport_residue(first_residue)
    second_values = second_split.transport_residue(second_residue)

    assert {
        place.residue_field.to_sympy(value)
        for place, value in zip(first_split.target_places, first_values, strict=True)
    } == {alpha, -alpha}
    assert {
        place.residue_field.to_sympy(value)
        for place, value in zip(second_split.target_places, second_values, strict=True)
    } == {2 * alpha, -2 * alpha}


def test_cubic_factor_that_does_not_split_in_its_root_field_is_unsupported() -> None:
    tower, _extension, model = _source_model()

    change = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        Poly(z**3 - 2, z, domain=QQ),
    )

    assert change.status is AlgebraicConstantBaseChangeStatus.UNSUPPORTED
    assert not change.certified
    assert change.target_extension is None
    assert "general splitting fields are unsupported" in change.detail


def test_unsupported_factor_does_not_return_a_partial_common_field() -> None:
    """A nonnormal factor cannot hide behind an otherwise valid quadratic."""
    tower, _extension, model = _source_model()

    change = build_algebraic_constant_compositum_base_change(
        model,
        tower.derivation(),
        (
            Poly(z**2 - 2, z, domain=QQ),
            Poly(z**3 - 3, z, domain=QQ),
        ),
    )

    assert change.status is AlgebraicConstantBaseChangeStatus.UNSUPPORTED
    assert not change.certified
    assert change.constant_compositum is None
    assert change.target_extension is None


def test_algebraic_coefficient_factor_is_an_explicit_boundary() -> None:
    """Foreign constant coefficients must not escape as coercion exceptions."""
    constants, tower, _extension, model = _algebraic_source_model()
    factor = Poly(z**2 - sqrt(2), z, domain=constants)

    change = build_algebraic_constant_compositum_base_change(
        model,
        tower.derivation(),
        (factor,),
    )

    assert change.status is AlgebraicConstantBaseChangeStatus.UNSUPPORTED
    assert not change.certified
    assert "cannot be normalized over QQ" in change.detail


def test_infinite_places_are_an_explicit_boundary_of_the_finite_layer() -> None:
    tower, _extension, model = _source_model()
    change = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        _quadratic_factor(),
    )
    infinity = model.infinite_places()[0]

    split = split_algebraic_place(change, infinity)

    assert split.status is AlgebraicPlaceSplitStatus.UNSUPPORTED
    assert "infinity" in split.detail


def test_unramified_target_places_use_the_base_prime_for_local_residues() -> None:
    """Split target fibres retain a quotient-field invertible uniformizer."""
    tower, extension, model = _source_model()
    change = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        _quadratic_factor(),
    )
    assert change.target_extension is not None
    assert change.target_derivation is not None

    integrand = change.embed(
        extension.field.from_sympy(y / ((x - 2) * (x - 8))),
    )
    result = collect_algebraic_all_place_residues(
        integrand,
        change.target_derivation,
        IntegralBasis.power_basis(change.target_extension),
    )

    assert result.status is AlgebraicAllPlaceResidueStatus.COLLECTED
    assert result.reconstructs_input
    finite = tuple(
        record
        for record in result.residues
        if isinstance(record.place, FiniteAlgebraicPlace)
    )
    assert finite
    assert all(
        record.place.uniformizer
        == change.target_extension.field.from_sympy(record.place.base_prime.as_expr())
        for record in finite
    )


def test_higher_degree_target_residue_field_splits_over_algebraic_constants() -> None:
    """The target residue compositum retains its exact constant embedding."""
    tower, _extension, model = _source_model()
    change = build_algebraic_constant_base_change(
        model,
        tower.derivation(),
        _quadratic_factor(),
    )
    source_place = model.finite_places(Poly(x**2 + 1, x, domain=QQ))[0]

    split = split_algebraic_place(change, source_place)

    assert split.status is AlgebraicPlaceSplitStatus.SPLIT
    assert split.split
    assert len(split.target_places) == 2
    assert split.ramification_indices == (1, 1)
    assert sum(place.degree for place in split.target_places) == source_place.degree
