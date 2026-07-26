"""Certified orders, ideals, places, valuations, and divisors."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, S, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFractionalIdeal,
    AlgebraicFunctionField,
    AlgebraicOrder,
    DifferentialTower,
    IntegralBasis,
    MonomialExtension,
    integral_closure,
    polynomial_order,
    simple_radical_integral_basis,
)
from sympy_risch.algebraic_geometry.closure import (
    IntegralClosureResult,
    IntegralClosureStep,
    OrderMaximalityCertificate,
)

x, y, s = symbols("x y s")
distinct_shifts = st.tuples(
    st.integers(min_value=-3, max_value=3),
    st.integers(min_value=-3, max_value=3),
).filter(lambda shifts: shifts[0] != shifts[1])


def algebraic_extension(modulus, constants=QQ):
    """Construct ``constants(x)(y)`` with the standard rational base."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
    )
    return AlgebraicExtension(
        Poly(modulus, y, domain=rational.field()),
    )


def test_polynomial_order_scales_a_rational_function_model() -> None:
    extension = algebraic_extension(y**2 - x / (x + 1))
    field = extension.field

    order = polynomial_order(extension)

    assert order.contains(field.one)
    assert order.contains(field.from_sympy(y * (x + 1)))
    assert not order.contains(field.generator)
    assert order.discriminant == Poly(
        4 * x * (x + 1),
        x,
        domain=QQ,
    )


def test_order_contract_rejects_a_nonmultiplicative_lattice() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    nonorder = IntegralBasis(
        extension,
        (field.one, field.from_sympy(y / x)),
    )

    with pytest.raises(ValueError, match="not closed"):
        AlgebraicOrder(nonorder)


def test_integral_closure_finds_hidden_quadratic_integrality() -> None:
    extension = algebraic_extension(y**2 - x**2 * (x + 1))
    field = extension.field
    explicit = AlgebraicOrder(simple_radical_integral_basis(extension))

    result = integral_closure(extension)

    assert result.changed
    assert result.is_certified
    assert len(result.steps) == 1
    assert result.steps[0].prime == Poly(x, x, domain=QQ)
    assert result.steps[0].index == Poly(x, x, domain=QQ)
    assert result.steps[0].verifies_discriminant_index_formula
    assert result.order.discriminant == Poly(4 * x + 4, x, domain=QQ)
    assert result.order.contains(field.from_sympy(y / x))
    assert result.order.equivalent(explicit)
    assert result.maximality_certificates == ()


def test_integral_closure_step_rejects_tampered_evidence() -> None:
    extension = algebraic_extension(y**2 - x**2 * (x + 1))
    result = integral_closure(extension)
    step = result.steps[0]
    tampered = IntegralClosureStep(
        prime=step.prime,
        previous_discriminant=step.previous_discriminant,
        new_discriminant=step.new_discriminant,
        index=Poly(1, x, domain=QQ),
        previous_order=step.previous_order,
        new_order=step.new_order,
    )
    forged_result = IntegralClosureResult(
        initial_order=result.initial_order,
        order=result.order,
        steps=(tampered,),
        tested_primes=result.tested_primes,
        maximality_certificates=result.maximality_certificates,
    )

    assert not tampered.verifies
    assert not forged_result.is_certified


def test_maximality_certificate_recomputes_the_trace_radical() -> None:
    extension = algebraic_extension(y**3 - x)
    order = polynomial_order(extension)
    prime = Poly(x, x, domain=QQ)
    forged = OrderMaximalityCertificate(
        order=order,
        prime=prime,
        trace_radical=order.unit_ideal(),
        idealizer=order,
    )
    forged_result = IntegralClosureResult(
        initial_order=order,
        order=order,
        steps=(),
        tested_primes=(prime,),
        maximality_certificates=(forged,),
    )

    assert order.trace_radical(prime) != order.unit_ideal()
    assert not forged.verifies
    assert not forged_result.is_certified


@pytest.mark.parametrize(
    ("modulus", "expected_steps"),
    [
        (y**3 - x**3 * (x + 1), 2),
        (y**3 - x**3 * (x + 1) ** 2, 3),
        (y**4 - x**4 * (x + 1), 3),
    ],
)
def test_integral_closure_handles_deep_hidden_radicals(
    modulus,
    expected_steps,
) -> None:
    extension = algebraic_extension(modulus)
    field = extension.field

    result = integral_closure(extension)

    assert len(result.steps) == expected_steps
    assert result.is_certified
    assert result.order.contains(field.from_sympy(y / x))
    assert all(step.verifies_discriminant_index_formula for step in result.steps)
    assert all(certificate.verifies for certificate in result.maximality_certificates)


@given(distinct_shifts)
@settings(max_examples=14, deadline=None)
def test_generated_hidden_quadratic_normalizations(shifts) -> None:
    repeated_shift, residual_shift = shifts
    repeated = x - repeated_shift
    residual = x - residual_shift
    extension = algebraic_extension(y**2 - repeated**2 * residual)
    field = extension.field

    result = integral_closure(extension)

    assert result.is_certified
    assert result.order.contains(field.from_sympy(y / repeated))
    assert result.order.equivalent(
        AlgebraicOrder(simple_radical_integral_basis(extension))
    )


def test_already_maximal_power_order_is_unchanged() -> None:
    extension = algebraic_extension(y**3 - y - x)

    result = integral_closure(extension)

    assert not result.changed
    assert result.order.equivalent(polynomial_order(extension))
    assert result.order.discriminant == Poly(
        4 - 27 * x**2,
        x,
        domain=QQ,
    )


def test_fractional_ideal_arithmetic_in_a_ramified_quadratic_order() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    order = integral_closure(extension).order
    ideal_y = order.principal_ideal(field.generator)
    ideal_x = order.principal_ideal(field.from_sympy(x))
    unit = order.unit_ideal()

    assert ideal_y.multiply(ideal_y) == ideal_x
    assert ideal_y.colon(ideal_y) == unit
    assert ideal_y.multiply(ideal_y.inverse()) == unit
    assert ideal_x.is_subset_of(ideal_y)
    assert ideal_x.add(ideal_y) == ideal_y
    assert ideal_x.intersection(ideal_y) == ideal_x
    assert ideal_x.index_in(ideal_y) == Poly(x, x, domain=QQ)
    assert ideal_y.power(0) == unit
    assert ideal_y.power(-1) == ideal_y.inverse()


def test_nondiagonal_ideal_intersection_satisfies_duality() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    order = integral_closure(extension).order
    left = order.principal_ideal(field.generator + field.one)
    right = order.principal_ideal(field.from_sympy(x))

    intersection = left.intersection(right)

    assert intersection.is_subset_of(left)
    assert intersection.is_subset_of(right)
    assert intersection.inverse() == left.inverse().add(right.inverse())


def test_fractional_ideal_rejects_an_unstable_full_rank_lattice() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    order = integral_closure(extension).order

    with pytest.raises(ValueError, match="not stable"):
        AlgebraicFractionalIdeal(
            order,
            (field.one, field.from_sympy(y / x**2)),
        )


def test_codifferent_and_different_are_mutual_inverses() -> None:
    extension = algebraic_extension(y**3 - y - x)
    order = integral_closure(extension).order

    codifferent = order.codifferent
    different = order.different

    assert codifferent.multiply(different) == order.unit_ideal()
    assert different.is_integral


def test_ramified_finite_place_has_nonbase_uniformizer() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)

    (place,) = model.finite_places(Poly(x, x, domain=QQ))

    assert place.ramification_index == 2
    assert place.relative_degree == 1
    assert place.degree == 1
    assert place.is_ramified
    assert place.valuation(field.from_sympy(x)) == 2
    assert place.valuation(field.generator) == 1
    assert place.valuation(place.uniformizer) == 1
    assert place.different_exponent == 1
    assert place.reduce_integral(field.generator) == place.residue_field.zero


def test_hidden_plane_branch_splits_in_the_maximal_order() -> None:
    extension = algebraic_extension(y**2 - x**2 * (x + 1))
    field = extension.field
    model = AlgebraicFunctionField(extension)

    places = model.finite_places(Poly(x, x, domain=QQ))

    assert len(places) == 2
    assert all(place.ramification_index == 1 for place in places)
    assert {
        place.residue_field.to_sympy(place.reduce_integral(field.from_sympy(y / x)))
        for place in places
    } == {-1, 1}


def test_split_and_inert_closed_fibers_preserve_degree_identity() -> None:
    extension = algebraic_extension(y**2 - x)
    model = AlgebraicFunctionField(extension)

    split = model.finite_places(Poly(x - 1, x, domain=QQ))
    inert = model.finite_places(Poly(x**2 + 1, x, domain=QQ))

    assert len(split) == 2
    assert all(
        place.ramification_index == place.relative_degree == place.degree == 1
        for place in split
    )
    assert len(inert) == 1
    assert inert[0].ramification_index == 1
    assert inert[0].relative_degree == 2
    assert inert[0].degree == 4
    assert (
        sum(place.ramification_index * place.relative_degree for place in inert)
        == extension.field.rank
    )


def test_local_reduction_clears_denominators_away_from_the_place() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    (place,) = model.finite_places(Poly(x, x, domain=QQ))

    regular = field.from_sympy((y + 1) / (x + 1))

    assert place.valuation(regular) == 0
    assert place.residue_field.to_sympy(place.reduce_local(regular)) == 1
    with pytest.raises(ValueError, match="pole"):
        place.reduce_local(field.from_sympy(1 / x))


def test_local_laurent_expansion_is_exact_across_negative_powers() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    (place,) = model.finite_places(Poly(x, x, domain=QQ))
    value = (field.one + field.generator) / field.from_sympy(x)

    expansion = place.local_expansion(value, stop=3)

    assert expansion.start == -2
    assert [
        place.residue_field.to_sympy(expansion.coefficient(exponent))
        for exponent in range(-2, 3)
    ] == [1, 1, 0, 0, 0]
    assert expansion.remainder_valuation is S.Infinity
    assert expansion.coefficient(100) == place.residue_field.zero
    assert expansion.reconstructs


def test_truncated_laurent_expansion_rejects_uncomputed_coefficients() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    place = next(
        place
        for place in model.finite_places(Poly(x - 1, x, domain=QQ))
        if place.residue_field.to_sympy(place.reduce_integral(field.generator)) == 1
    )
    value = field.one / (field.from_sympy(2) - field.generator)

    short = place.local_expansion(value, stop=2)
    long = place.local_expansion(value, stop=9)

    assert short.coefficient(-1) == place.residue_field.zero
    assert short.coefficient(1) == long.coefficient(1)
    assert long.coefficient(8) != place.residue_field.zero
    with pytest.raises(IndexError, match="not computed"):
        short.coefficient(8)


def test_infinity_chart_finds_ramification_and_exact_expansions() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension, infinity_variable=s)

    (place,) = model.infinite_places()
    expansion = place.local_expansion(field.from_sympy(x), stop=1)

    assert model.reciprocal_chart.variable == s
    assert place.ramification_index == 2
    assert place.valuation(field.from_sympy(x)) == -2
    assert place.valuation(field.generator) == -1
    assert place.valuation(place.uniformizer) == 1
    assert expansion.start == -2
    assert place.residue_field.to_sympy(expansion.coefficient(-2)) == 1
    assert expansion.reconstructs


def test_unramified_infinity_splits_on_an_even_hyperelliptic_model() -> None:
    extension = algebraic_extension(y**2 - x**2 - 1)
    field = extension.field
    model = AlgebraicFunctionField(extension)

    places = model.infinite_places()

    assert len(places) == 2
    assert all(place.ramification_index == 1 for place in places)
    assert all(place.valuation(field.from_sympy(x)) == -1 for place in places)
    assert all(place.valuation(field.generator) == -1 for place in places)


@given(
    st.integers(min_value=-4, max_value=4),
    st.integers(min_value=-4, max_value=4),
)
@settings(max_examples=35, deadline=None)
def test_generated_monomial_valuations_match_the_ramification_law(
    x_exponent,
    y_exponent,
) -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    finite = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinite = model.infinite_places()[0]
    value = field.from_sympy(x) ** x_exponent * field.generator**y_exponent
    expected = 2 * x_exponent + y_exponent

    uncached_valuation = finite.valuation(value)
    assert finite.valuation(finite.uniformizer) == 1
    assert uncached_valuation == expected
    assert finite.valuation(value) == expected
    assert infinite.valuation(value) == -expected
    assert finite.valuation(value * field.generator) == expected + 1
    assert finite.valuation(field.one / value) == -expected


def test_general_differential_residues_use_local_uniformizers() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    finite = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinite = model.infinite_places()[0]
    logarithmic = model.differential(field.from_sympy(1 / x))

    assert finite.residue_field.to_sympy(logarithmic.residue(finite)) == 2
    assert infinite.residue_field.to_sympy(logarithmic.residue(infinite)) == -2
    assert (
        finite.residue_field.to_sympy(
            finite.differential_residue(field.one / field.generator)
        )
        == 0
    )


def test_unramified_differential_residues_are_placewise_exact() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    places = model.finite_places(Poly(x - 1, x, domain=QQ))
    differential = model.differential(field.generator / field.from_sympy(x - 1))

    residues = {
        place.residue_field.to_sympy(differential.residue(place)) for place in places
    }

    assert residues == {-1, 1}


def test_principal_divisors_are_complete_and_degree_zero() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)

    divisor_x = model.principal_divisor(field.from_sympy(x))
    divisor_y = model.principal_divisor(field.generator)
    finite = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinite = model.infinite_places()[0]

    assert divisor_x.degree == divisor_y.degree == 0
    assert divisor_x.coefficient(finite) == 2
    assert divisor_x.coefficient(infinite) == -2
    assert divisor_y.coefficient(finite) == 1
    assert divisor_y.coefficient(infinite) == -1
    assert divisor_x == divisor_y.scale(2)
    assert divisor_x.positive_part.is_effective
    assert divisor_x.negative_part.is_effective


@pytest.mark.parametrize("split_value", [1, 2, 3])
def test_principal_divisor_preserves_cancelling_split_support(
    split_value: int,
) -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    places = {
        place.residue_field.to_sympy(place.reduce_integral(field.generator)): place
        for place in model.finite_places(Poly(x - split_value**2, x, domain=QQ))
    }
    split_element = field.from_sympy(split_value)
    value = (field.generator - split_element) / (field.generator + split_element)

    divisor = model.principal_divisor(value)

    assert extension.norm(value) == extension.parent_field.one
    assert divisor.coefficient(places[split_value]) == 1
    assert divisor.coefficient(places[-split_value]) == -1
    assert divisor.degree == 0


def test_fractional_ideal_and_finite_divisor_round_trip() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    ideal = model.order.principal_ideal(field.generator / field.from_sympy(x - 1))

    divisor = model.ideal_divisor(ideal)
    reconstructed = model.finite_ideal(divisor)

    ramified = model.finite_places(Poly(x, x, domain=QQ))[0]
    split = model.finite_places(Poly(x - 1, x, domain=QQ))
    assert divisor.coefficient(ramified) == 1
    assert all(divisor.coefficient(place) == -1 for place in split)
    assert reconstructed == ideal


def test_fractional_ideal_divisor_preserves_cancelling_split_support() -> None:
    extension = algebraic_extension(y**2 - x)
    field = extension.field
    model = AlgebraicFunctionField(extension)
    places = {
        place.residue_field.to_sympy(place.reduce_integral(field.generator)): place
        for place in model.finite_places(Poly(x - 1, x, domain=QQ))
    }
    ideal = places[1].prime_ideal.multiply(places[-1].prime_ideal.inverse())

    divisor = model.ideal_divisor(ideal)

    assert ideal.coordinate_matrix.det() == extension.parent_field.one
    assert divisor.coefficient(places[1]) == 1
    assert divisor.coefficient(places[-1]) == -1
    assert model.finite_ideal(divisor) == ideal


def test_finite_ideal_rejects_infinity_support() -> None:
    extension = algebraic_extension(y**2 - x)
    model = AlgebraicFunctionField(extension)
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((infinity, 1),))

    with pytest.raises(ValueError, match="infinity"):
        model.finite_ideal(divisor)


def test_divisor_arithmetic_combines_and_cancels_support() -> None:
    extension = algebraic_extension(y**2 - x)
    model = AlgebraicFunctionField(extension)
    finite = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinite = model.infinite_places()[0]
    left = AlgebraicDivisor(model, ((finite, 2), (infinite, -1)))
    right = AlgebraicDivisor(model, ((finite, -2), (infinite, 1)))

    assert (left + right).is_zero
    assert left - left == AlgebraicDivisor(model)
    assert left.degree == 1
    assert left.support == (finite, infinite)


def test_canonical_divisor_and_genus_match_hyperelliptic_examples() -> None:
    rational_extension = algebraic_extension(y**2 - x)
    elliptic_extension = algebraic_extension(y**2 - x**3 + x)
    rational_model = AlgebraicFunctionField(rational_extension)
    elliptic_model = AlgebraicFunctionField(elliptic_extension)

    rational_canonical = rational_model.canonical_divisor()
    elliptic_canonical = elliptic_model.canonical_divisor()

    assert rational_canonical.degree == -2
    assert rational_model.genus == 0
    assert elliptic_canonical.degree == 0
    assert elliptic_model.genus == 1
    assert rational_model.differential(1).divisor() == rational_canonical
    assert elliptic_model.differential(1).divisor() == elliptic_canonical


def test_principal_divisor_zero_and_zero_differential_are_rejected() -> None:
    extension = algebraic_extension(y**2 - x)
    model = AlgebraicFunctionField(extension)

    with pytest.raises(ValueError, match="zero element"):
        model.principal_divisor(0)
    with pytest.raises(ValueError, match="zero differential"):
        model.differential(0).divisor()


def test_place_contract_rejects_invalid_base_primes() -> None:
    extension = algebraic_extension(y**2 - x)
    model = AlgebraicFunctionField(extension)

    with pytest.raises(TypeError, match="SymPy Poly"):
        model.finite_places(x)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonconstant"):
        model.finite_places(Poly(1, x, domain=QQ))
    with pytest.raises(ValueError, match="irreducible"):
        model.finite_places(Poly(x**2 - 1, x, domain=QQ))


def test_refinement_limit_is_an_explicit_guard() -> None:
    extension = algebraic_extension(y**3 - x**3 * (x + 1) ** 2)

    with pytest.raises(RuntimeError, match="max_refinements"):
        integral_closure(extension, max_refinements=1)


def test_positive_characteristic_curve_is_explicitly_unsupported() -> None:
    from sympy.polys.domains import GF

    parent = GF(5).frac_field(x)
    with pytest.raises(ValueError, match="characteristic zero"):
        AlgebraicExtension(
            Poly(y**2 - x, y, domain=parent),
        )


def test_algebraic_constant_field_survives_place_construction() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    extension = algebraic_extension(y**2 - x, constants)
    field = extension.field
    model = AlgebraicFunctionField(extension)

    (place,) = model.finite_places(
        Poly(x, x, domain=constants),
    )

    assert place.ramification_index == 2
    assert place.valuation(field.generator) == 1
    assert place.residue_field.domain == constants


def test_genus_rejects_an_unaccounted_constant_field_extension() -> None:
    extension = algebraic_extension(y**2 - 2)
    model = AlgebraicFunctionField(extension)

    assert model.canonical_divisor().degree == -4
    with pytest.raises(ValueError, match="negative genus"):
        _ = model.genus
