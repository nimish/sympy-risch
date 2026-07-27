"""Dependency classification for proposed transcendental extensions."""

from math import lcm

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import I, Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
    PrimitiveExtensionAnalysis,
    analyze_hyperexponential_extension,
    analyze_hypertangent_extension,
    certify_primitive_extension,
)

x, t, y, z = symbols("x t y z")


def rational_base(
    *,
    generator=x,
    generator_derivative=1,
) -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(
            Poly(generator_derivative, generator, domain=QQ),
            ConstantFieldStatus.PRESERVED,
        ),
    )


def proposed_hyperexponential(
    eta,
    *,
    base=None,
    constant_field_status=ConstantFieldStatus.UNKNOWN,
) -> tuple[DifferentialTower, MonomialExtension]:
    if base is None:
        base = rational_base()
    extension = MonomialExtension(
        Poly(eta * t, t, domain=base.field()),
        constant_field_status,
    )
    return base, extension


def proposed_hypertangent(
    eta,
    *,
    base=None,
    constant_field_status=ConstantFieldStatus.UNKNOWN,
) -> tuple[DifferentialTower, MonomialExtension]:
    if base is None:
        base = rational_base()
    extension = MonomialExtension(
        Poly(eta * (1 + t**2), t, domain=base.field()),
        constant_field_status,
    )
    return base, extension


def test_primitive_factory_certifies_the_rational_base() -> None:
    analysis = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        DifferentialTower(QQ).derivation(),
    )

    assert isinstance(analysis, PrimitiveExtensionAnalysis)
    assert analysis.status is ConstantFieldStatus.PRESERVED
    assert not analysis.is_dependent
    assert analysis.dependency_witness is None
    assert analysis.constant_relation is None
    assert analysis.detail is None
    assert analysis.exact_result is not None
    assert not analysis.exact_result.has_solution
    assert analysis.lower_fundamental_signature is None


def test_primitive_factory_certifies_recursive_independence() -> None:
    base_analysis = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        DifferentialTower(QQ).derivation(),
    )
    base = DifferentialTower(QQ).extend(base_analysis.extension)
    logarithm_analysis = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )

    assert logarithm_analysis.status is ConstantFieldStatus.PRESERVED
    assert not logarithm_analysis.is_dependent
    assert logarithm_analysis.exact_result is not None
    assert logarithm_analysis.exact_result.obstruction == base.field().from_sympy(1 / x)
    assert logarithm_analysis.lower_fundamental_signature is not None
    assert tuple(
        direction.level
        for direction in logarithm_analysis.lower_fundamental_signature.directions
    ) == (0,)


def test_primitive_factory_finds_dependencies_between_generators() -> None:
    base_analysis = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        DifferentialTower(QQ).derivation(),
    )
    base = DifferentialTower(QQ).extend(base_analysis.extension)
    logarithm_analysis = certify_primitive_extension(
        Poly(1 / x, t, domain=base.field()),
        base.derivation(),
    )
    logarithmic_tower = base.extend(logarithm_analysis.extension)

    dependent = certify_primitive_extension(
        Poly(2 / x, z, domain=logarithmic_tower.field()),
        logarithmic_tower.derivation(),
    )

    assert dependent.status is ConstantFieldStatus.ENLARGED
    assert dependent.is_dependent
    assert dependent.exact_result is not None
    assert dependent.dependency_witness == logarithmic_tower.field().from_sympy(2 * t)
    assert dependent.constant_relation is not None
    proposed_tower = logarithmic_tower.extend(dependent.extension)
    assert (
        proposed_tower.derivation()(dependent.constant_relation)
        == proposed_tower.field().zero
    )


def test_primitive_factory_finds_a_direct_lower_field_antiderivative() -> None:
    base = rational_base()
    dependent = certify_primitive_extension(
        Poly(1, t, domain=base.field()),
        base.derivation(),
    )

    assert dependent.status is ConstantFieldStatus.ENLARGED
    assert dependent.dependency_witness == base.field().from_sympy(x)
    assert dependent.constant_relation is not None


def test_primitive_factory_detects_an_exact_derivative_over_a_simple_radical() -> None:
    base = rational_base()
    algebraic = base.extend(
        AlgebraicExtension(
            Poly(y**2 - x, y, domain=base.field()),
        )
    )

    analysis = certify_primitive_extension(
        Poly(y, z, domain=algebraic.field()),
        algebraic.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.ENLARGED
    assert analysis.exact_result is not None
    assert analysis.exact_result.has_solution
    assert analysis.dependency_witness is not None
    assert algebraic.derivation()(
        analysis.dependency_witness
    ) == algebraic.field().from_sympy(y)


def test_primitive_factory_keeps_nondiagonal_algebraic_parents_unknown() -> None:
    base = rational_base()
    algebraic = base.extend(
        AlgebraicExtension(
            Poly(y**3 + y - x, y, domain=base.field()),
        )
    )

    analysis = certify_primitive_extension(
        Poly(y, z, domain=algebraic.field()),
        algebraic.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.exact_result is None
    assert analysis.detail is not None
    assert "pure simple radical" in analysis.detail
    assert analysis.lower_fundamental_signature is None


def test_primitive_factory_replays_an_unannotated_lower_tower() -> None:
    uncertified_base = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ))
    )

    analysis = certify_primitive_extension(
        Poly(1 / x, t, domain=uncertified_base.field()),
        uncertified_base.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.PRESERVED
    assert analysis.detail is None
    assert analysis.lower_fundamental_signature is not None


def test_primitive_factory_leaves_an_unsupported_lower_family_unknown() -> None:
    base = rational_base()
    unsupported = base.extend(
        MonomialExtension(
            Poly(1 + t**3, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )

    analysis = certify_primitive_extension(
        Poly(1, z, domain=unsupported.field()),
        unsupported.derivation(),
    )

    assert analysis.status is ConstantFieldStatus.UNKNOWN
    assert analysis.exact_result is None
    assert analysis.lower_fundamental_signature is None
    assert analysis.detail is not None
    assert "without replayed constant preservation" in analysis.detail


@pytest.mark.parametrize(
    ("generator_derivative", "dependent_eta", "independent_eta"),
    [
        (t, t, t / (t - 1)),
        (1 + t**2, 1 + t**2, 2 * t),
    ],
)
def test_primitive_factory_uses_supported_transcendental_exact_decisions(
    generator_derivative,
    dependent_eta,
    independent_eta,
) -> None:
    base = rational_base()
    transcendental = base.extend(
        MonomialExtension(
            Poly(generator_derivative, t, domain=base.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )

    dependent = certify_primitive_extension(
        Poly(dependent_eta, z, domain=transcendental.field()),
        transcendental.derivation(),
    )
    independent = certify_primitive_extension(
        Poly(independent_eta, z, domain=transcendental.field()),
        transcendental.derivation(),
    )

    assert dependent.status is ConstantFieldStatus.ENLARGED
    assert dependent.dependency_witness == transcendental.field().from_sympy(t)
    assert independent.status is ConstantFieldStatus.PRESERVED
    assert independent.dependency_witness is None


def test_primitive_factory_rejects_invalid_proposals() -> None:
    base = rational_base()

    with pytest.raises(ValueError, match="requires Dt in the lower field"):
        certify_primitive_extension(
            Poly(t, t, domain=base.field()),
            base.derivation(),
        )
    with pytest.raises(ValueError, match="parent field does not match"):
        certify_primitive_extension(
            Poly(1, t, domain=QQ),
            base.derivation(),
        )
    with pytest.raises(TypeError, match="requires a Derivation"):
        certify_primitive_extension(
            Poly(1, x, domain=QQ),
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "eta",
        "expected_kind",
        "expected_power",
        "expected_argument",
        "expected_exponent",
    ),
    [
        (
            1 / x,
            HyperexponentialDependencyKind.RATIONAL,
            1,
            x,
            None,
        ),
        (
            -3 / x,
            HyperexponentialDependencyKind.RATIONAL,
            1,
            x**-3,
            None,
        ),
        (
            1 / (2 * x),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            x,
            None,
        ),
        (
            (2 * x + 1) / (2 * (x**2 + x + 1)),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            x**2 + x + 1,
            None,
        ),
        (
            1 / (2 * x) + 1 / (3 * (x - 1)),
            HyperexponentialDependencyKind.ALGEBRAIC,
            6,
            x**3 * (x - 1) ** 2,
            None,
        ),
        (
            1,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            x,
        ),
        (
            2 * x,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            x**2,
        ),
        (
            -1 / x**2,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            1 / x,
        ),
        (
            1 / (x**2 + 1),
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            None,
        ),
        (
            1 + 1 / (2 * x),
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            None,
        ),
    ],
)
def test_hyperexponential_dependency_examples(
    eta,
    expected_kind,
    expected_power,
    expected_argument,
    expected_exponent,
) -> None:
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is expected_kind
    assert result.minimal_power == expected_power
    assert result.eta == base.field().from_sympy(eta)
    if expected_argument is None:
        assert result.relation_argument is None
    else:
        assert cancel(result.relation_argument - expected_argument) == 0
        assert (
            cancel(
                expected_power * eta
                - result.relation_argument.diff(x) / result.relation_argument
            )
            == 0
        )
    if expected_exponent is None:
        assert result.exponential_exponent is None
    else:
        assert cancel(result.exponential_exponent - expected_exponent) == 0
        assert cancel(result.exponential_exponent.diff(x) - eta) == 0


@pytest.mark.parametrize(
    ("eta", "expected_power"),
    [
        (1 / x, 1),
        (1 / (2 * x), 2),
        (1 / (2 * x) + 1 / (3 * (x - 1)), 6),
        (-3 / x, 1),
    ],
)
def test_dependency_relation_is_constant_in_the_proposed_tower(
    eta,
    expected_power,
) -> None:
    base, extension = proposed_hyperexponential(eta)
    analysis = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )
    tower = base.extend(extension)
    field = tower.field()
    relation = field.from_sympy(t**expected_power / analysis.relation_argument)

    assert tower.derivation()(relation) == field.zero


def test_dependency_and_named_exponential_presentation_are_independent() -> None:
    dependent_base, dependent_extension = proposed_hyperexponential(1 / x)
    exponential_base, exponential_extension = proposed_hyperexponential(2 * x)

    dependent = analyze_hyperexponential_extension(
        dependent_extension,
        dependent_base.derivation(),
    )
    exponential = analyze_hyperexponential_extension(
        exponential_extension,
        exponential_base.derivation(),
    )

    assert dependent.is_dependent
    assert not dependent.has_exponential_presentation
    assert not exponential.is_dependent
    assert exponential.has_exponential_presentation
    assert exponential.exponential_exponent == x**2


@pytest.mark.parametrize(
    (
        "eta",
        "declared",
        "expected_inferred",
        "expected_conflict",
    ),
    [
        (
            1,
            ConstantFieldStatus.UNKNOWN,
            ConstantFieldStatus.PRESERVED,
            False,
        ),
        (
            1,
            ConstantFieldStatus.PRESERVED,
            ConstantFieldStatus.PRESERVED,
            False,
        ),
        (
            1,
            ConstantFieldStatus.ENLARGED,
            ConstantFieldStatus.PRESERVED,
            True,
        ),
        (
            1 / x,
            ConstantFieldStatus.UNKNOWN,
            ConstantFieldStatus.ENLARGED,
            False,
        ),
        (
            1 / x,
            ConstantFieldStatus.PRESERVED,
            ConstantFieldStatus.ENLARGED,
            True,
        ),
        (
            1 / x,
            ConstantFieldStatus.ENLARGED,
            ConstantFieldStatus.ENLARGED,
            False,
        ),
    ],
)
def test_dependency_analysis_checks_declared_constant_field_status(
    eta,
    declared,
    expected_inferred,
    expected_conflict,
) -> None:
    base, extension = proposed_hyperexponential(
        eta,
        constant_field_status=declared,
    )

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.inferred_constant_field_status is expected_inferred
    assert result.conflicts_with_declared_constant_field_status is expected_conflict


def test_transcendental_obstructions_retain_their_certificates() -> None:
    cases = [
        (1, "has_polynomial_obstruction"),
        (1 / x**2, "has_repeated_pole_obstruction"),
        (1 / (x**2 + 1), "has_nonlogarithmic_residue_obstruction"),
    ]

    for eta, obstruction_property in cases:
        base, extension = proposed_hyperexponential(eta)
        result = analyze_hyperexponential_extension(
            extension,
            base.derivation(),
        )

        assert result.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
        assert getattr(result.logarithmic_result, obstruction_property)
        assert result.minimal_power is None
        assert result.relation_argument is None


@pytest.mark.parametrize(
    (
        "eta",
        "expected_kind",
        "expected_power",
        "expected_argument",
        "expected_exponent",
    ),
    [
        (
            1 / (x * t),
            HyperexponentialDependencyKind.RATIONAL,
            1,
            t,
            None,
        ),
        (
            1 / (2 * x * t),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            t,
            None,
        ),
        (
            1,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            x,
        ),
    ],
)
def test_dependency_analysis_descends_through_a_primitive_level(
    eta,
    expected_kind,
    expected_power,
    expected_argument,
    expected_exponent,
) -> None:
    rational = rational_base()
    primitive = rational.extend(
        MonomialExtension(
            Poly(1 / x, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        ),
    )
    extension = MonomialExtension(
        Poly(eta * z, z, domain=primitive.field()),
    )

    result = analyze_hyperexponential_extension(
        extension,
        primitive.derivation(),
    )

    assert result.kind is expected_kind
    assert result.minimal_power == expected_power
    assert result.relation_argument == expected_argument
    assert result.exponential_exponent == expected_exponent
    if result.is_dependent:
        tower = primitive.extend(extension)
        relation = tower.field().from_sympy(
            z**expected_power / result.relation_argument,
        )
        assert tower.derivation()(relation) == tower.field().zero


@given(
    x_valuation=st.integers(min_value=-5, max_value=5),
    linear_valuation=st.integers(min_value=-5, max_value=5),
    quadratic_valuation=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=40, deadline=None)
def test_generated_rational_dependencies_are_classified(
    x_valuation,
    linear_valuation,
    quadratic_valuation,
) -> None:
    assume(any((x_valuation, linear_valuation, quadratic_valuation)))
    argument = cancel(
        x**x_valuation
        * (x - 1) ** linear_valuation
        * (x**2 + x + 1) ** quadratic_valuation
    )
    eta = cancel(argument.diff(x) / argument)
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.RATIONAL
    assert result.minimal_power == 1
    assert cancel(result.relation_argument / argument).diff(x) == 0
    assert (
        cancel(result.relation_argument.diff(x) / result.relation_argument - eta) == 0
    )


@given(
    first_numerator=st.integers(min_value=-5, max_value=5),
    first_denominator=st.integers(min_value=1, max_value=6),
    second_numerator=st.integers(min_value=-5, max_value=5),
    second_denominator=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=45, deadline=None)
def test_generated_radical_dependencies_find_the_minimal_power(
    first_numerator,
    first_denominator,
    second_numerator,
    second_denominator,
) -> None:
    first = Rational(first_numerator, first_denominator)
    second = Rational(second_numerator, second_denominator)
    expected_power = lcm(first.q, second.q)
    assume(expected_power > 1)
    eta = first / x + second / (x - 1)
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.ALGEBRAIC
    assert result.minimal_power == expected_power
    expected_argument = x ** (expected_power * first) * (x - 1) ** (
        expected_power * second
    )
    assert cancel(result.relation_argument / expected_argument - 1) == 0
    assert (
        cancel(
            expected_power * eta
            - result.relation_argument.diff(x) / result.relation_argument
        )
        == 0
    )


@given(
    polynomial_coefficient=st.integers(min_value=-4, max_value=4),
    polynomial_degree=st.integers(min_value=1, max_value=4),
    pole_coefficient=st.integers(min_value=-4, max_value=4),
    pole_root=st.integers(min_value=-4, max_value=4),
    pole_power=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=40, deadline=None)
def test_generated_rational_exponents_get_named_presentations(
    polynomial_coefficient,
    polynomial_degree,
    pole_coefficient,
    pole_root,
    pole_power,
) -> None:
    exponent = (
        polynomial_coefficient * x**polynomial_degree
        + pole_coefficient / (x - pole_root) ** pole_power
    )
    eta = cancel(exponent.diff(x))
    assume(eta != 0)
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.TRANSCENDENTAL
    assert result.has_exponential_presentation
    assert cancel(result.exponential_exponent.diff(x) - eta) == 0
    assert cancel((result.exponential_exponent - exponent).diff(x)) == 0


def test_dependency_analysis_validates_extension_shape_and_parent() -> None:
    base = rational_base()
    primitive = MonomialExtension(
        Poly(1 / x, t, domain=base.field()),
    )
    hypertangent = MonomialExtension(
        Poly((1 + t**2) / x, t, domain=base.field()),
    )

    for extension in (primitive, hypertangent):
        with pytest.raises(ValueError, match=r"requires Dt = eta\*t"):
            analyze_hyperexponential_extension(
                extension,
                base.derivation(),
            )

    with pytest.raises(TypeError, match="requires a monomial extension"):
        analyze_hyperexponential_extension(
            object(),  # type: ignore[arg-type]
            base.derivation(),
        )

    _, extension = proposed_hyperexponential(1 / x, base=base)
    with pytest.raises(TypeError, match="requires a Derivation"):
        analyze_hyperexponential_extension(
            extension,
            object(),  # type: ignore[arg-type]
        )

    other_base = rational_base(generator=z)
    with pytest.raises(ValueError, match="parent field does not match"):
        analyze_hyperexponential_extension(
            extension,
            other_base.derivation(),
        )


def test_dependency_analysis_rejects_nonstandard_rational_bases() -> None:
    base = rational_base(generator_derivative=2)
    _, extension = proposed_hyperexponential(1 / x, base=base)

    with pytest.raises(NotImplementedError, match=r"C\(x\) with Dx = 1"):
        analyze_hyperexponential_extension(
            extension,
            base.derivation(),
        )


@pytest.mark.parametrize(
    ("eta", "expected_kind", "expected_power", "expected_argument"),
    [
        (
            1 / (x**2 + 1),
            HyperexponentialDependencyKind.RATIONAL,
            1,
            (x - I) / (x + I),
        ),
        (
            1 / (2 * (x**2 + 1)),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            (x - I) / (x + I),
        ),
        (
            1 / x,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
        ),
        (
            1 / x**2,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
        ),
        (
            x + 1 / (x**2 + 1),
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
        ),
    ],
)
def test_hypertangent_dependency_uses_the_cayley_logarithmic_derivative(
    eta,
    expected_kind,
    expected_power,
    expected_argument,
) -> None:
    base, extension = proposed_hypertangent(eta)

    result = analyze_hypertangent_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is expected_kind
    assert result.minimal_power == expected_power
    assert result.is_dependent == (
        expected_kind is not HyperexponentialDependencyKind.TRANSCENDENTAL
    )
    if expected_argument is None:
        assert result.relation_argument is None
    else:
        assert cancel(result.relation_argument - expected_argument) == 0
        assert (
            cancel(
                result.relation_argument.diff(x) / result.relation_argument
                - result.minimal_power * 2 * I * eta
            )
            == 0
        )


def test_hypertangent_dependency_detects_constant_field_conflict() -> None:
    base, extension = proposed_hypertangent(
        1 / (x**2 + 1),
        constant_field_status=ConstantFieldStatus.PRESERVED,
    )

    result = analyze_hypertangent_extension(
        extension,
        base.derivation(),
    )

    assert result.is_dependent
    assert result.inferred_constant_field_status is ConstantFieldStatus.ENLARGED
    assert result.conflicts_with_declared_constant_field_status


def test_hypertangent_dependency_validates_shape_parent_and_base() -> None:
    base = rational_base()
    _, extension = proposed_hypertangent(1 / x, base=base)
    hyperexponential = MonomialExtension(
        Poly(t / x, t, domain=base.field()),
    )

    with pytest.raises(ValueError, match="requires Dt = eta"):
        analyze_hypertangent_extension(
            hyperexponential,
            base.derivation(),
        )
    with pytest.raises(TypeError, match="requires a monomial extension"):
        analyze_hypertangent_extension(
            object(),  # type: ignore[arg-type]
            base.derivation(),
        )
    with pytest.raises(TypeError, match="requires a Derivation"):
        analyze_hypertangent_extension(
            extension,
            object(),  # type: ignore[arg-type]
        )

    other_base = rational_base(generator=z)
    with pytest.raises(ValueError, match="parent field does not match"):
        analyze_hypertangent_extension(
            extension,
            other_base.derivation(),
        )

    nonstandard_base = rational_base(generator_derivative=2)
    _, nonstandard_extension = proposed_hypertangent(
        1 / x,
        base=nonstandard_base,
    )
    with pytest.raises(NotImplementedError, match=r"QQ\(x\) with Dx = 1"):
        analyze_hypertangent_extension(
            nonstandard_extension,
            nonstandard_base.derivation(),
        )
