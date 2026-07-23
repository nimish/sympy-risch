"""Dependency classification for proposed hyperexponential extensions."""

from math import lcm

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import Poly, Rational, cancel, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    ConstantFieldStatus,
    DifferentialTower,
    HyperexponentialDependencyKind,
    MonomialExtension,
    analyze_hyperexponential_extension,
)


x, t, z = symbols("x t z")


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
        Poly(eta*t, t, domain=base.field()),
        constant_field_status,
    )
    return base, extension


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
            1/x,
            HyperexponentialDependencyKind.RATIONAL,
            1,
            x,
            None,
        ),
        (
            -3/x,
            HyperexponentialDependencyKind.RATIONAL,
            1,
            x**-3,
            None,
        ),
        (
            1/(2*x),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            x,
            None,
        ),
        (
            (2*x + 1)/(2*(x**2 + x + 1)),
            HyperexponentialDependencyKind.ALGEBRAIC,
            2,
            x**2 + x + 1,
            None,
        ),
        (
            1/(2*x) + 1/(3*(x - 1)),
            HyperexponentialDependencyKind.ALGEBRAIC,
            6,
            x**3*(x - 1)**2,
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
            2*x,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            x**2,
        ),
        (
            -1/x**2,
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            1/x,
        ),
        (
            1/(x**2 + 1),
            HyperexponentialDependencyKind.TRANSCENDENTAL,
            None,
            None,
            None,
        ),
        (
            1 + 1/(2*x),
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
        assert cancel(
            expected_power*eta
            - result.relation_argument.diff(x)/result.relation_argument
        ) == 0
    if expected_exponent is None:
        assert result.exponential_exponent is None
    else:
        assert cancel(result.exponential_exponent - expected_exponent) == 0
        assert cancel(result.exponential_exponent.diff(x) - eta) == 0


@pytest.mark.parametrize(
    ("eta", "expected_power"),
    [
        (1/x, 1),
        (1/(2*x), 2),
        (1/(2*x) + 1/(3*(x - 1)), 6),
        (-3/x, 1),
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
    relation = field.from_sympy(
        t**expected_power/analysis.relation_argument
    )

    assert tower.derivation()(relation) == field.zero


def test_dependency_and_named_exponential_presentation_are_independent() -> None:
    dependent_base, dependent_extension = proposed_hyperexponential(1/x)
    exponential_base, exponential_extension = proposed_hyperexponential(2*x)

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
            1/x,
            ConstantFieldStatus.UNKNOWN,
            ConstantFieldStatus.ENLARGED,
            False,
        ),
        (
            1/x,
            ConstantFieldStatus.PRESERVED,
            ConstantFieldStatus.ENLARGED,
            True,
        ),
        (
            1/x,
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
    assert (
        result.conflicts_with_declared_constant_field_status
        is expected_conflict
    )


def test_transcendental_obstructions_retain_their_certificates() -> None:
    cases = [
        (1, "has_polynomial_obstruction"),
        (1/x**2, "has_repeated_pole_obstruction"),
        (1/(x**2 + 1), "has_nonlogarithmic_residue_obstruction"),
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
        * (x - 1)**linear_valuation
        * (x**2 + x + 1)**quadratic_valuation
    )
    eta = cancel(argument.diff(x)/argument)
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.RATIONAL
    assert result.minimal_power == 1
    assert cancel(result.relation_argument/argument).diff(x) == 0
    assert cancel(
        result.relation_argument.diff(x)/result.relation_argument - eta
    ) == 0


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
    eta = first/x + second/(x - 1)
    base, extension = proposed_hyperexponential(eta)

    result = analyze_hyperexponential_extension(
        extension,
        base.derivation(),
    )

    assert result.kind is HyperexponentialDependencyKind.ALGEBRAIC
    assert result.minimal_power == expected_power
    expected_argument = (
        x**(expected_power*first)
        * (x - 1)**(expected_power*second)
    )
    assert cancel(result.relation_argument/expected_argument - 1) == 0
    assert cancel(
        expected_power*eta
        - result.relation_argument.diff(x)/result.relation_argument
    ) == 0


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
        polynomial_coefficient*x**polynomial_degree
        + pole_coefficient/(x - pole_root)**pole_power
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
        Poly(1/x, t, domain=base.field()),
    )
    hypertangent = MonomialExtension(
        Poly((1 + t**2)/x, t, domain=base.field()),
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

    _, extension = proposed_hyperexponential(1/x, base=base)
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
    _, extension = proposed_hyperexponential(1/x, base=base)

    with pytest.raises(NotImplementedError, match=r"QQ\(x\) with Dx = 1"):
        analyze_hyperexponential_extension(
            extension,
            base.derivation(),
        )
