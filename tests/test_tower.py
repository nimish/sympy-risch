from dataclasses import FrozenInstanceError

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sympy import Poly, cancel, diff, sqrt, symbols
from sympy.polys.domains import GF, QQ, QQ_I, ZZ

from sympy_risch import (
    AlgebraicExtension,
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
)

x, t, u = symbols("x t u")


def rational_tower() -> DifferentialTower:
    return DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ))
    )


def logarithmic_tower() -> DifferentialTower:
    kx = rational_tower()
    return kx.extend(
        MonomialExtension(Poly(1/x, t, domain=kx.field()))
    )


def test_empty_tower_is_the_constant_field() -> None:
    tower = DifferentialTower(QQ)

    assert tower.top == -1
    assert tower.field() == QQ
    assert tower.derivation()(QQ(7)) == QQ.zero


def test_extending_a_tower_does_not_mutate_it() -> None:
    constants = DifferentialTower(QQ)
    extension = MonomialExtension(Poly(1, x, domain=QQ))

    extended = constants.extend(extension)

    assert constants.levels == ()
    assert constants.field() == QQ
    assert extended.levels == (extension,)
    assert extended.field() == QQ.frac_field(x)

    with pytest.raises(FrozenInstanceError):
        extended.constants = QQ  # type: ignore[misc]


def test_level_sequence_is_copied_to_an_immutable_tuple() -> None:
    extension = MonomialExtension(Poly(1, x, domain=QQ))
    source = [extension]

    tower = DifferentialTower(QQ, source)  # type: ignore[arg-type]
    source.clear()

    assert tower.levels == (extension,)


def test_tower_rejects_an_incompatible_parent_field() -> None:
    kx = rational_tower()
    wrong_parent = MonomialExtension(Poly(1, t, domain=QQ))

    with pytest.raises(ValueError, match="parent does not match"):
        kx.extend(wrong_parent)


def test_monomial_extension_validates_its_derivative_law() -> None:
    with pytest.raises(TypeError, match="SymPy Poly"):
        MonomialExtension(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one generator"):
        MonomialExtension(Poly(x + t, x, t, domain=QQ))
    with pytest.raises(ValueError, match="parent domain must be a field"):
        MonomialExtension(Poly(1, x, domain=ZZ))
    with pytest.raises(ValueError, match="characteristic zero"):
        MonomialExtension(Poly(1, x, domain=GF(5)))
    with pytest.raises(ValueError, match="zero-derivative"):
        MonomialExtension(Poly(0, x, domain=QQ))
    with pytest.raises(TypeError, match="ConstantFieldStatus"):
        MonomialExtension(
            Poly(1, x, domain=QQ),
            "preserved",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="SpecialPolynomialStatus"):
        MonomialExtension(
            Poly(1, x, domain=QQ),
            special_polynomial_status="absent",  # type: ignore[arg-type]
        )


def test_monomial_constant_field_status_is_explicit_and_immutable() -> None:
    unknown = MonomialExtension(Poly(1, x, domain=QQ))
    preserved = MonomialExtension(
        Poly(1, x, domain=QQ),
        ConstantFieldStatus.PRESERVED,
    )

    assert unknown.constant_field_status is ConstantFieldStatus.UNKNOWN
    assert preserved.constant_field_status is ConstantFieldStatus.PRESERVED
    with pytest.raises(ValueError, match="status is unknown"):
        unknown.require_preserved_constants("test algorithm")
    preserved.require_preserved_constants("test algorithm")
    with pytest.raises(FrozenInstanceError):
        preserved.constant_field_status = ConstantFieldStatus.ENLARGED  # type: ignore[misc]


def test_monomial_special_polynomial_status_is_explicit_and_immutable() -> None:
    unknown = MonomialExtension(Poly(1, x, domain=QQ))
    absent = MonomialExtension(
        Poly(1, x, domain=QQ),
        special_polynomial_status=SpecialPolynomialStatus.ABSENT,
    )

    assert unknown.special_polynomial_status is SpecialPolynomialStatus.UNKNOWN
    assert absent.special_polynomial_status is SpecialPolynomialStatus.ABSENT
    with pytest.raises(ValueError, match="status is unknown"):
        unknown.require_no_special_polynomials("test algorithm")
    absent.require_no_special_polynomials("test algorithm")
    with pytest.raises(FrozenInstanceError):
        absent.special_polynomial_status = SpecialPolynomialStatus.PRESENT  # type: ignore[misc]


def test_monomial_extension_rejects_a_generator_already_in_parent() -> None:
    kxt = QQ.frac_field(x).frac_field(t)

    with pytest.raises(ValueError, match="already occurs"):
        MonomialExtension(Poly(1, x, domain=kxt))


def test_algebraic_parent_compositions_fail_at_the_documented_boundary() -> None:
    algebraic_generator, nested_generator = symbols("y z")
    base = rational_tower()
    algebraic = base.extend(
        AlgebraicExtension(
            Poly(
                algebraic_generator**2 - x,
                algebraic_generator,
                domain=base.field(),
            ),
        ),
    )
    monomial = MonomialExtension(
        Poly(algebraic_generator*t, t, domain=algebraic.field()),
    )

    # A proposal remains inspectable by dependency factories, but SymPy's
    # native domain backend cannot represent a usable K(y)(t) field yet.
    assert monomial.family is MonomialFamily.HYPEREXPONENTIAL
    with pytest.raises(NotImplementedError, match="algebraic parent fields"):
        algebraic.extend(monomial)
    with pytest.raises(NotImplementedError, match="nested algebraic"):
        AlgebraicExtension(
            Poly(
                nested_generator**2 - algebraic_generator,
                nested_generator,
                domain=algebraic.field(),
            ),
        )


@pytest.mark.parametrize(
    ("derivative", "family"),
    [
        (1/x, MonomialFamily.PRIMITIVE),
        (t/(x + 1), MonomialFamily.HYPEREXPONENTIAL),
        ((1 + t**2)/x, MonomialFamily.HYPERTANGENT),
        (t - 1, MonomialFamily.POLYNOMIAL),
        (1 + t + t**2, MonomialFamily.POLYNOMIAL),
        (t**3, MonomialFamily.POLYNOMIAL),
    ],
)
def test_monomial_family_is_derived_from_the_derivative_shape(
    derivative,
    family,
) -> None:
    parent = QQ.frac_field(x)
    extension = MonomialExtension(Poly(derivative, t, domain=parent))

    assert extension.family is family


@pytest.mark.parametrize(
    "parent",
    [QQ_I, QQ.algebraic_field(sqrt(2))],
)
@pytest.mark.parametrize(
    ("derivative", "family"),
    [
        (t, MonomialFamily.HYPEREXPONENTIAL),
        (1 + t**2, MonomialFamily.HYPERTANGENT),
    ],
)
def test_monomial_family_coerces_algebraic_domain_coefficients(
    parent,
    derivative,
    family,
) -> None:
    extension = MonomialExtension(Poly(derivative, t, domain=parent))

    assert extension.family is family


def test_level_access_and_derivations_are_explicit() -> None:
    tower = logarithmic_tower()

    assert tower.field(-1) == QQ
    assert tower.field(0) == QQ.frac_field(x)
    assert tower.field(1) == QQ.frac_field(x, t)
    assert tower.derivation().level == 1
    assert tower.derivation().lower().level == 0
    assert tower.derivation(0).lower().level == -1

    with pytest.raises(IndexError, match="out of range"):
        tower.field(2)
    with pytest.raises(IndexError, match="out of range"):
        Derivation(tower, -2)
    with pytest.raises(ValueError, match="no lower derivation"):
        tower.derivation(-1).lower()


def test_recursive_polynomial_derivative() -> None:
    tower = logarithmic_tower()
    derivative = tower.derivation()
    parent = tower.field(0)
    polynomial = Poly(t**2 + x*t, t, domain=parent)

    assert derivative.coefficients(polynomial) == Poly(t, t, domain=parent)
    assert derivative.polynomial(polynomial) == Poly(
        2*t/x + t + 1,
        t,
        domain=parent,
    )


def test_recursive_fraction_derivative() -> None:
    tower = logarithmic_tower()
    derivative = tower.derivation()
    field = tower.field()
    expression = (t**2 + x*t + 1)/(x*t + 1)
    expected = cancel(diff(expression, x) + diff(expression, t)/x)

    result = derivative(field.from_sympy(expression))

    assert result == field.from_sympy(expected)


def test_derivation_recurses_through_three_levels() -> None:
    kxt = logarithmic_tower()
    kxtu = kxt.extend(
        MonomialExtension(Poly(t*u, u, domain=kxt.field()))
    )
    field = kxtu.field()

    assert kxtu.derivation()(field.from_sympy(t*u)) == field.from_sympy(
        t**2*u + u/x
    )


def test_fraction_helper_uses_the_same_quotient_rule() -> None:
    tower = logarithmic_tower()
    derivative = tower.derivation()
    parent = tower.field(0)
    numerator = Poly(t**2 + x, t, domain=parent)
    denominator = Poly(x*t + 1, t, domain=parent)

    result = derivative.fraction(numerator, denominator)
    expected = derivative(tower.field().from_sympy((t**2 + x)/(x*t + 1)))

    assert result == expected


def test_nested_ground_field_element_is_embedded_exactly() -> None:
    tower = logarithmic_tower()
    parent_element = tower.field(0).from_sympy(1/x)

    assert tower.derivation()(parent_element) == tower.field().from_sympy(-1/x**2)


@given(
    st.lists(
        st.integers(min_value=-5, max_value=5),
        min_size=1,
        max_size=8,
    )
)
def test_generated_polynomials_satisfy_the_chain_rule(coefficients) -> None:
    tower = logarithmic_tower()
    field = tower.field()
    expression = sum(
        coefficient*x**index*t**(len(coefficients) - index - 1)
        for index, coefficient in enumerate(coefficients)
    )
    expected = diff(expression, x) + diff(expression, t)/x

    assert tower.derivation()(field.from_sympy(expression)) == field.from_sympy(expected)


@given(
    st.lists(st.integers(min_value=-3, max_value=3), min_size=1, max_size=3),
    st.lists(st.integers(min_value=-3, max_value=3), min_size=1, max_size=3),
)
@settings(max_examples=30, deadline=None)
def test_generated_elements_satisfy_derivation_laws(left_coeffs, right_coeffs) -> None:
    tower = logarithmic_tower()
    derivative = tower.derivation()
    field = tower.field()
    left_expression = sum(
        coefficient*x**index*t**(len(left_coeffs) - index - 1)
        for index, coefficient in enumerate(left_coeffs)
    )
    right_expression = sum(
        coefficient*x**index*t**(len(right_coeffs) - index - 1)
        for index, coefficient in enumerate(right_coeffs)
    )
    left = field.from_sympy(left_expression)
    right = field.from_sympy(right_expression)
    denominator = field.one + right**2
    left_derivative = derivative(left)
    right_derivative = derivative(right)
    denominator_derivative = derivative(denominator)

    assert derivative(left + right) == left_derivative + right_derivative
    assert derivative(left*right) == left_derivative*right + left*right_derivative
    assert derivative(left/denominator) == (
        left_derivative*denominator - left*denominator_derivative
    )/denominator**2
