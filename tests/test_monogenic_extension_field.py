import pytest
from sympy import Poly, symbols
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import GF, QQ, ZZ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, DomainError

from sympy_risch.domains import MonogenicExtensionField

x, a, b, t = symbols("x a b t")


def test_checked_extension_rejects_nonfields_and_reducible_moduli() -> None:
    with pytest.raises(TypeError, match="exact field parent"):
        MonogenicExtensionField(Poly(a**2 - 2, a, domain=ZZ))
    with pytest.raises(ValueError, match="positive degree"):
        MonogenicExtensionField(Poly(1, a, domain=QQ))
    with pytest.raises(ValueError, match="irreducible"):
        MonogenicExtensionField(Poly(a**2 - 1, a, domain=QQ))


def test_quadratic_field_arithmetic_agrees_with_known_answer() -> None:
    field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    alpha = field.generator

    quotient = (alpha + field.one) / (alpha - field.one)

    assert field.is_Field
    assert field.is_PID
    assert field.get_field() is field
    with pytest.raises(DomainError, match="no ring associated"):
        field.get_ring()
    assert alpha**2 == field.convert(2)
    assert quotient == field.from_sympy(3 + 2 * a)
    assert quotient * (alpha - field.one) == alpha + field.one


def test_rational_function_parent_embeds_without_expression_roundtrip() -> None:
    parent = QQ.frac_field(x)
    field = MonogenicExtensionField(Poly(a**2 - x, a, domain=parent))
    alpha = field.generator
    native_x = parent.from_sympy(x)

    embedded_x = field.convert(native_x)
    quotient = field.exquo(alpha + field.one, alpha - field.one)

    assert alpha**2 == embedded_x
    assert quotient * (alpha - field.one) == alpha + field.one
    assert field.to_dict(embedded_x) == {(0,): native_x}


def test_foreign_extension_elements_require_an_explicit_embedding() -> None:
    first = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    second = MonogenicExtensionField(Poly(a**2 - 3, a, domain=QQ))

    with pytest.raises(CoercionFailed, match="explicit embedding"):
        first.convert(second.generator)


def test_conversion_canonicalizes_owned_elements_and_preserves_provenance() -> None:
    field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    equal_field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    malformed = ExtensionElement(field.ring.convert(a**2), field)

    canonical = field.convert(malformed)
    transferred = field.convert(equal_field.generator)

    assert canonical == field.convert(2)
    assert canonical.ext is field
    assert transferred == field.generator
    assert transferred.ext is field


def test_polynomial_euclid_uses_the_extension_as_a_field() -> None:
    field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    alpha = field.generator
    linear = Poly.from_dict(
        {(1,): field.one, (0,): -alpha},
        t,
        domain=field,
    )
    other = Poly.from_dict(
        {(1,): field.one, (0,): field.one},
        t,
        domain=field,
    )

    assert (linear * other).gcd(linear).monic() == linear.monic()
    bezout_left, bezout_right, gcd = field.gcdex(alpha, field.one)
    assert bezout_left * alpha + bezout_right == gcd == field.one


def test_polynomial_and_fraction_fields_embed_extension_constants() -> None:
    field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    alpha = field.generator

    polynomial_ring = field.poly_ring(t)
    rational_field = field.frac_field(t)
    polynomial_alpha = polynomial_ring.convert(alpha)
    rational_alpha = rational_field.convert(alpha)
    rational_t = rational_field.gens[0]

    assert polynomial_alpha.is_ground
    assert alpha == polynomial_alpha.LC
    assert rational_alpha == rational_field.new(alpha)
    assert rational_field.from_sympy((t + a) / (t - a)) == (
        rational_t + rational_alpha
    ) / (rational_t - rational_alpha)


def test_domain_matrix_inversion_uses_field_division() -> None:
    field = MonogenicExtensionField(Poly(a**2 - 2, a, domain=QQ))
    alpha = field.generator
    matrix = DomainMatrix(
        [[alpha, field.one], [field.one, alpha]],
        (2, 2),
        field,
    )

    product = matrix * matrix.inv()
    assert product.to_list() == [
        [field.one, field.zero],
        [field.zero, field.one],
    ]

    nullspace = DomainMatrix([[alpha, field.one]], (1, 2), field).nullspace()
    expected = DomainMatrix([[-field.one, alpha]], (1, 2), field)
    assert nullspace.unify_eq(expected)


def test_high_degree_arithmetic_stays_in_the_power_basis() -> None:
    field = MonogenicExtensionField(Poly(a**8 - 2, a, domain=QQ))
    alpha = field.generator

    value = (alpha + field.one) ** 17

    assert value.rep.degree() < field.rank
    assert value / value == field.one


def test_positive_characteristic_field_uses_the_same_contract() -> None:
    field = MonogenicExtensionField(Poly(a**2 + 1, a, domain=GF(3)))
    alpha = field.generator

    assert field.characteristic() == 3
    assert not field.has_CharacteristicZero
    assert alpha**2 == -field.one
    assert (alpha + field.one) / (alpha - field.one) == -alpha
