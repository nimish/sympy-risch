"""Exact polynomial operations over nested algebraic fraction fields.

SymPy reduces a fraction-field element by cancelling common polynomial
factors, but it does not make the denominator monic when the ground domain is
an ``AlgebraicField``.  Equal constants can consequently retain different
numerator/denominator units.  Most arithmetic tolerates that representation;
polynomial equality, gcd, and modular inversion do not always do so.

The helpers here canonicalize only that affected coefficient shape.  They
remain ordinary SymPy ``Poly`` operations and do not introduce a parallel
field representation.
"""

from __future__ import annotations

from typing import cast

from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement


def _has_algebraic_fraction_ground(domain: Domain) -> bool:
    """Whether ``domain`` is a fraction tower grounded in an algebraic field."""
    current = domain
    saw_fraction = False
    while isinstance(current, FractionField):
        saw_fraction = True
        current = current.domain
    return saw_fraction and isinstance(current, AlgebraicField)


def _canonicalize_element(value: object, domain: Domain) -> FieldElement:
    """Make every algebraic fraction-field denominator monic recursively."""
    converted = cast(FieldElement, domain.convert(value))
    if not isinstance(domain, FractionField) or not _has_algebraic_fraction_ground(
        domain
    ):
        return converted
    if not isinstance(converted, FracElement):
        raise TypeError("a FractionField conversion did not return a FracElement")

    ground = cast(Domain, domain.domain)
    ring = domain.field.ring
    numerator = ring.from_dict(
        {
            monomial: _canonicalize_element(coefficient, ground)
            for monomial, coefficient in converted.numer.items()
        }
    )
    denominator = ring.from_dict(
        {
            monomial: _canonicalize_element(coefficient, ground)
            for monomial, coefficient in converted.denom.items()
        }
    )
    leading = _canonicalize_element(denominator.LC, ground)
    inverse = _canonicalize_element(ground.one / leading, ground)
    numerator = numerator.mul_ground(inverse)
    denominator = denominator.mul_ground(inverse)

    # Multiplication by the inverse may itself have produced nonmonic nested
    # coefficient fractions, so normalize those coefficients once more.
    numerator = ring.from_dict(
        {
            monomial: _canonicalize_element(coefficient, ground)
            for monomial, coefficient in numerator.items()
        }
    )
    denominator = ring.from_dict(
        {
            monomial: _canonicalize_element(coefficient, ground)
            for monomial, coefficient in denominator.items()
        }
    )
    return cast(FieldElement, domain.field.raw_new(numerator, denominator))


def canonicalize_polynomial(polynomial: Poly) -> Poly:
    """Return ``polynomial`` with canonical nested fraction coefficients."""
    if not _has_algebraic_fraction_ground(polynomial.domain):
        return polynomial
    return Poly.from_dict(
        {
            monomial: _canonicalize_element(coefficient, polynomial.domain)
            for monomial, coefficient in polynomial.terms()
        },
        *polynomial.gens,
        domain=polynomial.domain,
    )


def exact_field_equal(left: object, right: object, field: Domain) -> bool:
    """Compare two elements by their exact value in ``field``."""
    difference = field.convert(left) - field.convert(right)
    return difference == field.zero or cancel(field.to_sympy(difference)) == 0


def exact_polynomial_equal(left: Poly, right: Poly) -> bool:
    """Compare polynomials by exact coefficient values, not stored units."""
    if left.gens != right.gens or left.domain != right.domain:
        return False
    difference = left - right
    return all(
        exact_field_equal(coefficient, left.domain.zero, left.domain)
        for coefficient in difference.coeffs()
    )


def polynomial_gcd(left: Poly, right: Poly) -> Poly:
    """Compute a monic gcd after canonicalizing affected coefficients."""
    return canonicalize_polynomial(
        canonicalize_polynomial(left).gcd(canonicalize_polynomial(right))
    ).monic()


def polynomial_inverse_modulo(value: Poly, modulus: Poly) -> Poly:
    """Return the exact inverse of ``value`` modulo ``modulus``."""
    canonical_value = canonicalize_polynomial(value)
    canonical_modulus = canonicalize_polynomial(modulus)
    coefficient, _, gcd = canonical_value.gcdex(canonical_modulus)
    if gcd.is_zero or gcd.degree() != 0:
        raise NotInvertible("zero divisor")
    inverse = canonicalize_polynomial(
        coefficient.mul_ground(
            canonical_value.domain.one / canonical_value.domain.convert(gcd.LC()),
        ).rem(canonical_modulus)
    )
    if not exact_polynomial_equal(
        canonicalize_polynomial((canonical_value * inverse).rem(canonical_modulus)),
        canonical_value.one,
    ):
        raise NotInvertible("zero divisor")
    return inverse
