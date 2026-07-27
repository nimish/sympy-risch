"""Normal forms for rational functions in a monomial generator.

These routines are shared by monomial reduction, exact-derivative solving,
and the RDE algorithms.  They know only about the tower representation and
polynomial arithmetic; recursive Risch decisions remain in their theorem
family modules.
"""

from __future__ import annotations

from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction

from .._typing import FieldElement
from ..tower.core import Derivation, MonomialExtension


def split_normal_special(
    polynomial: Poly,
    derivation: Derivation,
) -> tuple[Poly, Poly]:
    r"""Return the normal and special parts of a nonzero polynomial.

    A square-free factor ``p`` is normal when ``gcd(p, Dp) = 1`` and special
    when ``p`` divides ``Dp``.  The two returned factors retain the original
    multiplicities and multiply exactly to ``polynomial``.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if polynomial.is_zero:
        raise ValueError("the zero polynomial has no normal/special decomposition")

    derivation.polynomial(polynomial)
    coefficient, square_free_factors = polynomial.sqf_list()
    normal = polynomial.one.mul_ground(coefficient)
    special = polynomial.one

    for factor, multiplicity in square_free_factors:
        factor_derivative = derivation.polynomial(factor)
        special_factor = factor.gcd(factor_derivative)
        normal *= factor.exquo(special_factor) ** multiplicity
        special *= special_factor**multiplicity

    return normal, special


def _reduced_polynomial_fraction(
    element: FieldElement,
    extension: MonomialExtension,
) -> tuple[Poly, Poly]:
    """Return relatively prime numerator and denominator polynomials in ``t``."""
    numerator_expression, denominator_expression = fraction(
        extension.field.to_sympy(element)
    )
    numerator = Poly(
        numerator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    return numerator.mul_ground(coefficient), denominator


def _polynomial_and_proper_fraction(
    element: FieldElement,
    extension: MonomialExtension,
) -> tuple[Poly, Poly, Poly]:
    """Return polynomial part and proper fraction with a monic denominator."""
    numerator, denominator = _reduced_polynomial_fraction(element, extension)
    leading_inverse = denominator.domain.one / denominator.domain.convert(
        denominator.LC(),
    )
    numerator = numerator.mul_ground(leading_inverse)
    denominator = denominator.monic()
    polynomial, proper_numerator = numerator.div(denominator)
    if proper_numerator.is_zero:
        return polynomial, denominator.zero, denominator.one
    return polynomial, proper_numerator, denominator


def _normal_fraction_part(
    element: FieldElement,
    extension: MonomialExtension,
    derivation: Derivation,
) -> tuple[Poly, Poly]:
    r"""Return ``A, N`` for the normal proper fraction ``A/N`` in ``element``.

    For a reduced denominator ``N*S``, Bezout reduction gives
    ``C/(N*S) = A/N + B/S``.  This is the normal-fraction input to the
    generalized Rothstein--Trager residue calculation.
    """
    _, proper_numerator, denominator = _polynomial_and_proper_fraction(
        element,
        extension,
    )
    if proper_numerator.is_zero:
        return denominator.zero, denominator.one

    normal_denominator, special_denominator = split_normal_special(
        denominator,
        derivation,
    )
    if normal_denominator.degree() == 0:
        return denominator.zero, denominator.one

    normal_numerator = (
        proper_numerator * special_denominator.invert(normal_denominator)
    ).rem(normal_denominator)
    if normal_numerator.is_zero:
        return denominator.zero, denominator.one
    return normal_numerator, normal_denominator.monic()
