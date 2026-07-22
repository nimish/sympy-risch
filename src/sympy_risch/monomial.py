r"""Reduction helpers for transcendental monomial extensions."""

from __future__ import annotations

from typing import Any

from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction

from .tower import Derivation, MonomialExtension, _coerce


def split_normal_special(
    polynomial: Poly,
    derivation: Derivation,
) -> tuple[Poly, Poly]:
    r"""Return the normal and special parts of a nonzero polynomial.

    A square-free factor ``p`` is normal when ``gcd(p, Dp) = 1`` and special
    when ``p`` divides ``Dp``, where ``D`` is the total tower derivation.  For
    an arbitrary polynomial, its square-free factors are classified separately
    and retain their original multiplicities.  The returned pair multiplies
    exactly to ``polynomial``; its coefficient is assigned to the normal part.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if polynomial.is_zero:
        raise ValueError("the zero polynomial has no normal/special decomposition")

    # Besides checking the generator and coefficient domain, this makes the
    # required level explicit even for a constant polynomial with no factors.
    derivation.polynomial(polynomial)

    coefficient, square_free_factors = polynomial.sqf_list()
    normal = polynomial.one.mul_ground(coefficient)
    special = polynomial.one

    for factor, multiplicity in square_free_factors:
        factor_derivative = derivation.polynomial(factor)
        special_factor = factor.gcd(factor_derivative)
        normal_factor = factor.exquo(special_factor)
        special *= special_factor**multiplicity
        normal *= normal_factor**multiplicity

    return normal, special


def _reduced_polynomial_fraction(
    element: Any,
    extension: MonomialExtension,
) -> tuple[Poly, Poly]:
    numerator_expression, denominator_expression = fraction(element.as_expr())
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


def _normal_denominator_excess(
    denominator: Poly,
    derivation: Derivation,
) -> int:
    normal, _ = split_normal_special(denominator, derivation)
    return sum(
        (multiplicity - 1)*factor.degree()
        for factor, multiplicity in normal.sqf_list()[1]
    )


def hermite_reduce_monomial(
    integrand: Any,
    derivation: Derivation,
) -> tuple[Any, Any]:
    r"""Hermite-reduce the normal denominator in a monomial extension.

    Return ``(g, h)`` with ``integrand = D(g) + h`` and a square-free normal
    denominator in ``h``.  Polynomial terms and special denominator factors
    are deliberately retained in ``h`` for the extension-family reduction.
    """
    if derivation.level < 0:
        raise TypeError("monomial Hermite reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("monomial Hermite reduction requires a monomial level")

    field = extension.field
    exact_part = field.zero
    remainder = _coerce(field, integrand)

    while True:
        numerator, denominator = _reduced_polynomial_fraction(
            remainder,
            extension,
        )
        normal, _ = split_normal_special(denominator, derivation)
        repeated_factors = [
            (factor, multiplicity)
            for factor, multiplicity in normal.sqf_list()[1]
            if multiplicity > 1
        ]
        if not repeated_factors:
            return exact_part, remainder

        factor, multiplicity = max(
            repeated_factors,
            key=lambda item: item[1],
        )
        cofactor = denominator.exquo(factor**multiplicity)
        factor_derivative = derivation.polynomial(factor)
        modular_coefficient = (cofactor*factor_derivative).mul_ground(
            multiplicity - 1,
        )
        correction_numerator = (
            -numerator*modular_coefficient.invert(factor)
        ).rem(factor)
        correction_denominator = factor**(multiplicity - 1)
        correction = field.from_sympy(
            correction_numerator.as_expr()/correction_denominator.as_expr(),
        )

        previous_excess = _normal_denominator_excess(denominator, derivation)
        new_remainder = remainder - derivation(correction)
        _, new_denominator = _reduced_polynomial_fraction(
            new_remainder,
            extension,
        )
        new_excess = _normal_denominator_excess(
            new_denominator,
            derivation,
        )
        if new_excess >= previous_excess:
            raise ValueError("monomial Hermite reduction did not lower a normal pole")

        exact_part += correction
        remainder = new_remainder
