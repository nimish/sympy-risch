r"""Hermite reduction for normal denominator factors in monomial fields."""

from __future__ import annotations

from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..reduction.fractions import (
    _reduced_polynomial_fraction,
    split_normal_special,
)
from ..reduction.polynomials import polynomial_inverse_modulo
from ..tower.core import (
    Derivation,
    MonomialExtension,
    _coerce,
)


def _normal_denominator_excess(
    denominator: Poly,
    derivation: Derivation,
) -> int:
    normal, _ = split_normal_special(denominator, derivation)
    return sum(
        (multiplicity - 1) * factor.degree()
        for factor, multiplicity in normal.sqf_list()[1]
    )


def hermite_reduce_monomial(
    integrand: object,
    derivation: Derivation,
) -> tuple[FieldElement, FieldElement]:
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
        modular_coefficient = (cofactor * factor_derivative).mul_ground(
            multiplicity - 1,
        )
        correction_numerator = (
            -numerator * polynomial_inverse_modulo(modular_coefficient, factor)
        ).rem(factor)
        correction_denominator = factor ** (multiplicity - 1)
        correction = field.from_sympy(
            correction_numerator.as_expr() / correction_denominator.as_expr(),
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
