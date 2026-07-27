r"""Normal-pole Hermite reduction for a terminal pure radical.

For ``L = K(theta)``, ``theta**n = u``, and
``D(theta) = eta*theta``, every element has a power-basis expansion

``f = sum(f_j*theta**j)``.

The derivative is diagonal in that basis:

``D(g_j*theta**j) = (D(g_j) + j*eta*g_j)*theta**j``.

This module applies Hermite's repeated-normal-pole step to those scalar
operators over the *immediate* lower monomial field.  It deliberately does
not reuse the direct ``C(x)`` algebraic-curve implementation: the lower
coefficients here are differential elements, not constants.  Special poles
are retained, as are resonant leading equations that need later relative
logarithmic/RDE machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction

from .._typing import FieldElement
from ..reduction.fractions import split_normal_special
from ..tower import AlgebraicExtension, Derivation, MonomialExtension
from ._field import _extension_element
from .relative import (
    RelativeSimpleRadicalData,
    _diagonal_generator_logarithmic_derivative,
    recognize_relative_simple_radical,
)


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalHermiteMode:
    """The reduced lower-field coefficient of one ``theta**power`` mode."""

    power: int
    coefficient: FieldElement
    exact_part: FieldElement
    remainder: FieldElement


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalHermiteReduction:
    r"""A relative normal-Hermite decomposition ``f = D(g) + h``.

    The normal denominator in every power-basis coefficient of ``h`` is
    square-free.  Special factors are intentionally not part of this
    guarantee.
    """

    radical: RelativeSimpleRadicalData
    derivation: Derivation
    integrand: ExtensionElement
    exact_part: ExtensionElement
    remainder: ExtensionElement
    modes: tuple[RelativeSimpleRadicalHermiteMode, ...]

    @property
    def reconstructs(self) -> bool:
        """Whether the stored exact part reconstructs the integrand."""
        return self.derivation(self.exact_part) + self.remainder == self.integrand

    @property
    def normal_denominators_are_square_free(self) -> bool:
        """Whether every remaining lower-coordinate normal part is square-free."""
        _extension, lower_extension, lower_derivation = _relative_lower_monomial(
            self.derivation,
        )
        algebraic_extension = self.radical.extension
        return all(
            _normal_denominator_excess(coordinate, lower_extension, lower_derivation)
            == 0
            for coordinate in algebraic_extension.power_basis_coordinates(
                self.remainder
            )
        )


def _relative_lower_monomial(
    derivation: Derivation,
) -> tuple[AlgebraicExtension, MonomialExtension, Derivation]:
    """Return the terminal algebraic extension and its monomial lower level."""
    if not isinstance(derivation, Derivation):
        raise TypeError("relative Hermite reduction needs a derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "relative Hermite reduction needs a terminal algebraic extension "
            "over a monomial lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("relative Hermite reduction needs an algebraic top level")
    lower_derivation = derivation.lower()
    lower_extension = derivation.tower.levels[lower_derivation.level]
    if not isinstance(lower_extension, MonomialExtension):
        raise NotImplementedError(
            "relative Hermite reduction needs an immediate monomial lower field"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError("the algebraic extension does not match its lower derivation")
    return extension, lower_extension, lower_derivation


def _lower_fraction_polynomials(
    value: FieldElement,
    extension: MonomialExtension,
) -> tuple[Poly, Poly]:
    """Return relatively prime numerator and denominator in the lower generator.

    Native mixed fields are flattened by SymPy (for example ``QQ(x, t)``),
    while normal/special factorization needs the immediate view ``QQ(x)[t]``.
    Converting through the exact expression is the small, explicit bridge.
    """
    numerator_expression, denominator_expression = fraction(
        extension.field.to_sympy(value),
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


def _lower_fraction_element(
    numerator: Poly,
    denominator: Poly,
    extension: MonomialExtension,
) -> FieldElement:
    """Build a native lower-field element from immediate-coordinate polynomials."""
    return extension.field.from_sympy(
        numerator.as_expr() / denominator.as_expr(),
    )


def _normal_denominator_excess(
    value: FieldElement,
    extension: MonomialExtension,
    derivation: Derivation,
) -> int:
    """Return the repeated-pole degree of the normal denominator part."""
    _numerator, denominator = _lower_fraction_polynomials(value, extension)
    normal, _special = split_normal_special(denominator, derivation)
    return sum(
        (multiplicity - 1) * factor.degree()
        for factor, multiplicity in normal.sqf_list()[1]
    )


def _rational_modulo(
    value: FieldElement,
    factor: Poly,
    extension: MonomialExtension,
) -> Poly:
    r"""Return a lower-field rational value modulo an irreducible ``factor``.

    The denominator must be coprime to ``factor``.  A remaining denominator
    factor would mean that the mode coefficient has a higher-order pole at
    the factor, outside this first normal-Hermite cell.
    """
    numerator, denominator = _lower_fraction_polynomials(value, extension)
    if denominator.gcd(factor).degree() != 0:
        raise NotImplementedError(
            "relative Hermite reduction needs a regular leading coefficient "
            f"at the normal factor {factor.as_expr()}"
        )
    return (numerator.rem(factor) * denominator.invert(factor)).rem(factor)


def _reduce_mode_normal_poles(
    value: FieldElement,
    coefficient: FieldElement,
    extension: MonomialExtension,
    derivation: Derivation,
) -> tuple[FieldElement, FieldElement]:
    r"""Hermite-reduce ``D(g) + coefficient*g`` at normal repeated poles.

    At a factor ``p**m`` of the denominator, write the correction as
    ``c / p**(m - 1)``.  Its leading coefficient is

    ``c * (p*coefficient - (m - 1)*D(p))``.

    Solving this equation modulo ``p`` cancels the pole of order ``m``.
    """
    field = extension.field
    exact_part = field.zero
    remainder = value

    while True:
        numerator, denominator = _lower_fraction_polynomials(remainder, extension)
        normal, _special = split_normal_special(denominator, derivation)
        repeated_factors = [
            (factor, multiplicity)
            for factor, multiplicity in normal.sqf_list()[1]
            if multiplicity > 1
        ]
        if not repeated_factors:
            return exact_part, remainder

        factor, multiplicity = max(repeated_factors, key=lambda item: item[1])
        cofactor = denominator.exquo(factor**multiplicity)
        factor_derivative = derivation.polynomial(factor)
        factor_element = field.from_sympy(factor.as_expr())
        leading = factor_element * coefficient - field.from_sympy(
            (multiplicity - 1) * factor_derivative.as_expr(),
        )
        leading_modulo = _rational_modulo(leading, factor, extension)
        if leading_modulo.gcd(factor).degree() != 0:
            raise NotImplementedError(
                "relative Hermite reduction encountered a resonant normal pole; "
                "relative logarithmic/RDE reduction is required"
            )

        right_hand_side = (numerator.rem(factor) * cofactor.invert(factor)).rem(factor)
        correction_numerator = (right_hand_side * leading_modulo.invert(factor)).rem(
            factor
        )
        correction_denominator = factor ** (multiplicity - 1)
        correction = _lower_fraction_element(
            correction_numerator,
            correction_denominator,
            extension,
        )

        previous_excess = _normal_denominator_excess(
            remainder,
            extension,
            derivation,
        )
        new_remainder = remainder - (derivation(correction) + coefficient * correction)
        new_excess = _normal_denominator_excess(
            new_remainder,
            extension,
            derivation,
        )
        if new_excess >= previous_excess:
            raise ValueError(
                "relative Hermite reduction did not lower a normal repeated pole",
            )

        exact_part += correction
        remainder = new_remainder


def hermite_reduce_relative_simple_radical(
    integrand: object,
    derivation: Derivation,
) -> RelativeSimpleRadicalHermiteReduction:
    r"""Write a terminal pure-radical integrand as ``D(g) + h``.

    This eager first scope requires an immediate monomial lower field and a
    pure radical with diagonal derivative.  It reduces repeated *normal*
    denominator factors in every power-basis coordinate and leaves special
    factors for later relative reduction.  Unsupported leading equations
    raise :class:`NotImplementedError`; they never produce a false claim that
    a remainder is Hermite-reduced.
    """
    extension, lower_extension, lower_derivation = _relative_lower_monomial(
        derivation,
    )
    radical = recognize_relative_simple_radical(extension)
    if radical is None:
        raise NotImplementedError(
            "relative Hermite reduction needs a terminal pure simple-radical extension",
        )
    eta = _diagonal_generator_logarithmic_derivative(extension, derivation)
    if eta is None:
        raise NotImplementedError(
            "relative Hermite reduction needs a diagonal simple-radical "
            "generator derivative",
        )

    element = _extension_element(extension, integrand)
    exact_coordinates: list[FieldElement] = []
    remainder_coordinates: list[FieldElement] = []
    modes: list[RelativeSimpleRadicalHermiteMode] = []

    # Import locally: ``monomial`` is a broader facade that eventually uses
    # algebraic-aware exact dispatch, whereas this module is imported by the
    # algebraic facade itself.
    from ..monomial.hermite import hermite_reduce_monomial

    for power, coordinate in enumerate(extension.power_basis_coordinates(element)):
        mode_coefficient = extension.parent_field.convert(power) * eta
        if power == 0:
            coordinate_exact, coordinate_remainder = hermite_reduce_monomial(
                coordinate,
                lower_derivation,
            )
        else:
            coordinate_exact, coordinate_remainder = _reduce_mode_normal_poles(
                coordinate,
                mode_coefficient,
                lower_extension,
                lower_derivation,
            )
        exact_coordinates.append(coordinate_exact)
        remainder_coordinates.append(coordinate_remainder)
        modes.append(
            RelativeSimpleRadicalHermiteMode(
                power,
                mode_coefficient,
                coordinate_exact,
                coordinate_remainder,
            ),
        )

    result = RelativeSimpleRadicalHermiteReduction(
        radical,
        derivation,
        element,
        extension.from_power_basis_coordinates(tuple(exact_coordinates)),
        extension.from_power_basis_coordinates(tuple(remainder_coordinates)),
        tuple(modes),
    )
    if not result.reconstructs:
        raise ValueError("relative Hermite reduction failed to reconstruct")
    if not result.normal_denominators_are_square_free:
        raise ValueError("relative Hermite reduction left a repeated normal pole")
    return result
