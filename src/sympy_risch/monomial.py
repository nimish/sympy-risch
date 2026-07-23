r"""Reduction helpers for transcendental monomial extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polytools import Poly
from sympy.polys.polytools import resultant
from sympy.simplify.radsimp import fraction

from .tower import Derivation, MonomialExtension, _coerce


@dataclass(frozen=True, slots=True)
class BronsteinResidueFactor:
    r"""One square-free residue factor and its constant-root split.

    If ``residue_factor`` is Bronstein's :math:`R_i`, then
    ``constant_factor`` is :math:`Q_i = \gcd(R_i, \kappa(R_i))`.  The
    complementary factor records the nonconstant residues that obstruct an
    elementary integral by Liouville's theorem.
    """

    residue_factor: Poly
    multiplicity: int
    constant_factor: Poly
    nonconstant_factor: Poly

    @property
    def all_roots_constant(self) -> bool:
        """Whether this factor contains no nonconstant residue obstruction."""
        return self.nonconstant_factor.degree() == 0


@dataclass(frozen=True, slots=True)
class BronsteinResidueCertificate:
    r"""Inspectable result of the generalized normal-residue criterion."""

    normal_numerator: Poly
    normal_denominator: Poly
    residue_polynomial: Poly
    kappa_residue_polynomial: Poly
    factors: tuple[BronsteinResidueFactor, ...]

    @property
    def all_residues_constant(self) -> bool:
        r"""Whether every root of the residue polynomial passes the test.

        This is necessary, but not sufficient, for the entire integrand to
        have an elementary integral: polynomial and special parts still need
        their extension-family reductions.
        """
        return all(factor.all_roots_constant for factor in self.factors)


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


def _normal_fraction_part(
    element: Any,
    extension: MonomialExtension,
    derivation: Derivation,
) -> tuple[Poly, Poly]:
    r"""Return ``A, N`` for the normal proper fraction ``A/N`` in ``element``.

    For a reduced denominator ``N*S``, Bezout reduction gives

    ``C/(N*S) = A/N + B/S`` with ``A = C*S**(-1) mod N``.

    This is the decomposition used before the generalized Rothstein--Trager
    resultant in section 3.4 of Bronstein's ISSAC'98 tutorial.
    """
    numerator, denominator = _reduced_polynomial_fraction(element, extension)

    # A coefficient from the lower field is a unit in K[t].  Moving the
    # leading coefficient into the numerator makes the factor decomposition
    # canonical; for example 1/(x*t) becomes (1/x)/t.
    denominator_leading_coefficient = denominator.domain.convert(
        denominator.LC(),
    )
    denominator_leading_inverse = (
        denominator.domain.one/denominator_leading_coefficient
    )
    numerator = numerator.mul_ground(denominator_leading_inverse)
    denominator = denominator.monic()

    _, proper_numerator = numerator.div(denominator)
    if proper_numerator.is_zero:
        return denominator.zero, denominator.one

    normal_denominator, special_denominator = split_normal_special(
        denominator,
        derivation,
    )
    if normal_denominator.degree() == 0:
        return denominator.zero, denominator.one

    normal_numerator = (
        proper_numerator*special_denominator.invert(normal_denominator)
    ).rem(normal_denominator)
    if normal_numerator.is_zero:
        return denominator.zero, denominator.one
    return normal_numerator, normal_denominator.monic()


def _coefficient_derivative(
    polynomial: Poly,
    derivation: Derivation,
) -> Poly:
    r"""Apply Bronstein's ``kappa``: differentiate coefficients, fixing ``z``."""
    return Poly.from_dict(
        {
            monomial: derivation(coefficient)
            for monomial, coefficient in polynomial.terms()
        },
        polynomial.gens[0],
        domain=polynomial.domain,
    )


def bronstein_residue_criterion(
    remainder: Any,
    derivation: Derivation,
    residue: Symbol | None = None,
) -> BronsteinResidueCertificate:
    r"""Test whether the normal residues of a monomial remainder are constant.

    ``remainder`` should already have undergone monomial Hermite reduction.
    The function isolates its normal proper fraction ``A/N`` and constructs

    ``R(z) = resultant_t(N, A - z*D(N))``.

    For every square-free factor :math:`R_i`, it computes
    :math:`Q_i = \gcd(R_i, \kappa(R_i))`, where :math:`\kappa` differentiates
    lower-field coefficients and fixes ``z``.  A proper quotient
    :math:`R_i/Q_i` certifies a nonconstant residue, contradicting the
    Liouville form of an elementary integral.

    Passing the criterion only settles the normal fraction.  It does not yet
    integrate, or prove integrability of, polynomial and special parts.  The
    theorem also assumes the differential tower has the intended constant
    field (the usual no-new-constants hypothesis).
    """
    if derivation.level < 0:
        raise TypeError("the residue criterion requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("the residue criterion requires a monomial level")

    residue = residue or Dummy("z")
    tower_generators = {level.generator for level in derivation.tower.levels}
    if residue in tower_generators:
        raise ValueError("the residue symbol must differ from every tower generator")

    normal_numerator, normal_denominator = _normal_fraction_part(
        _coerce(extension.field, remainder),
        extension,
        derivation,
    )
    if normal_denominator.degree() == 0:
        residue_polynomial = Poly(1, residue, domain=extension.parent_field)
        return BronsteinResidueCertificate(
            normal_numerator,
            normal_denominator,
            residue_polynomial,
            residue_polynomial.zero,
            (),
        )

    denominator_derivative = derivation.polynomial(normal_denominator)
    if normal_denominator.gcd(denominator_derivative).degree() > 0:
        raise ValueError(
            "the residue criterion requires a Hermite-reduced normal denominator"
        )

    residue_expression = resultant(
        normal_denominator.as_expr(),
        normal_numerator.as_expr()
        - residue*denominator_derivative.as_expr(),
        extension.generator,
    )
    residue_polynomial = Poly(
        residue_expression,
        residue,
        domain=extension.parent_field,
    )
    if residue_polynomial.is_zero:
        raise ValueError("the generalized residue resultant unexpectedly vanished")
    residue_polynomial = residue_polynomial.monic()
    kappa_residue_polynomial = _coefficient_derivative(
        residue_polynomial,
        derivation.lower(),
    )

    factors = []
    for residue_factor, multiplicity in residue_polynomial.sqf_list()[1]:
        residue_factor = residue_factor.monic()
        kappa_factor = _coefficient_derivative(
            residue_factor,
            derivation.lower(),
        )
        constant_factor = residue_factor.gcd(kappa_factor).monic()
        nonconstant_factor = residue_factor.exquo(constant_factor).monic()
        factors.append(
            BronsteinResidueFactor(
                residue_factor,
                multiplicity,
                constant_factor,
                nonconstant_factor,
            )
        )

    return BronsteinResidueCertificate(
        normal_numerator,
        normal_denominator,
        residue_polynomial,
        kappa_residue_polynomial,
        tuple(factors),
    )


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
