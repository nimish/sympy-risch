r"""Bronstein normal-residue criterion and logarithmic residue reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polytools import Poly, resultant

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..lrt import _differential_lazard_rioboo_trager
from ..reduction.fractions import (
    _normal_fraction_part,
)
from ..tower.core import (
    Derivation,
    MonomialExtension,
    _coerce,
    _domain_symbols,
)


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


@dataclass(frozen=True, slots=True)
class BronsteinResidueReduction:
    r"""Generalized logarithmic reduction of the normal simple fraction."""

    certificate: BronsteinResidueCertificate
    logarithmic_parts: tuple[tuple[Poly, Poly], ...]
    logarithmic_derivative: FieldElement
    remainder: FieldElement

    @property
    def all_normal_residues_removed(self) -> bool:
        """Whether no nonconstant normal-residue obstruction remains."""
        return self.certificate.all_residues_constant


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


def _differentiate_logand(
    logand: Poly,
    derivation: Derivation,
    residue: Symbol,
) -> Expr:
    r"""Differentiate an LRT logand while treating its residue as constant."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("logand differentiation requires a monomial level")

    expression: Expr = S.Zero
    for (power,), coefficient in logand.terms():
        coefficient_expression = logand.domain.to_sympy(coefficient)
        coefficient_polynomial = Poly(
            coefficient_expression,
            residue,
            domain=extension.parent_field,
        )
        coefficient_derivative = _coefficient_derivative(
            coefficient_polynomial,
            derivation.lower(),
        )
        expression += coefficient_derivative.as_expr() * extension.generator**power
        if power:
            expression += (
                power
                * coefficient_polynomial.as_expr()
                * extension.generator ** (power - 1)
                * extension.generator_derivative.as_expr()
            )
    return expression


def _logarithmic_part_derivative(
    logand: Poly,
    residue_factor: Poly,
    derivation: Derivation,
    residue: Symbol,
) -> FieldElement:
    r"""Return ``sum(alpha * D(log(s(alpha))))`` by a quotient-ring trace."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("logarithmic reduction requires a monomial level")

    field = extension.field
    modulus = Poly(
        residue_factor.as_expr(),
        residue,
        domain=field,
    ).monic()
    logand_modulus = Poly(
        logand.as_expr(),
        residue,
        domain=field,
    ).rem(modulus)
    logand_derivative = Poly(
        _differentiate_logand(logand, derivation, residue),
        residue,
        domain=field,
    ).rem(modulus)
    residue_polynomial = Poly(residue, residue, domain=field)
    trace_element = (
        residue_polynomial * logand_derivative * logand_modulus.invert(modulus)
    ).rem(modulus)

    # The trace of multiplication by u in F[z]/(q) is the sum of u(alpha)
    # over the roots alpha of the square-free q.  Computing the diagonal in
    # the power basis avoids constructing any algebraic residue explicitly.
    trace = field.zero
    for power in range(modulus.degree()):
        basis_element = Poly(residue**power, residue, domain=field)
        product = (trace_element * basis_element).rem(modulus)
        trace += _coerce(field, product.nth(power))
    return trace


def bronstein_residue_criterion(
    remainder: object,
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
    top extension must pass the derivation-bound no-new-constants replay.
    """
    if derivation.level < 0:
        raise TypeError("the residue criterion requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("the residue criterion requires a monomial level")
    require_preserved_constant_field(derivation, "the residue criterion")

    residue = Dummy("z") if residue is None else residue
    if not isinstance(residue, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")
    tower_symbols = _domain_symbols(extension.field) | {
        level.generator for level in derivation.tower.levels
    }
    if residue in tower_symbols:
        raise ValueError("the residue symbol must be new over the tower field")

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
        normal_numerator.as_expr() - residue * denominator_derivative.as_expr(),
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


def residue_reduce_monomial(
    remainder: object,
    derivation: Derivation,
    residue: Symbol | None = None,
) -> BronsteinResidueReduction:
    r"""Extract constant-residue logarithms from a monomial remainder.

    This completes the constructive part of ``ResidueReduce`` in Bronstein,
    *Symbolic Integration I*, Section 5.6.  Each returned ``(s_i, q_i)`` in
    ``logarithmic_parts`` represents

    .. math::

        \sum_{q_i(\alpha)=0} \alpha\log(s_i(\alpha, t)).

    The result also contains the derivative of that encoded logarithmic sum
    and the remainder after subtracting it.  Nonconstant residue factors are
    left in the remainder and recorded by the attached certificate.
    """
    if derivation.level < 0:
        raise TypeError("residue reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("residue reduction requires a monomial level")

    element = _coerce(extension.field, remainder)
    certificate = bronstein_residue_criterion(element, derivation, residue)
    residue = cast(Symbol, certificate.residue_polynomial.gens[0])
    selected_factors = tuple(
        (factor.constant_factor, factor.multiplicity)
        for factor in certificate.factors
        if factor.constant_factor.degree() > 0
    )
    if not selected_factors:
        return BronsteinResidueReduction(
            certificate,
            (),
            extension.field.zero,
            element,
        )

    denominator_derivative = derivation.polynomial(
        certificate.normal_denominator,
    )
    logarithmic_parts = _differential_lazard_rioboo_trager(
        certificate.normal_numerator,
        certificate.normal_denominator,
        denominator_derivative,
        selected_factors,
        extension.generator,
        residue,
    )
    logarithmic_derivative = extension.field.zero
    for logand, residue_factor in logarithmic_parts:
        logarithmic_derivative += _logarithmic_part_derivative(
            logand,
            residue_factor,
            derivation,
            residue,
        )

    return BronsteinResidueReduction(
        certificate,
        tuple(logarithmic_parts),
        logarithmic_derivative,
        element - logarithmic_derivative,
    )
