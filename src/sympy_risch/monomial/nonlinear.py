r"""Common reduction for nonlinear monomial extensions."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from .._typing import FieldElement, is_field_zero
from ..constant_proofs import require_preserved_constant_field
from ..reduction.fractions import (
    _polynomial_and_proper_fraction,
    split_normal_special,
)
from ..tower.core import (
    Derivation,
    MonomialExtension,
    _coerce,
)
from .hermite import hermite_reduce_monomial
from .residues import BronsteinResidueReduction, residue_reduce_monomial


@dataclass(frozen=True, slots=True)
class ReducedFunctionReduction:
    r"""Reduction of the residue at infinity of a reduced monomial function.

    ``infinity_coefficient`` is Bronstein's :math:`b`, the coefficient of
    :math:`t^{\delta-1}` in the reduced polynomial part, and
    ``infinity_residue`` is :math:`c=b/\lambda`, where
    :math:`\lambda=\operatorname{lc}_t(Dt)`.  A nonzero
    ``infinity_residue_derivative`` is a Liouville obstruction.  Otherwise a
    nonzero residue can be removed as a logarithmic derivative when
    ``logarithmic_argument`` supplies a nonconstant special polynomial.
    """

    polynomial_part: Poly
    special_numerator: Poly
    special_denominator: Poly
    infinity_coefficient: FieldElement
    infinity_residue: FieldElement
    infinity_residue_derivative: FieldElement
    logarithmic_coefficient: FieldElement | None
    logarithmic_argument: Poly | None
    logarithmic_derivative: FieldElement
    remainder: FieldElement

    @property
    def has_infinity_obstruction(self) -> bool:
        """Whether the candidate residue at infinity is nonconstant."""
        return not is_field_zero(self.infinity_residue_derivative)

    @property
    def needs_special_polynomial(self) -> bool:
        """Whether logarithmic removal is blocked only by an unknown special."""
        return (
            not is_field_zero(self.infinity_residue)
            and not self.has_infinity_obstruction
            and self.logarithmic_argument is None
        )


@dataclass(frozen=True, slots=True)
class NonlinearMonomialReduction:
    r"""End-to-end common reduction of a nonlinear monomial integrand.

    The nested results retain the finite-residue and residue-at-infinity
    certificates.  The flattened ``exact_part``, ``logarithmic_derivative``,
    and ``remainder`` satisfy

    ``f = D(exact_part) + logarithmic_derivative + remainder``.

    Family-specific reduction of the final special part is deliberately a
    later step.
    """

    hermite_exact_part: FieldElement
    residue_reduction: BronsteinResidueReduction
    polynomial_exact_part: Poly
    polynomial_remainder: Poly
    reduced_function_reduction: ReducedFunctionReduction | None
    exact_part: FieldElement
    logarithmic_derivative: FieldElement
    remainder: FieldElement

    @property
    def has_finite_residue_obstruction(self) -> bool:
        """Whether a nonconstant normal residue proves non-elementarity."""
        return not self.residue_reduction.all_normal_residues_removed

    @property
    def has_infinity_obstruction(self) -> bool:
        """Whether the reduced polynomial has a nonconstant infinity residue."""
        return (
            self.reduced_function_reduction is not None
            and self.reduced_function_reduction.has_infinity_obstruction
        )

    @property
    def has_obstruction(self) -> bool:
        """Whether either common structure theorem proves non-elementarity."""
        return self.has_finite_residue_obstruction or self.has_infinity_obstruction

    @property
    def needs_special_polynomial(self) -> bool:
        """Whether infinity reduction awaits a known nontrivial special."""
        return (
            self.reduced_function_reduction is not None
            and self.reduced_function_reduction.needs_special_polynomial
        )


def polynomial_reduce_monomial(
    polynomial: Poly,
    derivation: Derivation,
) -> tuple[Poly, Poly]:
    r"""Reduce a polynomial at a nonlinear monomial level.

    For ``delta = degree(Dt) >= 2``, return ``(q, r)`` satisfying

    ``polynomial = D(q) + r`` and ``degree(r) < delta``.

    This is the polynomial reduction of Bronstein, *Symbolic Integration I*,
    Section 5.4.  Primitive and hyperexponential polynomial reductions need
    lower-field integration problems and are handled separately later.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if derivation.level < 0:
        raise TypeError("polynomial reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("polynomial reduction requires a monomial level")

    # Validate both the generator and coefficient domain before inspecting
    # leading terms.  This also validates the zero polynomial.
    derivation.polynomial(polynomial)
    generator_derivative = extension.generator_derivative
    delta = generator_derivative.degree()
    if delta < 2:
        raise ValueError(
            "nonlinear monomial polynomial reduction requires degree(Dt) >= 2"
        )

    domain = polynomial.domain
    derivative_leading_coefficient = domain.convert(
        generator_derivative.LC(),
    )
    exact_part = polynomial.zero
    remainder = polynomial

    while not remainder.is_zero and remainder.degree() >= delta:
        degree = remainder.degree()
        correction_degree = degree - delta + 1
        leading_coefficient = domain.convert(remainder.LC())
        leading_derivative_coefficient = (
            domain.convert(correction_degree) * derivative_leading_coefficient
        )
        correction_coefficient = domain.exquo(
            leading_coefficient,
            leading_derivative_coefficient,
        )
        correction = Poly.from_dict(
            {(correction_degree,): correction_coefficient},
            extension.generator,
            domain=domain,
        )
        new_remainder = remainder - derivation.polynomial(correction)
        if not new_remainder.is_zero and new_remainder.degree() >= degree:
            raise ValueError("polynomial reduction did not lower the degree")

        exact_part += correction
        remainder = new_remainder

    return exact_part, remainder


def reduce_monomial_reduced_function(
    remainder: object,
    derivation: Derivation,
    special_polynomial: Poly | None = None,
) -> ReducedFunctionReduction:
    r"""Reduce the residue at infinity of a nonlinear monomial remainder.

    This implements the common nonlinear step in Bronstein, *Symbolic
    Integration I*, Section 5.7.  The input must already be reduced:

    * its proper denominator has no normal part;
    * its polynomial part has degree less than
      :math:`\delta=\deg_t(Dt)`.

    Write the polynomial part as ``p`` and set

    ``b = [t**(delta - 1)]p``, ``c = b/lambda``,

    where ``lambda`` is the leading coefficient of ``Dt``.  If ``D(c)`` is
    nonzero, Liouville's theorem certifies non-elementarity.  If ``c`` is a
    nonzero constant and ``q`` is a nonconstant special polynomial, then

    ``(c/degree(q))*D(q)/q``

    removes the leading term.  A special denominator is preferred as ``q``;
    otherwise the special part of ``Dt`` is tried.  ``special_polynomial`` can
    supply a different known special.  When none is known, the result records
    ``needs_special_polynomial`` and leaves the input unchanged; no complete
    general no-special continuation is implemented.
    """
    if derivation.level < 0:
        raise TypeError("reduced-function reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("reduced-function reduction requires a monomial level")
    require_preserved_constant_field(derivation, "reduced-function reduction")

    generator_derivative = extension.generator_derivative
    delta = generator_derivative.degree()
    if delta < 2:
        raise ValueError("reduced-function reduction requires degree(Dt) >= 2")

    field = extension.field
    parent = extension.parent_field
    element = _coerce(field, remainder)
    polynomial, proper_numerator, proper_denominator = _polynomial_and_proper_fraction(
        element, extension
    )
    if proper_denominator.degree() > 0:
        normal_denominator, _ = split_normal_special(
            proper_denominator,
            derivation,
        )
        if normal_denominator.degree() > 0:
            raise ValueError(
                "reduced-function reduction requires all normal residues "
                "to be removed first"
            )

    if not polynomial.is_zero and polynomial.degree() >= delta:
        raise ValueError(
            "reduced-function reduction requires polynomial reduction first"
        )

    infinity_coefficient = _coerce(
        parent,
        polynomial.nth(delta - 1),
    )
    derivative_leading_coefficient = parent.convert(
        generator_derivative.LC(),
    )
    infinity_residue = parent.exquo(
        infinity_coefficient,
        derivative_leading_coefficient,
    )
    infinity_residue_derivative = derivation.lower()(infinity_residue)
    zero = parent.zero

    def result(
        *,
        logarithmic_coefficient: FieldElement | None = None,
        logarithmic_argument: Poly | None = None,
        logarithmic_derivative: FieldElement = field.zero,
        reduced_remainder: FieldElement = element,
    ) -> ReducedFunctionReduction:
        return ReducedFunctionReduction(
            polynomial,
            proper_numerator,
            proper_denominator,
            infinity_coefficient,
            infinity_residue,
            infinity_residue_derivative,
            logarithmic_coefficient,
            logarithmic_argument,
            logarithmic_derivative,
            reduced_remainder,
        )

    if infinity_residue == zero or infinity_residue_derivative != zero:
        return result()

    logarithmic_argument = special_polynomial
    if logarithmic_argument is None and proper_denominator.degree() > 0:
        logarithmic_argument = proper_denominator
    if logarithmic_argument is None:
        _, derivative_special_part = split_normal_special(
            generator_derivative,
            derivation,
        )
        if derivative_special_part.degree() > 0:
            logarithmic_argument = derivative_special_part
    if logarithmic_argument is None:
        return result()

    if not isinstance(logarithmic_argument, Poly):
        raise TypeError("special_polynomial must be a SymPy Poly")
    derivation.polynomial(logarithmic_argument)
    if logarithmic_argument.degree() <= 0:
        raise ValueError("special_polynomial must be nonconstant")
    logarithmic_argument = logarithmic_argument.monic()
    argument_derivative = derivation.polynomial(logarithmic_argument)
    if not argument_derivative.rem(logarithmic_argument).is_zero:
        raise ValueError("special_polynomial must divide its derivative")

    logarithmic_coefficient = parent.exquo(
        infinity_residue,
        parent.convert(logarithmic_argument.degree()),
    )
    logarithmic_derivative_polynomial = argument_derivative.exquo(
        logarithmic_argument,
    ).mul_ground(logarithmic_coefficient)
    logarithmic_derivative = _coerce(
        field,
        logarithmic_derivative_polynomial.as_expr(),
    )
    reduced_remainder = element - logarithmic_derivative

    reduced_polynomial, _, _ = _polynomial_and_proper_fraction(
        reduced_remainder,
        extension,
    )
    if not reduced_polynomial.is_zero and reduced_polynomial.degree() >= delta - 1:
        raise ValueError("special polynomial did not remove the residue at infinity")

    return result(
        logarithmic_coefficient=logarithmic_coefficient,
        logarithmic_argument=logarithmic_argument,
        logarithmic_derivative=logarithmic_derivative,
        reduced_remainder=reduced_remainder,
    )


def reduce_nonlinear_monomial(
    integrand: object,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
    special_polynomial: Poly | None = None,
) -> NonlinearMonomialReduction:
    r"""Apply all common reductions for a nonlinear monomial extension.

    The stages are, in order:

    1. Hermite reduction of repeated normal poles;
    2. generalized residue reduction of the simple normal fraction;
    3. nonlinear polynomial reduction;
    4. residue-at-infinity reduction when no finite obstruction remains.

    The function requires the top extension to pass the no-new-constants
    replay. It does not perform the final primitive, hyperexponential,
    hypertangent, or general nonlinear-special family reduction.
    """
    if derivation.level < 0:
        raise TypeError("nonlinear reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("nonlinear reduction requires a monomial level")
    require_preserved_constant_field(derivation, "nonlinear monomial reduction")
    if extension.generator_derivative.degree() < 2:
        raise ValueError("nonlinear reduction requires degree(Dt) >= 2")

    field = extension.field
    element = _coerce(field, integrand)
    hermite_exact_part, hermite_remainder = hermite_reduce_monomial(
        element,
        derivation,
    )
    residue_reduction = residue_reduce_monomial(
        hermite_remainder,
        derivation,
        residue,
    )

    polynomial_part, _, _ = _polynomial_and_proper_fraction(
        residue_reduction.remainder,
        extension,
    )
    polynomial_exact_part, polynomial_remainder = polynomial_reduce_monomial(
        polynomial_part,
        derivation,
    )
    polynomial_exact_element = _coerce(
        field,
        polynomial_exact_part.as_expr(),
    )
    exact_part = hermite_exact_part + polynomial_exact_element
    after_polynomial_reduction = residue_reduction.remainder - derivation(
        polynomial_exact_element
    )

    reduced_function_reduction = None
    infinity_logarithmic_derivative = field.zero
    remainder = after_polynomial_reduction
    if residue_reduction.all_normal_residues_removed:
        reduced_function_reduction = reduce_monomial_reduced_function(
            after_polynomial_reduction,
            derivation,
            special_polynomial,
        )
        infinity_logarithmic_derivative = (
            reduced_function_reduction.logarithmic_derivative
        )
        remainder = reduced_function_reduction.remainder

    return NonlinearMonomialReduction(
        hermite_exact_part,
        residue_reduction,
        polynomial_exact_part,
        polynomial_remainder,
        reduced_function_reduction,
        exact_part,
        (residue_reduction.logarithmic_derivative + infinity_logarithmic_derivative),
        remainder,
    )
