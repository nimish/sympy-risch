r"""Primitive monomial reduction, exactness, and logarithmic signatures."""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..reduction.fractions import (
    _polynomial_and_proper_fraction,
    split_normal_special,
)
from ..reduction.polynomials import exact_field_equal, exact_polynomial_equal
from ..tower.core import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)
from .hermite import hermite_reduce_monomial
from .logarithmic import RadicalLogarithmicDerivativeResult
from .primitive_polynomial import (
    PrimitivePolynomialReduction,
    polynomial_reduce_primitive,
)
from .principal_parts import (
    _factor_principal_numerators,
    _qq_constant_polynomial_multiple,
)
from .residues import BronsteinResidueReduction, residue_reduce_monomial


@dataclass(frozen=True, slots=True)
class PrimitiveMonomialReduction:
    r"""End-to-end reduction at a primitive monomial level.

    The flattened fields satisfy

    ``f = D(exact_part) + logarithmic_derivative + remainder``.

    A nonconstant normal residue or a positive-degree polynomial remainder
    proves non-elementarity.  Otherwise ``lower_remainder`` is the remaining
    element of the parent field for recursive integration.
    """

    hermite_exact_part: FieldElement
    residue_reduction: BronsteinResidueReduction
    polynomial_reduction: PrimitivePolynomialReduction
    exact_part: FieldElement
    logarithmic_derivative: FieldElement
    remainder: FieldElement
    lower_remainder: FieldElement | None

    @property
    def has_finite_residue_obstruction(self) -> bool:
        """Whether a nonconstant normal residue proves non-elementarity."""
        return not self.residue_reduction.all_normal_residues_removed

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether a positive-degree reduced polynomial remains."""
        polynomial = self.polynomial_reduction.remainder
        return not polynomial.is_zero and polynomial.degree() > 0

    @property
    def is_non_elementary(self) -> bool:
        """Whether this level has supplied a Liouville obstruction."""
        return self.has_finite_residue_obstruction or self.has_polynomial_obstruction

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only recursive integration in the parent field remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


@dataclass(frozen=True, slots=True)
class PrimitiveExactDerivativeResult:
    r"""Certified exact-derivative result in a primitive field ``K(t)``.

    The fields satisfy

    ``f = D(exact_part) + obstruction``.

    Unlike elementary integration, logarithmic derivatives remain in
    ``obstruction`` because their antiderivatives do not belong to ``K(t)``.
    """

    hermite_exact_part: FieldElement
    polynomial_reduction: PrimitivePolynomialReduction
    exact_part: FieldElement
    proper_numerator: Poly
    proper_denominator: Poly
    obstruction: FieldElement

    @property
    def has_proper_fraction_obstruction(self) -> bool:
        """Whether a nonzero square-free proper fraction remains."""
        return not self.proper_numerator.is_zero

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether a reduced polynomial coefficient remains."""
        return self.polynomial_reduction.has_obstruction

    @property
    def has_solution(self) -> bool:
        """Whether the input is a derivative of an element of ``K(t)``."""
        return not (
            self.has_proper_fraction_obstruction or self.has_polynomial_obstruction
        )

    @property
    def antiderivative(self) -> FieldElement | None:
        """Return the in-field antiderivative, or ``None`` if none exists."""
        return self.exact_part if self.has_solution else None


@dataclass(frozen=True, slots=True)
class PrimitiveLogarithmicDerivativeFactor:
    r"""The complete principal part at one irreducible ``p`` in ``K[t]``.

    ``principal_numerators[j - 1]`` is the reduced numerator of ``1/p**j``.
    A radical logarithmic derivative may have only a simple pole, whose
    numerator is a rational constant multiple of ``Dp`` modulo ``p``.
    ``logarithmic_multiplicity`` stores that constant when it exists.
    """

    factor: Poly
    multiplicity: int
    principal_numerators: tuple[Poly, ...]
    factor_derivative: Poly
    logarithmic_multiplicity: FieldElement | None

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether a nonzero principal part has order greater than one."""
        return any(not numerator.is_zero for numerator in self.principal_numerators[1:])

    @property
    def has_logarithmic_residue(self) -> bool:
        """Whether the simple numerator is a rational multiple of ``Dp``."""
        return self.logarithmic_multiplicity is not None


@dataclass(frozen=True, slots=True)
class PrimitiveLogarithmicDerivativeSignature:
    r"""Canonical radical-logarithmic data for one primitive tower level.

    The signature records the unique decomposition

    .. math::

        f=P(t)+\sum_{p}\sum_{j=1}^{m_p} A_{p,j}/p^j

    over monic irreducible ``p`` in ``K[t]``.  When ``K`` is a recursively
    supported primitive tower, ``f`` is a rational multiple of a logarithmic
    derivative precisely when:

    * ``P`` is constant in ``t`` and its value is a radical logarithmic
      derivative in ``K``;
    * every higher principal part ``A[p, j]`` for ``j > 1`` vanishes; and
    * every simple numerator is a rational multiple of ``Dp`` modulo ``p``.

    On success, ``minimal_power*f = D(relation_argument)/relation_argument``.
    The stored decomposition remains useful on failure: it is the linear
    obstruction data needed by the later parametric solver.
    """

    derivation: Derivation
    integrand: FieldElement
    polynomial_part: Poly
    lower_logarithmic_result: RadicalLogarithmicDerivativeResult
    factors: tuple[PrimitiveLogarithmicDerivativeFactor, ...]
    minimal_power: int | None
    relation_argument: FieldElement | None

    @property
    def has_positive_degree_polynomial_obstruction(self) -> bool:
        """Whether the polynomial part depends nontrivially on ``t``."""
        return not self.polynomial_part.is_zero and self.polynomial_part.degree() > 0

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether some finite place has a pole of order greater than one."""
        return any(factor.has_repeated_pole_obstruction for factor in self.factors)

    @property
    def has_nonlogarithmic_residue_obstruction(self) -> bool:
        """Whether some simple principal part is not rational times ``Dp``."""
        return any(not factor.has_logarithmic_residue for factor in self.factors)

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether the constant polynomial part is not radical logarithmic."""
        return not self.lower_logarithmic_result.has_solution

    @property
    def has_solution(self) -> bool:
        """Whether a positive integer multiple is a logarithmic derivative."""
        return self.minimal_power is not None

    @property
    def reconstructs_decomposition(self) -> bool:
        """Whether the stored principal parts reconstruct ``integrand``."""
        extension = self.derivation.tower.levels[self.derivation.level]
        if not isinstance(extension, MonomialExtension):
            return False
        expression = self.polynomial_part.as_expr()
        for factor in self.factors:
            for pole_order, numerator in enumerate(
                factor.principal_numerators,
                start=1,
            ):
                expression += (
                    numerator.as_expr() / factor.factor.as_expr() ** pole_order
                )
        return exact_field_equal(
            _coerce(extension.field, expression),
            _coerce(extension.field, self.integrand),
            extension.field,
        )

    @property
    def reconstructs_relation(self) -> bool:
        """Whether the successful radical-logarithmic identity is exact."""
        if not self.has_solution or self.relation_argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.relation_argument)
        return exact_field_equal(
            self.minimal_power * _coerce(field, self.integrand),
            self.derivation(argument) / argument,
            field,
        )


def reduce_primitive_monomial(
    integrand: object,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> PrimitiveMonomialReduction:
    r"""Apply the complete reduction for a primitive monomial over ``K``.

    The stages are:

    1. Hermite reduction of repeated normal poles;
    2. generalized residue reduction of the simple normal fraction;
    3. primitive polynomial reduction modulo ``D(K) + C*Dt``;
    4. reduction to the parent field when no obstruction remains.

    In a primitive extension with preserved constants there are no
    nonconstant special polynomials.  Consequently, after all constant normal
    residues have been removed, no proper fraction may remain.  A polynomial
    remainder of positive degree in ``t`` is non-elementary; a constant
    polynomial belongs to the parent field and is returned for recursion.
    """
    if derivation.level < 0:
        raise TypeError("primitive reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive reduction requires a monomial level")
    if extension.generator_derivative.degree() != 0:
        raise ValueError("primitive reduction requires Dt in K")
    require_preserved_constant_field(derivation, "primitive reduction")

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
    polynomial_reduction = polynomial_reduce_primitive(
        polynomial_part,
        derivation,
    )
    polynomial_exact_element = _coerce(
        field,
        polynomial_reduction.exact_part.as_expr(),
    )
    exact_part = hermite_exact_part + polynomial_exact_element
    remainder = residue_reduction.remainder - derivation(polynomial_exact_element)
    final_polynomial, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(remainder, extension)
    )
    if not exact_polynomial_equal(
        final_polynomial,
        polynomial_reduction.remainder,
    ):
        raise ValueError("primitive reduction produced an inconsistent remainder")

    lower_remainder = None
    if residue_reduction.all_normal_residues_removed:
        if not proper_numerator.is_zero:
            normal_denominator, special_denominator = split_normal_special(
                proper_denominator,
                derivation,
            )
            if special_denominator.degree() > 0:
                raise ValueError(
                    "a primitive extension with preserved constants cannot "
                    "have a nonconstant special denominator"
                )
            if normal_denominator.degree() > 0:
                raise ValueError("residue reduction left an unexpected normal fraction")

        if final_polynomial.is_zero:
            lower_remainder = extension.parent_field.zero
        elif final_polynomial.degree() == 0:
            lower_remainder = extension.parent_field.convert(final_polynomial.nth(0))

    return PrimitiveMonomialReduction(
        hermite_exact_part,
        residue_reduction,
        polynomial_reduction,
        exact_part,
        residue_reduction.logarithmic_derivative,
        remainder,
        lower_remainder,
    )


def primitive_logarithmic_derivative_signature(
    integrand: object,
    derivation: Derivation,
) -> PrimitiveLogarithmicDerivativeSignature:
    r"""Compute a radical-logarithmic signature at a primitive tower level.

    The top level must be primitive, with ``Dt`` in the lower field, and must
    pass the replayed no-new-constants criterion. The lower field is decided
    recursively down to the standard rational base ``QQ(x)``.
    Factor and principal-part data do not depend on a proposed parametric
    ratio, so the parametric structure fallback can compare signatures.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("primitive logarithmic signatures require a Derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "primitive logarithmic signatures require a primitive level "
            "above the rational base"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive logarithmic signatures require a monomial level")
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError(
            "primitive logarithmic signatures require Dt in the lower field"
        )
    require_preserved_constant_field(derivation, "primitive logarithmic signatures")
    field = extension.field
    element = _coerce(field, integrand)
    polynomial_part, proper_numerator, denominator = _polynomial_and_proper_fraction(
        element, extension
    )
    lower_component = (
        extension.parent_field.zero
        if polynomial_part.is_zero
        else extension.parent_field.convert(polynomial_part.nth(0))
    )
    from .logarithmic import solve_radical_logarithmic_derivative

    lower_logarithmic_result = solve_radical_logarithmic_derivative(
        lower_component,
        derivation.lower(),
    )
    lower_power = lower_logarithmic_result.minimal_power

    factor_signatures = []
    for raw_factor, multiplicity in denominator.factor_list()[1]:
        factor = raw_factor.monic()
        principal_numerators = _factor_principal_numerators(
            proper_numerator,
            denominator,
            factor,
            multiplicity,
        )
        factor_derivative = derivation.polynomial(factor).rem(factor)
        if factor_derivative.is_zero:
            raise ValueError(
                "a primitive extension with preserved constants cannot "
                "have a nonconstant special factor"
            )
        logarithmic_multiplicity = _qq_constant_polynomial_multiple(
            principal_numerators[0],
            factor_derivative,
            derivation.lower(),
        )
        factor_signatures.append(
            PrimitiveLogarithmicDerivativeFactor(
                factor,
                multiplicity,
                principal_numerators,
                factor_derivative,
                logarithmic_multiplicity,
            )
        )

    factors = tuple(factor_signatures)
    has_positive_degree = not polynomial_part.is_zero and polynomial_part.degree() > 0
    has_repeated_pole = any(factor.has_repeated_pole_obstruction for factor in factors)
    has_nonlogarithmic_residue = any(
        not factor.has_logarithmic_residue for factor in factors
    )

    minimal_power = None
    relation_argument = None
    if (
        lower_power is not None
        and not has_positive_degree
        and not has_repeated_pole
        and not has_nonlogarithmic_residue
    ):
        minimal_power = lower_power
        for factor in factors:
            minimal_power = lcm(
                minimal_power,
                int(QQ.denom(QQ.convert(factor.logarithmic_multiplicity))),
            )

        scaled_lower_result = solve_radical_logarithmic_derivative(
            minimal_power * lower_component,
            derivation.lower(),
        )
        if (
            not scaled_lower_result.has_solution
            or scaled_lower_result.minimal_power != 1
            or scaled_lower_result.argument is None
        ):
            raise ValueError(
                "clearing lower logarithmic residues did not produce an argument"
            )
        relation_argument = _coerce(
            field,
            scaled_lower_result.argument,
        )
        for factor in factors:
            exponent = QQ.convert(minimal_power) * QQ.convert(
                factor.logarithmic_multiplicity
            )
            if QQ.denom(exponent) != 1:
                raise ValueError(
                    "clearing primitive logarithmic residues left a "
                    "nonintegral valuation"
                )
            relation_argument *= _coerce(field, factor.factor.as_expr()) ** int(
                QQ.numer(exponent)
            )

    signature = PrimitiveLogarithmicDerivativeSignature(
        derivation,
        element,
        polynomial_part,
        lower_logarithmic_result,
        factors,
        minimal_power,
        relation_argument,
    )
    if not signature.reconstructs_decomposition:
        raise ValueError(
            "primitive logarithmic signature did not reconstruct its input"
        )
    if signature.has_solution and not signature.reconstructs_relation:
        raise ValueError("primitive logarithmic signature witness did not reconstruct")
    return signature


def solve_primitive_exact_derivative(
    integrand: object,
    derivation: Derivation,
) -> PrimitiveExactDerivativeResult:
    r"""Decide the exact-derivative problem in a primitive field ``K(t)``.

    Hermite reduction first removes every possible repeated-pole derivative.
    In a primitive extension with preserved constants, every nonconstant
    polynomial is normal.  A remaining proper square-free fraction therefore
    cannot be the derivative of a rational function in ``t``.  The polynomial
    part is reduced separately modulo derivatives by
    :func:`polynomial_reduce_primitive`.

    This solver intentionally does not perform residue or logarithmic
    reduction: for example, ``1/(x*t) = D(t)/t`` is elementary after adjoining
    ``log(t)``, but it is not the derivative of an element already in
    ``K(t)``.
    """
    if derivation.level < 0:
        raise TypeError("primitive exact solving requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive exact solving requires a monomial level")
    if extension.generator_derivative.degree() != 0:
        raise ValueError("primitive exact solving requires Dt in K")
    require_preserved_constant_field(derivation, "primitive exact solving")

    field = extension.field
    element = _coerce(field, integrand)
    hermite_exact_part, hermite_remainder = hermite_reduce_monomial(
        element,
        derivation,
    )
    polynomial_part, _, _ = _polynomial_and_proper_fraction(
        hermite_remainder,
        extension,
    )
    polynomial_reduction = polynomial_reduce_primitive(
        polynomial_part,
        derivation,
    )
    polynomial_exact_element = _coerce(
        field,
        polynomial_reduction.exact_part.as_expr(),
    )
    exact_part = hermite_exact_part + polynomial_exact_element
    obstruction = hermite_remainder - derivation(polynomial_exact_element)
    final_polynomial, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(obstruction, extension)
    )
    if not exact_polynomial_equal(
        final_polynomial,
        polynomial_reduction.remainder,
    ):
        raise ValueError("primitive exact solving produced an inconsistent obstruction")

    if not proper_numerator.is_zero:
        normal_denominator, special_denominator = split_normal_special(
            proper_denominator,
            derivation,
        )
        if special_denominator.degree() > 0:
            raise ValueError(
                "a primitive extension with preserved constants cannot "
                "have a nonconstant special denominator"
            )
        if normal_denominator.degree() == 0:
            raise ValueError(
                "primitive exact solving produced an invalid proper fraction"
            )

    return PrimitiveExactDerivativeResult(
        hermite_exact_part,
        polynomial_reduction,
        exact_part,
        proper_numerator,
        proper_denominator,
        obstruction,
    )
