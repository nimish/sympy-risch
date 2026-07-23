r"""Reduction helpers for transcendental monomial extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polytools import Poly
from sympy.polys.polytools import resultant
from sympy.simplify.radsimp import fraction

from .lrt import _differential_lazard_rioboo_trager
from .tower import Derivation, MonomialExtension, _coerce, _domain_symbols


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
    logarithmic_derivative: Any
    remainder: Any

    @property
    def all_normal_residues_removed(self) -> bool:
        """Whether no nonconstant normal-residue obstruction remains."""
        return self.certificate.all_residues_constant


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
    infinity_coefficient: Any
    infinity_residue: Any
    infinity_residue_derivative: Any
    logarithmic_coefficient: Any | None
    logarithmic_argument: Poly | None
    logarithmic_derivative: Any
    remainder: Any

    @property
    def has_infinity_obstruction(self) -> bool:
        """Whether the candidate residue at infinity is nonconstant."""
        return self.infinity_residue_derivative != 0

    @property
    def needs_special_polynomial(self) -> bool:
        """Whether logarithmic removal is blocked only by an unknown special."""
        return (
            self.infinity_residue != 0
            and not self.has_infinity_obstruction
            and self.logarithmic_argument is None
        )


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
            domain.convert(correction_degree)*derivative_leading_coefficient
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


def _polynomial_and_proper_fraction(
    element: Any,
    extension: MonomialExtension,
) -> tuple[Poly, Poly, Poly]:
    """Return the polynomial part and a proper fraction over a monic denominator."""
    numerator, denominator = _reduced_polynomial_fraction(element, extension)
    leading_inverse = denominator.domain.one/denominator.domain.convert(
        denominator.LC(),
    )
    numerator = numerator.mul_ground(leading_inverse)
    denominator = denominator.monic()
    polynomial, proper_numerator = numerator.div(denominator)
    if proper_numerator.is_zero:
        return polynomial, denominator.zero, denominator.one
    return polynomial, proper_numerator, denominator


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


def _differentiate_logand(
    logand: Poly,
    derivation: Derivation,
    residue: Symbol,
) -> Any:
    r"""Differentiate an LRT logand while treating its residue as constant."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("logand differentiation requires a monomial level")

    expression = 0
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
        expression += coefficient_derivative.as_expr()*extension.generator**power
        if power:
            expression += (
                power
                * coefficient_polynomial.as_expr()
                * extension.generator**(power - 1)
                * extension.generator_derivative.as_expr()
            )
    return expression


def _logarithmic_part_derivative(
    logand: Poly,
    residue_factor: Poly,
    derivation: Derivation,
    residue: Symbol,
) -> Any:
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
        residue_polynomial
        * logand_derivative
        * logand_modulus.invert(modulus)
    ).rem(modulus)

    # The trace of multiplication by u in F[z]/(q) is the sum of u(alpha)
    # over the roots alpha of the square-free q.  Computing the diagonal in
    # the power basis avoids constructing any algebraic residue explicitly.
    trace = field.zero
    for power in range(modulus.degree()):
        basis_element = Poly(residue**power, residue, domain=field)
        product = (trace_element*basis_element).rem(modulus)
        trace += _coerce(field, product.nth(power))
    return trace


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
    top extension must explicitly record the no-new-constants hypothesis.
    """
    if derivation.level < 0:
        raise TypeError("the residue criterion requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("the residue criterion requires a monomial level")
    extension.require_preserved_constants("the residue criterion")

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


def residue_reduce_monomial(
    remainder: Any,
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
    residue = certificate.residue_polynomial.gens[0]
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


def reduce_monomial_reduced_function(
    remainder: Any,
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
    ``needs_special_polynomial`` and leaves the input unchanged for the
    nonlinear-no-specials branch.
    """
    if derivation.level < 0:
        raise TypeError("reduced-function reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("reduced-function reduction requires a monomial level")
    extension.require_preserved_constants("reduced-function reduction")

    generator_derivative = extension.generator_derivative
    delta = generator_derivative.degree()
    if delta < 2:
        raise ValueError(
            "reduced-function reduction requires degree(Dt) >= 2"
        )

    field = extension.field
    parent = extension.parent_field
    element = _coerce(field, remainder)
    polynomial, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(element, extension)
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
        logarithmic_coefficient: Any | None = None,
        logarithmic_argument: Poly | None = None,
        logarithmic_derivative: Any = field.zero,
        reduced_remainder: Any = element,
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
    if (
        not reduced_polynomial.is_zero
        and reduced_polynomial.degree() >= delta - 1
    ):
        raise ValueError(
            "special polynomial did not remove the residue at infinity"
        )

    return result(
        logarithmic_coefficient=logarithmic_coefficient,
        logarithmic_argument=logarithmic_argument,
        logarithmic_derivative=logarithmic_derivative,
        reduced_remainder=reduced_remainder,
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
