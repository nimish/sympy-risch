r"""Reduction helpers for transcendental monomial extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.polytools import cancel, invert, resultant
from sympy.simplify.radsimp import fraction

from .lrt import _differential_lazard_rioboo_trager
from .rational import solve_rational_exact_derivative
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

    hermite_exact_part: Any
    residue_reduction: BronsteinResidueReduction
    polynomial_exact_part: Poly
    polynomial_remainder: Poly
    reduced_function_reduction: ReducedFunctionReduction | None
    exact_part: Any
    logarithmic_derivative: Any
    remainder: Any

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
        return (
            self.has_finite_residue_obstruction
            or self.has_infinity_obstruction
        )

    @property
    def needs_special_polynomial(self) -> bool:
        """Whether infinity reduction awaits a known nontrivial special."""
        return (
            self.reduced_function_reduction is not None
            and self.reduced_function_reduction.needs_special_polynomial
        )


class NonlinearNoSpecialStatus(Enum):
    """Conclusion of the nonlinear no-special decision procedure."""

    REDUCED_TO_LOWER_FIELD = "reduced_to_lower_field"
    FINITE_RESIDUE_OBSTRUCTION = "finite_residue_obstruction"
    INFINITY_RESIDUE_OBSTRUCTION = "infinity_residue_obstruction"
    NONCONSTANT_REDUCED_POLYNOMIAL = "nonconstant_reduced_polynomial"


@dataclass(frozen=True, slots=True)
class NonlinearNoSpecialReduction:
    r"""Decision result for Bronstein's nonlinear no-special case.

    ``common_reduction`` reconstructs the input as an exact derivative, a
    logarithmic derivative, and a reduced remainder.  When ``status`` is
    ``REDUCED_TO_LOWER_FIELD``, ``lower_remainder`` is that remainder coerced
    to :math:`K`; otherwise the status identifies a Liouville obstruction.
    """

    common_reduction: NonlinearMonomialReduction
    status: NonlinearNoSpecialStatus
    lower_remainder: Any | None = None

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether integration has recursively reduced from ``K(t)`` to ``K``."""
        return self.status is NonlinearNoSpecialStatus.REDUCED_TO_LOWER_FIELD

    @property
    def is_non_elementary(self) -> bool:
        """Whether the result contains a proof of non-elementarity."""
        return not self.reduced_to_lower_field


@dataclass(frozen=True, slots=True)
class PrimitivePolynomialStep:
    r"""One coefficient reduction modulo ``D(K) + C*Dt``.

    For the current coefficient ``a``, the fields record

    ``a = D(exact_coefficient) + eta_multiple*Dt + obstruction``.

    All field-valued entries are native elements of the parent SymPy domain.
    """

    degree: int
    coefficient: Any
    exact_coefficient: Any
    eta_multiple: Any
    obstruction: Any


@dataclass(frozen=True, slots=True)
class PrimitivePolynomialReduction:
    r"""Certified polynomial reduction in a primitive extension."""

    exact_part: Poly
    remainder: Poly
    steps: tuple[PrimitivePolynomialStep, ...]

    @property
    def has_obstruction(self) -> bool:
        """Whether the polynomial is not an exact derivative in the extension."""
        return not self.remainder.is_zero


@dataclass(frozen=True, slots=True)
class PrimitiveMonomialReduction:
    r"""End-to-end reduction at a primitive monomial level.

    The flattened fields satisfy

    ``f = D(exact_part) + logarithmic_derivative + remainder``.

    A nonconstant normal residue or a positive-degree polynomial remainder
    proves non-elementarity.  Otherwise ``lower_remainder`` is the remaining
    element of the parent field for recursive integration.
    """

    hermite_exact_part: Any
    residue_reduction: BronsteinResidueReduction
    polynomial_reduction: PrimitivePolynomialReduction
    exact_part: Any
    logarithmic_derivative: Any
    remainder: Any
    lower_remainder: Any | None

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
        return (
            self.has_finite_residue_obstruction
            or self.has_polynomial_obstruction
        )

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only recursive integration in the parent field remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


@dataclass(frozen=True, slots=True)
class _RationalObstructionPivot:
    """One nonzero rational residue coordinate used to quotient by ``C*eta``."""

    factor: Poly
    degree: int
    value: Any


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


def _standard_rational_base_variable(derivation: Derivation) -> Symbol:
    """Return ``x`` when ``derivation`` is the standard ``d/dx`` on ``QQ(x)``."""
    if derivation.level != 0:
        raise NotImplementedError(
            "primitive polynomial reduction currently requires the lower "
            "field QQ(x)"
        )
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or extension.parent_field != QQ
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=QQ)
    ):
        raise NotImplementedError(
            "primitive polynomial reduction currently requires QQ(x) "
            "with Dx = 1"
        )
    return extension.generator


def _qq_reduced_fraction(
    expression: Any,
    variable: Symbol,
) -> tuple[Poly, Poly]:
    """Return a fraction over ``QQ`` with a monic denominator."""
    numerator_expression, denominator_expression = cancel(
        expression,
    ).as_numer_denom()
    numerator = Poly(numerator_expression, variable, domain=QQ)
    denominator = Poly(denominator_expression, variable, domain=QQ)
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(coefficient)
    leading_inverse = QQ.one/denominator.LC()
    return numerator.mul_ground(leading_inverse), denominator.monic()


def _partial_numerator(
    obstruction: Any,
    factor: Poly,
    variable: Symbol,
) -> Poly:
    r"""Return the numerator of the ``1/factor`` partial-fraction component."""
    numerator, denominator = _qq_reduced_fraction(
        obstruction.as_expr(),
        variable,
    )
    if not denominator.rem(factor).is_zero:
        return factor.zero
    cofactor = denominator.exquo(factor)
    return (numerator*invert(cofactor, factor)).rem(factor)


def _rational_obstruction_pivot(
    eta_obstruction: Any,
    variable: Symbol,
) -> _RationalObstructionPivot:
    """Choose a deterministic nonzero coordinate of the class of ``eta``."""
    _, denominator = _qq_reduced_fraction(
        eta_obstruction.as_expr(),
        variable,
    )
    factors = denominator.factor_list()[1]
    if not factors:
        raise ValueError("a nonzero rational obstruction needs a pole")
    factor = factors[0][0]
    partial_numerator = _partial_numerator(
        eta_obstruction,
        factor,
        variable,
    )
    (degree,), value = partial_numerator.terms()[0]
    return _RationalObstructionPivot(
        factor,
        degree,
        QQ.convert(value),
    )


def _rational_coefficient_reduction(
    coefficient: Any,
    eta_exact_part: Any,
    eta_obstruction: Any,
    pivot: _RationalObstructionPivot,
    parent: Any,
    variable: Symbol,
) -> tuple[Any, Any, Any]:
    r"""Return ``b, c, r`` such that ``a = Db + c*eta + r``."""
    result = solve_rational_exact_derivative(
        coefficient.as_expr(),
        variable,
    )
    coefficient_exact_part = _coerce(parent, result.exact_part)
    coefficient_obstruction = _coerce(parent, result.obstruction)

    if coefficient_obstruction == parent.zero:
        eta_multiple = QQ.zero
    else:
        coordinate = _partial_numerator(
            coefficient_obstruction,
            pivot.factor,
            variable,
        ).nth(pivot.degree)
        eta_multiple = QQ.exquo(
            QQ.convert(coordinate),
            pivot.value,
        )

    exact_coefficient = (
        coefficient_exact_part
        - parent.convert(eta_multiple)*eta_exact_part
    )
    obstruction = (
        coefficient_obstruction
        - parent.convert(eta_multiple)*eta_obstruction
    )
    return exact_coefficient, eta_multiple, obstruction


def polynomial_reduce_primitive(
    polynomial: Poly,
    derivation: Derivation,
) -> PrimitivePolynomialReduction:
    r"""Reduce a polynomial over a primitive ``QQ(x)(t)`` extension.

    Let ``Dt = eta``.  At each degree this solves

    .. math::

        a = Db + c\eta + r,\qquad b\in\mathbb{Q}(x),\ c\in\mathbb{Q},

    using rational Hermite remainders as canonical representatives modulo
    derivatives.  It then subtracts

    .. math::

        D\left(bt^n + \frac{c}{n+1}t^{n+1}\right).

    The returned fields satisfy ``polynomial = D(exact_part) + remainder``.
    Every nonzero coefficient of ``remainder`` is reduced modulo
    ``D(QQ(x)) + QQ*eta``.

    This first recursive base case deliberately requires the standard lower
    field ``QQ(x)`` with ``Dx = 1``.  Later exact-derivative dispatch will
    replace that restriction for deeper towers.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if derivation.level < 0:
        raise TypeError("primitive polynomial reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive polynomial reduction requires a monomial level")

    derivation.polynomial(polynomial)
    if extension.generator_derivative.degree() != 0:
        raise ValueError("primitive polynomial reduction requires Dt in K")
    extension.require_preserved_constants("primitive polynomial reduction")

    lower_derivation = derivation.lower()
    variable = _standard_rational_base_variable(lower_derivation)
    parent = extension.parent_field
    eta = parent.convert(extension.generator_derivative.nth(0))
    eta_result = solve_rational_exact_derivative(eta.as_expr(), variable)
    if eta_result.has_solution:
        raise ValueError(
            "constant-field status is preserved, but Dt is exact in the "
            "lower field"
        )
    eta_exact_part = _coerce(parent, eta_result.exact_part)
    eta_obstruction = _coerce(parent, eta_result.obstruction)
    pivot = _rational_obstruction_pivot(
        eta_obstruction,
        variable,
    )

    exact_part = polynomial.zero
    remainder = polynomial.zero
    working = polynomial
    steps: list[PrimitivePolynomialStep] = []

    while not working.is_zero:
        degree = working.degree()
        coefficient = parent.convert(working.nth(degree))
        exact_coefficient, eta_multiple, obstruction = (
            _rational_coefficient_reduction(
                coefficient,
                eta_exact_part,
                eta_obstruction,
                pivot,
                parent,
                variable,
            )
        )
        correction = Poly.from_dict(
            {
                (degree,): exact_coefficient,
                (degree + 1,): parent.convert(
                    eta_multiple/QQ.convert(degree + 1)
                ),
            },
            extension.generator,
            domain=parent,
        )
        obstruction_term = Poly.from_dict(
            {(degree,): obstruction},
            extension.generator,
            domain=parent,
        )
        new_working = (
            working
            - derivation.polynomial(correction)
            - obstruction_term
        )
        if not new_working.is_zero and new_working.degree() >= degree:
            raise ValueError("primitive polynomial reduction did not lower the degree")

        exact_part += correction
        remainder += obstruction_term
        steps.append(
            PrimitivePolynomialStep(
                degree,
                coefficient,
                exact_coefficient,
                eta_multiple,
                obstruction,
            )
        )
        working = new_working

    return PrimitivePolynomialReduction(
        exact_part,
        remainder,
        tuple(steps),
    )


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


def reduce_nonlinear_monomial(
    integrand: Any,
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

    The function requires the top extension to record the no-new-constants
    hypothesis.  It does not perform the final primitive, hyperexponential,
    hypertangent, or nonlinear-no-special family reduction.
    """
    if derivation.level < 0:
        raise TypeError("nonlinear reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("nonlinear reduction requires a monomial level")
    extension.require_preserved_constants("nonlinear monomial reduction")
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
    after_polynomial_reduction = (
        residue_reduction.remainder
        - derivation(polynomial_exact_element)
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
        (
            residue_reduction.logarithmic_derivative
            + infinity_logarithmic_derivative
        ),
        remainder,
    )


def reduce_primitive_monomial(
    integrand: Any,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> PrimitiveMonomialReduction:
    r"""Apply the complete reduction for a primitive monomial over ``QQ(x)``.

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
    extension.require_preserved_constants("primitive reduction")

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
    remainder = (
        residue_reduction.remainder
        - derivation(polynomial_exact_element)
    )
    final_polynomial, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(remainder, extension)
    )
    if final_polynomial != polynomial_reduction.remainder:
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
                raise ValueError(
                    "residue reduction left an unexpected normal fraction"
                )

        if final_polynomial.is_zero:
            lower_remainder = extension.parent_field.zero
        elif final_polynomial.degree() == 0:
            lower_remainder = extension.parent_field.convert(
                final_polynomial.nth(0)
            )

    return PrimitiveMonomialReduction(
        hermite_exact_part,
        residue_reduction,
        polynomial_reduction,
        exact_part,
        residue_reduction.logarithmic_derivative,
        remainder,
        lower_remainder,
    )


def reduce_nonlinear_no_special(
    integrand: Any,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> NonlinearNoSpecialReduction:
    r"""Decide the nonlinear case with no nonconstant special polynomials.

    This is the corollary following Theorem 5.11 in Bronstein, *Symbolic
    Integration I*.  After the common Hermite, residue, polynomial, and
    residue-at-infinity reductions, an elementary remainder must belong to
    the lower field :math:`K` when
    :math:`S^{\mathrm{irr}}=\varnothing`.  A remaining nonconstant polynomial
    in ``t`` therefore proves non-elementarity.

    The absence of special polynomials and preservation of constants are
    structure-theorem hypotheses, not facts inferable from one integrand.
    They must be recorded on the immutable top extension.
    """
    if derivation.level < 0:
        raise TypeError("nonlinear no-special reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("nonlinear no-special reduction requires a monomial level")
    extension.require_no_special_polynomials("nonlinear no-special reduction")

    # This inexpensive check catches inconsistent manually supplied metadata.
    # It is not intended to replace a structure-theorem proof: special
    # polynomials need not divide Dt.
    _, obvious_special = split_normal_special(
        extension.generator_derivative,
        derivation,
    )
    if obvious_special.degree() > 0:
        raise ValueError(
            "special-polynomial status is absent, but Dt has a "
            "nonconstant special factor"
        )

    common = reduce_nonlinear_monomial(
        integrand,
        derivation,
        residue=residue,
    )
    if common.has_finite_residue_obstruction:
        return NonlinearNoSpecialReduction(
            common,
            NonlinearNoSpecialStatus.FINITE_RESIDUE_OBSTRUCTION,
        )
    if common.has_infinity_obstruction:
        return NonlinearNoSpecialReduction(
            common,
            NonlinearNoSpecialStatus.INFINITY_RESIDUE_OBSTRUCTION,
        )

    polynomial, proper_numerator, proper_denominator = (
        _polynomial_and_proper_fraction(common.remainder, extension)
    )
    if not proper_numerator.is_zero:
        _, special_denominator = split_normal_special(
            proper_denominator,
            derivation,
        )
        if special_denominator.degree() > 0:
            raise ValueError(
                "special-polynomial status is absent, but the reduced "
                "remainder has a nonconstant special denominator"
            )
        raise ValueError(
            "common reduction left an unexpected normal proper fraction"
        )

    if polynomial.is_zero or polynomial.degree() == 0:
        lower_remainder = (
            extension.parent_field.zero
            if polynomial.is_zero
            else extension.parent_field.convert(polynomial.nth(0))
        )
        return NonlinearNoSpecialReduction(
            common,
            NonlinearNoSpecialStatus.REDUCED_TO_LOWER_FIELD,
            lower_remainder,
        )

    return NonlinearNoSpecialReduction(
        common,
        NonlinearNoSpecialStatus.NONCONSTANT_REDUCED_POLYNOMIAL,
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
