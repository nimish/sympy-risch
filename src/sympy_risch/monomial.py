r"""Reduction helpers for transcendental monomial extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import lcm
from typing import Any

from sympy.core.numbers import I
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly
from sympy.polys.polytools import cancel, invert, resultant
from sympy.simplify.radsimp import fraction

from .dependency import (
    analyze_hyperexponential_extension,
    analyze_hypertangent_extension,
)
from .lrt import _differential_lazard_rioboo_trager
from .rational import (
    RationalLogDerivativeResult,
    solve_rational_exact_derivative,
    solve_rational_logarithmic_derivative,
)
from .rde import (
    CoupledRationalRDEResult,
    RischDifferentialEquationResult,
    solve_coupled_rational_rde,
    solve_risch_differential_equation,
)
from .tower import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
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
class PrimitiveExactDerivativeResult:
    r"""Certified exact-derivative result in a primitive field ``K(t)``.

    The fields satisfy

    ``f = D(exact_part) + obstruction``.

    Unlike elementary integration, logarithmic derivatives remain in
    ``obstruction`` because their antiderivatives do not belong to ``K(t)``.
    """

    hermite_exact_part: Any
    polynomial_reduction: PrimitivePolynomialReduction
    exact_part: Any
    proper_numerator: Poly
    proper_denominator: Poly
    obstruction: Any

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
            self.has_proper_fraction_obstruction
            or self.has_polynomial_obstruction
        )

    @property
    def antiderivative(self) -> Any | None:
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
    logarithmic_multiplicity: Any | None

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether a nonzero principal part has order greater than one."""
        return any(
            not numerator.is_zero
            for numerator in self.principal_numerators[1:]
        )

    @property
    def has_logarithmic_residue(self) -> bool:
        """Whether the simple numerator is a rational multiple of ``Dp``."""
        return self.logarithmic_multiplicity is not None


@dataclass(frozen=True, slots=True)
class RadicalLogarithmicDerivativeResult:
    r"""Recursive decision for ``n*f = D(v)/v`` in a primitive tower.

    ``minimal_power`` is the least positive integer ``n`` for which such a
    nonzero ``v`` exists.  The certificate is the complete rational result at
    the base level or the primitive signature at the current level.
    """

    derivation: Derivation
    integrand: Any
    minimal_power: int | None
    argument: Any | None
    certificate: (
        RationalLogDerivativeResult
        | PrimitiveLogarithmicDerivativeSignature
    )

    @property
    def has_solution(self) -> bool:
        """Whether some positive integer multiple is logarithmic."""
        return self.minimal_power is not None

    @property
    def reconstructs(self) -> bool:
        """Whether the returned radical-logarithmic identity is exact."""
        if not self.has_solution or self.argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.argument)
        return (
            self.minimal_power*_coerce(field, self.integrand)
            == self.derivation(argument)/argument
        )


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
    integrand: Any
    polynomial_part: Poly
    lower_logarithmic_result: RadicalLogarithmicDerivativeResult
    factors: tuple[PrimitiveLogarithmicDerivativeFactor, ...]
    minimal_power: int | None
    relation_argument: Any | None

    @property
    def has_positive_degree_polynomial_obstruction(self) -> bool:
        """Whether the polynomial part depends nontrivially on ``t``."""
        return (
            not self.polynomial_part.is_zero
            and self.polynomial_part.degree() > 0
        )

    @property
    def has_repeated_pole_obstruction(self) -> bool:
        """Whether some finite place has a pole of order greater than one."""
        return any(
            factor.has_repeated_pole_obstruction
            for factor in self.factors
        )

    @property
    def has_nonlogarithmic_residue_obstruction(self) -> bool:
        """Whether some simple principal part is not rational times ``Dp``."""
        return any(
            not factor.has_logarithmic_residue
            for factor in self.factors
        )

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
                    numerator.as_expr()/factor.factor.as_expr()**pole_order
                )
        return (
            _coerce(extension.field, expression)
            == _coerce(extension.field, self.integrand)
        )

    @property
    def reconstructs_relation(self) -> bool:
        """Whether the successful radical-logarithmic identity is exact."""
        if not self.has_solution or self.relation_argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.relation_argument)
        return (
            self.minimal_power*_coerce(field, self.integrand)
            == self.derivation(argument)/argument
        )


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentStep:
    r"""One nonzero Laurent mode reduced by a lower-field Risch equation.

    For ``Dt/t = eta`` and the input coefficient ``b`` at power ``n``, the
    nested result decides

    ``Dy + n*eta*y = b``.

    A solution contributes ``y*t**n`` to the exact part.  Otherwise the
    original ``b*t**n`` remains as a certified special-mode obstruction.
    """

    power: int
    coefficient: Any
    exact_coefficient: Any | None
    rde_result: RischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether this Laurent mode is an exact derivative."""
        return self.rde_result.has_solution


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentReduction:
    r"""Mode-by-mode reduction of a Laurent polynomial in ``t``.

    The result satisfies

    ``integrand = D(exact_part) + remainder``.

    The power-zero coefficient belongs to the lower field and is returned as
    ``lower_remainder``.  Every nonzero mode is represented by one
    :class:`HyperexponentialLaurentStep`.
    """

    exact_part: Any
    remainder: Any
    lower_remainder: Any
    steps: tuple[HyperexponentialLaurentStep, ...]

    @property
    def obstructed_steps(self) -> tuple[HyperexponentialLaurentStep, ...]:
        """Return all nonzero modes whose rational RDE has no solution."""
        return tuple(step for step in self.steps if not step.has_solution)

    @property
    def has_obstruction(self) -> bool:
        """Whether a nonzero Laurent mode cannot be integrated in the field."""
        return bool(self.obstructed_steps)

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only the power-zero lower-field coefficient remains."""
        return not self.has_obstruction


@dataclass(frozen=True, slots=True)
class HyperexponentialMonomialReduction:
    r"""Complete elementary reduction at one hyperexponential level.

    The flattened fields satisfy

    ``f = D(exact_part) + logarithmic_derivative + remainder``.

    Normal finite poles are handled by generalized residue reduction.  Once
    those residues are constant, the remaining special Laurent modes are
    handled by :class:`HyperexponentialLaurentReduction`.
    """

    hermite_exact_part: Any
    residue_reduction: BronsteinResidueReduction
    laurent_reduction: HyperexponentialLaurentReduction | None
    exact_part: Any
    logarithmic_derivative: Any
    remainder: Any
    lower_remainder: Any | None

    @property
    def has_finite_residue_obstruction(self) -> bool:
        """Whether a nonconstant normal residue proves non-elementarity."""
        return not self.residue_reduction.all_normal_residues_removed

    @property
    def has_laurent_obstruction(self) -> bool:
        """Whether a special Laurent mode has an unsolvable rational RDE."""
        return (
            self.laurent_reduction is not None
            and self.laurent_reduction.has_obstruction
        )

    @property
    def is_non_elementary(self) -> bool:
        """Whether this level supplies a Liouville obstruction."""
        return (
            self.has_finite_residue_obstruction
            or self.has_laurent_obstruction
        )

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only elementary integration in ``QQ(x)`` remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


@dataclass(frozen=True, slots=True)
class HypertangentCayleyStep:
    r"""One positive Cayley mode and its real coupled RDE certificate."""

    power: int
    real_coefficient: Any
    imaginary_coefficient: Any
    exact_real_coefficient: Any | None
    exact_imaginary_coefficient: Any | None
    rde_result: CoupledRationalRDEResult

    @property
    def has_solution(self) -> bool:
        """Whether this conjugate pair is an exact derivative."""
        return self.rde_result.has_solution


@dataclass(frozen=True, slots=True)
class HypertangentSpecialReduction:
    r"""Real paired-mode reduction at the special place ``1 + t**2``."""

    exact_part: Any
    remainder: Any
    lower_remainder: Any
    steps: tuple[HypertangentCayleyStep, ...]

    @property
    def obstructed_steps(self) -> tuple[HypertangentCayleyStep, ...]:
        """Return all Cayley pairs with unsolvable coupled equations."""
        return tuple(step for step in self.steps if not step.has_solution)

    @property
    def has_obstruction(self) -> bool:
        """Whether some nonzero Cayley pair remains reduced."""
        return bool(self.obstructed_steps)

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only the Cayley power-zero coefficient remains."""
        return not self.has_obstruction


@dataclass(frozen=True, slots=True)
class HypertangentMonomialReduction:
    r"""Complete real reduction at one hypertangent level over ``QQ(x)``."""

    common_reduction: NonlinearMonomialReduction
    special_reduction: HypertangentSpecialReduction | None
    exact_part: Any
    logarithmic_derivative: Any
    remainder: Any
    lower_remainder: Any | None

    @property
    def has_common_obstruction(self) -> bool:
        """Whether a normal residue or infinity residue is nonconstant."""
        return self.common_reduction.has_obstruction

    @property
    def has_special_obstruction(self) -> bool:
        """Whether a Cayley pair has no rational coupled solution."""
        return (
            self.special_reduction is not None
            and self.special_reduction.has_obstruction
        )

    @property
    def is_non_elementary(self) -> bool:
        """Whether this level supplies a Liouville obstruction."""
        return self.has_common_obstruction or self.has_special_obstruction

    @property
    def reduced_to_lower_field(self) -> bool:
        """Whether only elementary rational integration remains."""
        return not self.is_non_elementary and self.lower_remainder is not None


@dataclass(frozen=True, slots=True)
class _RationalObstructionPivot:
    """One nonzero rational residue coordinate used to quotient by ``C*eta``."""

    factor: Poly
    degree: int
    value: Any


@dataclass(frozen=True, slots=True)
class _PrimitiveObstructionPivot:
    """An iterated Laurent coordinate on a primitive exact remainder."""

    exponents: tuple[int, ...]
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
            "this reduction currently requires the lower field QQ(x)"
        )
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or extension.parent_field != QQ
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=QQ)
    ):
        raise NotImplementedError(
            "this reduction currently requires QQ(x) with Dx = 1"
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


def _laurent_coefficient_at_infinity(
    element: Any,
    extension: MonomialExtension,
    exponent: int,
) -> Any:
    r"""Return the coefficient of ``t**exponent`` at the place at infinity.

    For ``f = N(t)/Q(t)``, substitute ``t = 1/z`` and divide the reversed
    numerator series by the reversed denominator series.  The denominator's
    constant term is its nonzero leading coefficient, so the recurrence is
    exact over the parent field and defines this coefficient for every
    rational function in ``K(t)``.
    """
    parent = extension.parent_field
    numerator, denominator = _reduced_polynomial_fraction(
        _coerce(extension.field, element),
        extension,
    )
    if numerator.is_zero:
        return parent.zero

    numerator_degree = numerator.degree()
    denominator_degree = denominator.degree()
    leading_exponent = numerator_degree - denominator_degree
    if exponent > leading_exponent:
        return parent.zero

    target_order = leading_exponent - exponent
    quotient_coefficients: list[Any] = []
    denominator_leading = parent.convert(denominator.LC())
    for order in range(target_order + 1):
        numerator_index = numerator_degree - order
        right_hand_side = (
            parent.convert(numerator.nth(numerator_index))
            if numerator_index >= 0
            else parent.zero
        )
        for denominator_order in range(
            1,
            min(order, denominator_degree) + 1,
        ):
            right_hand_side -= (
                parent.convert(
                    denominator.nth(
                        denominator_degree - denominator_order
                    )
                )
                * quotient_coefficients[order - denominator_order]
            )
        quotient_coefficients.append(
            parent.exquo(right_hand_side, denominator_leading)
        )
    return quotient_coefficients[target_order]


def _primitive_obstruction_pivot(
    eta_obstruction: Any,
    derivation: Derivation,
) -> _PrimitiveObstructionPivot:
    r"""Choose a deterministic nonzero constant coordinate of ``[eta]``.

    At each lower monomial level, choose the leading exponent in the Laurent
    expansion at infinity and descend through its nonzero coefficient.  The
    resulting iterated coefficient is in the constant field.  Holding those
    exponents fixed defines a constant-linear functional on every element of
    the lower rational-function tower.

    The complement to ``C*[eta]`` is not unique: any constant-linear
    functional nonzero on ``[eta]`` defines a valid projection.  Iterated
    Laurent coefficients are used here because, unlike point evaluation,
    they are defined on every rational function and introduce neither
    factorization nor choices of pole-avoiding values.
    """
    exponents: list[int] = []
    coefficient = eta_obstruction
    current = derivation
    while current.level >= 0:
        extension = current.tower.levels[current.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "recursive primitive coordinates require monomial lower levels"
            )
        numerator, denominator = _reduced_polynomial_fraction(
            _coerce(extension.field, coefficient),
            extension,
        )
        if numerator.is_zero:
            raise ValueError("a primitive obstruction pivot must be nonzero")
        exponent = numerator.degree() - denominator.degree()
        exponents.append(exponent)
        coefficient = _laurent_coefficient_at_infinity(
            coefficient,
            extension,
            exponent,
        )
        current = current.lower()

    value = _coerce(derivation.tower.constants, coefficient)
    if value == derivation.tower.constants.zero:
        raise ValueError("primitive obstruction pivot unexpectedly vanished")
    return _PrimitiveObstructionPivot(tuple(exponents), value)


def _primitive_obstruction_coordinate(
    obstruction: Any,
    derivation: Derivation,
    pivot: _PrimitiveObstructionPivot,
) -> Any:
    """Apply a fixed iterated Laurent coordinate to an obstruction."""
    coefficient = obstruction
    current = derivation
    for exponent in pivot.exponents:
        if current.level < 0:
            raise ValueError("primitive obstruction pivot exceeds the tower")
        extension = current.tower.levels[current.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "recursive primitive coordinates require monomial lower levels"
            )
        coefficient = _laurent_coefficient_at_infinity(
            coefficient,
            extension,
            exponent,
        )
        current = current.lower()
    if current.level != -1:
        raise ValueError("primitive obstruction pivot does not reach constants")
    return _coerce(derivation.tower.constants, coefficient)


def _recursive_coefficient_reduction(
    coefficient: Any,
    eta: Any,
    eta_exact_part: Any,
    eta_obstruction: Any,
    pivot: _PrimitiveObstructionPivot,
    parent: Any,
    lower_derivation: Derivation,
) -> tuple[Any, Any, Any]:
    r"""Return ``b, c, r`` with ``a = Db + c*eta + r`` over a lower tower.

    The recursive exact-derivative solver supplies representatives of the
    classes of ``a`` and ``eta`` modulo ``D(K)``.  As in the optimized
    rational base case, ``pivot`` is a fixed nonzero constant-valued
    coordinate of the class of ``eta``.  Subtracting the matching multiple
    makes the returned obstruction's pivot coordinate zero and therefore
    gives an idempotent representative modulo ``D(K) + C*eta``.
    """
    # Local import avoids an import cycle: exact-derivative dispatch uses the
    # primitive solver defined in this module, while recursion strictly lowers
    # the tower level at runtime.
    from .exact import solve_exact_derivative

    result = solve_exact_derivative(coefficient, lower_derivation)
    coefficient_exact_part = _coerce(parent, result.exact_part)
    coefficient_obstruction = _coerce(parent, result.obstruction)

    constants = lower_derivation.tower.constants
    coordinate = _primitive_obstruction_coordinate(
        coefficient_obstruction,
        lower_derivation,
        pivot,
    )
    eta_multiple = constants.exquo(coordinate, pivot.value)
    exact_coefficient = (
        coefficient_exact_part
        - parent.convert(eta_multiple)*eta_exact_part
    )
    obstruction = (
        coefficient_obstruction
        - parent.convert(eta_multiple)*eta_obstruction
    )
    if (
        _primitive_obstruction_coordinate(
            obstruction,
            lower_derivation,
            pivot,
        )
        != constants.zero
    ):
        raise ValueError(
            "primitive obstruction projection did not remove its pivot"
        )

    if (
        lower_derivation(exact_coefficient)
        + parent.convert(eta_multiple)*eta
        + obstruction
        != coefficient
    ):
        raise ValueError(
            "recursive primitive coefficient reduction failed to reconstruct"
        )
    return exact_coefficient, eta_multiple, obstruction


def polynomial_reduce_primitive(
    polynomial: Poly,
    derivation: Derivation,
) -> PrimitivePolynomialReduction:
    r"""Reduce a polynomial over a primitive extension ``K(t)``.

    Let ``Dt = eta``.  At each degree this solves

    .. math::

        a = Db + c\eta + r,\qquad b\in K,\ c\in C,

    using exact-derivative remainders as representatives modulo ``D(K)``.  It
    then subtracts

    .. math::

        D\left(bt^n + \frac{c}{n+1}t^{n+1}\right).

    The returned fields satisfy ``polynomial = D(exact_part) + remainder``.
    Every nonzero coefficient of ``remainder`` is reduced modulo
    ``D(K) + C*eta``.

    Over the standard rational base, Bronstein's residue-pivot normalization
    supplies a constant coordinate of the obstruction space.  Over a
    supported primitive lower tower, a fixed iterated Laurent coefficient at
    infinity supplies the analogous coordinate.  Both paths remove the
    ``eta`` direction even when additional obstruction coordinates remain.
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
    parent = extension.parent_field
    eta = parent.convert(extension.generator_derivative.nth(0))

    variable = None
    if lower_derivation.level == 0:
        variable = _standard_rational_base_variable(lower_derivation)
        eta_result = solve_rational_exact_derivative(
            eta.as_expr(),
            variable,
        )
    else:
        # See the local-import note in _recursive_coefficient_reduction.
        from .exact import solve_exact_derivative

        eta_result = solve_exact_derivative(eta, lower_derivation)

    if eta_result.has_solution:
        raise ValueError(
            "constant-field status is preserved, but Dt is exact in the "
            "lower field"
        )
    eta_exact_part = _coerce(parent, eta_result.exact_part)
    eta_obstruction = _coerce(parent, eta_result.obstruction)
    pivot = (
        _rational_obstruction_pivot(eta_obstruction, variable)
        if variable is not None
        else _primitive_obstruction_pivot(
            eta_obstruction,
            lower_derivation,
        )
    )

    exact_part = polynomial.zero
    remainder = polynomial.zero
    working = polynomial
    steps: list[PrimitivePolynomialStep] = []

    while not working.is_zero:
        degree = working.degree()
        coefficient = parent.convert(working.nth(degree))
        if variable is not None:
            if not isinstance(pivot, _RationalObstructionPivot):
                raise TypeError(
                    "rational primitive reduction needs a rational pivot"
                )
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
        else:
            if not isinstance(pivot, _PrimitiveObstructionPivot):
                raise TypeError(
                    "recursive primitive reduction needs a Laurent pivot"
                )
            exact_coefficient, eta_multiple, obstruction = (
                _recursive_coefficient_reduction(
                    coefficient,
                    eta,
                    eta_exact_part,
                    eta_obstruction,
                    pivot,
                    parent,
                    lower_derivation,
                )
            )
        correction = Poly.from_dict(
            {
                (degree,): exact_coefficient,
                (degree + 1,): parent.convert(
                    eta_multiple
                    / derivation.tower.constants.convert(degree + 1)
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


def _qq_constant_polynomial_multiple(
    numerator: Poly,
    direction: Poly,
    lower_derivation: Derivation,
) -> Any | None:
    r"""Return ``c in QQ`` when ``numerator = c*direction``.

    The quotient first has to be a differential constant in the lower field.
    Under the preserved-constants hypotheses of the recursive primitive
    tower, that constant then belongs to ``QQ``.
    """
    if numerator.is_zero:
        return QQ.zero
    if direction.is_zero or numerator.degree() != direction.degree():
        return None

    parent = numerator.domain
    candidate = parent.exquo(
        parent.convert(numerator.LC()),
        parent.convert(direction.LC()),
    )
    if lower_derivation(candidate) != lower_derivation.field.zero:
        return None
    try:
        constant = QQ.from_sympy(parent.to_sympy(candidate))
    except CoercionFailed:
        return None
    if numerator != direction.mul_ground(parent.convert(constant)):
        return None
    return constant


def _factor_principal_numerators(
    proper_numerator: Poly,
    denominator: Poly,
    factor: Poly,
    multiplicity: int,
) -> tuple[Poly, ...]:
    r"""Return ``A_j`` in the local expansion ``sum(A_j/p**j)``."""
    factor_power = factor**multiplicity
    cofactor = denominator.exquo(factor_power)
    local_numerator = (
        proper_numerator*cofactor.invert(factor_power)
    ).rem(factor_power)
    principal_numerators = [factor.zero]*multiplicity
    working = local_numerator
    for pole_order in range(multiplicity, 0, -1):
        coefficient = working.rem(factor)
        principal_numerators[pole_order - 1] = coefficient
        working = (working - coefficient).exquo(factor)
    if not working.is_zero:
        raise ValueError("principal-part expansion did not terminate")
    return tuple(principal_numerators)


def primitive_logarithmic_derivative_signature(
    integrand: Any,
    derivation: Derivation,
) -> PrimitiveLogarithmicDerivativeSignature:
    r"""Compute a radical-logarithmic signature at a primitive tower level.

    The top level must be primitive, with ``Dt`` in the lower field, and must
    carry explicit evidence that it introduces no constants.  The lower field
    is decided recursively down to the standard rational base ``QQ(x)``.
    Factor and principal-part data do not depend on a proposed parametric
    ratio, so the parametric structure fallback can compare signatures.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError(
            "primitive logarithmic signatures require a Derivation"
        )
    if derivation.level < 1:
        raise NotImplementedError(
            "primitive logarithmic signatures require a primitive level "
            "above the rational base"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "primitive logarithmic signatures require a monomial level"
        )
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError(
            "primitive logarithmic signatures require Dt in the lower field"
        )
    extension.require_preserved_constants(
        "primitive logarithmic signatures"
    )
    field = extension.field
    element = _coerce(field, integrand)
    polynomial_part, proper_numerator, denominator = (
        _polynomial_and_proper_fraction(element, extension)
    )
    lower_component = (
        extension.parent_field.zero
        if polynomial_part.is_zero
        else extension.parent_field.convert(polynomial_part.nth(0))
    )
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
    has_positive_degree = (
        not polynomial_part.is_zero and polynomial_part.degree() > 0
    )
    has_repeated_pole = any(
        factor.has_repeated_pole_obstruction
        for factor in factors
    )
    has_nonlogarithmic_residue = any(
        not factor.has_logarithmic_residue
        for factor in factors
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
                int(
                    QQ.denom(
                        QQ.convert(factor.logarithmic_multiplicity)
                    )
                ),
            )

        scaled_lower_result = solve_radical_logarithmic_derivative(
            minimal_power*lower_component,
            derivation.lower(),
        )
        if (
            not scaled_lower_result.has_solution
            or scaled_lower_result.minimal_power != 1
            or scaled_lower_result.argument is None
        ):
            raise ValueError(
                "clearing lower logarithmic residues did not produce "
                "an argument"
            )
        relation_argument = _coerce(
            field,
            scaled_lower_result.argument,
        )
        for factor in factors:
            exponent = (
                QQ.convert(minimal_power)
                * QQ.convert(factor.logarithmic_multiplicity)
            )
            if QQ.denom(exponent) != 1:
                raise ValueError(
                    "clearing primitive logarithmic residues left a "
                    "nonintegral valuation"
                )
            relation_argument *= (
                _coerce(field, factor.factor.as_expr())
                ** int(QQ.numer(exponent))
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
        raise ValueError(
            "primitive logarithmic signature witness did not reconstruct"
        )
    return signature


def solve_radical_logarithmic_derivative(
    integrand: Any,
    derivation: Derivation,
) -> RadicalLogarithmicDerivativeResult:
    r"""Decide ``n*f = D(v)/v`` recursively in a primitive tower.

    The base case is the complete rational logarithmic-derivative decision in
    ``QQ(x)``.  Every higher supported level applies the primitive structure
    theorem and descends on its constant polynomial coordinate.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError(
            "radical logarithmic derivatives require a Derivation"
        )

    field = derivation.field
    element = _coerce(field, integrand)
    if derivation.level == 0:
        variable = _standard_rational_base_variable(derivation)
        rational_result = solve_rational_logarithmic_derivative(
            field.to_sympy(element),
            variable,
        )
        minimal_power = rational_result.minimal_radical_power
        argument = None
        if minimal_power is not None:
            scaled_result = solve_rational_logarithmic_derivative(
                cancel(minimal_power*field.to_sympy(element)),
                variable,
            )
            if not scaled_result.has_solution:
                raise ValueError(
                    "clearing rational logarithmic residues did not "
                    "produce an argument"
                )
            argument = field.from_sympy(scaled_result.argument)
        result = RadicalLogarithmicDerivativeResult(
            derivation,
            element,
            minimal_power,
            argument,
            rational_result,
        )
    else:
        extension = derivation.tower.levels[derivation.level]
        if (
            not isinstance(extension, MonomialExtension)
            or extension.family is not MonomialFamily.PRIMITIVE
        ):
            raise NotImplementedError(
                "radical logarithmic derivatives currently support only "
                "the rational base and primitive tower levels"
            )
        signature = primitive_logarithmic_derivative_signature(
            element,
            derivation,
        )
        result = RadicalLogarithmicDerivativeResult(
            derivation,
            element,
            signature.minimal_power,
            signature.relation_argument,
            signature,
        )

    if result.has_solution and not result.reconstructs:
        raise ValueError(
            "radical logarithmic derivative witness did not reconstruct"
        )
    return result


def solve_primitive_exact_derivative(
    integrand: Any,
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
    extension.require_preserved_constants("primitive exact solving")

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
    if final_polynomial != polynomial_reduction.remainder:
        raise ValueError(
            "primitive exact solving produced an inconsistent obstruction"
        )

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


def _validate_hyperexponential_reduction(
    derivation: Derivation,
) -> tuple[MonomialExtension, Derivation, Any]:
    """Validate a supported transcendental hyperexponential extension."""
    if derivation.level < 1:
        raise TypeError(
            "hyperexponential reduction requires a nonconstant lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hyperexponential reduction requires a monomial level")
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError(
            "hyperexponential reduction requires Dt = eta*t"
        )
    extension.require_preserved_constants("hyperexponential reduction")

    lower_derivation = derivation.lower()
    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(1)
    )
    dependency = analyze_hyperexponential_extension(
        extension,
        lower_derivation,
    )
    if dependency.is_dependent:
        raise ValueError(
            "hyperexponential reduction requires a transcendental "
            "generator; the proposed generator is algebraically dependent"
        )

    return extension, lower_derivation, eta


def _reduce_hyperexponential_laurent_validated(
    element: Any,
    derivation: Derivation,
    extension: MonomialExtension,
    lower_derivation: Derivation,
    eta: Any,
) -> HyperexponentialLaurentReduction:
    """Reduce a validated Laurent element without repeating dependency tests."""
    field = extension.field
    parent = extension.parent_field
    generator = extension.generator
    numerator, denominator = _reduced_polynomial_fraction(
        element,
        extension,
    )
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise ValueError(
            "hyperexponential Laurent reduction requires a denominator "
            "that is a power of the generator"
        )
    (denominator_power,), denominator_coefficient = denominator_terms[0]
    denominator_coefficient = parent.convert(denominator_coefficient)
    coefficient_scale = parent.one/denominator_coefficient

    coefficients = {
        numerator_power - denominator_power: (
            parent.convert(coefficient)*coefficient_scale
        )
        for (numerator_power,), coefficient in numerator.terms()
    }
    exact_part = field.zero
    remainder = field.zero
    lower_remainder = parent.zero
    steps: list[HyperexponentialLaurentStep] = []

    for power in sorted(coefficients):
        coefficient = coefficients[power]
        coefficient_expression = parent.to_sympy(coefficient)
        term = _coerce(
            field,
            coefficient_expression*generator**power,
        )
        if power == 0:
            lower_remainder = coefficient
            remainder += term
            continue

        rde_result = solve_risch_differential_equation(
            parent.convert(power)*eta,
            coefficient,
            lower_derivation,
        )
        exact_coefficient = None
        if rde_result.has_solution:
            exact_coefficient = rde_result.solution
            exact_part += _coerce(
                field,
                parent.to_sympy(exact_coefficient)*generator**power,
            )
        else:
            remainder += term
        steps.append(
            HyperexponentialLaurentStep(
                power,
                coefficient,
                exact_coefficient,
                rde_result,
            )
        )

    if derivation(exact_part) + remainder != element:
        raise ValueError(
            "hyperexponential Laurent reduction reconstruction failed"
        )
    return HyperexponentialLaurentReduction(
        exact_part,
        remainder,
        lower_remainder,
        tuple(steps),
    )


def reduce_hyperexponential_laurent(
    integrand: Any,
    derivation: Derivation,
) -> HyperexponentialLaurentReduction:
    r"""Reduce a Laurent polynomial in a hyperexponential generator.

    For ``Dt/t = eta`` and

    ``f = sum(b_n*t**n)``,

    every nonzero mode is an exact derivative precisely when the lower-field
    Risch differential equation

    ``Dy_n + n*eta*y_n = b_n``

    has a solution in the supported lower field. The zero mode is returned for
    recursive lower-field integration.
    """
    extension, lower_derivation, eta = _validate_hyperexponential_reduction(
        derivation
    )
    element = _coerce(extension.field, integrand)
    return _reduce_hyperexponential_laurent_validated(
        element,
        derivation,
        extension,
        lower_derivation,
        eta,
    )


def reduce_hyperexponential_monomial(
    integrand: Any,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> HyperexponentialMonomialReduction:
    r"""Apply the complete reduction for a supported hyperexponential level.

    Repeated normal poles are Hermite-reduced, constant normal residues are
    extracted as logarithmic derivatives, and remaining special Laurent modes
    are reduced through lower-field Risch differential equations.
    """
    extension, lower_derivation, eta = _validate_hyperexponential_reduction(
        derivation
    )
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

    laurent_reduction = None
    exact_part = hermite_exact_part
    remainder = residue_reduction.remainder
    lower_remainder = None
    if residue_reduction.all_normal_residues_removed:
        laurent_reduction = _reduce_hyperexponential_laurent_validated(
            residue_reduction.remainder,
            derivation,
            extension,
            lower_derivation,
            eta,
        )
        exact_part += laurent_reduction.exact_part
        remainder = laurent_reduction.remainder
        if not laurent_reduction.has_obstruction:
            lower_remainder = laurent_reduction.lower_remainder

    if (
        derivation(exact_part)
        + residue_reduction.logarithmic_derivative
        + remainder
        != element
    ):
        raise ValueError(
            "hyperexponential monomial reduction reconstruction failed"
        )
    return HyperexponentialMonomialReduction(
        hermite_exact_part,
        residue_reduction,
        laurent_reduction,
        exact_part,
        residue_reduction.logarithmic_derivative,
        remainder,
        lower_remainder,
    )


def _validate_hypertangent_reduction(
    derivation: Derivation,
) -> tuple[MonomialExtension, Symbol, Any]:
    """Validate ``Dt = eta*(1+t**2)`` with ``eta`` in ``QQ(x)``."""
    if derivation.level < 1:
        raise TypeError(
            "hypertangent reduction requires a monomial over QQ(x)"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("hypertangent reduction requires a monomial level")
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent reduction requires Dt = eta*(1 + t**2)"
        )
    extension.require_preserved_constants("hypertangent reduction")

    lower_derivation = derivation.lower()
    variable = _standard_rational_base_variable(lower_derivation)
    dependency = analyze_hypertangent_extension(
        extension,
        lower_derivation,
    )
    if dependency.is_dependent:
        raise ValueError(
            "hypertangent reduction requires a transcendental generator; "
            "the Cayley generator is algebraically dependent"
        )
    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(2)
    )
    if eta == extension.parent_field.zero:
        raise ValueError("the hypertangent coefficient eta must be nonzero")
    return extension, variable, eta


def _gaussian_conjugate(coefficient: Any, gaussian_field: Any) -> Any:
    """Conjugate a member of ``QQ_I(x)`` while fixing the formal variable."""
    expression = gaussian_field.to_sympy(coefficient)
    return gaussian_field.from_sympy(expression.xreplace({I: -I}))


def _hypertangent_cayley_coefficients(
    element: Any,
    extension: MonomialExtension,
    variable: Symbol,
) -> tuple[Any, dict[int, Any]]:
    r"""Return Laurent coefficients after ``u=(1+i*t)/(1-i*t)``.

    The special irreducible ``1+t**2`` becomes a monomial in ``u``.  Thus an
    element whose only non-lower-field poles are special becomes a Laurent
    polynomial over ``QQ_I(x)``.  This is the Cayley-coordinate form of the
    hypertangent reduction in Bronstein, *Symbolic Integration I*, Chapter 5.
    """
    generator = extension.generator
    cayley_generator = Dummy("u")
    gaussian_field = QQ_I.frac_field(variable)
    inverse_cayley = I*(1 - cayley_generator)/(1 + cayley_generator)
    transformed = cancel(
        element.as_expr().subs(generator, inverse_cayley)
    )
    numerator_expression, denominator_expression = (
        transformed.as_numer_denom()
    )
    numerator = Poly(
        numerator_expression,
        cayley_generator,
        domain=gaussian_field,
    )
    denominator = Poly(
        denominator_expression,
        cayley_generator,
        domain=gaussian_field,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(coefficient)
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise ValueError(
            "hypertangent special reduction requires a Laurent polynomial "
            "in the Cayley generator"
        )

    (denominator_power,), denominator_coefficient = denominator_terms[0]
    coefficient_scale = (
        gaussian_field.one
        / gaussian_field.convert(denominator_coefficient)
    )
    coefficients = {
        numerator_power - denominator_power: (
            gaussian_field.convert(term_coefficient)*coefficient_scale
        )
        for (numerator_power,), term_coefficient in numerator.terms()
    }
    return gaussian_field, coefficients


def _reduce_hypertangent_special_validated(
    element: Any,
    derivation: Derivation,
    extension: MonomialExtension,
    variable: Symbol,
    eta: Any,
) -> HypertangentSpecialReduction:
    """Reduce validated special modes without repeating tower checks."""
    field = extension.field
    parent = extension.parent_field
    generator = extension.generator
    gaussian_field, coefficients = _hypertangent_cayley_coefficients(
        element,
        extension,
        variable,
    )
    zero = gaussian_field.zero

    # A real rational function of t has b[-n] = conjugate(b[n]) in Cayley
    # coordinates because real conjugation sends u to u**-1.
    powers = set(coefficients)
    for power in (candidate for candidate in powers if candidate != 0):
        if (
            coefficients.get(-power, zero)
            - _gaussian_conjugate(
                coefficients[power],
                gaussian_field,
            )
            != zero
        ):
            raise ValueError(
                "Cayley coefficients do not form real conjugate pairs"
            )

    lower_coefficient = coefficients.get(0, zero)
    if (
        lower_coefficient
        - _gaussian_conjugate(lower_coefficient, gaussian_field)
        != zero
    ):
        raise ValueError("the Cayley power-zero coefficient is not real")
    lower_remainder = _coerce(
        parent,
        gaussian_field.to_sympy(lower_coefficient),
    )

    cayley_expression = (
        (1 + I*generator)/(1 - I*generator)
    )
    exact_part = field.zero
    steps: list[HypertangentCayleyStep] = []
    for power in sorted(candidate for candidate in powers if candidate > 0):
        coefficient = coefficients[power]
        conjugate_coefficient = _gaussian_conjugate(
            coefficient,
            gaussian_field,
        )
        real_coefficient = _coerce(
            parent,
            cancel(
                (
                    gaussian_field.to_sympy(coefficient)
                    + gaussian_field.to_sympy(conjugate_coefficient)
                )
                / 2
            ),
        )
        imaginary_coefficient = _coerce(
            parent,
            cancel(
                (
                    gaussian_field.to_sympy(coefficient)
                    - gaussian_field.to_sympy(conjugate_coefficient)
                )
                / (2*I)
            ),
        )
        rde_result = solve_coupled_rational_rde(
            2*power*parent.to_sympy(eta),
            parent.to_sympy(real_coefficient),
            parent.to_sympy(imaginary_coefficient),
            variable,
        )
        exact_real_coefficient = None
        exact_imaginary_coefficient = None
        if rde_result.has_solution:
            exact_real_coefficient = _coerce(
                parent,
                rde_result.real_solution,
            )
            exact_imaginary_coefficient = _coerce(
                parent,
                rde_result.imaginary_solution,
            )
            real_expression = parent.to_sympy(exact_real_coefficient)
            imaginary_expression = parent.to_sympy(
                exact_imaginary_coefficient
            )
            paired_exact_expression = cancel(
                (real_expression + I*imaginary_expression)
                * cayley_expression**power
                + (real_expression - I*imaginary_expression)
                * cayley_expression**(-power)
            )
            exact_part += _coerce(field, paired_exact_expression)

        steps.append(
            HypertangentCayleyStep(
                power,
                real_coefficient,
                imaginary_coefficient,
                exact_real_coefficient,
                exact_imaginary_coefficient,
                rde_result,
            )
        )

    remainder = element - derivation(exact_part)
    if derivation(exact_part) + remainder != element:
        raise ValueError(
            "hypertangent special reduction reconstruction failed"
        )
    if not any(not step.has_solution for step in steps):
        if remainder != _coerce(field, lower_remainder):
            raise ValueError(
                "solved hypertangent modes did not reduce to the lower field"
            )
    return HypertangentSpecialReduction(
        exact_part,
        remainder,
        lower_remainder,
        tuple(steps),
    )


def reduce_hypertangent_special(
    integrand: Any,
    derivation: Derivation,
) -> HypertangentSpecialReduction:
    r"""Reduce the special part of a rational-coefficient tangent extension.

    For ``Dt = eta*(1+t**2)``, introduce

    ``u = (1 + i*t)/(1 - i*t)``, so ``Du/u = 2*i*eta``.

    Real inputs have paired Laurent coefficients.  A positive mode
    ``(a + i*b)*u**n`` and its conjugate are exact precisely when the real
    coupled rational system

    ``Dp - 2*n*eta*q = a`` and ``Dq + 2*n*eta*p = b``

    has a solution.  Pairing before reconstruction keeps the public result in
    the real field ``QQ(x,t)``.
    """
    extension, variable, eta = _validate_hypertangent_reduction(
        derivation
    )
    element = _coerce(extension.field, integrand)
    return _reduce_hypertangent_special_validated(
        element,
        derivation,
        extension,
        variable,
        eta,
    )


def reduce_hypertangent_monomial(
    integrand: Any,
    derivation: Derivation,
    *,
    residue: Symbol | None = None,
) -> HypertangentMonomialReduction:
    r"""Apply the complete reduction for ``Dt=eta*(1+t**2)``, ``eta in QQ(x)``.

    Common nonlinear Hermite, normal-residue, polynomial, and infinity
    reductions run first.  If they leave no Liouville obstruction, the
    remaining powers of the special polynomial are reduced through real
    conjugate Cayley modes and coupled rational Risch equations.
    """
    extension, variable, eta = _validate_hypertangent_reduction(
        derivation
    )
    field = extension.field
    element = _coerce(field, integrand)
    common_reduction = reduce_nonlinear_monomial(
        element,
        derivation,
        residue=residue,
    )

    special_reduction = None
    exact_part = common_reduction.exact_part
    remainder = common_reduction.remainder
    lower_remainder = None
    if (
        not common_reduction.has_obstruction
        and not common_reduction.needs_special_polynomial
    ):
        special_reduction = _reduce_hypertangent_special_validated(
            common_reduction.remainder,
            derivation,
            extension,
            variable,
            eta,
        )
        exact_part += special_reduction.exact_part
        remainder = special_reduction.remainder
        if not special_reduction.has_obstruction:
            lower_remainder = special_reduction.lower_remainder

    if (
        derivation(exact_part)
        + common_reduction.logarithmic_derivative
        + remainder
        != element
    ):
        raise ValueError(
            "hypertangent monomial reduction reconstruction failed"
        )
    return HypertangentMonomialReduction(
        common_reduction,
        special_reduction,
        exact_part,
        common_reduction.logarithmic_derivative,
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
