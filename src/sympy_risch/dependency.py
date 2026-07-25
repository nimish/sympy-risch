r"""Dependency analysis for proposed differential-field extensions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import lcm
from operator import index as integer_index
from typing import TYPE_CHECKING, SupportsIndex, cast

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.fields import FracElement
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, cancel, invert

from ._typing import FieldElement
from .rational import (
    RationalExactDerivativeResult,
    RationalLogDerivativeResult,
    _solve_rational_logarithmic_derivative_over_domain,
)
from .tower import (
    ConstantFieldStatus,
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
    _rebase_monomial_derivation,
)

if TYPE_CHECKING:
    from .exact import (
        ExactDerivativeResult,
        TranscendentalExactDerivativeCertificate,
    )
    from .monomial import (
        PrimitiveExactDerivativeResult,
        PrimitiveLogarithmicDerivativeSignature,
        RadicalLogarithmicDerivativeResult,
    )


class ParametricLogarithmicDerivativeOutcome(Enum):
    r"""Logical outcome of Bronstein's parametric logarithmic-derivative problem.

    ``PROVED_NO_SOLUTION`` is reserved for a complete mathematical
    obstruction. ``INCONCLUSIVE`` means that a heuristic such as Bronstein's
    Section 7.3 algorithm returned ``failed`` and a structure-theorem fallback
    is still required. ``UNSUPPORTED`` records a missing effective-field
    prerequisite rather than a mathematical negative answer.
    """

    SOLUTION = "solution"
    PROVED_NO_SOLUTION = "proved_no_solution"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


class ParametricLogarithmicDerivativeHeuristicBranch(Enum):
    """Coefficient comparison used by Bronstein's Section 7.3 heuristic."""

    POLYNOMIAL_PART = "polynomial_part"
    DENOMINATOR_CONGRUENCE = "denominator_congruence"


@dataclass(frozen=True, slots=True)
class PrimitiveExtensionAnalysis:
    r"""Checked constant-field certificate for a primitive proposal.

    For ``K(t)`` with ``Dt = eta`` in ``K``, constants are preserved exactly
    when ``eta`` is not a derivative in ``K``.  When ``eta = D(v)``,
    ``t - v`` is a new constant and the proposed generator is dependent.

    ``extension`` is a fresh immutable extension whose
    :class:`ConstantFieldStatus` records the proved outcome.  ``UNKNOWN`` is
    used only when a prerequisite lower level is uncertified or the lower
    exact-derivative decision is not implemented.
    """

    extension: MonomialExtension
    exact_result: ExactDerivativeResult | None
    detail: str | None = None

    @property
    def status(self) -> ConstantFieldStatus:
        """Return the certified constant-field status."""
        return self.extension.constant_field_status

    @property
    def is_dependent(self) -> bool:
        """Whether ``Dt`` is exact and therefore ``t`` introduces a constant."""
        return self.status is ConstantFieldStatus.ENLARGED

    @property
    def dependency_witness(self) -> FieldElement | None:
        """Return ``v`` when ``Dt = D(v)``, otherwise ``None``."""
        if not self.is_dependent or self.exact_result is None:
            return None
        return self.exact_result.antiderivative

    @property
    def constant_relation(self) -> FieldElement | None:
        """Return the certified new constant ``t - v`` in the proposed field."""
        witness = self.dependency_witness
        if witness is None:
            return None
        return _coerce(
            self.extension.field,
            self.extension.generator
            - self.extension.parent_field.to_sympy(witness),
        )


@dataclass(frozen=True, slots=True)
class ParametricLogarithmicDerivativeHeuristicCertificate:
    r"""Trace of the necessary-condition test in Bronstein's Lemma 7.3.1.

    For ``f = p + a/d`` and ``eta = q + b/e``, the lemma first compares
    polynomial coefficients above ``B = max(0, degree(Dx) - 1)``.  If those
    coefficients do not determine ``m/n``, it compares ``l*f`` and ``l*eta``
    modulo ``z``, where over ``QQ(x)``:

    ``l = lcm(d, e)`` and ``z = gcd(l, derivative(l, x))``.

    ``candidate_ratio`` is ``m/n`` when the comparison determines it.
    ``logarithmic_result`` certifies the final test that a positive integer
    multiple of ``n*f - m*eta`` is a rational logarithmic derivative.
    """

    variable: Symbol
    degree_bound: int
    f_polynomial_part: Poly
    eta_polynomial_part: Poly
    f_proper_numerator: Poly
    eta_proper_numerator: Poly
    f_denominator: Poly
    eta_denominator: Poly
    branch: ParametricLogarithmicDerivativeHeuristicBranch
    comparison_modulus: Poly | None
    candidate_ratio: FieldElement | None
    logarithmic_power: int | None
    logarithmic_result: RationalLogDerivativeResult | None
    reason: str


@dataclass(frozen=True, slots=True)
class RationalParametricLogarithmicDerivativeResidue:
    r"""One irreducible-factor constraint in the rational fallback.

    ``f_partial_numerator`` and ``eta_partial_numerator`` are the local
    numerators modulo ``factor``.  A rational radical logarithmic derivative
    may retain only a rational multiple of ``factor.diff()``.  The
    ``*_obstruction`` tuples are the remaining coordinates after projecting
    away that one-dimensional logarithmic direction.  ``*_residue`` retains
    the projected coefficient itself.  Over ``QQ_I`` its imaginary coordinate
    supplies an additional constraint: the final residue must still be
    rational because it is a valuation after an integer scaling.
    """

    factor: Poly
    f_partial_numerator: Poly
    eta_partial_numerator: Poly
    f_obstruction: tuple[FieldElement, ...]
    eta_obstruction: tuple[FieldElement, ...]
    f_residue: FieldElement | None = None
    eta_residue: FieldElement | None = None
    rationality_constraints: tuple[
        tuple[FieldElement, FieldElement], ...
    ] = ()


@dataclass(frozen=True, slots=True)
class RationalParametricLogarithmicDerivativeCertificate:
    r"""Exact linear certificate for the complete rational-base fallback.

    Every pair ``(a, b)`` in ``linear_constraints`` represents the equation
    ``a = c*b`` for ``c = m/n``.  The first pair compares the constant
    polynomial parts; subsequent pairs compare non-logarithmic residue
    coordinates and, over ``QQ_I``, enforce that the surviving residue is
    rational.  An inconsistent system proves that no solution exists.
    Otherwise ``candidate_ratio`` is the chosen ``c`` and
    ``logarithmic_result`` certifies the final radical logarithmic derivative.
    """

    heuristic_result: ParametricLogarithmicDerivativeResult
    common_denominator: Poly
    residues: tuple[RationalParametricLogarithmicDerivativeResidue, ...]
    linear_constraints: tuple[tuple[FieldElement, FieldElement], ...]
    candidate_ratio: FieldElement | None
    logarithmic_power: int | None
    logarithmic_result: RationalLogDerivativeResult | None
    reason: str


@dataclass(frozen=True, slots=True)
class PrimitiveParametricLogarithmicDerivativeFactor:
    r"""Linear constraints contributed by one irreducible ``p`` in ``K[t]``.

    The principal-numerator tuples are padded to the same multiplicity.
    ``linear_constraints`` contains equations ``a = c*b`` for ``c = m/n``.
    Higher-pole equations force cancellation.  The simple-pole equations
    project modulo the rational-constant direction spanned by ``Dp``.
    """

    factor: Poly
    factor_derivative: Poly
    f_principal_numerators: tuple[Poly, ...]
    eta_principal_numerators: tuple[Poly, ...]
    linear_constraints: tuple[tuple[FieldElement, FieldElement], ...]


@dataclass(frozen=True, slots=True)
class PrimitiveParametricLogarithmicDerivativeCertificate:
    r"""Exact recursive primitive structure-theorem certificate.

    ``polynomial_constraints`` compares every positive power of the primitive
    generator.  The factor certificates compare repeated principal parts and
    simple residues.  Together, ``linear_constraints`` determine, reject, or
    leave free the rational ratio ``candidate_ratio = m/n``.

    If the top-level constraints leave the ratio free,
    ``lower_parametric_result`` records the complete recursive decision on the
    constant-in-the-generator part.  ``residual_signature`` finally certifies
    whether the selected integer residual is a radical logarithmic derivative
    and constructs its argument.
    """

    f_signature: PrimitiveLogarithmicDerivativeSignature
    eta_signature: PrimitiveLogarithmicDerivativeSignature
    polynomial_constraints: tuple[tuple[FieldElement, FieldElement], ...]
    factors: tuple[PrimitiveParametricLogarithmicDerivativeFactor, ...]
    linear_constraints: tuple[tuple[FieldElement, FieldElement], ...]
    candidate_ratio: FieldElement | None
    lower_parametric_result: ParametricLogarithmicDerivativeResult | None
    residual_signature: PrimitiveLogarithmicDerivativeSignature | None
    reason: str


def _parametric_logarithmic_derivative_eta(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> FieldElement:
    r"""Validate ``Dtheta = eta*theta`` over the supplied lower field."""
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "parametric logarithmic derivatives require a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError(
            "parametric logarithmic derivatives require a Derivation"
        )
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError(
            "parametric logarithmic derivatives require Dtheta = eta*theta"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError(
            "extension parent field does not match the lower derivation"
        )
    return _coerce(
        lower_derivation.field,
        extension.generator_derivative.nth(1),
    )


def _integer_parameter(value: object, name: str) -> int:
    """Return an exact integer parameter, rejecting booleans and rationals."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def verify_parametric_logarithmic_derivative_witness(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
    n: int,
    m: int,
    argument: object,
) -> bool:
    r"""Verify one exact witness for Bronstein's equation (7.37).

    For ``Dtheta/theta = eta``, a witness consists of integers ``n != 0`` and
    ``m`` and a nonzero ``v`` in the lower field satisfying

    ``n*f = Dv/v + m*eta``.

    The comparison is performed entirely with native SymPy domain elements.
    No expression simplification or numerical equality is used.
    """
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    n_integer = _integer_parameter(n, "n")
    m_integer = _integer_parameter(m, "m")
    if n_integer == 0:
        return False

    field = lower_derivation.field
    element = _coerce(field, f)
    witness_argument = _coerce(field, argument)
    if witness_argument == field.zero:
        return False

    return (
        n_integer*element
        == lower_derivation(witness_argument)/witness_argument
        + m_integer*eta
    )


@dataclass(frozen=True, slots=True)
class ParametricLogarithmicDerivativeResult:
    r"""Certified outcome of the parametric logarithmic-derivative problem.

    The input is ``f`` in a differential field ``K`` and a hyperexponential
    monomial ``theta`` over ``K``. A successful result stores a normalized
    witness ``(n, m, argument)`` with ``n > 0`` and

    ``n*f = D(argument)/argument + m*Dtheta/theta``.

    A negative result must distinguish a proved impossibility from the
    inconclusive ``failed`` outcome of Bronstein's Section 7.3 heuristic and
    from an unsupported effective field. Only ``PROVED_NO_SOLUTION`` may be
    consumed as a mathematical negative answer.
    """

    extension: MonomialExtension
    lower_derivation: Derivation
    f: FieldElement
    outcome: ParametricLogarithmicDerivativeOutcome
    n: int | None = None
    m: int | None = None
    argument: FieldElement | None = None
    certificate: (
        ParametricLogarithmicDerivativeHeuristicCertificate
        | RationalParametricLogarithmicDerivativeCertificate
        | PrimitiveParametricLogarithmicDerivativeCertificate
        | None
    ) = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _parametric_logarithmic_derivative_eta(
            self.extension,
            self.lower_derivation,
        )
        if not isinstance(
            self.outcome,
            ParametricLogarithmicDerivativeOutcome,
        ):
            raise TypeError(
                "outcome must be a ParametricLogarithmicDerivativeOutcome"
            )

        field = self.lower_derivation.field
        object.__setattr__(self, "f", _coerce(field, self.f))

        if self.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION:
            if self.n is None or self.m is None or self.argument is None:
                raise ValueError(
                    "a parametric logarithmic-derivative solution needs "
                    "n, m, and a nonzero argument"
                )
            n = _integer_parameter(self.n, "n")
            m = _integer_parameter(self.m, "m")
            argument = _coerce(field, self.argument)
            if n == 0:
                raise ValueError("a solution requires n != 0")
            if argument == field.zero:
                raise ValueError("a solution requires a nonzero argument")

            # Multiplying the equation by -1 replaces v by 1/v. Store the
            # unique sign convention n > 0 so later solvers compare witnesses
            # consistently.
            if n < 0:
                n = -n
                m = -m
                argument = field.one/argument

            object.__setattr__(self, "n", n)
            object.__setattr__(self, "m", m)
            object.__setattr__(self, "argument", argument)
            if not self.reconstructs:
                raise ValueError(
                    "parametric logarithmic-derivative witness does not "
                    "satisfy its defining equation"
                )
            return

        if (
            self.n is not None
            or self.m is not None
            or self.argument is not None
        ):
            raise ValueError(
                "a nonsolution outcome cannot carry solution witness fields"
            )
        if (
            self.outcome
            is ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION
            and self.certificate is None
        ):
            raise ValueError(
                "a proved nonsolution requires an obstruction certificate"
            )
        if (
            self.outcome
            in {
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            }
            and not self.detail
        ):
            raise ValueError(
                "an inconclusive or unsupported result requires a detail"
            )

    @property
    def eta(self) -> FieldElement:
        """Return ``Dtheta/theta`` as an element of the lower field."""
        return _parametric_logarithmic_derivative_eta(
            self.extension,
            self.lower_derivation,
        )

    @property
    def has_solution(self) -> bool:
        """Whether the result contains an exactly verified witness."""
        return self.outcome is ParametricLogarithmicDerivativeOutcome.SOLUTION

    @property
    def proves_no_solution(self) -> bool:
        """Whether a complete obstruction proves that no witness exists."""
        return (
            self.outcome
            is ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION
        )

    @property
    def is_inconclusive(self) -> bool:
        """Whether a complete fallback must still decide the problem."""
        return (
            self.outcome
            is ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE
        )

    @property
    def is_unsupported(self) -> bool:
        """Whether an effective-field prerequisite is unavailable."""
        return (
            self.outcome
            is ParametricLogarithmicDerivativeOutcome.UNSUPPORTED
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the stored solution witness satisfies equation (7.37)."""
        if (
            not self.has_solution
            or self.n is None
            or self.m is None
            or self.argument is None
        ):
            return False
        return verify_parametric_logarithmic_derivative_witness(
            self.f,
            self.extension,
            self.lower_derivation,
            self.n,
            self.m,
            self.argument,
        )


class HyperexponentialDependencyKind(Enum):
    r"""How a proposed generator depends after constant normalization.

    The derivative law determines the generator only up to a nonzero
    multiplicative constant.  These kinds assume that constant is fixed by
    the source expression, or belongs to the existing constant field.
    """

    RATIONAL = "rational"
    ALGEBRAIC = "algebraic"
    TRANSCENDENTAL = "transcendental"


@dataclass(frozen=True, slots=True)
class HypertangentDependency:
    r"""Dependency certificate for ``Dt=eta*(1+t**2)`` over ``QQ(x)``.

    In the Cayley coordinate ``u=(1+i*t)/(1-i*t)``, one has

    ``Du/u = 2*i*eta``.

    If ``minimal_power`` and ``relation_argument`` are present, they satisfy

    ``D(u**minimal_power/relation_argument) = 0``.

    Thus a formal transcendental tangent generator would enlarge the constant
    field.  Power one is a rational Cayley dependency; larger powers are
    algebraic dependencies.
    """

    extension: MonomialExtension
    eta: FieldElement
    kind: HyperexponentialDependencyKind
    minimal_power: int | None
    relation_argument: Expr | None

    @property
    def is_dependent(self) -> bool:
        """Whether the Cayley generator is rationally or algebraically dependent."""
        return self.kind is not HyperexponentialDependencyKind.TRANSCENDENTAL

    @property
    def is_rationally_dependent(self) -> bool:
        """Whether the Cayley generator itself belongs to the lower field."""
        return self.kind is HyperexponentialDependencyKind.RATIONAL

    @property
    def is_algebraically_dependent(self) -> bool:
        """Whether a positive Cayley power belongs to the lower field."""
        return self.kind is HyperexponentialDependencyKind.ALGEBRAIC

    @property
    def inferred_constant_field_status(self) -> ConstantFieldStatus:
        """Infer whether the formal extension preserves its constants."""
        if self.is_dependent:
            return ConstantFieldStatus.ENLARGED
        return ConstantFieldStatus.PRESERVED

    @property
    def conflicts_with_declared_constant_field_status(self) -> bool:
        """Whether extension metadata contradicts the inferred status."""
        declared = self.extension.constant_field_status
        return (
            declared is not ConstantFieldStatus.UNKNOWN
            and declared is not self.inferred_constant_field_status
        )


@dataclass(frozen=True, slots=True)
class HyperexponentialDependency:
    r"""Dependency certificate for ``Dt = eta*t`` over a supported tower.

    In the rational and algebraic cases, ``minimal_power`` and
    ``relation_argument`` satisfy

    ``D(t**minimal_power/relation_argument) = 0``.

    Thus power one means that the proposed generator belongs to the lower
    field after fixing its multiplicative constant.  A larger power means
    that a normalized source expression should be represented by an
    algebraic extension instead of a transcendental monomial extension.

    For a purely formal transcendental ``t``, the relation only proves that
    ``t**n/relation_argument`` is constant.  If that constant is not known to
    belong to the existing constant field, adjoining ``t`` enlarges the
    constants rather than making ``t`` algebraic over the original field.

    Independently, ``exponential_exponent`` is present when ``eta = Du`` for a
    lower-field element ``u``.  It records the named presentation
    ``t = c*exp(u)`` but does not make ``t`` dependent over the lower field.
    """

    extension: MonomialExtension
    eta: FieldElement
    kind: HyperexponentialDependencyKind
    minimal_power: int | None
    relation_argument: Expr | None
    logarithmic_result: (
        RationalLogDerivativeResult
        | PrimitiveLogarithmicDerivativeSignature
        | RadicalLogarithmicDerivativeResult
    )
    exact_result: (
        RationalExactDerivativeResult
        | PrimitiveExactDerivativeResult
        | TranscendentalExactDerivativeCertificate
        | ExactDerivativeResult
        | None
    )
    exponential_exponent: Expr | None

    @property
    def is_dependent(self) -> bool:
        """Whether the proposed transcendental generator is dependent."""
        return self.kind is not HyperexponentialDependencyKind.TRANSCENDENTAL

    @property
    def is_rationally_dependent(self) -> bool:
        """Whether normalized ``t`` already belongs to the lower field."""
        return self.kind is HyperexponentialDependencyKind.RATIONAL

    @property
    def is_algebraically_dependent(self) -> bool:
        """Whether a positive power of normalized ``t`` is in the lower field."""
        return self.kind is HyperexponentialDependencyKind.ALGEBRAIC

    @property
    def has_exponential_presentation(self) -> bool:
        """Whether ``t`` can be presented as ``c*exp(u)`` for rational ``u``."""
        return self.exponential_exponent is not None

    @property
    def inferred_constant_field_status(self) -> ConstantFieldStatus:
        r"""Infer the constants of the formal transcendental extension.

        A dependency makes ``t**n/u`` a new constant, so a formal
        transcendental ``K(t)`` would enlarge the constant field.  In the
        genuinely hyperexponential case, the standard structure theorem gives
        preservation of constants.
        """
        if self.is_dependent:
            return ConstantFieldStatus.ENLARGED
        return ConstantFieldStatus.PRESERVED

    @property
    def conflicts_with_declared_constant_field_status(self) -> bool:
        """Whether the extension metadata contradicts the inferred status."""
        declared = self.extension.constant_field_status
        return (
            declared is not ConstantFieldStatus.UNKNOWN
            and declared is not self.inferred_constant_field_status
        )


def _standard_rational_variable(derivation: Derivation) -> Symbol:
    """Return ``x`` for the standard derivation on ``QQ(x)`` or ``QQ_I(x)``.

    The integer parameters in a parametric logarithmic derivative are still
    rational when the constants have been enlarged by ``i``.  Supporting the
    Gaussian rational base is therefore a genuine extension of the same
    decision problem, not a relaxation to arbitrary complex parameters.
    """
    if derivation.level != 0:
        raise NotImplementedError(
            "rational-base dependency analysis currently requires QQ(x) "
            "or QQ_I(x)"
        )
    constants = derivation.tower.constants
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or constants not in {QQ, QQ_I}
        or extension.parent_field != constants
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=constants)
    ):
        raise NotImplementedError(
            "rational-base dependency analysis currently requires QQ(x) "
            "with Dx = 1 or QQ_I(x) with Dx = 1"
        )
    return extension.generator


def _minimal_logarithmic_power(
    result: RationalLogDerivativeResult,
) -> int | None:
    r"""Return minimal ``n > 0`` making all logarithmic valuations integral."""
    return result.minimal_radical_power


def _rational_polynomial_parts(
    element: FracElement,
    variable: Symbol,
    domain: Domain,
) -> tuple[Poly, Poly, Poly]:
    """Return ``p, a, d`` with ``element = p + a/d`` over ``domain(x)``."""
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=domain,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=domain,
    )
    polynomial_part, proper_numerator = numerator.div(denominator)
    return polynomial_part, proper_numerator, denominator


def _rational_coefficient_ratio(
    left: Poly,
    right: Poly,
    degrees: range,
    constant_field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve ``coefficient(left, x**i) = c*coefficient(right, x**i)``.

    The boolean distinguishes an inconsistent system from a consistent one.
    A ``None`` ratio means that every compared coefficient vanishes, so the
    system is consistent but does not determine ``c``.  Although the
    coefficients may live in ``QQ_I``, the parameter ``c=m/n`` is required
    to belong to ``QQ``.
    """
    constraints = tuple(
        (
            constant_field.convert(left.nth(degree)),
            constant_field.convert(right.nth(degree)),
        )
        for degree in degrees
    )
    return _constant_field_linear_ratio(constraints, constant_field)


def _rational_linear_ratio(
    constraints: tuple[tuple[FieldElement, FieldElement], ...],
    constant_field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve the exact scalar system ``left = c*right`` with ``c`` in ``QQ``.

    ``(True, None)`` means that every equation is ``0 = c*0``, so the ratio
    remains free.  ``(False, None)`` means that no rational ratio satisfies
    all constraints.
    """
    return _constant_field_linear_ratio(constraints, constant_field)


def solve_parametric_logarithmic_derivative_heuristic(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Run Bronstein's Section 7.3 heuristic over ``QQ(x)`` or ``QQ_I(x)``.

    Given ``Dtheta/theta = eta``, seek integers ``n != 0`` and ``m`` and a
    nonzero ``v`` in ``QQ(x)`` such that

    ``n*f = Dv/v + m*eta``.

    Lemma 7.3.1 supplies necessary high-degree and denominator congruences
    that often determine the rational ratio ``m/n``.  Once a ratio is known,
    the exact rational logarithmic-derivative solver decides whether an
    integer multiple of the residual is ``Dv/v``.

    This routine intentionally preserves the heuristic's incompleteness.
    When the lemma does not determine ``m/n``, the result is ``INCONCLUSIVE``,
    not a false proof of nonsolvability.  Chapter 9 structure theorems are the
    complete fallback.
    """
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    field = lower_derivation.field
    constant_field = lower_derivation.tower.constants
    element = _coerce(field, f)

    try:
        variable = _standard_rational_variable(lower_derivation)
    except NotImplementedError as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    p, a, d = _rational_polynomial_parts(
        cast(FracElement, element),
        variable,
        constant_field,
    )
    q, b, e = _rational_polynomial_parts(
        cast(FracElement, eta),
        variable,
        constant_field,
    )
    degree_bound = 0  # Dx = 1, so max(0, degree(Dx) - 1) = 0.

    branch = ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    modulus = None
    candidate_ratio = None

    def certificate(
        reason: str,
        *,
        ratio: FieldElement | None,
        logarithmic_power: int | None = None,
        logarithmic_result: RationalLogDerivativeResult | None = None,
    ) -> ParametricLogarithmicDerivativeHeuristicCertificate:
        return ParametricLogarithmicDerivativeHeuristicCertificate(
            variable,
            degree_bound,
            p,
            q,
            a,
            b,
            d,
            e,
            branch,
            modulus,
            ratio,
            logarithmic_power,
            logarithmic_result,
            reason,
        )

    if q.degree() > degree_bound:
        top_degree = max(p.degree(), q.degree())
        consistent, candidate_ratio = _rational_coefficient_ratio(
            p,
            q,
            range(degree_bound + 1, int(top_degree) + 1),
            constant_field,
        )
        if not consistent:
            obstruction = certificate(
                "polynomial coefficients above the degree bound are not "
                "proportional",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                certificate=obstruction,
            )
        if candidate_ratio is None:
            raise ValueError(
                "a nonconstant eta polynomial part did not determine m/n"
            )
    elif p.degree() > degree_bound:
        obstruction = certificate(
            "f has polynomial degree above the bound while eta does not",
            ratio=None,
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=obstruction,
        )
    else:
        branch = (
            ParametricLogarithmicDerivativeHeuristicBranch.DENOMINATOR_CONGRUENCE
        )
        common_denominator = d.lcm(e)

        # In QQ[x] every nonconstant irreducible polynomial is normal for
        # D=d/dx, so Bronstein's z = l_s*gcd(l_n, D(l_n)) reduces to gcd(l,l').
        modulus = common_denominator.gcd(common_denominator.diff())
        if modulus.degree() <= 0:
            trace = certificate(
                "Lemma 7.3.1 supplies no nonconstant denominator modulus",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                certificate=trace,
                detail=(
                    "Section 7.3 could not determine m/n; a structure-theorem "
                    "fallback is required"
                ),
            )

        scaled_f = (
            common_denominator.exquo(d)*(p*d + a)
        ).rem(modulus)
        scaled_eta = (
            common_denominator.exquo(e)*(q*e + b)
        ).rem(modulus)
        consistent, candidate_ratio = _rational_coefficient_ratio(
            scaled_f,
            scaled_eta,
            range(modulus.degree()),
            constant_field,
        )
        if not consistent:
            obstruction = certificate(
                "denominator congruences are not proportional",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                certificate=obstruction,
            )
        if candidate_ratio is None:
            trace = certificate(
                "denominator congruences do not determine m/n",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                certificate=trace,
                detail=(
                    "Section 7.3 denominator congruences leave m/n free; "
                    "a structure-theorem fallback is required"
                ),
            )

    # The coefficient comparison determines s=m/n.  Write it in lowest terms
    # as M/N with N>0, then decide whether some positive integer Q makes
    # Q*(N*f - M*eta) a rational logarithmic derivative.
    m = int(QQ.numer(candidate_ratio))
    n = int(QQ.denom(candidate_ratio))
    residual = n*element - m*eta
    logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
        field.to_sympy(residual),
        variable,
        constant_field,
    )
    logarithmic_power = _minimal_logarithmic_power(logarithmic_result)
    if logarithmic_power is None:
        obstruction = certificate(
            "the uniquely determined residual is not the logarithmic "
            "derivative of a rational radical",
            ratio=candidate_ratio,
            logarithmic_result=logarithmic_result,
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=obstruction,
        )

    if logarithmic_power > 1:
        logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
            logarithmic_power*field.to_sympy(residual),
            variable,
            constant_field,
        )
        if not logarithmic_result.has_solution:
            raise ValueError(
                "clearing logarithmic residues did not produce an argument"
            )

    solution_certificate = certificate(
        "coefficient comparison and radical logarithmic-derivative test "
        "produced a verified witness",
        ratio=candidate_ratio,
        logarithmic_power=logarithmic_power,
        logarithmic_result=logarithmic_result,
    )
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power*n,
        m=logarithmic_power*m,
        argument=logarithmic_result.argument,
        certificate=solution_certificate,
    )


def _rational_residue_coefficient(
    partial_numerator: Poly,
    factor: Poly,
    constant_field: Domain,
) -> FieldElement:
    r"""Return the ``factor.diff()`` coefficient at the leading pivot."""
    factor_degree = factor.degree()
    if factor_degree <= 0:
        raise ValueError("a residue factor must be nonconstant")

    derivative = factor.diff()
    pivot_degree = factor_degree - 1
    pivot = constant_field.convert(derivative.nth(pivot_degree))
    if pivot == constant_field.zero:
        raise ValueError(
            "characteristic-zero factor derivative lost its leading term"
        )
    return cast(
        FieldElement,
        constant_field.convert(partial_numerator.nth(pivot_degree))/pivot
    )


def _rational_residue_obstruction(
    partial_numerator: Poly,
    factor: Poly,
    constant_field: Domain,
) -> tuple[FieldElement, ...]:
    r"""Project a local numerator modulo the span of ``factor.diff()``."""
    factor_degree = factor.degree()
    if factor_degree <= 0:
        raise ValueError("a residue factor must be nonconstant")

    derivative = factor.diff()
    pivot_degree = factor_degree - 1
    multiple = _rational_residue_coefficient(
        partial_numerator,
        factor,
        constant_field,
    )
    return tuple(
        constant_field.convert(partial_numerator.nth(degree))
        - multiple*constant_field.convert(derivative.nth(degree))
        for degree in range(pivot_degree)
    )


def _rational_residue_rationality_constraints(
    f_residue: FieldElement,
    eta_residue: FieldElement,
    constant_field: Domain,
) -> tuple[tuple[FieldElement, FieldElement], ...]:
    r"""Constrain ``f_residue-c*eta_residue`` to lie in ``QQ``.

    A local logarithmic-derivative coefficient is a valuation after a
    positive integer scaling, hence it must be rational even after the base
    constants are enlarged to ``QQ_I``.  The ordinary projection only says
    that the coefficient lies in the Gaussian field; its imaginary coordinate
    gives the missing linear equation for the rational parameter ``c=m/n``.
    """
    if constant_field == QQ:
        return ()
    if constant_field != QQ_I:
        raise ValueError(
            "rational residue constraints require QQ or QQ_I constants"
        )
    _, f_imaginary = constant_field.to_sympy(f_residue).as_real_imag()
    _, eta_imaginary = constant_field.to_sympy(
        eta_residue,
    ).as_real_imag()
    return ((QQ.from_sympy(f_imaginary), QQ.from_sympy(eta_imaginary)),)


def _solve_rational_parametric_logarithmic_derivative_fallback(
    heuristic_result: ParametricLogarithmicDerivativeResult,
) -> ParametricLogarithmicDerivativeResult:
    r"""Complete an inconclusive Section 7.3 result over a rational base.

    At the heuristic's ``failed`` exit, both polynomial parts have degree at
    most zero and the common denominator is square-free.  For
    ``c = m/n``, the residual ``f - c*eta`` is a logarithmic derivative of a
    rational radical exactly when:

    * its constant polynomial part vanishes; and
    * at every irreducible denominator factor, its partial numerator is a
      rational multiple of the factor derivative.

    Removing each factor-derivative direction leaves an exact scalar linear
    system for the rational parameter ``c``.  The coefficient field may be
    ``QQ_I``; a non-rational Gaussian quotient is an inconsistency, never a
    candidate parameter.
    """
    if not heuristic_result.is_inconclusive:
        raise ValueError(
            "the rational fallback requires an inconclusive heuristic result"
        )
    heuristic_certificate = heuristic_result.certificate
    if not isinstance(
        heuristic_certificate,
        ParametricLogarithmicDerivativeHeuristicCertificate,
    ):
        raise TypeError(
            "the rational fallback requires a Section 7.3 certificate"
        )

    extension = heuristic_result.extension
    lower_derivation = heuristic_result.lower_derivation
    field = lower_derivation.field
    constant_field = lower_derivation.tower.constants
    variable = heuristic_certificate.variable
    eta = heuristic_result.eta

    p = heuristic_certificate.f_polynomial_part
    q = heuristic_certificate.eta_polynomial_part
    a = heuristic_certificate.f_proper_numerator
    b = heuristic_certificate.eta_proper_numerator
    d = heuristic_certificate.f_denominator
    e = heuristic_certificate.eta_denominator

    if p.degree() > 0 or q.degree() > 0:
        raise ValueError(
            "an inconclusive Section 7.3 result has a nonconstant "
            "polynomial part"
        )

    common_denominator = d.lcm(e)
    factorization = tuple(common_denominator.factor_list()[1])
    if any(multiplicity != 1 for _, multiplicity in factorization):
        raise ValueError(
            "an inconclusive Section 7.3 result has a repeated denominator"
        )

    common_f_numerator = common_denominator.exquo(d)*a
    common_eta_numerator = common_denominator.exquo(e)*b
    constraints: list[tuple[FieldElement, FieldElement]] = [
        (
            constant_field.convert(p.nth(0)),
            constant_field.convert(q.nth(0)),
        )
    ]
    residue_certificates: list[
        RationalParametricLogarithmicDerivativeResidue
    ] = []

    for factor, _ in factorization:
        cofactor = common_denominator.exquo(factor)
        inverse_cofactor = invert(cofactor, factor)
        f_partial_numerator = (
            common_f_numerator*inverse_cofactor
        ).rem(factor)
        eta_partial_numerator = (
            common_eta_numerator*inverse_cofactor
        ).rem(factor)
        f_residue = _rational_residue_coefficient(
            f_partial_numerator,
            factor,
            constant_field,
        )
        eta_residue = _rational_residue_coefficient(
            eta_partial_numerator,
            factor,
            constant_field,
        )
        f_obstruction = _rational_residue_obstruction(
            f_partial_numerator,
            factor,
            constant_field,
        )
        eta_obstruction = _rational_residue_obstruction(
            eta_partial_numerator,
            factor,
            constant_field,
        )
        constraints.extend(zip(f_obstruction, eta_obstruction, strict=True))
        rationality_constraints = _rational_residue_rationality_constraints(
            f_residue,
            eta_residue,
            constant_field,
        )
        constraints.extend(rationality_constraints)
        residue_certificates.append(
            RationalParametricLogarithmicDerivativeResidue(
                factor,
                f_partial_numerator,
                eta_partial_numerator,
                f_obstruction,
                eta_obstruction,
                f_residue,
                eta_residue,
                rationality_constraints,
            )
        )

    linear_constraints = tuple(constraints)
    residues = tuple(residue_certificates)
    consistent, candidate_ratio = _rational_linear_ratio(
        linear_constraints,
        constant_field,
    )
    if not consistent:
        certificate = RationalParametricLogarithmicDerivativeCertificate(
            heuristic_result,
            common_denominator,
            residues,
            linear_constraints,
            None,
            None,
            None,
            "the rational structure constraints on m/n are inconsistent",
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            heuristic_result.f,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate,
        )

    # If every obstruction coordinate vanishes identically, f itself is a
    # radical logarithmic derivative and m=0 is the simplest witness.
    if candidate_ratio is None:
        candidate_ratio = QQ.zero

    m = int(QQ.numer(candidate_ratio))
    n = int(QQ.denom(candidate_ratio))
    residual = n*heuristic_result.f - m*eta
    logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
        field.to_sympy(residual),
        variable,
        constant_field,
    )
    logarithmic_power = _minimal_logarithmic_power(logarithmic_result)
    if logarithmic_power is None:
        raise ValueError(
            "rational structure constraints produced a nonlogarithmic residual"
        )
    if logarithmic_power > 1:
        logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
            logarithmic_power*field.to_sympy(residual),
            variable,
            constant_field,
        )
        if not logarithmic_result.has_solution:
            raise ValueError(
                "clearing fallback logarithmic residues did not produce "
                "an argument"
            )

    certificate = RationalParametricLogarithmicDerivativeCertificate(
        heuristic_result,
        common_denominator,
        residues,
        linear_constraints,
        candidate_ratio,
        logarithmic_power,
        logarithmic_result,
        "the rational structure constraints produced a verified witness",
    )
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        heuristic_result.f,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power*n,
        m=logarithmic_power*m,
        argument=logarithmic_result.argument,
        certificate=certificate,
    )


def _constant_field_linear_ratio(
    constraints: tuple[tuple[FieldElement, FieldElement], ...],
    field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve ``left = c*right`` for one ``c in QQ`` over ``field``."""
    ratio = None
    for left, right in constraints:
        left_element = _coerce(field, left)
        right_element = _coerce(field, right)
        if right_element == field.zero:
            if left_element != field.zero:
                return False, None
            continue

        quotient = left_element/right_element
        try:
            current = QQ.from_sympy(field.to_sympy(quotient))
        except CoercionFailed:
            return False, None
        if ratio is None:
            ratio = current
        elif current != ratio:
            return False, None
    return True, ratio


def _primitive_parametric_signature_constraints(
    f_signature: PrimitiveLogarithmicDerivativeSignature,
    eta_signature: PrimitiveLogarithmicDerivativeSignature,
    derivation: Derivation,
) -> tuple[
    tuple[tuple[FieldElement, FieldElement], ...],
    tuple[PrimitiveParametricLogarithmicDerivativeFactor, ...],
    tuple[tuple[FieldElement, FieldElement], ...],
]:
    r"""Align two primitive signatures and project away every ``QQ*Dp``."""
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "primitive parametric constraints require a monomial level"
        )
    parent = extension.parent_field
    f_polynomial = f_signature.polynomial_part
    eta_polynomial = eta_signature.polynomial_part
    maximum_degree = max(
        (
            polynomial.degree()
            if not polynomial.is_zero
            else 0
        )
        for polynomial in (f_polynomial, eta_polynomial)
    )
    polynomial_constraints = tuple(
        (
            parent.convert(f_polynomial.nth(degree)),
            parent.convert(eta_polynomial.nth(degree)),
        )
        for degree in range(1, int(maximum_degree) + 1)
    )

    f_by_expression = {
        factor.factor.as_expr(): factor
        for factor in f_signature.factors
    }
    eta_by_expression = {
        factor.factor.as_expr(): factor
        for factor in eta_signature.factors
    }
    factor_expressions = list(f_by_expression)
    factor_expressions.extend(
        expression
        for expression in eta_by_expression
        if expression not in f_by_expression
    )

    factor_certificates = []
    all_constraints = list(polynomial_constraints)
    lower_derivation = derivation.lower()
    for expression in factor_expressions:
        f_factor = f_by_expression.get(expression)
        eta_factor = eta_by_expression.get(expression)
        if f_factor is None and eta_factor is None:
            raise ValueError("a collected factor has no source signature")
        if f_factor is not None:
            factor = f_factor.factor
        else:
            assert eta_factor is not None
            factor = eta_factor.factor
        maximum_multiplicity = max(
            f_factor.multiplicity if f_factor is not None else 0,
            eta_factor.multiplicity if eta_factor is not None else 0,
        )

        if f_factor is None:
            f_principal_numerators = (
                factor.zero,
            )*maximum_multiplicity
        else:
            f_principal_numerators = f_factor.principal_numerators + (
                factor.zero,
            )*(
                maximum_multiplicity
                - len(f_factor.principal_numerators)
            )
        if eta_factor is None:
            eta_principal_numerators = (
                factor.zero,
            )*maximum_multiplicity
        else:
            eta_principal_numerators = eta_factor.principal_numerators + (
                factor.zero,
            )*(
                maximum_multiplicity
                - len(eta_factor.principal_numerators)
            )
        factor_derivative = derivation.polynomial(factor).rem(factor)
        if factor_derivative.is_zero:
            raise ValueError(
                "a primitive extension with preserved constants cannot "
                "have a nonconstant special factor"
            )

        factor_constraints = []
        for pole_order in range(2, maximum_multiplicity + 1):
            f_numerator = f_principal_numerators[pole_order - 1]
            eta_numerator = eta_principal_numerators[pole_order - 1]
            factor_constraints.extend(
                (
                    parent.convert(f_numerator.nth(degree)),
                    parent.convert(eta_numerator.nth(degree)),
                )
                for degree in range(factor.degree())
            )

        f_simple = f_principal_numerators[0]
        eta_simple = eta_principal_numerators[0]
        (pivot_degree,), pivot_value = factor_derivative.terms()[0]
        pivot_value = parent.convert(pivot_value)
        f_pivot = parent.convert(f_simple.nth(pivot_degree))
        eta_pivot = parent.convert(eta_simple.nth(pivot_degree))
        for degree in range(factor.degree()):
            if degree == pivot_degree:
                continue
            derivative_coefficient = parent.convert(
                factor_derivative.nth(degree)
            )
            factor_constraints.append(
                (
                    parent.convert(f_simple.nth(degree))*pivot_value
                    - f_pivot*derivative_coefficient,
                    parent.convert(eta_simple.nth(degree))*pivot_value
                    - eta_pivot*derivative_coefficient,
                )
            )

        # The cross-products above make the residual numerator a K-multiple
        # of Dp. Its multiplier belongs to QQ exactly when its lower
        # derivative vanishes, under the same-constants hypothesis.
        factor_constraints.append(
            (
                lower_derivation(parent.exquo(f_pivot, pivot_value)),
                lower_derivation(parent.exquo(eta_pivot, pivot_value)),
            )
        )

        factor_constraints_tuple = tuple(factor_constraints)
        all_constraints.extend(factor_constraints_tuple)
        factor_certificates.append(
            PrimitiveParametricLogarithmicDerivativeFactor(
                factor,
                factor_derivative,
                f_principal_numerators,
                eta_principal_numerators,
                factor_constraints_tuple,
            )
        )

    return (
        polynomial_constraints,
        tuple(factor_certificates),
        tuple(all_constraints),
    )


def _solve_primitive_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Decide the parametric problem over a recursive primitive tower."""
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    field = lower_derivation.field
    element = _coerce(field, f)
    if lower_derivation.level < 1:
        raise ValueError(
            "primitive parametric solving requires a primitive lower field"
        )
    primitive_extension = lower_derivation.tower.levels[
        lower_derivation.level
    ]
    if (
        not isinstance(primitive_extension, MonomialExtension)
        or primitive_extension.family is not MonomialFamily.PRIMITIVE
    ):
        raise ValueError(
            "primitive parametric solving requires a primitive lower field"
        )
    for level in range(1, lower_derivation.level + 1):
        recursive_extension = lower_derivation.tower.levels[level]
        if (
            not isinstance(recursive_extension, MonomialExtension)
            or recursive_extension.family is not MonomialFamily.PRIMITIVE
        ):
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                detail=(
                    "recursive primitive parametric solving requires every "
                    "transcendental level above QQ(x) to be primitive"
                ),
            )
        if (
            recursive_extension.constant_field_status
            is not ConstantFieldStatus.PRESERVED
        ):
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                detail=(
                    "primitive parametric solving requires Const(K(t)) = "
                    "Const(K) at every primitive level"
                ),
            )

    # Local import avoids the module cycle: monomial family reducers use the
    # dependency analyses defined in this module.
    from .monomial import primitive_logarithmic_derivative_signature

    try:
        f_signature = primitive_logarithmic_derivative_signature(
            element,
            lower_derivation,
        )
        eta_signature = primitive_logarithmic_derivative_signature(
            eta,
            lower_derivation,
        )
    except NotImplementedError as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    (
        polynomial_constraints,
        factor_certificates,
        linear_constraints,
    ) = _primitive_parametric_signature_constraints(
        f_signature,
        eta_signature,
        lower_derivation,
    )
    consistent, candidate_ratio = _constant_field_linear_ratio(
        linear_constraints,
        primitive_extension.parent_field,
    )
    lower_parametric_result = None
    residual_signature = None

    def certificate(
        reason: str,
    ) -> PrimitiveParametricLogarithmicDerivativeCertificate:
        return PrimitiveParametricLogarithmicDerivativeCertificate(
            f_signature,
            eta_signature,
            polynomial_constraints,
            factor_certificates,
            linear_constraints,
            candidate_ratio,
            lower_parametric_result,
            residual_signature,
            reason,
        )

    if not consistent:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate(
                "the primitive signature constraints on m/n are inconsistent"
            ),
        )

    if candidate_ratio is None:
        f_lower = primitive_extension.parent_field.convert(
            f_signature.polynomial_part.nth(0)
            if not f_signature.polynomial_part.is_zero
            else primitive_extension.parent_field.zero
        )
        eta_lower = primitive_extension.parent_field.convert(
            eta_signature.polynomial_part.nth(0)
            if not eta_signature.polynomial_part.is_zero
            else primitive_extension.parent_field.zero
        )
        if eta_lower == primitive_extension.parent_field.zero:
            candidate_ratio = QQ.zero
        else:
            lower_theta = Dummy("theta_lower")
            lower_extension = MonomialExtension(
                Poly(
                    primitive_extension.parent_field.to_sympy(eta_lower)
                    * lower_theta,
                    lower_theta,
                    domain=primitive_extension.parent_field,
                )
            )
            lower_parametric_result = solve_parametric_logarithmic_derivative(
                f_lower,
                lower_extension,
                lower_derivation.lower(),
            )
            if lower_parametric_result.proves_no_solution:
                return ParametricLogarithmicDerivativeResult(
                    extension,
                    lower_derivation,
                    element,
                    ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                    certificate=certificate(
                        "the top signature leaves m/n free, but the complete "
                        "recursive lower problem is inconsistent"
                    ),
                )
            if not lower_parametric_result.has_solution:
                return ParametricLogarithmicDerivativeResult(
                    extension,
                    lower_derivation,
                    element,
                    ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                    certificate=certificate(
                        "the top signature leaves m/n free, but the lower "
                        "field family is unsupported"
                    ),
                    detail=lower_parametric_result.detail,
                )
            candidate_ratio = QQ(
                lower_parametric_result.m,
                lower_parametric_result.n,
            )

    ratio_numerator = int(QQ.numer(candidate_ratio))
    ratio_denominator = int(QQ.denom(candidate_ratio))
    integer_residual = (
        ratio_denominator*element
        - ratio_numerator*eta
    )
    residual_signature = primitive_logarithmic_derivative_signature(
        integer_residual,
        lower_derivation,
    )
    if not residual_signature.has_solution:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate(
                "the selected primitive residual is not the logarithmic "
                "derivative of a radical"
            ),
        )

    logarithmic_power = residual_signature.minimal_power
    if logarithmic_power is None:
        raise ValueError("a solved radical signature lacks its minimal power")
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power*ratio_denominator,
        m=logarithmic_power*ratio_numerator,
        argument=residual_signature.relation_argument,
        certificate=certificate(
            "the primitive signature constraints produced a verified witness"
        ),
    )


def solve_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Decide the parametric logarithmic derivative in supported fields.

    Over ``QQ(x)`` or ``QQ_I(x)``, the Section 7.3 coefficient heuristic is
    the fast path and a complete rational structure fallback handles its
    inconclusive exit.  In the Gaussian case the integer ratio ``m/n`` and
    all radical-logarithmic residues remain rational.

    Over a recursive primitive tower, the solver compares positive-degree
    polynomial coordinates, repeated principal parts, and simple residues
    modulo ``QQ*Dp`` at each level.  If the current coordinates leave ``m/n``
    free, the same complete decision descends to the lower component.
    """
    if lower_derivation.level >= 1:
        lower_extension = lower_derivation.tower.levels[
            lower_derivation.level
        ]
        if (
            isinstance(lower_extension, MonomialExtension)
            and lower_extension.family is MonomialFamily.PRIMITIVE
        ):
            return _solve_primitive_parametric_logarithmic_derivative(
                f,
                extension,
                lower_derivation,
            )

    heuristic_result = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        lower_derivation,
    )
    if not heuristic_result.is_inconclusive:
        return heuristic_result
    return _solve_rational_parametric_logarithmic_derivative_fallback(
        heuristic_result,
    )


def _gaussian_logarithmic_relation(
    logarithmic_derivative: Expr,
    variable: Symbol,
) -> tuple[int, Expr] | None:
    r"""Find minimal ``n>0`` with ``n*f=Dv/v`` in ``QQ_I(x)``.

    A rational logarithmic derivative has no polynomial or repeated-pole
    part, and its residue at every irreducible denominator factor is an
    integer.  Rational residues become integral after multiplication by their
    denominator least common multiple.
    """
    field = QQ_I.frac_field(variable)
    element = field.from_sympy(logarithmic_derivative)
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=QQ_I,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=QQ_I,
    )
    polynomial_part, proper_numerator = numerator.div(denominator)
    if not polynomial_part.is_zero:
        return None
    if proper_numerator.is_zero:
        return 1, S.One
    if denominator.gcd(denominator.diff()).degree() > 0:
        return None

    residues: list[tuple[Poly, FieldElement]] = []
    power = 1
    for factor, multiplicity in denominator.factor_list()[1]:
        if multiplicity != 1:
            return None
        cofactor = denominator.exquo(factor)
        local_denominator = (cofactor*factor.diff()).rem(factor)
        residue = (
            proper_numerator*invert(local_denominator, factor)
        ).rem(factor)
        if residue.degree() > 0:
            return None
        value = QQ_I.convert(residue.nth(0))
        if value.y != 0:
            return None
        rational_value = QQ.convert(value.x)
        residues.append((factor, rational_value))
        power = lcm(power, int(QQ.denom(rational_value)))

    relation_argument = S.One
    for factor, residue in residues:
        multiplicity = QQ.convert(power)*residue
        if QQ.denom(multiplicity) != 1:
            raise ValueError(
                "Gaussian logarithmic relation did not clear residues"
            )
        relation_argument *= factor.as_expr()**int(QQ.numer(multiplicity))
    relation_argument = cancel(relation_argument)
    if cancel(
        relation_argument.diff(variable)/relation_argument
        - power*logarithmic_derivative
    ) != 0:
        raise ValueError(
            "Gaussian logarithmic dependency reconstruction failed"
        )
    return power, relation_argument


def analyze_hypertangent_extension(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> HypertangentDependency:
    r"""Classify ``Dt=eta*(1+t**2)`` over a supported lower tower.

    The Cayley coordinate has logarithmic derivative ``2*i*eta``. Over the
    rational base the direct Gaussian residue test remains the fast path.
    Over recursively primitive towers, the tower is immutably rebased over
    ``QQ_I`` and the recursive radical logarithmic-derivative decision is
    applied there.
    """
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "hypertangent dependency analysis requires a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError(
            "hypertangent dependency analysis requires a Derivation"
        )
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent dependency analysis requires "
            "Dt = eta*(1 + t**2)"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError(
            "extension parent field does not match the lower derivation"
        )

    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(2)
    )
    if lower_derivation.level == 0:
        variable = _standard_rational_variable(lower_derivation)
        relation = _gaussian_logarithmic_relation(
            2*I*extension.parent_field.to_sympy(eta),
            variable,
        )
        if relation is None:
            kind = HyperexponentialDependencyKind.TRANSCENDENTAL
            minimal_power = None
            relation_argument = None
        else:
            minimal_power, relation_argument = relation
            if minimal_power == 1:
                kind = HyperexponentialDependencyKind.RATIONAL
            else:
                kind = HyperexponentialDependencyKind.ALGEBRAIC
    else:
        gaussian_derivation = _rebase_monomial_derivation(
            lower_derivation,
            QQ_I,
        )
        gaussian_field = gaussian_derivation.field
        gaussian_eta = gaussian_field.from_sympy(
            2*I*extension.parent_field.to_sympy(eta)
        )
        from .monomial import solve_radical_logarithmic_derivative

        radical_result = solve_radical_logarithmic_derivative(
            gaussian_eta,
            gaussian_derivation,
        )
        minimal_power = radical_result.minimal_power
        relation_argument = (
            gaussian_field.to_sympy(radical_result.argument)
            if radical_result.argument is not None
            else None
        )
        if minimal_power is None:
            kind = HyperexponentialDependencyKind.TRANSCENDENTAL
        elif minimal_power == 1:
            kind = HyperexponentialDependencyKind.RATIONAL
        else:
            kind = HyperexponentialDependencyKind.ALGEBRAIC
    return HypertangentDependency(
        extension,
        eta,
        kind,
        minimal_power,
        relation_argument,
    )


def certify_primitive_extension(
    generator_derivative: Poly,
    lower_derivation: Derivation,
) -> PrimitiveExtensionAnalysis:
    r"""Construct ``K(t)``, certifying whether it preserves constants.

    The proposed derivative must have degree zero in ``t`` and its coefficient
    domain must equal the field of ``lower_derivation``.  The lower
    exact-derivative decision then gives the complete primitive criterion:

    - ``Dt = D(v)`` gives the new constant ``t - v`` and status ``ENLARGED``;
    - a proved non-exact ``Dt`` gives status ``PRESERVED``;
    - an uncertified or unsupported lower tower gives status ``UNKNOWN``.

    Invalid derivative laws and mismatched parent fields are input errors, not
    unknown mathematical outcomes, and are therefore rejected.
    """
    if not isinstance(lower_derivation, Derivation):
        raise TypeError(
            "primitive extension certification requires a Derivation"
        )
    extension = MonomialExtension(generator_derivative)
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError(
            "primitive extension certification requires Dt in the lower field"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError(
            "extension parent field does not match the lower derivation"
        )

    lower_levels = lower_derivation.tower.levels[
        : lower_derivation.level + 1
    ]
    uncertified_level = next(
        (
            level
            for level in lower_levels
            if isinstance(level, MonomialExtension)
            and level.constant_field_status
            is not ConstantFieldStatus.PRESERVED
        ),
        None,
    )
    if uncertified_level is not None:
        return PrimitiveExtensionAnalysis(
            extension,
            None,
            "the lower tower contains a monomial level without certified "
            "constant preservation",
        )

    eta = _coerce(
        extension.parent_field,
        extension.generator_derivative.nth(0),
    )
    # Local import avoids the module cycle: primitive exact reduction uses
    # dependency decisions from this module.
    from .exact import solve_exact_derivative

    try:
        exact_result = solve_exact_derivative(eta, lower_derivation)
    except NotImplementedError as exc:
        return PrimitiveExtensionAnalysis(extension, None, str(exc))

    status = (
        ConstantFieldStatus.ENLARGED
        if exact_result.has_solution
        else ConstantFieldStatus.PRESERVED
    )
    certified_extension = replace(
        extension,
        constant_field_status=status,
    )
    return PrimitiveExtensionAnalysis(
        certified_extension,
        exact_result,
    )


def analyze_hyperexponential_extension(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> HyperexponentialDependency:
    r"""Classify a proposed ``Dt = eta*t`` over a supported lower tower.

    The recursive logarithmic-derivative test detects both direct and radical
    dependencies.  Its ``minimal_power`` is the least positive ``n`` for
    which ``n*eta = Du/u``.

    The ordinary exact-derivative test is independent: ``eta = Dv`` records
    the presentation ``t = c*exp(v)`` for a genuinely transcendental
    candidate.
    """
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "hyperexponential dependency analysis requires a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError(
            "hyperexponential dependency analysis requires a Derivation"
        )
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError(
            "hyperexponential dependency analysis requires Dt = eta*t"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError(
            "extension parent field does not match the lower derivation"
        )

    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(1)
    )
    # Local imports avoid the module cycle: primitive monomial reductions use
    # the dependency analyses defined in this module.
    from .exact import solve_exact_derivative
    from .monomial import solve_radical_logarithmic_derivative

    radical_result = solve_radical_logarithmic_derivative(
        eta,
        lower_derivation,
    )
    minimal_power = radical_result.minimal_power
    relation_argument = None
    if minimal_power is None:
        kind = HyperexponentialDependencyKind.TRANSCENDENTAL
    elif minimal_power == 1:
        kind = HyperexponentialDependencyKind.RATIONAL
        relation_argument = lower_derivation.field.to_sympy(
            radical_result.argument,
        )
    else:
        kind = HyperexponentialDependencyKind.ALGEBRAIC
        relation_argument = lower_derivation.field.to_sympy(
            radical_result.argument,
        )

    recursive_exact_result = solve_exact_derivative(
        eta,
        lower_derivation,
    )
    exponential_exponent = (
        lower_derivation.field.to_sympy(
            recursive_exact_result.antiderivative,
        )
        if recursive_exact_result.has_solution
        else None
    )
    logarithmic_result = (
        radical_result.certificate
        if lower_derivation.level == 0
        else radical_result
    )
    exact_result = (
        recursive_exact_result.certificate
        if lower_derivation.level == 0
        else recursive_exact_result
    )

    return HyperexponentialDependency(
        extension,
        eta,
        kind,
        minimal_power,
        relation_argument,
        logarithmic_result,
        exact_result,
        exponential_exponent,
    )
