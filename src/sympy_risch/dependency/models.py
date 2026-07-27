r"""Immutable results and certificates for dependency decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..rational import RationalExactDerivativeResult, RationalLogDerivativeResult
from ..relations.linear import LogarithmicRelationSystemResult
from ..relations.signature import FundamentalSignature
from ..tower.core import ConstantFieldStatus, Derivation, MonomialExtension, _coerce
from ._common import (
    _integer_parameter,
    _parametric_logarithmic_derivative_eta,
    verify_parametric_logarithmic_derivative_witness,
)

if TYPE_CHECKING:
    from ..exact import (
        ExactDerivativeResult,
        HypertangentGaussianExactDerivativeCertificate,
        TranscendentalExactDerivativeCertificate,
    )
    from ..gaussian import HypertangentGaussianNormalForm
    from ..monomial import (
        HyperexponentialLogarithmicDerivativeSignature,
        HypertangentLogarithmicDerivativeSignature,
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

    ``lower_fundamental_signature`` records the certified P/E/T structure of
    the lower field when it lies within that narrower relation-theorem scope.
    It is evidence only: the extension factory keeps its existing exact
    derivative decision and never treats a missing signature as preservation.
    """

    extension: MonomialExtension
    exact_result: ExactDerivativeResult | None
    detail: str | None = None
    lower_fundamental_signature: FundamentalSignature | None = None

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
            self.extension.generator - self.extension.parent_field.to_sympy(witness),
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
    rationality_constraints: tuple[tuple[FieldElement, FieldElement], ...] = ()


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


@dataclass(frozen=True, slots=True)
class MonomialRelationSystemParametricCertificate:
    r"""Chapter-9 rational relation-system certificate for mixed towers.

    The system compiles normal valuations and hyperexponential special
    valuations into one homogeneous matrix over ``QQ``.  A row with a nonzero
    coefficient of ``f`` is exactly a solution of the parametric
    logarithmic-derivative problem; absence of such a row is a complete
    obstruction in the supported primitive/hyperexponential tower scope.
    """

    system: LogarithmicRelationSystemResult
    reason: str


@dataclass(frozen=True, slots=True)
class HypertangentGaussianParametricLogarithmicDerivativeCertificate:
    r"""Real PLogD certificate obtained through the Cayley Gaussian form."""

    normal_form: HypertangentGaussianNormalForm
    target_result: ParametricLogarithmicDerivativeResult
    reason: str


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
        | MonomialRelationSystemParametricCertificate
        | HypertangentGaussianParametricLogarithmicDerivativeCertificate
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
            raise TypeError("outcome must be a ParametricLogarithmicDerivativeOutcome")

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
                argument = field.one / argument

            object.__setattr__(self, "n", n)
            object.__setattr__(self, "m", m)
            object.__setattr__(self, "argument", argument)
            if not self.reconstructs:
                raise ValueError(
                    "parametric logarithmic-derivative witness does not "
                    "satisfy its defining equation"
                )
            return

        if self.n is not None or self.m is not None or self.argument is not None:
            raise ValueError(
                "a nonsolution outcome cannot carry solution witness fields"
            )
        if (
            self.outcome is ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION
            and self.certificate is None
        ):
            raise ValueError("a proved nonsolution requires an obstruction certificate")
        if (
            self.outcome
            in {
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            }
            and not self.detail
        ):
            raise ValueError("an inconclusive or unsupported result requires a detail")

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
        return self.outcome is ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION

    @property
    def is_inconclusive(self) -> bool:
        """Whether a complete fallback must still decide the problem."""
        return self.outcome is ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE

    @property
    def is_unsupported(self) -> bool:
        """Whether an effective-field prerequisite is unavailable."""
        return self.outcome is ParametricLogarithmicDerivativeOutcome.UNSUPPORTED

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
        | HyperexponentialLogarithmicDerivativeSignature
        | HypertangentLogarithmicDerivativeSignature
        | RadicalLogarithmicDerivativeResult
    )
    exact_result: (
        RationalExactDerivativeResult
        | PrimitiveExactDerivativeResult
        | TranscendentalExactDerivativeCertificate
        | HypertangentGaussianExactDerivativeCertificate
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


@dataclass(frozen=True, slots=True)
class TranscendentalExtensionAnalysis:
    r"""Checked constant-field certificate for an exponential or tangent proposal.

    A hyperexponential proposal is admissible precisely when its logarithmic
    derivative has no rational logarithmic relation in the lower field. A
    hypertangent proposal has the same criterion after its Cayley transform.
    In either dependent case, the formal generator would introduce a new
    constant, so the returned immutable extension records ``ENLARGED``.

    ``UNKNOWN`` is reserved for an uncertified lower tower or a lower family
    outside the implemented decision procedures. This keeps a missing
    implementation distinct from a proved transcendence result.

    ``lower_fundamental_signature`` retains P/E/T structural evidence only
    after the lower tower has independently certified its constants.  It does
    not broaden the supported dependency decision.
    """

    extension: MonomialExtension
    dependency: HyperexponentialDependency | HypertangentDependency | None
    detail: str | None = None
    lower_fundamental_signature: FundamentalSignature | None = None

    @property
    def status(self) -> ConstantFieldStatus:
        """Return the constant-field status cached on ``extension``."""
        return self.extension.constant_field_status

    @property
    def is_dependent(self) -> bool:
        """Whether the proposal produces a new constant relation."""
        return self.status is ConstantFieldStatus.ENLARGED
