r"""Dependency analysis for proposed differential-field extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import lcm
from operator import index as integer_index
from typing import Any

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.polytools import Poly, cancel, invert

from .rational import (
    RationalExactDerivativeResult,
    RationalLogDerivativeResult,
    solve_rational_exact_derivative,
    solve_rational_logarithmic_derivative,
)
from .tower import (
    ConstantFieldStatus,
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
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
    candidate_ratio: Any | None
    logarithmic_power: int | None
    logarithmic_result: RationalLogDerivativeResult | None
    reason: str


def _parametric_logarithmic_derivative_eta(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> Any:
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


def _integer_parameter(value: Any, name: str) -> int:
    """Return an exact integer parameter, rejecting booleans and rationals."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def verify_parametric_logarithmic_derivative_witness(
    f: Any,
    extension: MonomialExtension,
    lower_derivation: Derivation,
    n: int,
    m: int,
    argument: Any,
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
    f: Any
    outcome: ParametricLogarithmicDerivativeOutcome
    n: int | None = None
    m: int | None = None
    argument: Any | None = None
    certificate: Any | None = None
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
    def eta(self) -> Any:
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
        if not self.has_solution:
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
    eta: Any
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
    r"""Dependency certificate for ``Dt = eta*t`` over ``QQ(x)``.

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
    rational ``u``.  It records the named presentation ``t = c*exp(u)`` but
    does not make ``t`` dependent over the lower rational field.
    """

    extension: MonomialExtension
    eta: Any
    kind: HyperexponentialDependencyKind
    minimal_power: int | None
    relation_argument: Expr | None
    logarithmic_result: RationalLogDerivativeResult
    exact_result: RationalExactDerivativeResult
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
    """Return ``x`` for the standard derivation on ``QQ(x)``."""
    if derivation.level != 0:
        raise NotImplementedError(
            "hyperexponential dependency analysis currently requires QQ(x)"
        )
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or extension.parent_field != QQ
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=QQ)
    ):
        raise NotImplementedError(
            "hyperexponential dependency analysis currently requires "
            "QQ(x) with Dx = 1"
        )
    return extension.generator


def _minimal_logarithmic_power(
    result: RationalLogDerivativeResult,
) -> int | None:
    r"""Return minimal ``n > 0`` making all logarithmic valuations integral."""
    if result.has_solution:
        return 1
    if (
        result.has_polynomial_obstruction
        or result.has_repeated_pole_obstruction
        or result.has_nonlogarithmic_residue_obstruction
    ):
        return None

    if not result.factors:
        raise ValueError(
            "logarithmic-derivative failure has no obstruction certificate"
        )

    power = 1
    for factor in result.factors:
        if factor.multiplicity is None:
            raise ValueError(
                "missing logarithmic multiplicity in dependency certificate"
            )
        power = lcm(power, int(QQ.denom(QQ.convert(factor.multiplicity))))

    if power <= 1:
        raise ValueError(
            "nonintegral logarithmic-derivative certificate has no denominator"
        )
    return power


def _rational_polynomial_parts(
    element: Any,
    variable: Symbol,
) -> tuple[Poly, Poly, Poly]:
    """Return ``p, a, d`` with ``element = p + a/d`` in ``QQ(x)``."""
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=QQ,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=QQ,
    )
    polynomial_part, proper_numerator = numerator.div(denominator)
    return polynomial_part, proper_numerator, denominator


def _rational_coefficient_ratio(
    left: Poly,
    right: Poly,
    degrees: range,
) -> tuple[bool, Any | None]:
    r"""Solve ``coefficient(left, x**i) = c*coefficient(right, x**i)``.

    The boolean distinguishes an inconsistent system from a consistent one.
    A ``None`` ratio means that every compared coefficient vanishes, so the
    system is consistent but does not determine ``c``.
    """
    ratio = None
    for degree in degrees:
        left_coefficient = QQ.convert(left.nth(degree))
        right_coefficient = QQ.convert(right.nth(degree))
        if right_coefficient == QQ.zero:
            if left_coefficient != QQ.zero:
                return False, None
            continue

        current = left_coefficient/right_coefficient
        if ratio is None:
            ratio = current
        elif current != ratio:
            return False, None

    return True, ratio


def solve_parametric_logarithmic_derivative_heuristic(
    f: Any,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Run Bronstein's Section 7.3 heuristic over ``QQ(x)``.

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

    p, a, d = _rational_polynomial_parts(element, variable)
    q, b, e = _rational_polynomial_parts(eta, variable)
    degree_bound = 0  # Dx = 1, so max(0, degree(Dx) - 1) = 0.

    branch = ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    modulus = None
    candidate_ratio = None

    def certificate(
        reason: str,
        *,
        ratio: Any | None,
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
            range(0, modulus.degree()),
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
    logarithmic_result = solve_rational_logarithmic_derivative(
        field.to_sympy(residual),
        variable,
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
        logarithmic_result = solve_rational_logarithmic_derivative(
            logarithmic_power*field.to_sympy(residual),
            variable,
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

    residues: list[tuple[Poly, Any]] = []
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
    r"""Classify a proposed ``Dt=eta*(1+t**2)`` extension over ``QQ(x)``.

    The analysis applies the rational logarithmic-derivative criterion to the
    Cayley coordinate, whose logarithmic derivative is ``2*i*eta``.
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

    variable = _standard_rational_variable(lower_derivation)
    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(2)
    )
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
    return HypertangentDependency(
        extension,
        eta,
        kind,
        minimal_power,
        relation_argument,
    )


def analyze_hyperexponential_extension(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> HyperexponentialDependency:
    r"""Classify a proposed ``Dt = eta*t`` extension over ``QQ(x)``.

    The logarithmic-derivative test detects both direct dependencies and
    radical dependencies.  If its factor valuations are rational but not all
    integral, their denominator least common multiple is the minimal positive
    ``n`` for which ``n*eta = Du/u``.

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

    variable = _standard_rational_variable(lower_derivation)
    eta = extension.parent_field.convert(
        extension.generator_derivative.nth(1)
    )
    eta_expression = eta.as_expr()

    logarithmic_result = solve_rational_logarithmic_derivative(
        eta_expression,
        variable,
    )
    minimal_power = _minimal_logarithmic_power(logarithmic_result)
    relation_argument = None
    if minimal_power is None:
        kind = HyperexponentialDependencyKind.TRANSCENDENTAL
    elif minimal_power == 1:
        kind = HyperexponentialDependencyKind.RATIONAL
        relation_argument = logarithmic_result.argument
    else:
        kind = HyperexponentialDependencyKind.ALGEBRAIC
        scaled_result = solve_rational_logarithmic_derivative(
            minimal_power*eta_expression,
            variable,
        )
        if not scaled_result.has_solution:
            raise ValueError(
                "minimal logarithmic power did not produce a dependency"
            )
        relation_argument = scaled_result.argument

    exact_result = solve_rational_exact_derivative(
        eta_expression,
        variable,
    )
    exponential_exponent = (
        exact_result.antiderivative
        if exact_result.has_solution
        else None
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
