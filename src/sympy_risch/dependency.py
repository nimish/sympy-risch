r"""Dependency analysis for proposed differential-field extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import lcm
from typing import Any

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

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
