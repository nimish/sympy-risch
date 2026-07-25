r"""Admission boundary for the certified elementary P/E/T contract.

The elementary dispatcher is intentionally broader than this boundary in a
few isolated, separately documented cases.  This module answers a different
question: whether an explicit active tower belongs to the finite
primitive/hyperexponential/hypertangent (P/E/T) class for which the library
currently makes its complete recursive elementary-integration claim.  It
never runs a reduction, so an unsupported answer is not a statement about
whether a particular integrand is elementary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from .hyperbolic import is_hyperbolic_mobius_law
from .tower import (
    AlgebraicExtension,
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
)

_ELEMENTARY_INTEGRATION_PROBLEM = "elementary_integration"


class TowerAdmissionOutcome(Enum):
    """Whether a tower is inside the declared algorithmic contract."""

    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"


class TowerAdmissionReason(Enum):
    """Stable reason for rejecting a declared elementary tower request."""

    NO_RATIONAL_BASE = "no_rational_base"
    BASE_FIELD = "base_field"
    BASE_DERIVATIVE = "base_derivative"
    ALGEBRAIC_EXTENSION = "algebraic_extension"
    CONSTANT_FIELD_UNKNOWN = "constant_field_unknown"
    CONSTANT_FIELD_ENLARGED = "constant_field_enlarged"
    HYPERBOLIC_LOWER_LEVEL = "hyperbolic_lower_level"
    SPECIAL_POLYNOMIAL_STATUS = "special_polynomial_status"
    UNSUPPORTED_MONOMIAL_FAMILY = "unsupported_monomial_family"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


@dataclass(frozen=True, slots=True)
class TowerAdmission:
    r"""A solver-free classification of one active tower prefix.

    ``tower`` and ``level`` describe the derivation on which an elementary
    integration request would operate.  Levels above ``level`` are inactive
    and deliberately have no effect on the classification.
    """

    problem: str
    tower: DifferentialTower
    level: int
    outcome: TowerAdmissionOutcome
    reason: TowerAdmissionReason | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, str) or not self.problem:
            raise ValueError("an admission result needs a nonempty problem name")
        if not isinstance(self.tower, DifferentialTower):
            raise TypeError("an admission result needs a DifferentialTower")
        if not isinstance(self.level, int):
            raise TypeError("an admission level must be an integer")
        if not -1 <= self.level <= self.tower.top:
            raise IndexError(f"tower level {self.level} is out of range")
        if not isinstance(self.outcome, TowerAdmissionOutcome):
            raise TypeError("an admission outcome must be a TowerAdmissionOutcome")
        if self.reason is not None and not isinstance(
            self.reason,
            TowerAdmissionReason,
        ):
            raise TypeError("an admission reason must be a TowerAdmissionReason")
        if not isinstance(self.detail, str):
            raise TypeError("an admission detail must be a string")
        if self.outcome is TowerAdmissionOutcome.ACCEPTED and self.reason is not None:
            raise ValueError("an accepted tower must not carry a rejection reason")
        if self.outcome is TowerAdmissionOutcome.UNSUPPORTED and self.reason is None:
            raise ValueError("an unsupported tower needs an admission reason")

    @property
    def accepted(self) -> bool:
        """Whether the active tower is in the declared elementary contract."""
        return self.outcome is TowerAdmissionOutcome.ACCEPTED

    @property
    def unsupported(self) -> bool:
        """Whether this is an algorithmic boundary rather than a proof."""
        return self.outcome is TowerAdmissionOutcome.UNSUPPORTED


def is_standard_rational_base(extension: MonomialExtension) -> bool:
    """Whether ``extension`` is the standard base ``QQ(x)`` with ``Dx = 1``."""
    return extension.parent_field == QQ and extension.generator_derivative == Poly(
        1,
        extension.generator,
        domain=QQ,
    )


def _accepted(derivation: Derivation) -> TowerAdmission:
    return TowerAdmission(
        _ELEMENTARY_INTEGRATION_PROBLEM,
        derivation.tower,
        derivation.level,
        TowerAdmissionOutcome.ACCEPTED,
        None,
        "the active tower is a certified finite P/E/T tower over QQ(x), "
        "possibly with a hyperbolic Möbius normalization at its top level",
    )


def _unsupported(
    derivation: Derivation,
    reason: TowerAdmissionReason,
    detail: str,
) -> TowerAdmission:
    return TowerAdmission(
        _ELEMENTARY_INTEGRATION_PROBLEM,
        derivation.tower,
        derivation.level,
        TowerAdmissionOutcome.UNSUPPORTED,
        reason,
        detail,
    )


def assess_elementary_admission(derivation: Derivation) -> TowerAdmission:
    r"""Classify an active tower for certified recursive P/E/T integration.

    The admitted class has the standard rational base ``QQ(x)``, ``Dx=1`` and
    any finite number of primitive, hyperexponential, or hypertangent levels
    with certified preserved constants.  The exact hyperbolic law
    ``Dt=eta*(1-t**2)`` is also admitted when it is the active top level,
    because the dispatcher normalizes that level immutably to a
    hyperexponential coordinate.

    This is an admission check, not an integration decision: it reads only
    exact tower metadata and never calls a solver.  In particular,
    ``UNSUPPORTED`` does not mean that an integrand has no elementary
    antiderivative.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("elementary admission requires a Derivation")

    if derivation.level < 0:
        return _unsupported(
            derivation,
            TowerAdmissionReason.NO_RATIONAL_BASE,
            "the constant field alone has no standard rational QQ(x) base",
        )

    active_levels = derivation.tower.levels[: derivation.level + 1]
    base = active_levels[0]
    if isinstance(base, AlgebraicExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.ALGEBRAIC_EXTENSION,
            "level 0 is algebraic; algebraic elementary integration is not admitted",
        )
    if not isinstance(base, MonomialExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_EXTENSION,
            "level 0 is not a monomial extension",
        )
    if base.parent_field != QQ:
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_FIELD,
            "the rational base must be over QQ",
        )
    if not is_standard_rational_base(base):
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_DERIVATIVE,
            "the rational base must have derivative Dx = 1",
        )

    for index, extension in enumerate(active_levels[1:], start=1):
        if isinstance(extension, AlgebraicExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.ALGEBRAIC_EXTENSION,
                f"level {index} is algebraic",
            )
        if not isinstance(extension, MonomialExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.UNSUPPORTED_EXTENSION,
                f"level {index} is not a monomial extension",
            )
        if extension.constant_field_status is ConstantFieldStatus.UNKNOWN:
            return _unsupported(
                derivation,
                TowerAdmissionReason.CONSTANT_FIELD_UNKNOWN,
                f"level {index} has no certified preserved-constants status",
            )
        if extension.constant_field_status is ConstantFieldStatus.ENLARGED:
            return _unsupported(
                derivation,
                TowerAdmissionReason.CONSTANT_FIELD_ENLARGED,
                f"level {index} is known to enlarge the constant field",
            )
        if extension.family in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            continue
        if is_hyperbolic_mobius_law(extension):
            if index == derivation.level:
                continue
            return _unsupported(
                derivation,
                TowerAdmissionReason.HYPERBOLIC_LOWER_LEVEL,
                f"level {index} is hyperbolic, but Möbius normalization is "
                "implemented only for the active top level",
            )
        if (
            extension.generator_derivative.degree() >= 2
            and extension.special_polynomial_status
            is not SpecialPolynomialStatus.ABSENT
        ):
            return _unsupported(
                derivation,
                TowerAdmissionReason.SPECIAL_POLYNOMIAL_STATUS,
                f"level {index} has an undecided or present special-polynomial "
                "hypothesis",
            )
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
            f"level {index} has unsupported monomial family {extension.family.value}",
        )

    return _accepted(derivation)


__all__ = [
    "TowerAdmission",
    "TowerAdmissionOutcome",
    "TowerAdmissionReason",
    "assess_elementary_admission",
]
