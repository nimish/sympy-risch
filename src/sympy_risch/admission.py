r"""Admission boundary for the certified elementary P/E/T contract.

The elementary dispatcher is intentionally broader than this boundary in a
few isolated, separately documented cases.  This module answers a different
question: whether an explicit active tower belongs to the finite
primitive/hyperexponential/hypertangent (P/E/T) class satisfying the
library's structural theorem preconditions.  It does not promise that every
recursive solver branch has been implemented, nor does it run an integrand
reduction.  It replays the structural constant-field criterion; an
unsupported answer is not a statement about whether a particular integrand
is elementary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.domains import QQ, QQ_I
from sympy.polys.polytools import Poly

from .constant_fields import is_supported_finite_constant_field
from .constant_proofs import replay_constant_field
from .hyperbolic import is_hyperbolic_mobius_law
from .tower.core import (
    AlgebraicExtension,
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
)

_ELEMENTARY_INTEGRATION_PROBLEM = "elementary_integration"
_SCALAR_RDE_PROBLEM = "scalar_rde"


class TowerAdmissionOutcome(Enum):
    """Whether a tower is inside the declared algorithmic contract."""

    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"


class TowerAdmissionReason(Enum):
    """Stable reason for rejecting a declared tower-algorithm request."""

    NO_RATIONAL_BASE = "no_rational_base"
    BASE_FIELD = "base_field"
    BASE_DERIVATIVE = "base_derivative"
    ALGEBRAIC_EXTENSION = "algebraic_extension"
    CONSTANT_FIELD_UNKNOWN = "constant_field_unknown"
    CONSTANT_FIELD_ENLARGED = "constant_field_enlarged"
    HYPERBOLIC_LOWER_LEVEL = "hyperbolic_lower_level"
    HYPERTANGENT_RDE_BASE = "hypertangent_rde_base"
    UNSUPPORTED_MONOMIAL_FAMILY = "unsupported_monomial_family"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


@dataclass(frozen=True, slots=True)
class TowerAdmission:
    r"""A structural classification of one active tower prefix.

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
        """Whether the active tower is in this result's declared contract."""
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


def _accepted(
    derivation: Derivation,
    *,
    problem: str = _ELEMENTARY_INTEGRATION_PROBLEM,
    detail: str = (
        "the active tower has a replay-certified finite P/E/T structural shape over QQ(x), "
        "possibly with a hyperbolic Möbius normalization at its top level"
    ),
) -> TowerAdmission:
    return TowerAdmission(
        problem,
        derivation.tower,
        derivation.level,
        TowerAdmissionOutcome.ACCEPTED,
        None,
        detail,
    )


def _unsupported(
    derivation: Derivation,
    reason: TowerAdmissionReason,
    detail: str,
    *,
    problem: str = _ELEMENTARY_INTEGRATION_PROBLEM,
) -> TowerAdmission:
    return TowerAdmission(
        problem,
        derivation.tower,
        derivation.level,
        TowerAdmissionOutcome.UNSUPPORTED,
        reason,
        detail,
    )


def _constant_field_rejection(
    derivation: Derivation,
    index: int,
    *,
    problem: str,
) -> TowerAdmission | None:
    """Reject a level whose independent replay is not preserved."""
    evidence = replay_constant_field(Derivation(derivation.tower, index))
    if evidence.status is ConstantFieldStatus.ENLARGED:
        return _unsupported(
            derivation,
            TowerAdmissionReason.CONSTANT_FIELD_ENLARGED,
            f"independent certification proves that level {index} enlarges "
            "the constant field",
            problem=problem,
        )
    if evidence.status is ConstantFieldStatus.UNKNOWN:
        suffix = f": {evidence.detail}" if evidence.detail else ""
        return _unsupported(
            derivation,
            TowerAdmissionReason.CONSTANT_FIELD_UNKNOWN,
            f"independent certification is unavailable for level {index}{suffix}",
            problem=problem,
        )
    return None


def assess_elementary_admission(derivation: Derivation) -> TowerAdmission:
    r"""Classify an active tower for certified recursive P/E/T integration.

    The admitted class has the standard rational base ``QQ(x)``, ``Dx=1`` and
    finite primitive, hyperexponential, or hypertangent prefixes with
    certified preserved constants.  This classifies structural theorem
    preconditions, not completion of every current recursive solver branch:
    primitive logarithmic-derivative recursion still requires a ``QQ(x)``
    lower field.  The exact hyperbolic law
    ``Dt=eta*(1-t**2)`` is also admitted when it is the active top level,
    because the dispatcher normalizes that level immutably to a
    hyperexponential coordinate.

    This is an admission check, not an integration decision: it replays the
    structural constant-field criteria but never reduces an integrand.  In
    particular, ``UNSUPPORTED`` does not mean that an integrand has no
    elementary antiderivative.
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
        if extension.family in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            rejection = _constant_field_rejection(
                derivation,
                index,
                problem=_ELEMENTARY_INTEGRATION_PROBLEM,
            )
            if rejection is not None:
                return rejection
            continue
        if is_hyperbolic_mobius_law(extension):
            rejection = _constant_field_rejection(
                derivation,
                index,
                problem=_ELEMENTARY_INTEGRATION_PROBLEM,
            )
            if rejection is not None:
                return rejection
            if index == derivation.level:
                continue
            return _unsupported(
                derivation,
                TowerAdmissionReason.HYPERBOLIC_LOWER_LEVEL,
                f"level {index} is hyperbolic, but Möbius normalization is "
                "implemented only for the active top level",
            )
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
            f"level {index} has unsupported monomial family {extension.family.value}; "
            "a caller-supplied no-special status is not a certificate",
        )

    return _accepted(derivation)


def assess_scalar_rde_admission(derivation: Derivation) -> TowerAdmission:
    r"""Classify a tower prefix for recursive scalar RDE dispatch.

    The current dispatcher supports ``C(x)`` bases with ``Dx=1`` when ``C``
    is an effective finite algebraic extension of ``QQ``.  Recursive P/E/T
    levels are currently admitted over ``QQ``; P/E levels are also admitted
    over ``QQ(i)``.  Unlike elementary integration admission, this narrower
    solver contract neither needs a special-polynomial hypothesis nor
    normalizes hyperbolic/Riccati polynomial levels.

    This check replays structural constant-field criteria, but neither solves
    a requested RDE nor promises that every recursive solver branch is
    implemented. An ``UNSUPPORTED`` result says only that the current scalar
    RDE dispatcher cannot be called safely on the prefix; it says nothing
    about solvability of a particular equation.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("scalar RDE admission requires a Derivation")

    if derivation.level < 0:
        return _unsupported(
            derivation,
            TowerAdmissionReason.NO_RATIONAL_BASE,
            "scalar RDE solving requires a nonconstant differential field",
            problem=_SCALAR_RDE_PROBLEM,
        )

    active_levels = derivation.tower.levels[: derivation.level + 1]
    base = active_levels[0]
    if isinstance(base, AlgebraicExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.ALGEBRAIC_EXTENSION,
            "level 0 is algebraic; scalar RDE dispatch has no algebraic base",
            problem=_SCALAR_RDE_PROBLEM,
        )
    if not isinstance(base, MonomialExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_EXTENSION,
            "level 0 is not a monomial extension",
            problem=_SCALAR_RDE_PROBLEM,
        )
    if not is_supported_finite_constant_field(base.parent_field):
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_FIELD,
            "the scalar RDE base must be over an effective finite algebraic "
            "extension of QQ",
            problem=_SCALAR_RDE_PROBLEM,
        )
    if base.generator_derivative != Poly(
        1,
        base.generator,
        domain=base.parent_field,
    ):
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_DERIVATIVE,
            "the scalar RDE rational base must have derivative Dx = 1",
            problem=_SCALAR_RDE_PROBLEM,
        )

    for index, extension in enumerate(active_levels[1:], start=1):
        if base.parent_field not in (QQ, QQ_I):
            return _unsupported(
                derivation,
                TowerAdmissionReason.BASE_FIELD,
                f"level {index} is recursive, but scalar RDE recursion over "
                "native finite algebraic constants is not yet implemented",
                problem=_SCALAR_RDE_PROBLEM,
            )
        if isinstance(extension, AlgebraicExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.ALGEBRAIC_EXTENSION,
                f"level {index} is algebraic",
                problem=_SCALAR_RDE_PROBLEM,
            )
        if not isinstance(extension, MonomialExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.UNSUPPORTED_EXTENSION,
                f"level {index} is not a monomial extension",
                problem=_SCALAR_RDE_PROBLEM,
            )
        if extension.family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            return _unsupported(
                derivation,
                TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
                f"level {index} has unsupported monomial family "
                f"{extension.family.value}",
                problem=_SCALAR_RDE_PROBLEM,
            )
        if (
            base.parent_field == QQ_I
            and extension.family is MonomialFamily.HYPERTANGENT
        ):
            return _unsupported(
                derivation,
                TowerAdmissionReason.HYPERTANGENT_RDE_BASE,
                f"level {index} is hypertangent, but scalar RDE dispatch "
                "normalizes hypertangent levels only over QQ(x)",
                problem=_SCALAR_RDE_PROBLEM,
            )
        rejection = _constant_field_rejection(
            derivation,
            index,
            problem=_SCALAR_RDE_PROBLEM,
        )
        if rejection is not None:
            return rejection

    return _accepted(
        derivation,
        problem=_SCALAR_RDE_PROBLEM,
        detail=(
            "the active tower is C(x) for an effective finite algebraic C, "
            "or a certified recursive P/E/T prefix over QQ(x), or a "
            "certified recursive P/E prefix over QQ(i)(x)"
        ),
    )


__all__ = [
    "TowerAdmission",
    "TowerAdmissionOutcome",
    "TowerAdmissionReason",
    "assess_elementary_admission",
    "assess_scalar_rde_admission",
]
