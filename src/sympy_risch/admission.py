r"""Operation admission for certified P/E/T cells.

Structural validity and algorithmic support are different questions.  This
module implements the latter as a small matrix indexed by an active tower
prefix and a :class:`RischOperation`.  An accepted result means that the
selected operation is total on its declared input cell; it does not predict
whether a particular equation has a solution.  An unsupported result is an
algorithm boundary, never a proof of nonexistence.

The elementary dispatcher has a few explicitly checked routes outside the
classical cell, notably terminal simple radicals.  Those routes retain their
own narrower contracts rather than being advertised here as total algebraic
integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.polytools import Poly

from .constant_fields import (
    finite_constant_field_profile,
    is_supported_rational_constant_field,
)
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


class TowerAdmissionOutcome(Enum):
    """Whether a tower is inside the declared algorithmic contract."""

    ACCEPTED = "accepted"
    UNSUPPORTED = "unsupported"


class RischOperation(Enum):
    """A public solver operation with an independently checkable input cell."""

    EXACT_DERIVATIVE = "exact_derivative"
    PARAMETRIC_LOGARITHMIC_DERIVATIVE = "parametric_logarithmic_derivative"
    SCALAR_RDE = "scalar_rde"
    PARAMETRIC_RDE = "parametric_rde"
    LIMITED_INTEGRATION = "limited_integration"
    COUPLED_RDE = "coupled_rde"
    ELEMENTARY_INTEGRATION = "elementary_integration"


class TowerAdmissionReason(Enum):
    """Stable reason for rejecting a declared tower-algorithm request."""

    NO_RATIONAL_BASE = "no_rational_base"
    BASE_FIELD = "base_field"
    BASE_DERIVATIVE = "base_derivative"
    ALGEBRAIC_EXTENSION = "algebraic_extension"
    CONSTANT_FIELD_UNKNOWN = "constant_field_unknown"
    CONSTANT_FIELD_ENLARGED = "constant_field_enlarged"
    COUPLED_RDE_BASE = "coupled_rde_base"
    HYPERBOLIC_LOWER_LEVEL = "hyperbolic_lower_level"
    HYPERTANGENT_REAL_CONSTANTS = "hypertangent_real_constants"
    UNSUPPORTED_MONOMIAL_FAMILY = "unsupported_monomial_family"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


@dataclass(frozen=True, slots=True)
class TowerAdmission:
    r"""Operation support for one active tower prefix.

    ``tower`` and ``level`` describe the derivation on which ``operation``
    would run. Levels above ``level`` are inactive and deliberately have no
    effect on the classification.
    """

    operation: RischOperation
    tower: DifferentialTower
    level: int
    outcome: TowerAdmissionOutcome
    reason: TowerAdmissionReason | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, RischOperation):
            raise TypeError("an admission result needs a RischOperation")
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
    """Whether ``extension`` is the standard base ``C(x)``, ``Dx = 1``."""
    constants = extension.parent_field
    return is_supported_rational_constant_field(constants) and is_rational_base(
        extension
    )


def is_rational_base(extension: MonomialExtension) -> bool:
    """Whether ``extension`` is ``C(x)``, ``Dx = 1`` over an exact field."""
    constants = extension.parent_field
    return extension.generator_derivative == Poly(
        1, extension.generator, domain=constants
    )


def _accepted(
    derivation: Derivation,
    *,
    operation: RischOperation,
    detail: str,
) -> TowerAdmission:
    return TowerAdmission(
        operation,
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
    operation: RischOperation,
) -> TowerAdmission:
    return TowerAdmission(
        operation,
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
    operation: RischOperation,
) -> TowerAdmission | None:
    """Reject a level whose independent replay is not preserved."""
    evidence = replay_constant_field(Derivation(derivation.tower, index))
    if evidence.status is ConstantFieldStatus.ENLARGED:
        return _unsupported(
            derivation,
            TowerAdmissionReason.CONSTANT_FIELD_ENLARGED,
            f"independent certification proves that level {index} enlarges "
            "the constant field",
            operation=operation,
        )
    if evidence.status is ConstantFieldStatus.UNKNOWN:
        suffix = f": {evidence.detail}" if evidence.detail else ""
        return _unsupported(
            derivation,
            TowerAdmissionReason.CONSTANT_FIELD_UNKNOWN,
            f"independent certification is unavailable for level {index}{suffix}",
            operation=operation,
        )
    return None


def assess_risch_operation(
    derivation: Derivation,
    operation: RischOperation,
) -> TowerAdmission:
    r"""Classify one active tower prefix for a selected Risch operation.

    The classical accepted cell has a standard rational base ``C(x)``,
    ``Dx=1``, over an effective exact rational constant field ``C``, followed
    by a replay-certified primitive, hyperexponential, or hypertangent prefix.
    Hypertangent and real coupled operations require provably real constants.

    Elementary integration additionally accepts an exact hyperbolic Möbius law
    at the active top level because that dispatcher owns a complete immutable
    normalization to the classical cell.  Other operations do not silently
    inherit that coordinate change.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("Risch operation admission requires a Derivation")
    if not isinstance(operation, RischOperation):
        raise TypeError("Risch operation admission requires a RischOperation")

    if derivation.level < 0:
        return _unsupported(
            derivation,
            TowerAdmissionReason.NO_RATIONAL_BASE,
            f"{operation.value} requires a standard rational C(x) base",
            operation=operation,
        )

    active_levels = derivation.tower.levels[: derivation.level + 1]
    base = active_levels[0]
    if isinstance(base, AlgebraicExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.ALGEBRAIC_EXTENSION,
            f"level 0 is algebraic; it is outside the classical {operation.value} cell",
            operation=operation,
        )
    if not isinstance(base, MonomialExtension):
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_EXTENSION,
            "level 0 is not a monomial extension",
            operation=operation,
        )
    constants = base.parent_field
    constant_profile = finite_constant_field_profile(constants)
    if not is_supported_rational_constant_field(constants):
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_FIELD,
            "the rational base must be over a supported finite algebraic "
            "constant field or a rational-function extension of one",
            operation=operation,
        )
    if not is_standard_rational_base(base):
        return _unsupported(
            derivation,
            TowerAdmissionReason.BASE_DERIVATIVE,
            "the rational base must have derivative Dx = 1",
            operation=operation,
        )
    if operation is RischOperation.COUPLED_RDE and (
        constant_profile is None or not constant_profile.is_real
    ):
        return _unsupported(
            derivation,
            TowerAdmissionReason.COUPLED_RDE_BASE,
            "real coupled RDE solving requires a provably real finite "
            "algebraic constant field",
            operation=operation,
        )

    for index, extension in enumerate(active_levels[1:], start=1):
        if isinstance(extension, AlgebraicExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.ALGEBRAIC_EXTENSION,
                f"level {index} is algebraic",
                operation=operation,
            )
        if not isinstance(extension, MonomialExtension):
            return _unsupported(
                derivation,
                TowerAdmissionReason.UNSUPPORTED_EXTENSION,
                f"level {index} is not a monomial extension",
                operation=operation,
            )
        if extension.family in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            if extension.family is MonomialFamily.HYPERTANGENT and (
                constant_profile is None or not constant_profile.is_real
            ):
                return _unsupported(
                    derivation,
                    TowerAdmissionReason.HYPERTANGENT_REAL_CONSTANTS,
                    f"level {index} is hypertangent, but {operation.value} "
                    "requires a provably real finite algebraic "
                    "constant field",
                    operation=operation,
                )
            rejection = _constant_field_rejection(
                derivation,
                index,
                operation=operation,
            )
            if rejection is not None:
                return rejection
            continue
        if is_hyperbolic_mobius_law(extension):
            rejection = _constant_field_rejection(
                derivation,
                index,
                operation=operation,
            )
            if rejection is not None:
                return rejection
            if (
                operation is RischOperation.ELEMENTARY_INTEGRATION
                and index == derivation.level
            ):
                continue
            if operation is RischOperation.ELEMENTARY_INTEGRATION:
                return _unsupported(
                    derivation,
                    TowerAdmissionReason.HYPERBOLIC_LOWER_LEVEL,
                    f"level {index} is hyperbolic, but Möbius normalization is "
                    "implemented only for the active top level",
                    operation=operation,
                )
        return _unsupported(
            derivation,
            TowerAdmissionReason.UNSUPPORTED_MONOMIAL_FAMILY,
            f"level {index} has monomial family {extension.family.value}, "
            f"which is outside the complete {operation.value} cell",
            operation=operation,
        )

    return _accepted(
        derivation,
        operation=operation,
        detail=(
            f"{operation.value} is total on this replay-certified "
            "P/E/T tower prefix"
            + (
                ", including its top-level hyperbolic Möbius normalization"
                if isinstance(active_levels[-1], MonomialExtension)
                and is_hyperbolic_mobius_law(active_levels[-1])
                else ""
            )
        ),
    )


def assess_elementary_admission(derivation: Derivation) -> TowerAdmission:
    r"""Classify a tower prefix for total recursive elementary integration."""
    return assess_risch_operation(
        derivation,
        RischOperation.ELEMENTARY_INTEGRATION,
    )


def assess_scalar_rde_admission(derivation: Derivation) -> TowerAdmission:
    r"""Classify a tower prefix for total recursive scalar RDE solving."""
    return assess_risch_operation(
        derivation,
        RischOperation.SCALAR_RDE,
    )


__all__ = [
    "RischOperation",
    "TowerAdmission",
    "TowerAdmissionOutcome",
    "TowerAdmissionReason",
    "assess_elementary_admission",
    "assess_risch_operation",
    "assess_scalar_rde_admission",
    "is_rational_base",
    "is_standard_rational_base",
]
