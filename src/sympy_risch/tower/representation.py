r"""Native SymPy representation checks for conceptual tower prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed

from .core import (
    AlgebraicExtension,
    DifferentialExtension,
    DifferentialTower,
    MonomialExtension,
    _coerce,
    _contains_finite_extension,
    _require_exact_characteristic_zero_field,
)


class TowerRepresentationOutcome(Enum):
    r"""Whether a conceptual algebraic/monomial prefix has a native field.

    ``REPRESENTABLE`` means that the current SymPy domain backend can build
    the prefix and perform one exact coercion and derivation at its top level.
    ``BACKEND_LIMITED`` is deliberately distinct from ``UNSUPPORTED``: the
    former is a mathematically meaningful ordering for which the present
    native-domain representation is unavailable, whereas the latter is not a
    well-formed algebraic/monomial prefix for this validation service.
    """

    REPRESENTABLE = "representable"
    BACKEND_LIMITED = "backend_limited"
    UNSUPPORTED = "unsupported"


class TowerRepresentationReason(Enum):
    """Stable explanation for a non-representable tower-prefix proposal."""

    MONOMIAL_OVER_ALGEBRAIC = "monomial_over_algebraic"
    NATIVE_OPERATION = "native_operation"
    PARENT_FIELD_MISMATCH = "parent_field_mismatch"
    DUPLICATE_GENERATOR = "duplicate_generator"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


@dataclass(frozen=True, slots=True)
class TowerRepresentationProposal:
    r"""One immutable conceptual prefix to check against SymPy's backend.

    This is intentionally only tower metadata: it neither stores custom
    elements nor tries to emulate a field that SymPy cannot represent. In
    particular, it can describe a proposed monomial level over an algebraic
    one, so that :func:`analyze_tower_representation` can return a structured
    backend limitation instead of requiring callers to interpret an exception
    from :meth:`DifferentialTower.extend`.
    """

    constants: Domain
    levels: tuple[DifferentialExtension, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_characteristic_zero_field(
            self.constants,
            role="the constant domain",
        )
        try:
            levels = tuple(self.levels)
        except TypeError as exc:
            raise TypeError("proposal levels must be an iterable") from exc
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_tower(
        cls,
        tower: DifferentialTower,
        level: int | None = None,
    ) -> TowerRepresentationProposal:
        """Copy one active prefix of an existing immutable tower.

        The returned proposal may be analyzed independently of levels above
        ``level``. It is useful to later solvers that only need a lower
        representable prefix while a broader conceptual tower is still being
        assembled.
        """
        if not isinstance(tower, DifferentialTower):
            raise TypeError(
                "a tower representation proposal requires a DifferentialTower"
            )
        if level is None:
            level = tower.top
        if not isinstance(level, int):
            raise TypeError("a tower representation level must be an integer")
        if not -1 <= level <= tower.top:
            raise IndexError(f"tower level {level} is out of range")
        return cls(tower.constants, tower.levels[: level + 1])


@dataclass(frozen=True, slots=True)
class TowerRepresentationAnalysis:
    r"""Checked native-SymPy representation result for a tower prefix.

    A representable result contains the reconstructed :attr:`native_tower`.
    A non-representable result never contains a partial tower. This keeps a
    backend limitation separate from mathematical integration decisions and
    prevents callers from accidentally dispatching against a partially built
    native field.
    """

    proposal: TowerRepresentationProposal
    outcome: TowerRepresentationOutcome
    reason: TowerRepresentationReason | None
    detail: str
    failed_level: int | None
    native_tower: DifferentialTower | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, TowerRepresentationProposal):
            raise TypeError(
                "a representation analysis needs a TowerRepresentationProposal"
            )
        if not isinstance(self.outcome, TowerRepresentationOutcome):
            raise TypeError(
                "a representation analysis needs a TowerRepresentationOutcome"
            )
        if self.reason is not None and not isinstance(
            self.reason,
            TowerRepresentationReason,
        ):
            raise TypeError(
                "a representation analysis reason must be a TowerRepresentationReason"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a representation analysis needs nonempty detail")
        if self.failed_level is not None and not isinstance(self.failed_level, int):
            raise TypeError("a failed representation level must be an integer")

        if self.outcome is TowerRepresentationOutcome.REPRESENTABLE:
            if self.reason is not None:
                raise ValueError(
                    "a representable tower must not carry a failure reason"
                )
            if self.failed_level is not None:
                raise ValueError("a representable tower must not carry a failed level")
            if not isinstance(self.native_tower, DifferentialTower):
                raise ValueError("a representable tower needs its native tower")
            if (
                self.native_tower.constants != self.proposal.constants
                or self.native_tower.levels != self.proposal.levels
            ):
                raise ValueError(
                    "a native tower must exactly represent the proposed prefix"
                )
            _exercise_native_tower(self.native_tower)
            return

        if self.reason is None:
            raise ValueError("a non-representable tower needs a failure reason")
        if self.native_tower is not None:
            raise ValueError("a non-representable tower must not carry a partial tower")
        if self.failed_level is None:
            raise ValueError("a non-representable tower needs a failed level")
        if not 0 <= self.failed_level < len(self.proposal.levels):
            raise IndexError(f"tower level {self.failed_level} is out of range")

        backend_reasons = {
            TowerRepresentationReason.MONOMIAL_OVER_ALGEBRAIC,
            TowerRepresentationReason.NATIVE_OPERATION,
        }
        if self.outcome is TowerRepresentationOutcome.BACKEND_LIMITED:
            if self.reason not in backend_reasons:
                raise ValueError(
                    "a backend-limited tower needs a backend limitation reason"
                )
            return
        if self.outcome is TowerRepresentationOutcome.UNSUPPORTED:
            if self.reason in backend_reasons:
                raise ValueError(
                    "a backend limitation must use the backend-limited outcome"
                )
            return
        raise AssertionError("unhandled tower representation outcome")

    @property
    def representable(self) -> bool:
        """Whether a complete native tower is available."""
        return self.outcome is TowerRepresentationOutcome.REPRESENTABLE

    @property
    def backend_limited(self) -> bool:
        """Whether SymPy's native domain backend blocks this valid ordering."""
        return self.outcome is TowerRepresentationOutcome.BACKEND_LIMITED

    @property
    def unsupported(self) -> bool:
        """Whether the proposal is outside this algebraic/monomial shape."""
        return self.outcome is TowerRepresentationOutcome.UNSUPPORTED


def _nonrepresentable_representation(
    proposal: TowerRepresentationProposal,
    outcome: TowerRepresentationOutcome,
    reason: TowerRepresentationReason,
    failed_level: int,
    detail: str,
) -> TowerRepresentationAnalysis:
    """Build a checked non-representable result without a partial tower."""
    return TowerRepresentationAnalysis(
        proposal,
        outcome,
        reason,
        detail,
        failed_level,
    )


def _exercise_native_tower(tower: DifferentialTower) -> None:
    """Exercise the top native coercion and derivation without integration.

    Building a ``FiniteExtension`` alone is not enough evidence for the
    algebraic-over-monomial ordering: the generator must also embed into the
    top field and its implicit derivative must traverse the lower tower. One
    generator derivative is a small, exact backend smoke test; it deliberately
    performs no Hermite reduction or integration dispatch.
    """
    if tower.top < 0:
        return
    extension = tower.levels[tower.top]
    generator = _coerce(tower.field(), extension.generator)
    _coerce(tower.field(), tower.derivation()(generator))


def analyze_tower_representation(
    proposal: TowerRepresentationProposal,
) -> TowerRepresentationAnalysis:
    r"""Classify the current native representation of one mixed prefix.

    The presently admitted physical orderings are any finite monomial prefix
    and an optional single terminal algebraic extension over that prefix.
    SymPy cannot yet form a rational-function field over a ``FiniteExtension``
    or nest finite extensions. Those mathematically legitimate cases return
    :attr:`TowerRepresentationOutcome.BACKEND_LIMITED`, while malformed or
    unknown tower shapes return ``UNSUPPORTED``. Neither non-representable
    outcome is an integration or elementary-function decision.
    """
    if not isinstance(proposal, TowerRepresentationProposal):
        raise TypeError(
            "tower representation analysis requires a TowerRepresentationProposal"
        )

    expected_parent = proposal.constants
    generators: set[Symbol] = set()

    for level, extension in enumerate(proposal.levels):
        if not isinstance(extension, (MonomialExtension, AlgebraicExtension)):
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.UNSUPPORTED_EXTENSION,
                level,
                f"level {level} is not a native monomial or algebraic extension",
            )
        if extension.parent_field != expected_parent:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.PARENT_FIELD_MISMATCH,
                level,
                f"level {level} does not use the preceding native field as parent",
            )
        if extension.generator in generators:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.DUPLICATE_GENERATOR,
                level,
                f"level {level} duplicates generator {extension.generator}",
            )

        generators.add(extension.generator)
        if isinstance(extension, MonomialExtension) and _contains_finite_extension(
            expected_parent
        ):
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.BACKEND_LIMITED,
                TowerRepresentationReason.MONOMIAL_OVER_ALGEBRAIC,
                level,
                "SymPy cannot form a native monomial field over an "
                "algebraic finite-extension parent",
            )
        try:
            expected_parent = extension.field
        except (CoercionFailed, NotImplementedError) as exc:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.BACKEND_LIMITED,
                TowerRepresentationReason.NATIVE_OPERATION,
                level,
                f"SymPy could not construct native level {level}: {exc}",
            )

    try:
        native_tower = DifferentialTower(proposal.constants, proposal.levels)
        return TowerRepresentationAnalysis(
            proposal,
            TowerRepresentationOutcome.REPRESENTABLE,
            None,
            "SymPy represents the complete prefix and its top generator coerces "
            "and differentiates exactly",
            None,
            native_tower,
        )
    except (CoercionFailed, NotImplementedError) as exc:
        if not proposal.levels:
            raise
        return _nonrepresentable_representation(
            proposal,
            TowerRepresentationOutcome.BACKEND_LIMITED,
            TowerRepresentationReason.NATIVE_OPERATION,
            len(proposal.levels) - 1,
            f"SymPy could not validate the native tower: {exc}",
        )


__all__ = [
    "TowerRepresentationAnalysis",
    "TowerRepresentationOutcome",
    "TowerRepresentationProposal",
    "TowerRepresentationReason",
    "analyze_tower_representation",
]
