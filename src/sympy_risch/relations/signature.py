r"""Certified fundamental directions for supported monomial towers.

This module owns the immutable direction models, the signature result, and the
builder that certifies a P/E/T tower prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ_I
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..constant_fields import (
    finite_constant_field_profile,
    is_supported_finite_constant_field,
)
from ..tower import (
    AlgebraicExtension,
    ConstantFieldStatus,
    Derivation,
    MonomialExtension,
    MonomialFamily,
)


class FundamentalSignatureOutcome(Enum):
    """Whether a tower has a certified P/E/T fundamental signature."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class FundamentalSignatureReason(Enum):
    """Stable reason a tower lacks a certified fundamental signature."""

    NO_RATIONAL_BASE = "no_rational_base"
    BASE_FIELD = "base_field"
    BASE_DERIVATIVE = "base_derivative"
    ALGEBRAIC_EXTENSION = "algebraic_extension"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    CONSTANT_FIELD_UNKNOWN = "constant_field_unknown"
    CONSTANT_FIELD_ENLARGED = "constant_field_enlarged"
    CONSTANT_FIELD_PROVENANCE = "constant_field_provenance"
    UNSUPPORTED_MONOMIAL_FAMILY = "unsupported_monomial_family"


@dataclass(frozen=True, slots=True)
class PrimitiveFundamentalDirection:
    r"""One primitive direction ``Dt`` in its native parent field."""

    level: int
    extension: MonomialExtension
    derivative: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.PRIMITIVE:
            raise ValueError("a primitive direction needs a primitive extension")
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(0),
        )
        if self.derivative != expected:
            raise ValueError("a primitive direction must equal Dt")

    @property
    def generator(self) -> Symbol:
        """Return the primitive generator."""
        return self.extension.generator


@dataclass(frozen=True, slots=True)
class HyperexponentialFundamentalDirection:
    r"""One exponential direction ``Dt/t`` in its native parent field."""

    level: int
    extension: MonomialExtension
    logarithmic_derivative: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.HYPEREXPONENTIAL:
            raise ValueError(
                "a hyperexponential direction needs a hyperexponential extension"
            )
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(1),
        )
        if self.logarithmic_derivative != expected:
            raise ValueError("a hyperexponential direction must equal Dt/t")

    @property
    def generator(self) -> Symbol:
        """Return the hyperexponential generator."""
        return self.extension.generator


@dataclass(frozen=True, slots=True)
class HypertangentFundamentalDirection:
    r"""One real tangent direction ``eta`` with ``Dt=eta*(1+t**2)``.

    ``cayley_coefficient`` remains in the real source parent field.  Its
    independently verified Gaussian image is the logarithmic derivative
    ``2*i*eta`` of the Cayley coordinate.
    """

    level: int
    extension: MonomialExtension
    cayley_coefficient: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.HYPERTANGENT:
            raise ValueError("a hypertangent direction needs a hypertangent extension")
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(2),
        )
        if self.cayley_coefficient != expected:
            raise ValueError("a hypertangent direction must equal eta")

    @property
    def generator(self) -> Symbol:
        """Return the hypertangent generator."""
        return self.extension.generator


FundamentalDirection = (
    PrimitiveFundamentalDirection
    | HyperexponentialFundamentalDirection
    | HypertangentFundamentalDirection
)


@dataclass(frozen=True, slots=True)
class FundamentalSignature:
    r"""Certified structural directions for one immutable P/E/T tower prefix.

    The signature records the standard rational base as its level-zero
    primitive direction, then each active primitive, hyperexponential, or
    hypertangent extension in order.  It is a structure-theorem input, not a
    search over all Darboux or special polynomials.  In particular, a
    hypertangent source direction is real data; its Cayley image is checked
    separately by :mod:`sympy_risch.gaussian`.
    """

    derivation: Derivation
    outcome: FundamentalSignatureOutcome
    reason: FundamentalSignatureReason | None
    detail: str
    certified_constant_levels: tuple[int, ...]
    primitive_directions: tuple[PrimitiveFundamentalDirection, ...]
    hyperexponential_directions: tuple[HyperexponentialFundamentalDirection, ...]
    hypertangent_directions: tuple[HypertangentFundamentalDirection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.derivation, Derivation):
            raise TypeError("a fundamental signature requires a Derivation")
        if not isinstance(self.outcome, FundamentalSignatureOutcome):
            raise TypeError(
                "a fundamental signature outcome must be a FundamentalSignatureOutcome"
            )
        if self.reason is not None and not isinstance(
            self.reason,
            FundamentalSignatureReason,
        ):
            raise TypeError(
                "a fundamental signature reason must be a FundamentalSignatureReason"
            )
        if not isinstance(self.detail, str):
            raise TypeError("a fundamental signature detail must be a string")
        direction_groups = (
            (self.primitive_directions, PrimitiveFundamentalDirection),
            (
                self.hyperexponential_directions,
                HyperexponentialFundamentalDirection,
            ),
            (self.hypertangent_directions, HypertangentFundamentalDirection),
        )
        for directions, direction_type in direction_groups:
            if not isinstance(directions, tuple) or not all(
                isinstance(direction, direction_type) for direction in directions
            ):
                raise TypeError(
                    "fundamental signature directions must be immutable "
                    "direction tuples"
                )
        if not isinstance(self.certified_constant_levels, tuple) or not all(
            isinstance(level, int) for level in self.certified_constant_levels
        ):
            raise TypeError(
                "fundamental signature certified levels must be an integer tuple"
            )
        if self.outcome is FundamentalSignatureOutcome.SUPPORTED:
            if self.reason is not None:
                raise ValueError("a supported signature has no rejection reason")
            if self.derivation.level < 0:
                raise ValueError("a supported signature requires a rational base level")
            expected_levels = tuple(range(self.derivation.level + 1))
            if self.certified_constant_levels != expected_levels:
                raise ValueError(
                    "a supported signature must certify every active tower level"
                )
            if (
                tuple(direction.level for direction in self.directions)
                != expected_levels
            ):
                raise ValueError(
                    "a supported signature needs one ordered direction per level"
                )
            base = self.derivation.tower.levels[0]
            if not isinstance(
                base, MonomialExtension
            ) or not _is_standard_relation_base(base):
                raise ValueError(
                    "a supported signature requires C(x) with Dx = 1 over an "
                    "effective finite constant field C"
                )
            profile = finite_constant_field_profile(base.parent_field)
            if profile is None:
                raise ValueError(
                    "a supported signature requires effective finite constants",
                )
            if (
                self.hypertangent_directions
                and not profile.is_real
                and base.parent_field != QQ_I
            ):
                raise ValueError(
                    "hypertangent directions require real constants or QQ_I",
                )
            for direction in self.directions:
                extension = self.derivation.tower.levels[direction.level]
                if direction.extension is not extension:
                    raise ValueError(
                        "a fundamental direction must retain its exact tower extension"
                    )
            if self.derivation.level > 0:
                # A supported signature is itself a structural theorem
                # certificate, so it cannot be authorized by advisory
                # extension metadata. Reuse the prefix replay rather than
                # independently scanning every direction.
                from ..constant_proofs import first_nonpreserved_constant_level

                evidence = first_nonpreserved_constant_level(self.derivation)
                if evidence is not None:
                    raise ValueError(
                        "a non-base fundamental direction requires replayed "
                        "constant preservation"
                    )
        else:
            if self.reason is None:
                raise ValueError("an unsupported signature needs a reason")
            if (
                self.certified_constant_levels
                or self.primitive_directions
                or self.hyperexponential_directions
                or self.hypertangent_directions
            ):
                raise ValueError(
                    "an unsupported signature must not expose partial directions"
                )

    @property
    def supported(self) -> bool:
        """Whether this is a certified P/E/T structural signature."""
        return self.outcome is FundamentalSignatureOutcome.SUPPORTED

    @property
    def unsupported(self) -> bool:
        """Whether the tower is outside this signature's theorem scope."""
        return self.outcome is FundamentalSignatureOutcome.UNSUPPORTED

    @property
    def directions(self) -> tuple[FundamentalDirection, ...]:
        """Return every source direction in ascending active-tower order."""
        return tuple(
            sorted(
                (
                    *self.primitive_directions,
                    *self.hyperexponential_directions,
                    *self.hypertangent_directions,
                ),
                key=lambda direction: direction.level,
            )
        )

    def direction_at(self, level: int) -> FundamentalDirection | None:
        """Return the exact source direction at one active tower level."""
        for direction in self.directions:
            if direction.level == level:
                return direction
        return None


def _unsupported_fundamental_signature(
    derivation: Derivation,
    reason: FundamentalSignatureReason,
    detail: str,
) -> FundamentalSignature:
    return FundamentalSignature(
        derivation,
        FundamentalSignatureOutcome.UNSUPPORTED,
        reason,
        detail,
        (),
        (),
        (),
        (),
    )


def _is_standard_relation_base(extension: MonomialExtension) -> bool:
    """Whether ``extension`` is a supported standard ``C(x)`` base."""
    return is_supported_finite_constant_field(
        extension.parent_field,
    ) and extension.generator_derivative == Poly(
        1, extension.generator, domain=extension.parent_field
    )


def build_fundamental_signature(derivation: Derivation) -> FundamentalSignature:
    r"""Describe the certified P/E/T directions of an active tower prefix.

    The construction replays the implemented constant-field criteria but never
    solves an input relation.  It reports ``UNSUPPORTED`` rather than a
    mathematical negative conclusion when a required structure-theorem
    hypothesis is absent.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("fundamental signatures require a Derivation")
    if derivation.level < 0:
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.NO_RATIONAL_BASE,
            "the constant field alone has no standard rational base",
        )

    active_levels = derivation.tower.levels[: derivation.level + 1]
    base = active_levels[0]
    if not isinstance(base, MonomialExtension):
        return _unsupported_fundamental_signature(
            derivation,
            (
                FundamentalSignatureReason.ALGEBRAIC_EXTENSION
                if isinstance(base, AlgebraicExtension)
                else FundamentalSignatureReason.UNSUPPORTED_EXTENSION
            ),
            "level 0 is not a monomial standard rational base",
        )
    if not is_supported_finite_constant_field(base.parent_field):
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.BASE_FIELD,
            "the supported relation base must be over an effective finite "
            "constant field over QQ",
        )
    if not _is_standard_relation_base(base):
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.BASE_DERIVATIVE,
            "the supported relation base must have Dx = 1",
        )

    primitive_directions: list[PrimitiveFundamentalDirection] = [
        PrimitiveFundamentalDirection(
            0,
            base,
            base.parent_field.convert(base.generator_derivative.nth(0)),
        ),
    ]
    hyperexponential_directions: list[HyperexponentialFundamentalDirection] = []
    hypertangent_directions: list[HypertangentFundamentalDirection] = []
    # Replay the full prefix once. The result cache is bottom-up and
    # identity-safe, so later factories and signatures reuse this check
    # instead of each rescanning every lower level.
    from ..constant_proofs import first_nonpreserved_constant_level

    constant_failure = first_nonpreserved_constant_level(derivation)
    profile = finite_constant_field_profile(base.parent_field)
    if profile is None:
        raise ValueError("the validated relation base lost its constant profile")

    for level, extension in enumerate(active_levels[1:], start=1):
        if not isinstance(extension, MonomialExtension):
            return _unsupported_fundamental_signature(
                derivation,
                (
                    FundamentalSignatureReason.ALGEBRAIC_EXTENSION
                    if isinstance(extension, AlgebraicExtension)
                    else FundamentalSignatureReason.UNSUPPORTED_EXTENSION
                ),
                f"level {level} is not a monomial extension",
            )
        if extension.family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
                f"level {level} has unsupported monomial family {extension.family.value}",
            )
        if (
            extension.family is MonomialFamily.HYPERTANGENT
            and not profile.is_real
            and base.parent_field != QQ_I
        ):
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
                "hypertangent fundamental directions require a provably real "
                "constant field or QQ_I",
            )
        if (
            constant_failure is not None
            and constant_failure.derivation.level == level
            and constant_failure.status is ConstantFieldStatus.UNKNOWN
        ):
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.CONSTANT_FIELD_UNKNOWN,
                f"level {level} has no replayed preserved-constants proof",
            )
        if (
            constant_failure is not None
            and constant_failure.derivation.level == level
            and constant_failure.status is ConstantFieldStatus.ENLARGED
        ):
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.CONSTANT_FIELD_ENLARGED,
                f"level {level} enlarges the constant field under replay",
            )
        if extension.family is MonomialFamily.PRIMITIVE:
            primitive_directions.append(
                PrimitiveFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(0),
                    ),
                ),
            )
            continue
        if extension.family is MonomialFamily.HYPEREXPONENTIAL:
            hyperexponential_directions.append(
                HyperexponentialFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(1),
                    ),
                ),
            )
            continue
        if extension.family is MonomialFamily.HYPERTANGENT:
            hypertangent_directions.append(
                HypertangentFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(2),
                    ),
                ),
            )
            continue
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
            f"level {level} has unsupported monomial family {extension.family.value}",
        )

    return FundamentalSignature(
        derivation,
        FundamentalSignatureOutcome.SUPPORTED,
        None,
        "the active tower has certified primitive, hyperexponential, and "
        "hypertangent fundamental directions",
        tuple(range(derivation.level + 1)),
        tuple(primitive_directions),
        tuple(hyperexponential_directions),
        tuple(hypertangent_directions),
    )
