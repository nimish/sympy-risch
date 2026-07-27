r"""Complete principal-divisor decisions for algebraic function fields.

For a divisor ``D`` of degree zero, ``D`` is principal exactly when the
complete Riemann--Roch space ``L(-D)`` is nonzero.  Indeed, a nonzero
``f in L(-D)`` has ``div(f) - D`` effective, and this divisor has degree
zero, hence is zero.  A complete empty space is therefore a genuine
non-principality certificate rather than a failed bounded search.

This module is deliberately neutral function-field infrastructure.  Risch
callers establish any differential-constant-field hypotheses before using a
decision in an elementary-integrability argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.agca.extensions import ExtensionElement

from ..algebraic._field import _extension_element
from .function_field import AlgebraicDivisor, AlgebraicFunctionField
from .riemann_roch import (
    RiemannRochSpace,
    RiemannRochSpaceStatus,
    compute_riemann_roch_space,
)


class AlgebraicPrincipalityStatus(Enum):
    """Outcome of a complete principality decision when supported."""

    PRINCIPAL = "principal"
    NONPRINCIPAL = "nonprincipal"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicPrincipalityDecision:
    r"""A verified answer to whether ``div(f) = divisor`` has a solution.

    A degree-zero negative answer carries the certified empty space
    ``L(-divisor)``.  A nonzero-degree negative answer follows directly from
    degree preservation of principal divisors.  An unsupported answer retains
    the function-field Riemann--Roch boundary that prevented a decision.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    status: AlgebraicPrincipalityStatus
    witness: ExtensionElement | None
    riemann_roch_space: RiemannRochSpace | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a principality decision needs a function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a principality decision needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(self.status, AlgebraicPrincipalityStatus):
            raise TypeError("the principality status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a principality decision needs a detail")

        witness = (
            None
            if self.witness is None
            else _extension_element(self.model.extension, self.witness)
        )
        object.__setattr__(self, "witness", witness)
        space = self.riemann_roch_space
        if space is not None and (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != -self.divisor
        ):
            raise ValueError("the Riemann--Roch space belongs to another divisor")

        if self.status is AlgebraicPrincipalityStatus.PRINCIPAL:
            if witness is None or witness == self.model.extension.field.zero:
                raise ValueError("a principal decision needs a nonzero witness")
            if self.divisor.degree != 0:
                raise ValueError("a principal divisor must have degree zero")
            if space is None or space.status is not RiemannRochSpaceStatus.CERTIFIED:
                raise ValueError(
                    "a principal decision needs a complete Riemann--Roch space"
                )
            if witness not in space.basis:
                raise ValueError("a principal witness must come from L(-D)")
            if self.model.principal_divisor(witness) != self.divisor:
                raise ValueError("the principal witness has another divisor")
            return

        if self.status is AlgebraicPrincipalityStatus.NONPRINCIPAL:
            if witness is not None:
                raise ValueError("a nonprincipal decision cannot carry a witness")
            if self.divisor.degree != 0:
                if space is not None:
                    raise ValueError(
                        "the degree obstruction needs no Riemann--Roch space"
                    )
                return
            if (
                space is None
                or space.status is not RiemannRochSpaceStatus.CERTIFIED
                or space.basis
            ):
                raise ValueError(
                    "a degree-zero nonprincipal decision needs a certified empty L(-D)"
                )
            return

        if self.status is AlgebraicPrincipalityStatus.UNSUPPORTED:
            if witness is not None:
                raise ValueError("an unsupported decision cannot carry a witness")
            if space is None or space.status is not RiemannRochSpaceStatus.UNSUPPORTED:
                raise ValueError(
                    "an unsupported decision needs its Riemann--Roch boundary"
                )
            return

        raise ValueError("a principality decision has an unknown status")

    @property
    def principal(self) -> bool:
        """Whether this decision carries a verified principal witness."""
        return self.status is AlgebraicPrincipalityStatus.PRINCIPAL

    @property
    def complete_negative(self) -> bool:
        """Whether this is a proof that the divisor is nonprincipal."""
        return self.status is AlgebraicPrincipalityStatus.NONPRINCIPAL


def decide_algebraic_principality(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> AlgebraicPrincipalityDecision:
    r"""Decide whether ``div(f) = divisor`` in an algebraic function field.

    The decision is complete whenever the underlying complete Riemann--Roch
    computation is available.  No bound or Risch-specific constant-field
    analysis is part of this reusable function-field API.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("principality needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("principality needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")

    if divisor.degree != 0:
        return AlgebraicPrincipalityDecision(
            model,
            divisor,
            AlgebraicPrincipalityStatus.NONPRINCIPAL,
            None,
            None,
            "a principal divisor must have degree zero",
        )

    space = compute_riemann_roch_space(model, -divisor)
    if space.status is RiemannRochSpaceStatus.UNSUPPORTED:
        return AlgebraicPrincipalityDecision(
            model,
            divisor,
            AlgebraicPrincipalityStatus.UNSUPPORTED,
            None,
            space,
            "principality requires a complete L(-D): " + space.detail,
        )
    if not space.basis:
        return AlgebraicPrincipalityDecision(
            model,
            divisor,
            AlgebraicPrincipalityStatus.NONPRINCIPAL,
            None,
            space,
            "the complete space L(-D) is zero, so D is nonprincipal",
        )

    witness = space.basis[0]
    return AlgebraicPrincipalityDecision(
        model,
        divisor,
        AlgebraicPrincipalityStatus.PRINCIPAL,
        witness,
        space,
        "a nonzero element of complete L(-D) has principal divisor D",
    )
