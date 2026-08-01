r"""Complete principal-divisor decisions for algebraic function fields.

For a divisor ``D`` of degree zero, ``D`` is principal exactly when its class
in ``Pic^0`` is zero.  The normal route reduces successive multiples of the
primitive coefficient divisor.  If ``D = c*D0``, only the first ``c``
multiples matter: the first positive order of ``[D0]`` divides ``c`` exactly
when ``D`` is principal.  This keeps every Riemann--Roch problem bounded even
when the final witness has high degree.

Direct computation of ``L(-D)`` remains the complete fallback when reduced
class arithmetic is unavailable.

This module is deliberately neutral function-field infrastructure.  Risch
callers establish any differential-constant-field hypotheses before using a
decision in an elementary-integrability argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import ClassVar, TypeAlias

from sympy.polys.agca.extensions import ExtensionElement

from ..algebraic._field import _extension_element
from .divisor_classes import (
    AlgebraicTorsionSearch,
    AlgebraicTorsionStatus,
    search_algebraic_divisor_torsion,
)
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


def _divisor_content(divisor: AlgebraicDivisor) -> int:
    """Greatest common divisor of the nonzero divisor coefficients."""
    content = 0
    for _place, coefficient in divisor.terms:
        content = gcd(content, abs(coefficient))
    return content


def _primitive_divisor(divisor: AlgebraicDivisor) -> AlgebraicDivisor:
    """Divide a nonzero divisor by its positive coefficient content."""
    content = _divisor_content(divisor)
    if content == 0:
        raise ValueError("the zero divisor has no primitive direction")
    return AlgebraicDivisor(
        divisor.model,
        tuple((place, coefficient // content) for place, coefficient in divisor.terms),
    )


def _torsion_search_matches_divisor(
    divisor: AlgebraicDivisor,
    search: AlgebraicTorsionSearch | None,
) -> bool:
    """Whether ``search`` covers the primitive class through the needed bound."""
    if search is None:
        return False
    content = _divisor_content(divisor)
    return (
        content > 0
        and search.divisor == _primitive_divisor(divisor)
        and search.max_order >= content
    )


def _torsion_search_proves_principal(
    divisor: AlgebraicDivisor,
    witness: ExtensionElement,
    search: AlgebraicTorsionSearch | None,
) -> bool:
    """Check a reconstructed relation without refactoring its large witness."""
    if (
        not _torsion_search_matches_divisor(divisor, search)
        or search is None
        or search.status is not AlgebraicTorsionStatus.TORSION
        or search.order is None
    ):
        return False
    content = _divisor_content(divisor)
    return content % search.order == 0 and witness == search.witness_for_multiple(
        content
    )


def _torsion_search_proves_nonprincipal(
    divisor: AlgebraicDivisor,
    search: AlgebraicTorsionSearch | None,
) -> bool:
    """Use the primitive class order or exhausted content bound."""
    if not _torsion_search_matches_divisor(divisor, search) or search is None:
        return False
    if search.status is AlgebraicTorsionStatus.INCONCLUSIVE:
        return True
    if search.status is AlgebraicTorsionStatus.TORSION and search.order is not None:
        return _divisor_content(divisor) % search.order != 0
    return False


class _AlgebraicPrincipalityDecision:
    r"""A verified answer to whether ``div(f) = divisor`` has a solution.

    A degree-zero answer normally carries the reduced-class search proving
    whether the primitive class order divides the divisor's coefficient
    content. A nonzero-degree negative follows directly from degree
    preservation of principal divisors. An unsupported answer retains the
    exact arithmetic boundary that prevented a decision.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    detail: str
    status: ClassVar[AlgebraicPrincipalityStatus]

    def _validate_common(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a principality decision needs a function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a principality decision needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a principality decision needs a detail")

    @property
    def principal(self) -> bool:
        """Whether this decision carries a verified principal witness."""
        return isinstance(self, AlgebraicPrincipal)

    @property
    def complete_negative(self) -> bool:
        """Whether this is a proof that the divisor is nonprincipal."""
        return isinstance(self, AlgebraicNonprincipal)


@dataclass(frozen=True, slots=True)
class AlgebraicPrincipal(_AlgebraicPrincipalityDecision):
    """A verified nonzero witness whose principal divisor is the target."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    witness: ExtensionElement
    riemann_roch_space: RiemannRochSpace | None
    divisor_class_search: AlgebraicTorsionSearch | None
    detail: str
    status: ClassVar[AlgebraicPrincipalityStatus] = (
        AlgebraicPrincipalityStatus.PRINCIPAL
    )

    def __post_init__(self) -> None:
        self._validate_common()

        witness = _extension_element(self.model.extension, self.witness)
        object.__setattr__(self, "witness", witness)
        space = self.riemann_roch_space
        if space is not None and (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != -self.divisor
        ):
            raise ValueError("the Riemann--Roch space belongs to another divisor")
        search = self.divisor_class_search
        if search is not None and (
            not isinstance(search, AlgebraicTorsionSearch)
            or search.model is not self.model
        ):
            raise ValueError("the divisor-class search belongs to another model")

        if witness == self.model.extension.field.zero:
            raise ValueError("a principal decision needs a nonzero witness")
        if self.divisor.degree != 0:
            raise ValueError("a principal divisor must have degree zero")
        if space is not None:
            if (
                space.status is not RiemannRochSpaceStatus.CERTIFIED
                or witness not in space.basis
                or self.model.principal_divisor(witness) != self.divisor
            ):
                raise ValueError(
                    "the Riemann--Roch witness does not prove principality"
                )
        elif not _torsion_search_proves_principal(self.divisor, witness, search):
            raise ValueError(
                "a principal decision needs a Riemann--Roch or divisor-class proof"
            )


@dataclass(frozen=True, slots=True)
class AlgebraicNonprincipal(_AlgebraicPrincipalityDecision):
    """A complete proof that the target divisor is not principal."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    riemann_roch_space: RiemannRochSpace | None
    divisor_class_search: AlgebraicTorsionSearch | None
    detail: str
    status: ClassVar[AlgebraicPrincipalityStatus] = (
        AlgebraicPrincipalityStatus.NONPRINCIPAL
    )
    witness: ClassVar[None] = None

    def __post_init__(self) -> None:
        self._validate_common()
        space = self.riemann_roch_space
        if space is not None and (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != -self.divisor
        ):
            raise ValueError("the Riemann--Roch space belongs to another divisor")
        search = self.divisor_class_search
        if search is not None and (
            not isinstance(search, AlgebraicTorsionSearch)
            or search.model is not self.model
        ):
            raise ValueError("the divisor-class search belongs to another model")
        if self.divisor.degree != 0:
            if space is not None:
                raise ValueError("the degree obstruction needs no Riemann--Roch space")
            return
        direct_proof = (
            space is not None
            and space.status is RiemannRochSpaceStatus.CERTIFIED
            and not space.basis
        )
        if not direct_proof and not _torsion_search_proves_nonprincipal(
            self.divisor, search
        ):
            raise ValueError(
                "a degree-zero nonprincipal decision needs a complete class proof"
            )


@dataclass(frozen=True, slots=True)
class AlgebraicPrincipalityUnsupported(_AlgebraicPrincipalityDecision):
    """An exact arithmetic boundary without a principality conclusion."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    riemann_roch_space: RiemannRochSpace | None
    divisor_class_search: AlgebraicTorsionSearch | None
    detail: str
    status: ClassVar[AlgebraicPrincipalityStatus] = (
        AlgebraicPrincipalityStatus.UNSUPPORTED
    )
    witness: ClassVar[None] = None

    def __post_init__(self) -> None:
        self._validate_common()
        space = self.riemann_roch_space
        search = self.divisor_class_search
        if space is not None and (
            not isinstance(space, RiemannRochSpace)
            or space.model is not self.model
            or space.divisor != -self.divisor
        ):
            raise ValueError("the Riemann--Roch space belongs to another divisor")
        if search is not None and (
            not isinstance(search, AlgebraicTorsionSearch)
            or search.model is not self.model
        ):
            raise ValueError("the divisor-class search belongs to another model")
        if (
            space is None or space.status is not RiemannRochSpaceStatus.UNSUPPORTED
        ) and search is None:
            raise ValueError(
                "an unsupported decision needs its exact arithmetic boundary"
            )


AlgebraicPrincipalityDecision: TypeAlias = (
    AlgebraicPrincipal | AlgebraicNonprincipal | AlgebraicPrincipalityUnsupported
)


def decide_algebraic_principality(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> AlgebraicPrincipalityDecision:
    r"""Decide whether ``div(f) = divisor`` in an algebraic function field.

    The decision is complete whenever the required Riemann--Roch reductions
    are available. The coefficient content of the supplied divisor provides
    the finite class-search bound, so this API has no heuristic order
    parameter or Risch-specific constant-field analysis.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("principality needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("principality needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")

    if divisor.degree != 0:
        return AlgebraicNonprincipal(
            model,
            divisor,
            None,
            None,
            "a principal divisor must have degree zero",
        )

    if divisor.is_zero:
        witness = model.extension.field.one
        space = compute_riemann_roch_space(
            model,
            -divisor,
            candidate_basis=(witness,),
        )
        return AlgebraicPrincipal(
            model,
            divisor,
            witness,
            space,
            None,
            "a direct complete Riemann--Roch witness proves principality",
        )

    content = _divisor_content(divisor)
    primitive = _primitive_divisor(divisor)
    search = search_algebraic_divisor_torsion(
        model,
        primitive,
        max_order=content,
    )
    if search.status is AlgebraicTorsionStatus.TORSION:
        if search.order is None:
            raise ValueError("a torsion search omitted its order")
        if content % search.order:
            return AlgebraicNonprincipal(
                model,
                divisor,
                None,
                search,
                "the primitive divisor has torsion order not dividing its content",
            )
        relation_witness = search.witness_for_multiple(content)
        return AlgebraicPrincipal(
            model,
            divisor,
            relation_witness,
            None,
            search,
            "reduced divisor-class arithmetic reconstructed a principal relation",
        )
    if search.status is AlgebraicTorsionStatus.INCONCLUSIVE:
        return AlgebraicNonprincipal(
            model,
            divisor,
            None,
            search,
            "the primitive class is nonzero at every order dividing the content",
        )

    # Keep the direct complete algorithm as a fallback when the reduced-class
    # route encounters an unsupported Riemann--Roch operation.
    space = compute_riemann_roch_space(
        model,
        -divisor,
    )
    if space.status is RiemannRochSpaceStatus.UNSUPPORTED:
        return AlgebraicPrincipalityUnsupported(
            model,
            divisor,
            space,
            search,
            "principality requires a complete L(-D): " + space.detail,
        )
    if not space.basis:
        return AlgebraicNonprincipal(
            model,
            divisor,
            space,
            search,
            "the complete space L(-D) is zero, so D is nonprincipal",
        )

    witness = space.basis[0]
    return AlgebraicPrincipal(
        model,
        divisor,
        witness,
        space,
        search,
        "a nonzero element of complete L(-D) has principal divisor D",
    )


def principality_from_torsion_search(
    search: AlgebraicTorsionSearch,
    multiple: int,
) -> AlgebraicPrincipalityDecision:
    r"""Decide principality of ``multiple * search.divisor`` from one search.

    This is the bridge used by logarithmic reconstruction: a residue divisor
    may itself be nonprincipal while a positive multiple is principal.
    """
    if not isinstance(search, AlgebraicTorsionSearch):
        raise TypeError("principality transport needs a divisor-class search")
    if not isinstance(multiple, int) or multiple <= 0:
        raise ValueError("the divisor multiple must be a positive integer")
    divisor = search.divisor.scale(multiple)

    if search.status is AlgebraicTorsionStatus.TORSION:
        if search.order is None:
            raise ValueError("a torsion search omitted its order")
        if multiple % search.order:
            return AlgebraicNonprincipal(
                search.model,
                divisor,
                None,
                search,
                "the found torsion order does not divide the requested multiple",
            )
        return AlgebraicPrincipal(
            search.model,
            divisor,
            search.witness_for_multiple(multiple),
            None,
            search,
            "the torsion relation reconstructs the requested principal divisor",
        )

    if (
        search.status is AlgebraicTorsionStatus.INCONCLUSIVE
        and search.max_order >= multiple
    ):
        return AlgebraicNonprincipal(
            search.model,
            divisor,
            None,
            search,
            "the class remains nonzero through the requested multiple",
        )

    return AlgebraicPrincipalityUnsupported(
        search.model,
        divisor,
        None,
        search,
        "the supplied torsion search does not decide this multiple",
    )
