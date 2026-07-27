r"""Reduced divisor classes and positive torsion relations.

Let ``D`` be a degree-zero divisor and let ``A`` be a fixed nonzero effective
divisor.  For the least ``n >= 0`` with

``L(n*A - D) != 0``,

choose ``f`` in that complete Riemann--Roch space.  Then

``D = (D - div(f)) + div(f)``

and the representative ``D - div(f)`` has bounded positive and negative
parts.  Riemann--Roch guarantees that the search stops once
``n*deg(A) >= genus``.  Repeating this reduction after adding another
degree-zero divisor implements class-group arithmetic without constructing
large powers of place ideals.

The bounded search below is a positive torsion semidecision.  An exhausted
order bound is explicitly inconclusive; finite-prime Jacobian reduction is
still needed for a total standalone torsion decision.  A supplied divisor's
*principality*, however, can use its coefficient content as a complete finite
bound; see :mod:`sympy_risch.algebraic_geometry.principal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement

from ..algebraic._field import (
    _base_polynomial_ring,
    _extension_element,
)
from .function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from .riemann_roch import (
    RiemannRochSpace,
    RiemannRochSpaceStatus,
    compute_riemann_roch_space,
)


def _normalize_constant_multiple(
    model: AlgebraicFunctionField,
    element: object,
) -> ExtensionElement:
    """Choose a compact representative modulo a nonzero constant factor."""
    extension = model.extension
    value = _extension_element(extension, element)
    if value == extension.field.zero:
        raise ValueError("a divisor-class witness must be nonzero")

    parent, constants, _variable = _base_polynomial_ring(extension)
    if not isinstance(parent, FractionField):
        raise TypeError("divisor-class reduction needs a rational-function base")

    coordinates = tuple(
        cast(FracElement, parent.convert(value.rep.nth(power)))
        for power in range(extension.field.rank)
    )
    nonzero = tuple(coordinate for coordinate in coordinates if coordinate)
    if not nonzero:
        raise ValueError("a nonzero extension element has zero coordinates")
    pivot = max(
        nonzero,
        key=lambda coordinate: coordinate.numer.degree() - coordinate.denom.degree(),
    )
    scale = constants.quo(pivot.denom.LC, pivot.numer.LC)
    return extension.field.from_sympy(constants.to_sympy(scale)) * value


def _default_anchor(model: AlgebraicFunctionField) -> AlgebraicDivisor:
    """Use one closed place at infinity as a positive-degree anchor."""
    places = model.infinite_places()
    if not places:
        raise NotImplementedError("the function-field model has no infinity place")
    return AlgebraicDivisor(model, ((places[0], 1),))


def _validate_anchor(
    model: AlgebraicFunctionField,
    anchor: AlgebraicDivisor | None,
) -> AlgebraicDivisor:
    result = _default_anchor(model) if anchor is None else anchor
    if not isinstance(result, AlgebraicDivisor):
        raise TypeError("divisor-class reduction needs an anchor divisor")
    if result.model is not model:
        raise ValueError("the anchor belongs to another function field")
    if not result.is_effective or result.degree <= 0:
        raise ValueError("the anchor must be an effective divisor of positive degree")
    return result


@dataclass(frozen=True, slots=True)
class ReducedAlgebraicDivisorClass:
    r"""One Riemann--Roch-certified reduction of a degree-zero divisor.

    ``correction_divisor`` is computed from ``correction`` by the reduction
    routine.  Retaining it avoids repeating expensive high-order valuation
    work when several reductions are composed into a torsion certificate.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    representative: AlgebraicDivisor
    correction: ExtensionElement
    correction_divisor: AlgebraicDivisor
    anchor_multiple: int
    riemann_roch_space: RiemannRochSpace

    def __post_init__(self) -> None:
        if self.divisor.model is not self.model:
            raise ValueError("the reduced divisor belongs to another model")
        if self.anchor.model is not self.model:
            raise ValueError("the reduction anchor belongs to another model")
        if self.representative.model is not self.model:
            raise ValueError("the representative belongs to another model")
        if self.correction_divisor.model is not self.model:
            raise ValueError("the correction divisor belongs to another model")
        if self.divisor.degree != 0 or self.representative.degree != 0:
            raise ValueError("divisor-class reduction requires degree-zero divisors")
        if not isinstance(self.anchor_multiple, int) or self.anchor_multiple < 0:
            raise ValueError("the anchor multiple must be a nonnegative integer")
        expected_space_divisor = self.anchor.scale(self.anchor_multiple) - self.divisor
        if (
            self.riemann_roch_space.model is not self.model
            or self.riemann_roch_space.divisor != expected_space_divisor
            or self.riemann_roch_space.status is not RiemannRochSpaceStatus.CERTIFIED
            or self.correction not in self.riemann_roch_space.basis
        ):
            raise ValueError("the reduction lacks its complete Riemann--Roch witness")
        if self.divisor != self.representative + self.correction_divisor:
            raise ValueError("the correction does not reconstruct the input divisor")


def reduce_algebraic_divisor_class(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    *,
    anchor: AlgebraicDivisor | None = None,
) -> ReducedAlgebraicDivisorClass:
    r"""Return a bounded representative of the class of ``divisor``.

    The anchor may have any positive degree.  A degree-one rational place
    gives the traditional representatives of degree at most the genus, while
    a higher-degree closed place remains sufficient for exact class
    arithmetic.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("divisor-class reduction needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("divisor-class reduction needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if divisor.degree != 0:
        raise ValueError("only degree-zero divisors define classes in Pic^0")
    reduction_anchor = _validate_anchor(model, anchor)

    genus = model.genus
    largest_multiple = (genus + reduction_anchor.degree - 1) // reduction_anchor.degree
    for multiple in range(largest_multiple + 1):
        space_divisor = reduction_anchor.scale(multiple) - divisor
        space = compute_riemann_roch_space(model, space_divisor)
        if space.status is RiemannRochSpaceStatus.UNSUPPORTED:
            raise NotImplementedError(
                "divisor-class reduction needs complete Riemann--Roch spaces: "
                + space.detail,
            )
        if not space.basis:
            continue

        correction = space.basis[0]
        correction_divisor = model.principal_divisor(correction)
        representative = divisor - correction_divisor
        return ReducedAlgebraicDivisorClass(
            model,
            divisor,
            reduction_anchor,
            representative,
            correction,
            correction_divisor,
            multiple,
            space,
        )

    raise ValueError(
        "Riemann--Roch failed to produce the guaranteed bounded representative"
    )


class AlgebraicTorsionStatus(Enum):
    """Outcome of a bounded positive divisor-class torsion search."""

    TORSION = "torsion"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicTorsionSearch:
    r"""A positive torsion relation or an explicit bounded-search boundary."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    max_order: int
    status: AlgebraicTorsionStatus
    order: int | None
    witness: ExtensionElement | None
    reductions: tuple[ReducedAlgebraicDivisorClass, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.divisor.model is not self.model or self.anchor.model is not self.model:
            raise ValueError("the torsion search data belongs to another model")
        if self.divisor.degree != 0:
            raise ValueError("torsion search requires a degree-zero divisor")
        if not isinstance(self.max_order, int) or self.max_order <= 0:
            raise ValueError("a torsion search bound must be a positive integer")
        if not isinstance(self.status, AlgebraicTorsionStatus):
            raise TypeError("the torsion status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a torsion search needs a detail")

        previous = AlgebraicDivisor(self.model)
        product = self.model.extension.field.one
        for reduction in self.reductions:
            if reduction.model is not self.model or reduction.anchor != self.anchor:
                raise ValueError("a torsion reduction belongs to another search")
            if reduction.divisor != previous + self.divisor:
                raise ValueError(
                    "torsion reductions do not form a class-addition chain"
                )
            previous = reduction.representative
            product = _normalize_constant_multiple(
                self.model,
                product * reduction.correction,
            )

        if self.status is AlgebraicTorsionStatus.TORSION:
            if (
                self.order is None
                or self.order != len(self.reductions)
                or not self.reductions
                or not self.reductions[-1].representative.is_zero
                or self.witness is None
                or self.witness != product
            ):
                raise ValueError("a torsion result lacks its reconstructed relation")
            return

        if self.order is not None or self.witness is not None:
            raise ValueError(
                "only a found torsion relation carries an order and witness"
            )
        if self.status is AlgebraicTorsionStatus.INCONCLUSIVE:
            if (
                len(self.reductions) != self.max_order
                or not self.reductions
                or self.reductions[-1].representative.is_zero
            ):
                raise ValueError("an inconclusive search did not exhaust its bound")
            return
        if self.status is AlgebraicTorsionStatus.UNSUPPORTED:
            return
        raise ValueError("the torsion search has an unknown status")

    @property
    def torsion(self) -> bool:
        """Whether the search found and reconstructed a positive relation."""
        return self.status is AlgebraicTorsionStatus.TORSION

    @property
    def last_representative(self) -> AlgebraicDivisor | None:
        """The final reduced class reached before the search stopped."""
        if not self.reductions:
            return None
        return self.reductions[-1].representative

    def witness_for_multiple(self, multiple: int) -> ExtensionElement:
        r"""Return a witness for ``div(f) = multiple*divisor``.

        This operation is available exactly when the found torsion order
        divides ``multiple``.
        """
        if (
            self.status is not AlgebraicTorsionStatus.TORSION
            or self.order is None
            or self.witness is None
        ):
            raise ValueError("no torsion relation was found")
        if not isinstance(multiple, int) or multiple <= 0:
            raise ValueError("the requested multiple must be positive")
        quotient, remainder = divmod(multiple, self.order)
        if remainder:
            raise ValueError("the torsion order does not divide this multiple")
        return _normalize_constant_multiple(self.model, self.witness**quotient)


def search_algebraic_divisor_torsion(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    *,
    max_order: int,
    anchor: AlgebraicDivisor | None = None,
) -> AlgebraicTorsionSearch:
    r"""Search the first ``max_order`` positive multiples of a divisor class.

    A found relation is exact.  Exhaustion is deliberately ``INCONCLUSIVE``:
    it proves only that the order is larger than the supplied bound (or
    infinite), not which alternative holds.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("torsion search needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("torsion search needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if divisor.degree != 0:
        raise ValueError("only degree-zero divisor classes can be torsion")
    if not isinstance(max_order, int) or max_order <= 0:
        raise ValueError("a torsion search bound must be a positive integer")
    reduction_anchor = _validate_anchor(model, anchor)

    reductions: list[ReducedAlgebraicDivisorClass] = []
    representative = AlgebraicDivisor(model)
    witness = model.extension.field.one
    for order in range(1, max_order + 1):
        try:
            reduction = reduce_algebraic_divisor_class(
                model,
                representative + divisor,
                anchor=reduction_anchor,
            )
        except NotImplementedError as exc:
            return AlgebraicTorsionSearch(
                model,
                divisor,
                reduction_anchor,
                max_order,
                AlgebraicTorsionStatus.UNSUPPORTED,
                None,
                None,
                tuple(reductions),
                str(exc),
            )
        reductions.append(reduction)
        representative = reduction.representative
        witness = _normalize_constant_multiple(
            model,
            witness * reduction.correction,
        )
        if representative.is_zero:
            return AlgebraicTorsionSearch(
                model,
                divisor,
                reduction_anchor,
                max_order,
                AlgebraicTorsionStatus.TORSION,
                order,
                witness,
                tuple(reductions),
                f"the first zero reduced representative occurs at order {order}",
            )

    return AlgebraicTorsionSearch(
        model,
        divisor,
        reduction_anchor,
        max_order,
        AlgebraicTorsionStatus.INCONCLUSIVE,
        None,
        None,
        tuple(reductions),
        f"no zero reduced representative occurs through order {max_order}",
    )


__all__ = [
    "AlgebraicTorsionSearch",
    "AlgebraicTorsionStatus",
    "ReducedAlgebraicDivisorClass",
    "reduce_algebraic_divisor_class",
    "search_algebraic_divisor_torsion",
]
