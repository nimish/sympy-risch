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
from math import isqrt
from typing import TypeAlias, cast

from sympy.ntheory import factorint
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
    CertifiedRiemannRochSpace,
    RiemannRochSpace,
    RiemannRochUnsupported,
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
            or not isinstance(self.riemann_roch_space, CertifiedRiemannRochSpace)
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
        match space:
            case RiemannRochUnsupported(detail=detail):
                raise NotImplementedError(
                    "divisor-class reduction needs complete Riemann--Roch spaces: "
                    + detail,
                )
            case CertifiedRiemannRochSpace(basis=()):
                continue
            case CertifiedRiemannRochSpace(basis=(correction, *_)):
                pass
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


class _AlgebraicTorsionSearch:
    """Shared behavior for the three bounded-search variants."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    max_order: int
    reductions: tuple[ReducedAlgebraicDivisorClass, ...]
    detail: str

    def _validate_common(self) -> None:
        if self.divisor.model is not self.model or self.anchor.model is not self.model:
            raise ValueError("the torsion search data belongs to another model")
        if self.divisor.degree != 0:
            raise ValueError("torsion search requires a degree-zero divisor")
        if not isinstance(self.max_order, int) or self.max_order <= 0:
            raise ValueError("a torsion search bound must be a positive integer")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a torsion search needs a detail")

        for reduction in self.reductions:
            if reduction.model is not self.model or reduction.anchor != self.anchor:
                raise ValueError("a torsion reduction belongs to another search")

    @property
    def last_representative(self) -> AlgebraicDivisor | None:
        """The final reduced class reached before the search stopped."""
        if not self.reductions:
            return None
        return self.reductions[-1].representative


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicTorsionRelation(_AlgebraicTorsionSearch):
    """A reconstructed positive divisor-class relation.

    The torsion search constructs relations from its certified class-reduction
    trace.  Keeping construction there avoids both caller-forged witnesses and
    a second, expensive global principal-divisor computation.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    max_order: int
    order: int
    witness: ExtensionElement
    reductions: tuple[ReducedAlgebraicDivisorClass, ...]
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "algebraic torsion relations must be created by the torsion search"
        )

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        divisor: AlgebraicDivisor,
        anchor: AlgebraicDivisor,
        max_order: int,
        order: int,
        witness: ExtensionElement,
        reductions: tuple[ReducedAlgebraicDivisorClass, ...],
        detail: str,
    ) -> AlgebraicTorsionRelation:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "divisor", divisor)
        object.__setattr__(result, "anchor", anchor)
        object.__setattr__(result, "max_order", max_order)
        object.__setattr__(result, "order", order)
        object.__setattr__(result, "witness", witness)
        object.__setattr__(result, "reductions", reductions)
        object.__setattr__(result, "detail", detail)
        result.__post_init__()
        return result

    def __post_init__(self) -> None:
        self._validate_common()
        if not 0 < self.order <= self.max_order:
            raise ValueError("a torsion order must lie within the search bound")
        witness = _extension_element(self.model.extension, self.witness)
        object.__setattr__(self, "witness", witness)
        if len(self.reductions) == self.order:
            previous = AlgebraicDivisor(self.model)
            product = self.model.extension.field.one
            for reduction in self.reductions:
                if reduction.divisor != previous + self.divisor:
                    raise ValueError(
                        "sequential torsion reductions do not form a chain"
                    )
                previous = reduction.representative
                product = _normalize_constant_multiple(
                    self.model, product * reduction.correction
                )
            if not previous.is_zero or witness != _normalize_constant_multiple(
                self.model, product
            ):
                raise ValueError("a sequential torsion result failed reconstruction")

    def witness_for_multiple(self, multiple: int) -> ExtensionElement:
        r"""Return a witness for ``div(f) = multiple*divisor``.

        This operation is available exactly when the found torsion order
        divides ``multiple``.
        """
        if not isinstance(multiple, int) or multiple <= 0:
            raise ValueError("the requested multiple must be positive")
        quotient, remainder = divmod(multiple, self.order)
        if remainder:
            raise ValueError("the torsion order does not divide this multiple")
        return _normalize_constant_multiple(self.model, self.witness**quotient)


@dataclass(frozen=True, slots=True)
class AlgebraicTorsionBoundExhausted(_AlgebraicTorsionSearch):
    """Proof that no relation occurs through a caller-supplied bound."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    max_order: int
    reductions: tuple[ReducedAlgebraicDivisorClass, ...]
    detail: str

    def __post_init__(self) -> None:
        self._validate_common()
        previous = AlgebraicDivisor(self.model)
        for reduction in self.reductions:
            if reduction.divisor != previous + self.divisor:
                raise ValueError(
                    "inconclusive reductions do not form a class-addition chain"
                )
            previous = reduction.representative
        if (
            len(self.reductions) != self.max_order
            or not self.reductions
            or self.reductions[-1].representative.is_zero
        ):
            raise ValueError("an inconclusive search did not exhaust its bound")


@dataclass(frozen=True, slots=True)
class AlgebraicTorsionSearchUnsupported(_AlgebraicTorsionSearch):
    """An arithmetic boundary encountered before completing the search."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    max_order: int
    reductions: tuple[ReducedAlgebraicDivisorClass, ...]
    detail: str

    def __post_init__(self) -> None:
        self._validate_common()


AlgebraicTorsionSearch: TypeAlias = (
    AlgebraicTorsionRelation
    | AlgebraicTorsionBoundExhausted
    | AlgebraicTorsionSearchUnsupported
)


@dataclass(frozen=True, slots=True)
class _ClassMultiple:
    """A reduced representative and principal witness for one multiple."""

    multiple: int
    representative: AlgebraicDivisor
    witness: ExtensionElement


def _add_class_multiples(
    model: AlgebraicFunctionField,
    left: _ClassMultiple,
    right: _ClassMultiple,
    anchor: AlgebraicDivisor,
) -> tuple[_ClassMultiple, ReducedAlgebraicDivisorClass]:
    """Add two certified multiples with one reduced class operation."""
    reduction = reduce_algebraic_divisor_class(
        model,
        left.representative + right.representative,
        anchor=anchor,
    )
    return (
        _ClassMultiple(
            left.multiple + right.multiple,
            reduction.representative,
            _normalize_constant_multiple(
                model,
                left.witness * right.witness * reduction.correction,
            ),
        ),
        reduction,
    )


def _scalar_class_multiple(
    model: AlgebraicFunctionField,
    base: _ClassMultiple,
    multiple: int,
    anchor: AlgebraicDivisor,
) -> tuple[_ClassMultiple, tuple[ReducedAlgebraicDivisorClass, ...]]:
    """Compute one nonnegative multiple by double-and-add."""
    if multiple < 0:
        raise ValueError("class scalar multiplication needs a nonnegative integer")
    result = _ClassMultiple(
        0,
        AlgebraicDivisor(model),
        model.extension.field.one,
    )
    addend = base
    reductions: list[ReducedAlgebraicDivisorClass] = []
    remaining = multiple
    while remaining:
        if remaining & 1:
            result, reduction = _add_class_multiples(
                model,
                result,
                addend,
                anchor,
            )
            reductions.append(reduction)
        remaining >>= 1
        if remaining:
            addend, reduction = _add_class_multiples(
                model,
                addend,
                addend,
                anchor,
            )
            reductions.append(reduction)
    return result, tuple(reductions)


def _divisor_identity_key(divisor: AlgebraicDivisor) -> tuple[tuple[int, int], ...]:
    """Hash a model-owned divisor by exact place identity and coefficient."""
    return tuple(
        sorted((id(place), coefficient) for place, coefficient in divisor.terms)
    )


def _search_algebraic_divisor_torsion_bsgs(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    *,
    max_order: int,
    anchor: AlgebraicDivisor,
) -> AlgebraicTorsionSearch:
    """Find an exact order through canonical reduced classes and BSGS."""
    if anchor.degree != 1:
        return search_algebraic_divisor_torsion(
            model,
            divisor,
            max_order=max_order,
            anchor=anchor,
        )

    zero = _ClassMultiple(
        0,
        AlgebraicDivisor(model),
        model.extension.field.one,
    )
    base, base_reduction = _add_class_multiples(
        model,
        zero,
        _ClassMultiple(1, divisor, model.extension.field.one),
        anchor,
    )
    trace: list[ReducedAlgebraicDivisorClass] = [base_reduction]
    if base.representative.is_zero:
        return AlgebraicTorsionRelation._create(
            model,
            divisor,
            anchor,
            max_order,
            1,
            base.witness,
            tuple(trace),
            "baby-step/giant-step found exact order 1",
        )

    step = isqrt(max_order)
    if step * step < max_order:
        step += 1
    babies = {_divisor_identity_key(zero.representative): zero}
    baby = zero
    for _ in range(1, step):
        baby, reduction = _add_class_multiples(model, baby, base, anchor)
        trace.append(reduction)
        if baby.representative.is_zero:
            return AlgebraicTorsionRelation._create(
                model,
                divisor,
                anchor,
                max_order,
                baby.multiple,
                baby.witness,
                tuple(trace),
                f"baby steps found exact order {baby.multiple}",
            )
        babies[_divisor_identity_key(baby.representative)] = baby

    giant_step, reduction = _add_class_multiples(model, baby, base, anchor)
    trace.append(reduction)
    giant = zero
    candidate: int | None = None
    for _ in range((max_order + step - 1) // step):
        giant, reduction = _add_class_multiples(
            model,
            giant,
            giant_step,
            anchor,
        )
        trace.append(reduction)
        matching_baby = babies.get(_divisor_identity_key(giant.representative))
        if matching_baby is None:
            continue
        proposed = giant.multiple - matching_baby.multiple
        if 0 < proposed <= max_order:
            candidate = proposed
            break

    if candidate is None:
        return AlgebraicTorsionSearchUnsupported(
            model,
            divisor,
            anchor,
            max_order,
            tuple(trace),
            "canonical baby-step/giant-step found no relation within the bound",
        )

    exact_order = candidate
    exact_state, exact_trace = _scalar_class_multiple(
        model,
        base,
        exact_order,
        anchor,
    )
    trace.extend(exact_trace)
    if not exact_state.representative.is_zero:
        raise ValueError("a baby-step/giant-step collision failed exact replay")

    for prime in factorint(candidate):
        while exact_order % prime == 0:
            quotient = exact_order // prime
            quotient_state, quotient_trace = _scalar_class_multiple(
                model,
                base,
                quotient,
                anchor,
            )
            trace.extend(quotient_trace)
            if not quotient_state.representative.is_zero:
                break
            exact_order = quotient
            exact_state = quotient_state

    return AlgebraicTorsionRelation._create(
        model,
        divisor,
        anchor,
        max_order,
        exact_order,
        exact_state.witness,
        tuple(trace),
        "certified baby-step/giant-step and prime-factor refinement found "
        f"exact order {exact_order}",
    )


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
            return AlgebraicTorsionSearchUnsupported(
                model,
                divisor,
                reduction_anchor,
                max_order,
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
            return AlgebraicTorsionRelation._create(
                model,
                divisor,
                reduction_anchor,
                max_order,
                order,
                witness,
                tuple(reductions),
                f"the first zero reduced representative occurs at order {order}",
            )

    return AlgebraicTorsionBoundExhausted(
        model,
        divisor,
        reduction_anchor,
        max_order,
        tuple(reductions),
        f"no zero reduced representative occurs through order {max_order}",
    )


__all__ = [
    "AlgebraicTorsionBoundExhausted",
    "AlgebraicTorsionRelation",
    "AlgebraicTorsionSearch",
    "AlgebraicTorsionSearchUnsupported",
    "ReducedAlgebraicDivisorClass",
    "reduce_algebraic_divisor_class",
    "search_algebraic_divisor_torsion",
]
