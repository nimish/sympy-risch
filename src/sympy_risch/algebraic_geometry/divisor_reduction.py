r"""Height-efficient divisor reduction following Hess's Algorithm 21.

For an effective divisor ``A`` of positive degree, the elementary reduction
step chooses ``a`` in ``L(D - r*A)`` and sets

``D_tilde = div(a) + D - r*A``.

Thus ``D = D_tilde + r*A - div(a)`` with ``D_tilde`` effective.  Applying
that step directly to a divisor with large coefficients is expensive because
the associated fractional ideals inherit the coefficient height.  Hess's
Algorithm 21 instead decomposes the coefficients in binary, reduces the
small-support digits, and combines them by Horner's rule.

The correcting function is retained as a power product.  This is the natural
exact representation for the algorithm: exponents may be large, but their
storage grows only logarithmically with the input height.  Call
:meth:`HessPowerProduct.evaluate` only when an expanded field element is
actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement

from .function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)
from .hess import hess_parametric_basis
from .riemann_roch import (
    RiemannRochSpaceStatus,
    compute_riemann_roch_space,
)


class HessReductionStrategy(Enum):
    """How the integer in each elementary reduction is selected."""

    DEGREE = "degree"
    MAXIMAL = "maximal"


@dataclass(frozen=True, slots=True)
class _HessPowerProductFactor:
    """One exact factor and its already-computed principal divisor."""

    element: ExtensionElement
    principal_divisor: AlgebraicDivisor
    exponent: int = 1


@dataclass(frozen=True, slots=True)
class HessPowerProduct:
    r"""A compact product ``prod(factor.element**factor.exponent)``."""

    model: AlgebraicFunctionField
    factors: tuple[_HessPowerProductFactor, ...] = ()

    @property
    def principal_divisor(self) -> AlgebraicDivisor:
        """Return the principal divisor without expanding the product."""
        result = AlgebraicDivisor(self.model)
        for factor in self.factors:
            result += factor.principal_divisor.scale(factor.exponent)
        return result

    def evaluate(self) -> ExtensionElement:
        """Expand the power product to one function-field element."""
        result = self.model.extension.field.one
        for factor in self.factors:
            result *= factor.element**factor.exponent
        return cast(ExtensionElement, result)

    def scaled(self, exponent: int) -> HessPowerProduct:
        """Raise the whole product to a nonnegative integer power."""
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("a power-product scale must be a nonnegative integer")
        if exponent == 0:
            return HessPowerProduct(self.model)
        return HessPowerProduct(
            self.model,
            tuple(
                _HessPowerProductFactor(
                    factor.element,
                    factor.principal_divisor,
                    factor.exponent * exponent,
                )
                for factor in self.factors
            ),
        )

    def multiply(self, other: HessPowerProduct) -> HessPowerProduct:
        """Multiply two products over the same function field."""
        if other.model is not self.model:
            raise ValueError("power products belong to different function fields")
        return HessPowerProduct(self.model, self.factors + other.factors)


@dataclass(frozen=True, slots=True, init=False)
class HessDivisorReduction:
    r"""An exact Algorithm 21 reduction of one arbitrary divisor.

    The stored identity is

    ``divisor = reduced_divisor + anchor_multiple*anchor - div(witness)``.

    The derived ``support_reductions`` and ``composition_reductions``
    properties describe the binary work without storing duplicate trace
    metadata.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    anchor: AlgebraicDivisor
    strategy: HessReductionStrategy
    reduced_divisor: AlgebraicDivisor
    anchor_multiple: int
    witness: HessPowerProduct

    def __init__(self) -> None:
        raise TypeError("Hess reductions must be created by reduce_hess_divisor")

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        divisor: AlgebraicDivisor,
        anchor: AlgebraicDivisor,
        strategy: HessReductionStrategy,
        reduced_divisor: AlgebraicDivisor,
        anchor_multiple: int,
        witness: HessPowerProduct,
    ) -> HessDivisorReduction:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "divisor", divisor)
        object.__setattr__(result, "anchor", anchor)
        object.__setattr__(result, "strategy", strategy)
        object.__setattr__(result, "reduced_divisor", reduced_divisor)
        object.__setattr__(result, "anchor_multiple", anchor_multiple)
        object.__setattr__(result, "witness", witness)
        result._validate()
        return result

    def _validate(self) -> None:
        if self.divisor.model is not self.model:
            raise ValueError("the input divisor belongs to another function field")
        if self.anchor.model is not self.model:
            raise ValueError("the reduction anchor belongs to another function field")
        if self.reduced_divisor.model is not self.model:
            raise ValueError("the reduced divisor belongs to another function field")
        if self.witness.model is not self.model:
            raise ValueError("the power product belongs to another function field")
        if not self.reduced_divisor.is_effective:
            raise ValueError("Hess reduction must produce an effective divisor")
        if self.reduced_divisor.degree >= self.model.genus + self.anchor.degree:
            raise ValueError("the reduced divisor exceeds the Hess degree bound")
        if self.divisor != (
            self.reduced_divisor
            + self.anchor.scale(self.anchor_multiple)
            - self.witness.principal_divisor
        ):
            raise ValueError("the Hess reduction identity does not reconstruct")

    @property
    def reconstructs(self) -> bool:
        """Independently expand and verify the principal-divisor witness."""
        return (
            self.model.principal_divisor(self.witness.evaluate())
            == self.witness.principal_divisor
        )

    @property
    def binary_layers(self) -> int:
        """Number of signed binary layers in the input divisor."""
        return max(
            (
                abs(coefficient).bit_length()
                for _place, coefficient in self.divisor.terms
            ),
            default=0,
        )

    @property
    def support_reductions(self) -> int:
        """Number of nonzero signed-place digits processed."""
        return sum(
            abs(coefficient).bit_count() for _place, coefficient in self.divisor.terms
        )

    @property
    def composition_reductions(self) -> int:
        """Number of Horner composition layers."""
        return max(self.binary_layers - 1, 0)


@dataclass(frozen=True, slots=True)
class _ElementaryReduction:
    """One elementary ``D = D_tilde + r*A - div(a)`` step."""

    divisor: AlgebraicDivisor
    anchor_multiple: int
    factor: HessPowerProduct


def _validate_inputs(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    anchor: AlgebraicDivisor | None,
    strategy: HessReductionStrategy,
) -> AlgebraicDivisor:
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("Hess reduction needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("Hess reduction needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(strategy, HessReductionStrategy):
        raise TypeError("the Hess reduction strategy has the wrong type")

    result = model.base_parameter_pole_divisor() if anchor is None else anchor
    if not isinstance(result, AlgebraicDivisor):
        raise TypeError("the reduction anchor must be an algebraic divisor")
    if result.model is not model:
        raise ValueError("the reduction anchor belongs to another function field")
    if not result.is_effective or result.degree <= 0:
        raise ValueError("the reduction anchor must be effective and nonzero")
    return result


def _degree_reduction_multiple(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    anchor: AlgebraicDivisor,
) -> int:
    """Choose ``r`` with ``g <= deg(D-rA) < g+deg(A)``."""
    return (divisor.degree - model.genus) // anchor.degree


def _complete_space_basis(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> tuple[ExtensionElement, ...]:
    """Return a complete basis, preserving unsupported as unsupported."""
    space = compute_riemann_roch_space(model, divisor)
    if space.status is RiemannRochSpaceStatus.UNSUPPORTED:
        raise NotImplementedError(
            "Hess divisor reduction needs a complete Riemann--Roch space: "
            + space.detail,
        )
    return space.basis


def _general_maximal_multiple(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    anchor: AlgebraicDivisor,
) -> tuple[int, ExtensionElement]:
    """Find the last nonzero ``L(D-rA)`` by monotone binary search."""
    lower = _degree_reduction_multiple(model, divisor, anchor)
    lower_basis = _complete_space_basis(model, divisor - anchor.scale(lower))
    if not lower_basis:
        raise ValueError("Riemann--Roch contradicted the Hess degree guarantee")

    upper = divisor.degree // anchor.degree + 1
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        basis = _complete_space_basis(model, divisor - anchor.scale(middle))
        if basis:
            lower = middle
            lower_basis = basis
        else:
            upper = middle
    return lower, lower_basis[0]


def _elementary_reduce(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    anchor: AlgebraicDivisor,
    strategy: HessReductionStrategy,
    parameter_anchor: bool,
) -> _ElementaryReduction:
    """Perform one theorem-backed elementary divisor reduction."""
    if divisor.is_zero:
        return _ElementaryReduction(divisor, 0, HessPowerProduct(model))

    if parameter_anchor:
        family = hess_parametric_basis(model, divisor)
        multiple = (
            _degree_reduction_multiple(model, divisor, anchor)
            if strategy is HessReductionStrategy.DEGREE
            else max(family.invariant_degrees)
        )
        basis = family.expand(-multiple)
        if not basis:
            raise ValueError("the parametric Hess basis lost its maximal member")
        element = basis[0]
    elif strategy is HessReductionStrategy.DEGREE:
        multiple = _degree_reduction_multiple(model, divisor, anchor)
        basis = _complete_space_basis(model, divisor - anchor.scale(multiple))
        if not basis:
            raise ValueError("Riemann--Roch contradicted the Hess degree guarantee")
        element = basis[0]
    else:
        multiple, element = _general_maximal_multiple(
            model,
            divisor,
            anchor,
        )

    principal = model.principal_divisor(element)
    reduced = divisor - anchor.scale(multiple) + principal
    if not reduced.is_effective:
        raise ValueError("the Riemann--Roch witness did not make the divisor effective")
    factor = HessPowerProduct(
        model,
        (_HessPowerProductFactor(element, principal),),
    )
    return _ElementaryReduction(reduced, multiple, factor)


def _binary_digits(divisor: AlgebraicDivisor) -> tuple[AlgebraicDivisor, ...]:
    """Decompose coefficients into signed binary digit divisors."""
    layers = max(
        (abs(coefficient).bit_length() for _place, coefficient in divisor.terms),
        default=0,
    )
    digits: list[list[tuple[AlgebraicPlace, int]]] = [[] for _layer in range(layers)]
    for place, coefficient in divisor.terms:
        sign = 1 if coefficient > 0 else -1
        magnitude = abs(coefficient)
        for layer in range(layers):
            if magnitude & (1 << layer):
                digits[layer].append((place, sign))
    return tuple(AlgebraicDivisor(divisor.model, tuple(digit)) for digit in digits)


def _reduce_digit(
    model: AlgebraicFunctionField,
    digit: AlgebraicDivisor,
    anchor: AlgebraicDivisor,
    strategy: HessReductionStrategy,
    parameter_anchor: bool,
) -> _ElementaryReduction:
    """Reduce one signed-support digit one place at a time."""
    current = AlgebraicDivisor(model)
    multiple = 0
    witness = HessPowerProduct(model)
    for place, coefficient in digit.terms:
        step = _elementary_reduce(
            model,
            current + AlgebraicDivisor(model, ((place, coefficient),)),
            anchor,
            strategy,
            parameter_anchor,
        )
        current = step.divisor
        multiple += step.anchor_multiple
        witness = witness.multiply(step.factor)
    return _ElementaryReduction(current, multiple, witness)


def reduce_hess_divisor(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    *,
    anchor: AlgebraicDivisor | None = None,
    strategy: HessReductionStrategy = HessReductionStrategy.DEGREE,
) -> HessDivisorReduction:
    r"""Reduce ``divisor`` with Hess's height-efficient Algorithm 21.

    The default anchor is the full pole divisor ``(x)_infinity``.  For that
    anchor, each elementary step uses one parametric Hess basis: the degree
    strategy selects the Riemann--Roch guaranteed shift, while the maximal
    strategy reads the last nonzero shift directly from the invariant
    degrees.  Other positive effective anchors use complete Riemann--Roch
    spaces and a monotone binary search for maximal reduction.
    """
    reduction_anchor = _validate_inputs(model, divisor, anchor, strategy)
    parameter_anchor = reduction_anchor == model.base_parameter_pole_divisor()
    digits = _binary_digits(divisor)
    if not digits:
        return HessDivisorReduction._create(
            model,
            divisor,
            reduction_anchor,
            strategy,
            divisor,
            0,
            HessPowerProduct(model),
        )

    reduced_digits: list[_ElementaryReduction] = []
    for digit in digits:
        reduced_digits.append(
            _reduce_digit(
                model,
                digit,
                reduction_anchor,
                strategy,
                parameter_anchor,
            ),
        )

    current = reduced_digits[-1]
    for digit in reversed(reduced_digits[:-1]):
        combined = _elementary_reduce(
            model,
            current.divisor.scale(2) + digit.divisor,
            reduction_anchor,
            strategy,
            parameter_anchor,
        )
        current = _ElementaryReduction(
            combined.divisor,
            2 * current.anchor_multiple
            + digit.anchor_multiple
            + combined.anchor_multiple,
            current.factor.scaled(2).multiply(digit.factor).multiply(combined.factor),
        )

    return HessDivisorReduction._create(
        model,
        divisor,
        reduction_anchor,
        strategy,
        current.divisor,
        current.anchor_multiple,
        current.factor,
    )
