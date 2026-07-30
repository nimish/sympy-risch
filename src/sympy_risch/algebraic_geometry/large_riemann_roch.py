r"""Short large-divisor Riemann--Roch spaces via Hess's Algorithm 24.

Algorithm 21 gives

``D = D_tilde + r*(x)_infinity - div(w)``,

where ``D_tilde`` has bounded degree and ``w`` is a compact power product.
Consequently,

``L(D) = w * L(D_tilde + r*(x)_infinity)``.

Combining that identity with :class:`HessParametricBasis` yields a short basis
without constructing every element of a potentially large vector space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement

from .divisor_reduction import (
    HessDivisorReduction,
    HessPowerProduct,
    HessReductionStrategy,
    reduce_hess_divisor,
)
from .function_field import AlgebraicDivisor, AlgebraicFunctionField
from .hess import HessParametricBasis, hess_parametric_basis
from .riemann_roch import constant_coordinate_matrix


@dataclass(frozen=True, slots=True)
class HessPowerProductElement:
    r"""One exact element represented as ``power_product * factor``."""

    power_product: HessPowerProduct
    factor: ExtensionElement

    def __post_init__(self) -> None:
        if not isinstance(self.power_product, HessPowerProduct):
            raise TypeError("a short Hess generator needs a power product")
        field = self.power_product.model.extension.field
        factor = cast(ExtensionElement, field.convert(self.factor))
        if factor == field.zero:
            raise ValueError("a power-product generator factor must be nonzero")
        object.__setattr__(self, "factor", factor)

    @property
    def model(self) -> AlgebraicFunctionField:
        """Underlying function-field model."""
        return self.power_product.model

    def evaluate(self) -> ExtensionElement:
        """Expand this compact element in the native function field."""
        return cast(ExtensionElement, self.power_product.evaluate() * self.factor)


@dataclass(frozen=True, slots=True, init=False)
class HessLargeRiemannRochSpace:
    r"""Algorithm 24's short representation of a complete ``L(D)``.

    If ``generators[j]`` represents ``v_j`` and ``invariant_degrees[j]`` is
    ``d_j``, the long basis is

    ``{x**i * v_j : 0 <= i <= d_j}``.

    Construction is factory-owned because the reduction and parametric basis
    together are the completeness certificate.
    """

    reduction: HessDivisorReduction
    reduced_basis: HessParametricBasis
    generators: tuple[HessPowerProductElement, ...]
    invariant_degrees: tuple[int, ...]

    def __init__(self) -> None:
        raise TypeError(
            "large Hess spaces must be created by hess_large_riemann_roch_space"
        )

    @classmethod
    def _create(
        cls,
        reduction: HessDivisorReduction,
        reduced_basis: HessParametricBasis,
        generators: tuple[HessPowerProductElement, ...],
        invariant_degrees: tuple[int, ...],
    ) -> HessLargeRiemannRochSpace:
        result = object.__new__(cls)
        object.__setattr__(result, "reduction", reduction)
        object.__setattr__(result, "reduced_basis", reduced_basis)
        object.__setattr__(result, "generators", generators)
        object.__setattr__(result, "invariant_degrees", invariant_degrees)
        result._validate()
        return result

    def _validate(self) -> None:
        if self.reduction.strategy is not HessReductionStrategy.DEGREE:
            raise ValueError("Algorithm 24 requires Hess degree reduction")
        if self.reduction.anchor != self.reduction.model.base_parameter_pole_divisor():
            raise ValueError("Algorithm 24 requires the base-parameter pole divisor")
        if (
            self.reduced_basis.model is not self.reduction.model
            or self.reduced_basis.divisor != self.reduction.reduced_divisor
        ):
            raise ValueError("the short basis does not describe the reduced divisor")

        rank = self.reduction.model.extension.field.rank
        if not (
            len(self.generators)
            == len(self.invariant_degrees)
            == len(self.reduced_basis.generators)
            == rank
        ):
            raise ValueError("a short Hess space needs one generator per field rank")
        expected_degrees = tuple(
            degree + self.reduction.anchor_multiple
            for degree in self.reduced_basis.invariant_degrees
        )
        if self.invariant_degrees != expected_degrees:
            raise ValueError("the Algorithm 24 invariant degrees are inconsistent")
        if any(
            generator.model is not self.reduction.model
            or generator.power_product != self.reduction.witness
            or generator.factor != reduced_generator
            for generator, reduced_generator in zip(
                self.generators,
                self.reduced_basis.generators,
                strict=True,
            )
        ):
            raise ValueError("an Algorithm 24 generator is inconsistent")

    @property
    def model(self) -> AlgebraicFunctionField:
        """Underlying function-field model."""
        return self.reduction.model

    @property
    def divisor(self) -> AlgebraicDivisor:
        """The original, possibly high-height divisor."""
        return self.reduction.divisor

    @property
    def parameter(self) -> ExtensionElement:
        """The separating base parameter used by the short basis."""
        return self.reduced_basis.parameter

    @property
    def dimension(self) -> int:
        """Dimension over the displayed constant field without expansion."""
        return sum(max(degree + 1, 0) for degree in self.invariant_degrees)

    def expand(self) -> tuple[ExtensionElement, ...]:
        """Expand the complete long basis only when a caller requests it."""
        witness = self.reduction.witness.evaluate()
        return tuple(
            cast(ExtensionElement, witness * element)
            for element in self.reduced_basis.expand(
                self.reduction.anchor_multiple,
            )
        )

    def contains(self, element: object) -> bool:
        """Check membership through the complete principal divisor."""
        value = self.model.extension.field.convert(element)
        if value == self.model.extension.field.zero:
            return True
        return (self.model.principal_divisor(value) + self.divisor).is_effective

    @property
    def reconstructs(self) -> bool:
        """Verify the compact reduction and generator correspondence."""
        return (
            self.reduction.reconstructs
            and self.dimension
            == self.reduced_basis.dimension(self.reduction.anchor_multiple)
            and all(
                generator.power_product == self.reduction.witness
                and generator.factor == reduced_generator
                for generator, reduced_generator in zip(
                    self.generators,
                    self.reduced_basis.generators,
                    strict=True,
                )
            )
        )

    def verify_expansion(self) -> bool:
        """Expand and independently verify membership and constant rank."""
        expanded = self.expand()
        return (
            len(expanded) == self.dimension
            and all(self.contains(element) for element in expanded)
            and constant_coordinate_matrix(self.model, expanded).rank()
            == self.dimension
        )


def hess_large_riemann_roch_space(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> HessLargeRiemannRochSpace:
    r"""Compute Algorithm 24's short complete representation of ``L(D)``."""
    reduction = reduce_hess_divisor(
        model,
        divisor,
        strategy=HessReductionStrategy.DEGREE,
    )
    reduced_basis = hess_parametric_basis(model, reduction.reduced_divisor)
    generators = tuple(
        HessPowerProductElement(reduction.witness, generator)
        for generator in reduced_basis.generators
    )
    invariant_degrees = tuple(
        degree + reduction.anchor_multiple for degree in reduced_basis.invariant_degrees
    )
    return HessLargeRiemannRochSpace._create(
        reduction,
        reduced_basis,
        generators,
        invariant_degrees,
    )


__all__ = [
    "HessLargeRiemannRochSpace",
    "HessPowerProductElement",
    "hess_large_riemann_roch_space",
]
