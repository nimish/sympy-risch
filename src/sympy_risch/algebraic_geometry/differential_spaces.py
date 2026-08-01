r"""Differential spaces from Hess Riemann--Roch spaces.

For a divisor ``D`` on an algebraic function field, use the convention

``Omega(D) = {omega : div(omega) >= D}``.

Writing every differential as ``omega = f*dx`` identifies this space with

``L(div(dx) - D)``.

The implementation retains Algorithm 24's compact coefficient-space
representation.  Individual :class:`AlgebraicDifferential` objects are only
constructed when a caller explicitly expands the space.
"""

from __future__ import annotations

from dataclasses import dataclass

from .function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from .large_riemann_roch import (
    HessLargeRiemannRochSpace,
    HessPowerProductElement,
    hess_large_riemann_roch_space,
)


@dataclass(frozen=True, slots=True)
class HessDifferentialSpace:
    r"""A compact complete representation of ``Omega(D)``.

    ``coefficient_space`` is the complete Hess representation of
    ``L(div(dx) - D)``.  The result is factory-owned so the divisor identity
    cannot be separated from that completeness certificate.
    """

    lower_bound: AlgebraicDivisor
    canonical_divisor: AlgebraicDivisor
    coefficient_space: HessLargeRiemannRochSpace

    def __post_init__(self) -> None:
        model = self.lower_bound.model
        if (
            self.canonical_divisor.model is not model
            or self.coefficient_space.model is not model
        ):
            raise ValueError("a differential space must use one global model")
        if self.coefficient_space.divisor != (
            self.canonical_divisor - self.lower_bound
        ):
            raise ValueError("the coefficient space does not represent Omega(D)")

    @property
    def model(self) -> AlgebraicFunctionField:
        """Underlying algebraic function field."""
        return self.lower_bound.model

    @property
    def coefficient_divisor(self) -> AlgebraicDivisor:
        r"""The divisor ``div(dx) - D`` defining the coefficient space."""
        return self.coefficient_space.divisor

    @property
    def generators(self) -> tuple[HessPowerProductElement, ...]:
        """Rank-sized compact generators for differential coefficients."""
        return self.coefficient_space.generators

    @property
    def invariant_degrees(self) -> tuple[int, ...]:
        """Polynomial degree bounds attached to the compact generators."""
        return self.coefficient_space.invariant_degrees

    @property
    def dimension(self) -> int:
        """Dimension over the constant field without basis expansion."""
        return self.coefficient_space.dimension

    @property
    def reconstructs(self) -> bool:
        """Whether the compact coefficient space reconstructs its divisor."""
        return (
            self.coefficient_space.reconstructs
            and self.coefficient_divisor == self.canonical_divisor - self.lower_bound
        )

    def expand(self) -> tuple[AlgebraicDifferential, ...]:
        """Expand the complete differential basis on request."""
        return tuple(
            self.model.differential(coefficient)
            for coefficient in self.coefficient_space.expand()
        )

    def contains(self, differential: AlgebraicDifferential) -> bool:
        r"""Return whether ``div(differential) >= D``.

        The zero differential belongs to every differential space even though
        it has no finite divisor.
        """
        if not isinstance(differential, AlgebraicDifferential):
            raise TypeError("differential-space membership needs a differential")
        if differential.model is not self.model:
            raise ValueError("the differential belongs to another global model")
        if differential.coefficient == self.model.extension.field.zero:
            return True
        return (differential.divisor() - self.lower_bound).is_effective

    def verify_expansion(self) -> bool:
        """Expand and independently verify the coefficient-space basis."""
        # Multiplication by dx is an injective linear identification, so the
        # complete membership and rank check on L(div(dx) - D) also verifies
        # the corresponding differential basis without expanding it twice.
        return self.coefficient_space.verify_expansion()


def hess_differential_space(
    model: AlgebraicFunctionField,
    lower_bound: AlgebraicDivisor,
) -> HessDifferentialSpace:
    r"""Compute ``Omega(D)`` as ``L(div(dx) - D)`` using Algorithm 24."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a differential space needs an algebraic function field")
    if not isinstance(lower_bound, AlgebraicDivisor):
        raise TypeError("a differential space needs a divisor lower bound")
    if lower_bound.model is not model:
        raise ValueError("the divisor belongs to another global model")

    canonical_divisor = model.canonical_divisor()
    coefficient_space = hess_large_riemann_roch_space(
        model,
        canonical_divisor - lower_bound,
    )
    return HessDifferentialSpace(
        lower_bound,
        canonical_divisor,
        coefficient_space,
    )


def holomorphic_differential_space(
    model: AlgebraicFunctionField,
) -> HessDifferentialSpace:
    r"""Compute the space of holomorphic differentials ``Omega(0)``."""
    return hess_differential_space(model, AlgebraicDivisor(model))


__all__ = [
    "HessDifferentialSpace",
    "hess_differential_space",
    "holomorphic_differential_space",
]
