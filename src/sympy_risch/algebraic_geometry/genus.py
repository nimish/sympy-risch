r"""Independent genus consistency checks for Hess function fields.

The relative different gives a canonical divisor ``K`` and therefore the
Riemann--Hurwitz value

``g = (deg(K) + 2) / 2``.

Independently, Hess Algorithm 13 computes ``L(K)`` directly from the finite
and infinite ideal lattices. The canonical-space theorem identifies
``dim L(K)`` with the genus. Agreement checks the different, both maximal
orders, and the direct Riemann--Roch lattice construction without routing
through Algorithm 24 or :attr:`AlgebraicFunctionField.genus`.

Both identities are stated over the full constant field. The public factory
therefore requires model-owned :class:`FunctionFieldConstantCertificate`
evidence rather than treating a displayed coefficient field as exact by
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constant_fields import FunctionFieldConstantCertificate
from .function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from .hess import HessParametricBasis, hess_parametric_basis


@dataclass(frozen=True, slots=True)
class HessGenusConsistency:
    """Two independently computed genus values for one global model."""

    model: AlgebraicFunctionField
    constant_field_certificate: FunctionFieldConstantCertificate
    canonical_divisor: AlgebraicDivisor
    canonical_space: HessParametricBasis

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a genus consistency result needs a function field")
        if not isinstance(
            self.constant_field_certificate,
            FunctionFieldConstantCertificate,
        ):
            raise TypeError("a genus consistency result needs full-constant evidence")
        if self.constant_field_certificate.model is not self.model:
            raise ValueError("the constant-field certificate belongs to another model")
        if not isinstance(self.canonical_divisor, AlgebraicDivisor):
            raise TypeError("a genus consistency result needs a canonical divisor")
        if self.canonical_divisor.model is not self.model:
            raise ValueError("the canonical divisor belongs to another model")
        if not isinstance(self.canonical_space, HessParametricBasis):
            raise TypeError("a genus consistency result needs an Algorithm 13 space")
        if self.canonical_space.model is not self.model:
            raise ValueError("the canonical space belongs to another model")
        if self.canonical_space.divisor != self.canonical_divisor:
            raise ValueError("the Algorithm 13 space is not L(K)")

    @property
    def canonical_degree(self) -> int:
        """Degree of the different-derived canonical divisor."""
        return self.canonical_divisor.degree

    @property
    def riemann_hurwitz_genus(self) -> int:
        """Genus obtained from ``deg(K) = 2*g - 2``."""
        numerator = self.canonical_degree + 2
        if numerator % 2:
            raise ValueError("the canonical divisor has impossible odd degree")
        genus = numerator // 2
        if genus < 0:
            raise ValueError("the canonical divisor gives a negative genus")
        return genus

    @property
    def holomorphic_dimension(self) -> int:
        """Dimension of ``L(K)`` computed directly by Algorithm 13."""
        return self.canonical_space.dimension()

    @property
    def consistent(self) -> bool:
        """Whether Riemann--Hurwitz and the direct canonical space agree."""
        return self.riemann_hurwitz_genus == self.holomorphic_dimension


def hess_genus_consistency(
    model: AlgebraicFunctionField,
    constant_field_certificate: FunctionFieldConstantCertificate,
) -> HessGenusConsistency:
    r"""Compare the different genus with a certified direct Algorithm 13 ``L(K)``."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("genus consistency needs an algebraic function field")
    if not isinstance(
        constant_field_certificate,
        FunctionFieldConstantCertificate,
    ):
        raise TypeError("genus consistency needs a constant-field certificate")
    if constant_field_certificate.model is not model:
        raise ValueError("the constant-field certificate belongs to another model")
    canonical_divisor = model.canonical_divisor()
    canonical_space = hess_parametric_basis(model, canonical_divisor)
    return HessGenusConsistency(
        model,
        constant_field_certificate,
        canonical_divisor,
        canonical_space,
    )


__all__ = [
    "HessGenusConsistency",
    "hess_genus_consistency",
]
