r"""Exact-derivative dispatch across implemented differential tower levels."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ._typing import FieldElement
from .monomial import PrimitiveExactDerivativeResult, solve_primitive_exact_derivative
from .rational import RationalExactDerivativeResult, solve_rational_exact_derivative
from .tower import Derivation, MonomialExtension, _coerce

ExactDerivativeCertificate = (
    RationalExactDerivativeResult | PrimitiveExactDerivativeResult
)


@dataclass(frozen=True, slots=True)
class ExactDerivativeResult:
    r"""Level-independent exact-derivative result using native domain elements."""

    exact_part: FieldElement
    obstruction: FieldElement
    certificate: ExactDerivativeCertificate | None

    @property
    def has_solution(self) -> bool:
        """Whether the input is a derivative in the current field."""
        return self.obstruction == 0

    @property
    def antiderivative(self) -> FieldElement | None:
        """Return the in-field antiderivative, or ``None`` if none exists."""
        return self.exact_part if self.has_solution else None


def _is_standard_rational_level(extension: MonomialExtension) -> bool:
    """Whether ``extension`` represents ``QQ(x)`` with ``Dx = 1``."""
    return (
        extension.parent_field == QQ
        and extension.generator_derivative
        == Poly(1, extension.generator, domain=QQ)
    )


def solve_exact_derivative(
    integrand: object,
    derivation: Derivation,
) -> ExactDerivativeResult:
    r"""Dispatch the exact-derivative problem at an implemented tower level.

    Results are normalized into ``derivation.field`` even though the rational
    base solver has an expression API and the primitive solver works directly
    with native fraction-field elements.
    """
    field = derivation.field
    element = _coerce(field, integrand)

    if derivation.level == -1:
        return ExactDerivativeResult(field.zero, element, None)

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "exact-derivative dispatch does not yet support algebraic levels"
        )

    certificate: RationalExactDerivativeResult | PrimitiveExactDerivativeResult
    if _is_standard_rational_level(extension):
        certificate = solve_rational_exact_derivative(
            field.to_sympy(element),
            extension.generator,
        )
    elif derivation.level > 0 and extension.generator_derivative.degree() == 0:
        certificate = solve_primitive_exact_derivative(
            element,
            derivation,
        )
    else:
        raise NotImplementedError(
            "exact-derivative dispatch currently supports rational and "
            "primitive levels"
        )

    return ExactDerivativeResult(
        _coerce(field, certificate.exact_part),
        _coerce(field, certificate.obstruction),
        certificate,
    )
