r"""Shared top-generator representations and base-tower recognition."""

from __future__ import annotations

from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..reduction.polynomials import canonicalize_polynomial
from ..tower import Derivation, MonomialExtension, _coerce
from ._common import _field_elements_equal


def _as_primitive_fraction(
    value: object,
    extension: MonomialExtension,
) -> tuple[FieldElement, Poly, Poly]:
    """Return an element and coprime polynomials in the top generator."""
    field = extension.field
    parent = extension.parent_field
    element = _coerce(
        field,
        value.as_expr() if isinstance(value, Poly) else value,
    )
    numerator_expression, denominator_expression = cancel(
        field.to_sympy(element)
    ).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        extension.generator,
        domain=parent,
    )
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=parent,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(parent.convert(coefficient))
    leading_inverse = parent.one / parent.convert(denominator.LC())
    numerator = canonicalize_polynomial(numerator.mul_ground(leading_inverse))
    denominator = canonicalize_polynomial(denominator.monic())
    if not _field_elements_equal(
        _coerce(field, numerator.as_expr() / denominator.as_expr()),
        element,
        field,
    ):
        raise ValueError("primitive fraction normalization failed")
    return element, numerator, denominator


def _as_primitive_polynomial(
    value: object,
    extension: MonomialExtension,
) -> Poly:
    """Coerce a top-field element known to be polynomial in the generator."""
    _, numerator, denominator = _as_primitive_fraction(
        value,
        extension,
    )
    if denominator.degree() > 0:
        raise NotImplementedError(
            "primitive RDE coefficients with top-generator poles are not implemented"
        )
    scale = extension.parent_field.one / extension.parent_field.convert(
        denominator.LC()
    )
    return numerator.mul_ground(scale)


def _positive_integer_rational(value: object) -> int | None:
    """Return a positive rational integer, or ``None`` otherwise."""
    rational = QQ.convert(value)
    if QQ.denom(rational) != 1 or QQ.numer(rational) <= 0:
        return None
    return int(QQ.numer(rational))


def _positive_integer_in_domain(value: object, domain: Domain) -> int | None:
    """Return a positive integer represented in ``domain``, if any."""
    expression = domain.to_sympy(domain.convert(value))
    if expression.is_Integer is not True or expression.is_positive is not True:
        return None
    return int(expression)


def _is_standard_rational_derivation_over(
    derivation: Derivation,
    constants: Domain,
) -> bool:
    """Whether ``derivation`` is ``d/dx`` on ``constants(x)``."""
    if derivation.level != 0:
        return False
    extension = derivation.tower.levels[0]
    return (
        isinstance(extension, MonomialExtension)
        and extension.parent_field == constants
        and extension.generator_derivative
        == Poly(1, extension.generator, domain=constants)
    )


def _is_standard_rational_derivation(derivation: Derivation) -> bool:
    """Whether the derivation is ``d/dx`` on the standard ``QQ(x)`` base."""
    return _is_standard_rational_derivation_over(derivation, QQ)
