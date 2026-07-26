r"""Exact low-level field conversions for algebraic algorithms."""

from __future__ import annotations

from typing import cast

from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly

from .._typing import SupportsAsExpr
from ..tower import AlgebraicExtension


def _extension_element(
    extension: AlgebraicExtension,
    value: object,
) -> ExtensionElement:
    if getattr(value, "ext", None) == extension.field:
        return cast(ExtensionElement, value)
    if isinstance(value, SupportsAsExpr):
        value = value.as_expr()
    return cast(ExtensionElement, extension.field.from_sympy(value))


def _evaluate_constant_polynomial(
    extension: AlgebraicExtension,
    polynomial: Poly,
    value: ExtensionElement,
) -> ExtensionElement:
    """Evaluate a polynomial over the displayed constants in ``extension``."""
    result = extension.field.zero
    for power in range(polynomial.degree(), -1, -1):
        coefficient = polynomial.domain.convert(polynomial.nth(power))
        result = result * value + _extension_element(
            extension,
            polynomial.domain.to_sympy(coefficient),
        )
    return cast(ExtensionElement, result)


def _same_univariate_polynomial_coefficients(left: Poly, right: Poly) -> bool:
    """Compare univariate polynomials while allowing different dummy names."""
    return (
        left.domain == right.domain
        and left.degree() == right.degree()
        and all(
            left.nth(power) == right.nth(power) for power in range(left.degree() + 1)
        )
    )


def _base_polynomial_ring(
    extension: AlgebraicExtension,
) -> tuple[FractionField, Domain, Symbol]:
    parent = extension.parent_field
    if not isinstance(parent, FractionField):
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires a one-variable "
            "rational parent field",
        )
    symbols = parent.symbols
    constants = parent.domain
    if symbols is None or len(symbols) != 1:
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires a one-variable "
            "rational parent field",
        )
    if constants is None or not constants.is_Field:
        raise ValueError("the rational function field must be over a field")
    variable = symbols[0]
    if not isinstance(variable, Symbol):
        raise TypeError("the rational parent generator must be a Symbol")
    return parent, constants, variable


def _parent_element_polynomial(
    element: object,
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> Poly:
    element = parent.convert(element)
    denominator_terms = element.denom.to_dict()
    if any(
        any(exponent != 0 for exponent in monomial) for monomial in denominator_terms
    ):
        raise ValueError("expected a polynomial element of the parent field")
    denominator_constant = denominator_terms.get((0,), constants.one)
    return Poly.from_dict(
        {
            monomial: coefficient / denominator_constant
            for monomial, coefficient in element.numer.to_dict().items()
        },
        variable,
        domain=constants,
    )


def _polynomial_parent_element(
    polynomial: Poly,
    parent: FractionField,
) -> FracElement:
    """Embed a polynomial directly into its fraction field."""
    ring = parent.field.ring
    numerator = ring.from_dict(polynomial.rep.to_dict())
    dtype = parent.dtype
    if dtype is None:
        raise TypeError("the rational parent field has no element type")
    return cast(FracElement, dtype(numerator, ring.one))


def _common_denominator(
    elements: tuple[FracElement, ...],
    constants: Domain,
    variable: Symbol,
) -> Poly:
    denominator = Poly(1, variable, domain=constants)
    for element in elements:
        element_denominator = Poly.from_dict(
            element.denom.to_dict(),
            variable,
            domain=constants,
        )
        denominator = denominator.lcm(element_denominator)
    return denominator


def _algebraic_constant_domain(constants: Domain) -> Domain:
    """Return a constant domain that supports algebraic root adjunction."""
    if constants.is_QQ or constants.is_AlgebraicField:
        return constants
    as_algebraic_field = getattr(constants, "as_AlgebraicField", None)
    if as_algebraic_field is None:
        raise NotImplementedError(
            "finite-place factorization currently requires rational or "
            "algebraic-number constants"
        )
    return cast(Domain, as_algebraic_field())


def _embed_parent(
    extension: AlgebraicExtension,
    element: object,
) -> ExtensionElement:
    parent = extension.parent_field
    if not isinstance(parent, FractionField):
        raise NotImplementedError(
            "integral-basis refinement requires a rational parent field"
        )
    if isinstance(element, Poly):
        element = _polynomial_parent_element(element, parent)
    else:
        element = parent.convert(element)
    representative = DMP([element], parent, 0)
    return extension.field.dtype(representative, extension.field)
