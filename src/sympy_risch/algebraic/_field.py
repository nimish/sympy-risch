r"""Exact low-level field conversions for algebraic algorithms."""

from __future__ import annotations

from typing import cast

from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed, PolynomialError
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction

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


def _quotient_extension_elements(
    extension: AlgebraicExtension,
    numerator: object,
    denominator: object,
) -> ExtensionElement:
    r"""Return ``numerator / denominator`` using exact polynomial Euclid.

    A separable algebraic extension is a field, but SymPy's native
    ``FiniteExtension`` inverse can incorrectly report a zero divisor when
    its rational-function parent has a general algebraic constant field.
    Working modulo the minimal polynomial over the parent field is the same
    operation, without that backend path.  This mirrors the construction used
    for :meth:`AlgebraicExtension.generator_derivative`.
    """
    top = _extension_element(extension, numerator)
    bottom = _extension_element(extension, denominator)
    if bottom == extension.field.zero:
        raise ZeroDivisionError("division by zero in an algebraic function field")

    numerator_polynomial = Poly(
        extension.field.to_sympy(top),
        extension.generator,
        domain=extension.parent_field,
    )
    denominator_polynomial = Poly(
        extension.field.to_sympy(bottom),
        extension.generator,
        domain=extension.parent_field,
    )
    inverse, _cofactor, gcd = denominator_polynomial.gcdex(
        extension.minimal_polynomial,
    )
    if gcd.degree() != 0:
        raise ZeroDivisionError("a nonzero algebraic field element was not invertible")
    inverse = inverse.mul_ground(
        extension.parent_field.one / extension.parent_field.convert(gcd.nth(0)),
    )
    return cast(
        ExtensionElement,
        extension._from_polynomial(numerator_polynomial * inverse),
    )


def _native_or_exact_quotient(
    extension: AlgebraicExtension,
    numerator: object,
    denominator: object,
) -> ExtensionElement:
    """Use native division, falling back only for its false zero divisors."""
    top = _extension_element(extension, numerator)
    bottom = _extension_element(extension, denominator)
    try:
        return cast(ExtensionElement, top / bottom)
    except ZeroDivisionError:
        return _quotient_extension_elements(extension, top, bottom)


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
    *,
    constants: Domain | None = None,
    variable: Symbol | None = None,
) -> tuple[FractionField, Domain, Symbol]:
    parent = extension.parent_field
    if not isinstance(parent, FractionField):
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires a one-variable "
            "rational parent field",
        )
    if (constants is None) != (variable is None):
        raise ValueError("constants and variable must be supplied together")
    if constants is not None and variable is not None:
        if not constants.is_Field or not constants.is_Exact:
            raise ValueError("the displayed constants must be an exact field")
        if isinstance(constants, FractionField):
            constant_symbols = constants.symbols
            flattened = (
                constant_symbols is not None
                and parent.domain == constants.domain
                and parent.symbols == (*constant_symbols, variable)
            )
        else:
            flattened = False
        nested = parent.domain == constants and parent.symbols == (variable,)
        if not (flattened or nested):
            raise ValueError(
                "the displayed constants and variable do not describe the parent field"
            )
        return parent, constants, variable

    symbols = parent.symbols
    inferred_constants = parent.domain
    if symbols is None or len(symbols) != 1:
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires a one-variable "
            "rational parent field",
        )
    if inferred_constants is None or not inferred_constants.is_Field:
        raise ValueError("the rational function field must be over a field")
    inferred_variable = symbols[0]
    if not isinstance(inferred_variable, Symbol):
        raise TypeError("the rational parent generator must be a Symbol")
    return parent, inferred_constants, inferred_variable


def _parent_element_polynomial(
    element: object,
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> Poly:
    element = parent.convert(element)
    if isinstance(constants, FractionField) and (
        element.numer.ring.domain != constants or element.numer.ring.gens != (variable,)
    ):
        try:
            return Poly(parent.to_sympy(element), variable, domain=constants)
        except (CoercionFailed, PolynomialError) as exc:
            raise ValueError(
                "expected a polynomial element of the parent field"
            ) from exc
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
    try:
        numerator = ring.from_dict(polynomial.rep.to_dict())
    except CoercionFailed:
        if not isinstance(polynomial.domain, FractionField):
            raise
        return cast(FracElement, parent.from_sympy(polynomial.as_expr()))
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
        if element.denom.ring.domain == constants and element.denom.ring.gens == (
            variable,
        ):
            element_denominator = Poly.from_dict(
                element.denom.to_dict(),
                variable,
                domain=constants,
            )
        elif isinstance(constants, FractionField):
            _numerator, denominator_expression = fraction(element.as_expr())
            element_denominator = Poly(
                denominator_expression,
                variable,
                domain=constants,
            )
        else:
            element_denominator = Poly.from_dict(
                element.denom.to_dict(),
                variable,
                domain=constants,
            )
        denominator = denominator.lcm(element_denominator)
    return denominator


def _algebraic_constant_domain(constants: Domain) -> Domain:
    """Return a constant domain that supports algebraic root adjunction."""
    if constants.is_QQ or constants.is_AlgebraicField or constants.is_FiniteField:
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
