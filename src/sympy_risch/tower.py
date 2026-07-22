r"""Immutable differential towers over exact SymPy domains.

This module contains the small kernel shared by later Risch reductions.  A
tower is a tuple of differential extensions, and a :class:`Derivation` is
bound to one explicit level of that tuple.  Recursive differentiation creates
a lower-level derivation instead of mutating shared tower state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Protocol

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction


class DifferentialExtension(Protocol):
    """Structural interface implemented by one differential tower level."""

    @property
    def generator(self) -> Symbol: ...

    @property
    def parent_field(self) -> Domain: ...

    @property
    def field(self) -> Domain: ...

    def differentiate(
        self,
        element: Any,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Any:
        """Differentiate ``element`` using a derivation on the parent."""


def _coerce(domain: Domain, element: Any) -> Any:
    """Coerce to ``domain``, including between nested fraction fields.

    SymPy's generic ``Domain.convert`` does not currently embed every
    ``FracElement`` from a nested ground field.  Conversion through the exact
    SymPy expression is a reliable boundary fallback and still returns a
    native domain element.
    """
    try:
        return domain.convert(element)
    except (CoercionFailed, ValueError):
        if isinstance(element, Expr):
            expression = element
        else:
            try:
                expression = element.as_expr()
            except AttributeError:
                raise
        return domain.from_sympy(expression)


def _domain_symbols(domain: Domain) -> set[Symbol]:
    """Return generators used throughout a nested SymPy domain."""
    symbols: set[Symbol] = set()
    current: Domain | None = domain

    while current is not None:
        symbols.update(getattr(current, "symbols", ()) or ())
        current = getattr(current, "domain", None)

    return symbols


def _adjoin_transcendental(parent_field: Domain, generator: Symbol) -> Domain:
    """Adjoin a generator while keeping consecutive fraction fields flat.

    A native ``QQ(x, t)`` domain is mathematically the same field as the
    conceptual tower level ``QQ(x)(t)``.  The flat representation avoids
    coercion failures in SymPy's normalization of deeply nested fraction
    fields while the tower metadata retains the coefficient-field hierarchy.
    """
    if isinstance(parent_field, FractionField):
        return parent_field.domain.frac_field(*parent_field.symbols, generator)
    return parent_field.frac_field(generator)


@dataclass(frozen=True, slots=True)
class MonomialExtension:
    r"""A transcendental extension ``K(t)`` with ``Dt`` in ``K[t]``.

    The univariate polynomial simultaneously identifies the generator, the
    parent field, and its derivative law.  This avoids separately stored
    metadata becoming inconsistent.

    Examples
    ========

    ``QQ(x)`` with ``Dx = 1`` is represented by
    ``MonomialExtension(Poly(1, x, domain=QQ))``.
    """

    generator_derivative: Poly
    _field: Domain = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        derivative = self.generator_derivative

        if not isinstance(derivative, Poly):
            raise TypeError("generator_derivative must be a SymPy Poly")
        if len(derivative.gens) != 1:
            raise ValueError("a monomial extension needs exactly one generator")
        if not derivative.domain.is_Field:
            raise ValueError("the parent domain must be a field")
        if derivative.domain.characteristic() != 0:
            raise ValueError("differential towers require characteristic zero")
        if derivative.is_zero:
            raise ValueError("a zero-derivative generator belongs in the constants")
        if self.generator in _domain_symbols(self.parent_field):
            raise ValueError(f"generator {self.generator} already occurs in the parent field")

        object.__setattr__(
            self,
            "_field",
            _adjoin_transcendental(self.parent_field, self.generator),
        )

    @property
    def generator(self) -> Symbol:
        """The transcendental generator ``t``."""
        return self.generator_derivative.gens[0]

    @property
    def parent_field(self) -> Domain:
        """The coefficient field ``K``."""
        return self.generator_derivative.domain

    @property
    def field(self) -> Domain:
        """The rational function field ``K(t)``."""
        return self._field

    def _validate_polynomial(self, polynomial: Poly) -> None:
        if polynomial.gens != (self.generator,):
            raise ValueError("polynomial generator does not match the tower level")
        if polynomial.domain != self.parent_field:
            raise ValueError("polynomial coefficients are not in the parent field")

    def differentiate_coefficients(
        self,
        polynomial: Poly,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Poly:
        """Differentiate only the coefficients of ``polynomial``."""
        self._validate_polynomial(polynomial)
        return Poly.from_dict(
            {
                monomial: coefficient_derivation(coefficient)
                for monomial, coefficient in polynomial.terms()
            },
            self.generator,
            domain=self.parent_field,
        )

    def differentiate_polynomial(
        self,
        polynomial: Poly,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Poly:
        """Differentiate a polynomial in the extension generator."""
        coefficient_part = self.differentiate_coefficients(
            polynomial,
            coefficient_derivation,
        )
        return coefficient_part + polynomial.diff()*self.generator_derivative

    def differentiate(
        self,
        element: Any,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Any:
        """Differentiate an element of ``K(t)`` by the quotient rule."""
        numerator_expression, denominator_expression = fraction(element.as_expr())
        numerator = Poly(
            numerator_expression,
            self.generator,
            domain=self.parent_field,
        )
        denominator = Poly(
            denominator_expression,
            self.generator,
            domain=self.parent_field,
        )
        numerator_derivative = self.differentiate_polynomial(
            numerator,
            coefficient_derivation,
        )
        denominator_derivative = self.differentiate_polynomial(
            denominator,
            coefficient_derivation,
        )
        result_numerator = (
            numerator_derivative*denominator
            - numerator*denominator_derivative
        )
        result_denominator = denominator**2

        return self.field.from_sympy(
            result_numerator.as_expr()/result_denominator.as_expr()
        )


@dataclass(frozen=True, slots=True)
class AlgebraicExtension:
    r"""A finite algebraic extension ``K[y]/(m(y))``.

    The minimal polynomial must be monic and irreducible over the parent
    field.  The derivative of the generator is computed by implicit
    differentiation whenever the extension derivation is evaluated.
    """

    minimal_polynomial: Poly
    _field: Domain = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        modulus = self.minimal_polynomial

        if not isinstance(modulus, Poly):
            raise TypeError("minimal_polynomial must be a SymPy Poly")
        if len(modulus.gens) != 1:
            raise ValueError("an algebraic extension needs exactly one generator")
        if not modulus.domain.is_Field:
            raise ValueError("the parent domain must be a field")
        if modulus.domain.characteristic() != 0:
            raise ValueError("differential towers require characteristic zero")
        if modulus.degree() < 2:
            raise ValueError("an algebraic extension must have degree at least two")
        if modulus.LC() != modulus.domain.one:
            raise ValueError("the minimal polynomial must be monic")
        if modulus.gcd(modulus.diff()).degree() != 0:
            raise ValueError("the minimal polynomial must be square-free")
        if modulus.is_irreducible is not True:
            raise ValueError("the minimal polynomial must be irreducible")
        if self.generator in _domain_symbols(self.parent_field):
            raise ValueError(f"generator {self.generator} already occurs in the parent field")

        object.__setattr__(self, "_field", FiniteExtension(modulus))

    @property
    def generator(self) -> Symbol:
        """The algebraic generator ``y``."""
        return self.minimal_polynomial.gens[0]

    @property
    def parent_field(self) -> Domain:
        """The coefficient field ``K``."""
        return self.minimal_polynomial.domain

    @property
    def field(self) -> Domain:
        """The native SymPy finite-extension field."""
        return self._field

    def _differentiate_coefficients(
        self,
        polynomial: Poly,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Poly:
        return Poly.from_dict(
            {
                monomial: coefficient_derivation(coefficient)
                for monomial, coefficient in polynomial.terms()
            },
            self.generator,
            domain=self.parent_field,
        )

    def generator_derivative(
        self,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Any:
        r"""Return ``Dy = -D_K(m)/partial_y(m)``."""
        coefficient_part = self._differentiate_coefficients(
            self.minimal_polynomial,
            coefficient_derivation,
        )
        partial_generator = self.minimal_polynomial.diff()
        return -self.field.from_sympy(coefficient_part.as_expr()) / self.field.from_sympy(
            partial_generator.as_expr()
        )

    def differentiate(
        self,
        element: Any,
        coefficient_derivation: Callable[[Any], Any],
    ) -> Any:
        """Differentiate a reduced representative modulo the minimal polynomial."""
        representative = Poly(
            element.as_expr(),
            self.generator,
            domain=self.parent_field,
        )
        coefficient_part = self._differentiate_coefficients(
            representative,
            coefficient_derivation,
        )
        generator_part = (
            self.field.from_sympy(representative.diff().as_expr())
            * self.generator_derivative(coefficient_derivation)
        )
        return self.field.from_sympy(coefficient_part.as_expr()) + generator_part


@dataclass(frozen=True, slots=True)
class DifferentialTower:
    """An immutable sequence of differential-field extensions."""

    constants: Domain
    levels: tuple[DifferentialExtension, ...] = ()

    def __post_init__(self) -> None:
        if not self.constants.is_Field:
            raise ValueError("the constant domain must be a field")
        if self.constants.characteristic() != 0:
            raise ValueError("differential towers require characteristic zero")

        levels = tuple(self.levels)
        object.__setattr__(self, "levels", levels)

        expected_parent = self.constants
        generators: set[Symbol] = set()

        for level in levels:
            try:
                parent_field = level.parent_field
                generator = level.generator
                extension_field = level.field
                level.differentiate
            except AttributeError as exc:
                raise TypeError("tower levels must be differential extensions") from exc

            if parent_field != expected_parent:
                raise ValueError("extension parent does not match the top tower field")
            if generator in generators:
                raise ValueError(f"duplicate tower generator {generator}")

            generators.add(generator)
            expected_parent = extension_field

    @property
    def top(self) -> int:
        """Index of the top extension, or ``-1`` for the constants."""
        return len(self.levels) - 1

    def field(self, level: int | None = None) -> Domain:
        """Return the field at ``level``; the constant field is level ``-1``."""
        if level is None:
            level = self.top

        if level == -1:
            return self.constants
        if not 0 <= level < len(self.levels):
            raise IndexError(f"tower level {level} is out of range")

        return self.levels[level].field

    def extend(self, extension: DifferentialExtension) -> DifferentialTower:
        """Return a new tower with ``extension`` appended."""
        return DifferentialTower(self.constants, self.levels + (extension,))

    def derivation(self, level: int | None = None) -> Derivation:
        """Return an immutable derivation bound to ``level``."""
        if level is None:
            level = self.top
        return Derivation(self, level)


@dataclass(frozen=True, slots=True)
class Derivation:
    """The derivation on one explicit level of a differential tower."""

    tower: DifferentialTower
    level: int

    def __post_init__(self) -> None:
        if not -1 <= self.level <= self.tower.top:
            raise IndexError(f"tower level {self.level} is out of range")

    @property
    def field(self) -> Domain:
        """Field on which this derivation operates."""
        return self.tower.field(self.level)

    def lower(self) -> Derivation:
        """Return the derivation on the next lower field."""
        if self.level == -1:
            raise ValueError("the constant field has no lower derivation")
        return Derivation(self.tower, self.level - 1)

    def __call__(self, element: Any) -> Any:
        """Differentiate ``element`` and return a native domain element."""
        element = _coerce(self.field, element)

        if self.level == -1:
            return self.field.zero

        extension = self.tower.levels[self.level]
        return extension.differentiate(element, self.lower())

    def coefficients(self, polynomial: Poly) -> Poly:
        """Differentiate only a polynomial's lower-field coefficients."""
        if self.level == -1:
            raise ValueError("the constant field has no polynomial generator")

        extension = self.tower.levels[self.level]
        if not isinstance(extension, MonomialExtension):
            raise TypeError("coefficient differentiation requires a monomial level")
        return extension.differentiate_coefficients(polynomial, self.lower())

    def polynomial(self, polynomial: Poly) -> Poly:
        """Differentiate a polynomial at this monomial level."""
        if self.level == -1:
            raise ValueError("the constant field has no polynomial generator")

        extension = self.tower.levels[self.level]
        if not isinstance(extension, MonomialExtension):
            raise TypeError("polynomial differentiation requires a monomial level")
        return extension.differentiate_polynomial(polynomial, self.lower())

    def fraction(self, numerator: Poly, denominator: Poly) -> Any:
        """Differentiate the fraction ``numerator/denominator``."""
        if self.level == -1:
            raise ValueError("the constant field has no polynomial generator")

        extension = self.tower.levels[self.level]
        if not isinstance(extension, MonomialExtension):
            raise TypeError("fraction differentiation requires a monomial level")

        extension._validate_polynomial(numerator)
        extension._validate_polynomial(denominator)
        if denominator.is_zero:
            raise ZeroDivisionError("fraction denominator is zero")
        element = extension.field.from_sympy(
            numerator.as_expr()/denominator.as_expr()
        )
        return extension.differentiate(element, self.lower())
