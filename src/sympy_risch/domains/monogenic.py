r"""A checked monogenic extension implementing SymPy's field contract.

SymPy's :class:`~sympy.polys.agca.extensions.FiniteExtension` represents the
general quotient ``K[y]/(m)``.  That quotient can be a ring with zero divisors,
so the class cannot unconditionally provide field-domain operations.  It also
delegates some coercion and exact division to its polynomial representative,
which makes it unreliable as the coefficient field of another polynomial or
rational-function domain.

``MonogenicExtensionField`` is intentionally narrower: its parent is an exact
field and its modulus must be proved irreducible by the available SymPy
backend.  In exchange it implements genuine field division by inversion
modulo the modulus and supplies provenance-preserving embeddings into the
polynomial and rational-function domains constructed from it.

This module is independent of differential towers and integration.  It is a
candidate reusable replacement substrate; existing Risch call sites continue
to use SymPy's finite extension until they are migrated explicitly.
"""

from __future__ import annotations

from typing import cast

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.old_polynomialring import GlobalPolynomialRing
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
    CoercionFailed,
    DomainError,
    GeneratorsError,
    NotInvertible,
    NotReversible,
)
from sympy.polys.polytools import Poly


class _ExtensionPolynomialRing(PolynomialRing):
    """Polynomial ring that embeds its extension-field constants natively."""

    def from_MonogenicExtensionField(
        self,
        element: ExtensionElement,
        source: MonogenicExtensionField,
    ) -> object | None:
        target = cast(Domain, self.domain)
        if target == source and source.of_type(element):
            return self.new(target.convert(element))
        return None


class _ExtensionFractionField(FractionField):
    """Rational-function field with native extension-constant embedding."""

    def from_MonogenicExtensionField(
        self,
        element: ExtensionElement,
        source: MonogenicExtensionField,
    ) -> object | None:
        target = cast(Domain, self.domain)
        if target == source and source.of_type(element):
            return self.new(target.convert(element))
        return None


class MonogenicExtensionField(Field):
    r"""The field ``K[y]/(m)`` for a certified irreducible modulus ``m``.

    Construction is deliberately total only on the cell where SymPy can prove
    irreducibility.  A reducible modulus raises :class:`ValueError`; an
    unavailable irreducibility decision raises :class:`NotImplementedError`.
    The latter is a capability boundary, not evidence that the modulus is
    reducible.
    """

    alias = "MonogenicExtensionField"
    dtype = ExtensionElement
    is_FiniteExtension = True

    modulus: Poly
    mod: object
    domain: Domain
    dom: Domain
    ring: GlobalPolynomialRing
    rank: int
    symbol: Basic
    generator: ExtensionElement
    basis: tuple[ExtensionElement, ...]
    zero: ExtensionElement
    one: ExtensionElement
    gens: tuple[ExtensionElement, ...]
    ngens: int

    def __init__(self, modulus: Poly) -> None:
        if not isinstance(modulus, Poly) or not modulus.is_univariate:
            raise TypeError("modulus must be a univariate SymPy Poly")
        if modulus.degree() < 1:
            raise ValueError("modulus must have positive degree")
        parent = modulus.domain
        if not parent.is_Field or not parent.is_Exact:
            raise TypeError("a monogenic extension field needs an exact field parent")

        modulus = modulus.monic(auto=False)
        try:
            irreducible = modulus.is_irreducible
        except (CoercionFailed, DomainError, NotImplementedError) as exc:
            raise NotImplementedError(
                "irreducibility is unavailable over this exact parent field"
            ) from exc
        if irreducible is False:
            raise ValueError("the extension modulus must be irreducible")
        if irreducible is not True:
            raise NotImplementedError(
                "irreducibility is unavailable over this exact parent field"
            )

        self.modulus = modulus
        self.mod = modulus.rep
        self.domain = self.dom = parent
        self.rank = modulus.degree()
        self.ring = parent.old_poly_ring(*modulus.gens)
        self.symbol = modulus.gens[0]

        generator_rep = self.ring.gens[0]
        self.zero = self._from_rep(self.ring.zero)
        self.one = self._from_rep(self.ring.one)
        self.generator = self._from_rep(generator_rep)
        self.gens = (self.generator,)
        self.ngens = 1
        self.basis = tuple(
            self._from_rep(generator_rep**power) for power in range(self.rank)
        )

    def _from_rep(self, representative: object) -> ExtensionElement:
        """Construct an element from a representative in the internal ring."""
        converted = self.ring.convert(representative)
        return ExtensionElement(converted % self.mod, self)

    def _embed_parent(
        self, value: object, source: Domain | None = None
    ) -> ExtensionElement:
        """Embed a value from the exact parent field as a constant."""
        converted = (
            self.domain.convert(value)
            if source is None
            else self.domain.convert(value, source)
        )
        return ExtensionElement(self.ring._ground_new(converted), self)

    def new(self, *args: object) -> ExtensionElement:
        """Construct from a representative or a proven parent-field value."""
        if len(args) != 1:
            raise TypeError("an extension element needs exactly one representative")
        value = args[0]
        if self.of_type(value):
            return self._from_rep(cast(ExtensionElement, value).rep)
        if isinstance(value, DomainElement) and value.parent() == self.domain:
            return self._embed_parent(value)
        return self._from_rep(value)

    def of_type(self, element: object) -> bool:
        """Require this field as provenance, not merely the element class."""
        return isinstance(element, ExtensionElement) and element.ext is self

    def convert(
        self,
        element: object,
        base: Domain | None = None,
    ) -> ExtensionElement:
        """Convert through checked embeddings and canonicalize owned elements."""
        if base is None and self.of_type(element):
            return self._from_rep(cast(ExtensionElement, element).rep)
        return cast(ExtensionElement, super().convert(element, base))

    def convert_from(self, element: object, base: Domain) -> ExtensionElement:
        """Convert only through this field or its explicitly embedded parent."""
        if base == self:
            if not isinstance(element, ExtensionElement) or element.ext != base:
                raise CoercionFailed("element does not belong to this extension field")
            return self._from_rep(element.rep)
        if base == self.domain:
            return self._embed_parent(element, base)
        if isinstance(base, MonogenicExtensionField):
            raise CoercionFailed(
                "finite extensions require an explicit embedding between fields"
            )
        try:
            return self._embed_parent(element, base)
        except (CoercionFailed, TypeError, ValueError) as exc:
            raise CoercionFailed(
                f"cannot embed an element of {base} into {self} without an explicit map"
            ) from exc

    def from_MonogenicExtensionField(
        self,
        element: ExtensionElement,
        source: MonogenicExtensionField,
    ) -> ExtensionElement | None:
        """Embed the immediate parent extension and reject unrelated fields."""
        if source == self:
            return self.convert_from(element, source)
        if source == self.domain:
            return self._embed_parent(element, source)
        return None

    def to_sympy(self, a: ExtensionElement) -> Basic:
        """Convert an owned element to a SymPy expression."""
        a = self.convert(a)
        return self.ring.to_sympy(a.rep)

    def from_sympy(self, a: Basic) -> ExtensionElement:
        """Convert a polynomial or rational expression in the generator."""
        if not isinstance(a, Expr):
            raise CoercionFailed(f"cannot convert {a} to {self}")
        try:
            parent_value = self.domain.from_sympy(a)
        except (CoercionFailed, TypeError, ValueError):
            pass
        else:
            return self._embed_parent(parent_value)

        numerator, denominator = a.as_numer_denom()
        try:
            top = self._from_rep(self.ring.convert(numerator))
            bottom = self._from_rep(self.ring.convert(denominator))
        except (CoercionFailed, TypeError, ValueError) as exc:
            raise CoercionFailed(f"cannot convert {a} to {self}") from exc
        return cast(ExtensionElement, top / bottom)

    def characteristic(self) -> int:
        """Return the characteristic inherited from the parent field."""
        return int(self.domain.characteristic())

    @property
    def has_CharacteristicZero(self) -> bool:
        """Whether the parent, and hence this extension, has characteristic zero."""
        return bool(self.domain.has_CharacteristicZero)

    def to_dict(self, element: ExtensionElement) -> dict[tuple[int, ...], object]:
        """Return power-basis coefficients as a sparse monomial dictionary."""
        element = self.convert(element)
        return element.rep.to_dict()

    def exquo(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> ExtensionElement:
        """Divide in the field by modular polynomial inversion."""
        top = self.convert(a)
        bottom = self.convert(b)
        if not bottom:
            raise ZeroDivisionError("division by zero in a monogenic extension field")
        try:
            inverse = bottom.rep.invert(self.mod)
        except NotInvertible as exc:
            raise ZeroDivisionError(
                "a nonzero element was not invertible modulo an irreducible modulus"
            ) from exc
        return ExtensionElement((top.rep * inverse) % self.mod, self)

    quo = exquo

    def rem(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> ExtensionElement:
        """Return zero because every defined field quotient is exact."""
        self.exquo(a, b)
        return self.zero

    def div(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> tuple[ExtensionElement, ExtensionElement]:
        """Return exact quotient and zero remainder."""
        return self.exquo(a, b), self.zero

    def gcd(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> ExtensionElement:
        """Return the normalized field gcd."""
        self.convert(a)
        self.convert(b)
        return self.one

    def gcdex(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> tuple[ExtensionElement, ExtensionElement, ExtensionElement]:
        """Return Bezout coefficients for the normalized field gcd."""
        a = self.convert(a)
        b = self.convert(b)
        if not a:
            if not b:
                return self.zero, self.one, self.zero
            return self.zero, self.exquo(self.one, b), self.one
        return self.exquo(self.one, a), self.zero, self.one

    def lcm(
        self,
        a: ExtensionElement,
        b: ExtensionElement,
    ) -> ExtensionElement:
        """Return the field-normalized least common multiple."""
        return cast(ExtensionElement, self.convert(a) * self.convert(b))

    def revert(self, a: ExtensionElement) -> ExtensionElement:
        """Return the multiplicative inverse of a nonzero element."""
        try:
            return self.exquo(self.one, a)
        except ZeroDivisionError as exc:
            raise NotReversible(f"{a} is not reversible in {self}") from exc

    def is_unit(self, a: ExtensionElement) -> bool:
        """Every nonzero element of this field is a unit."""
        return bool(self.convert(a))

    def is_negative(self, a: ExtensionElement) -> bool:
        """Algebraic extension fields have no canonical ordering."""
        self.convert(a)
        return False

    def get_field(self) -> MonogenicExtensionField:
        """Return this field domain."""
        return self

    def set_domain(self, parent: Domain) -> MonogenicExtensionField:
        """Rebuild over another parent, rechecking irreducibility."""
        return type(self)(self.modulus.set_domain(parent))

    def drop(self, *symbols: Basic) -> MonogenicExtensionField:
        """Drop symbols from the parent without dropping this generator."""
        if self.symbol in symbols:
            raise GeneratorsError("cannot drop the algebraic generator")
        return self.set_domain(self.domain.drop(*symbols))

    def poly_ring(self, *symbols: Basic, order: object = lex) -> PolynomialRing:
        """Construct a polynomial ring with native constant embedding."""
        return _ExtensionPolynomialRing(self, symbols, order)

    def frac_field(self, *symbols: Basic, order: object = lex) -> FractionField:
        """Construct a rational-function field with native constant embedding."""
        return _ExtensionFractionField(self, symbols, order)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MonogenicExtensionField):
            return NotImplemented
        return self.modulus == other.modulus

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.modulus))

    def __str__(self) -> str:
        return f"{self.domain}<{self.symbol}>/({self.modulus.as_expr()})"

    __repr__ = __str__


__all__ = ["MonogenicExtensionField"]
