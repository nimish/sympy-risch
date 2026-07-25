r"""Immutable differential towers over exact SymPy domains.

This module contains the small kernel shared by later Risch reductions.  A
tower is a tuple of differential extensions, and a :class:`Derivation` is
bound to one explicit level of that tuple.  Recursive differentiation creates
a lower-level derivation instead of mutating shared tower state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Protocol, cast

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import fraction

from ._typing import FieldElement, SupportsAsExpr

CoefficientDerivation = Callable[[object], FieldElement]


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
        element: FieldElement,
        coefficient_derivation: CoefficientDerivation,
    ) -> FieldElement:
        """Differentiate ``element`` using a derivation on the parent."""


def _coerce(domain: Domain, element: object) -> FieldElement:
    """Coerce to ``domain``, including between nested fraction fields.

    SymPy's generic ``Domain.convert`` does not currently embed every
    ``FracElement`` from a nested ground field.  Conversion through the exact
    SymPy expression is a reliable boundary fallback and still returns a
    native domain element.
    """
    try:
        return cast(FieldElement, domain.convert(element))
    except (CoercionFailed, ValueError):
        if isinstance(element, Expr):
            expression = element
        else:
            expression = cast(SupportsAsExpr, element).as_expr()
        return cast(FieldElement, domain.from_sympy(expression))


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
        ground = cast(Domain, parent_field.domain)
        generators = cast(tuple[Symbol, ...], parent_field.symbols)
        return ground.frac_field(*generators, generator)
    return parent_field.frac_field(generator)


class MonomialFamily(Enum):
    """Structural family determined by the polynomial derivative law."""

    PRIMITIVE = "primitive"
    HYPEREXPONENTIAL = "hyperexponential"
    HYPERTANGENT = "hypertangent"
    POLYNOMIAL = "polynomial"


class ConstantFieldStatus(Enum):
    r"""Knowledge about constants introduced by a transcendental level.

    The structure theorems used for negative Risch decisions require
    :math:`\operatorname{Const}(K(t))=\operatorname{Const}(K)`.  Until an
    automatic tower builder proves that fact, manually constructed levels must
    record whether it is asserted, disproved, or still unknown.
    """

    UNKNOWN = "unknown"
    PRESERVED = "preserved"
    ENLARGED = "enlarged"


class SpecialPolynomialStatus(Enum):
    r"""Knowledge about nonconstant special polynomials at a monomial level.

    A polynomial :math:`p\in K[t]` is special when :math:`p\mid Dp`.
    Bronstein's nonlinear no-special case assumes that no nonconstant
    irreducible special polynomial exists.  Until a tower builder proves that
    structure theorem hypothesis, a manually constructed level must record it
    explicitly.
    """

    UNKNOWN = "unknown"
    ABSENT = "absent"
    PRESENT = "present"


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

    The derivative law alone cannot establish that adjoining the formal
    generator preserves the constant field.  Set ``constant_field_status`` to
    :attr:`ConstantFieldStatus.PRESERVED` only when that hypothesis has been
    proved or is an explicit construction assumption.
    """

    generator_derivative: Poly
    constant_field_status: ConstantFieldStatus = ConstantFieldStatus.UNKNOWN
    special_polynomial_status: SpecialPolynomialStatus = (
        SpecialPolynomialStatus.UNKNOWN
    )
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
        if not isinstance(self.constant_field_status, ConstantFieldStatus):
            raise TypeError(
                "constant_field_status must be a ConstantFieldStatus"
            )
        if not isinstance(self.special_polynomial_status, SpecialPolynomialStatus):
            raise TypeError(
                "special_polynomial_status must be a SpecialPolynomialStatus"
            )
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
        return cast(Symbol, self.generator_derivative.gens[0])

    @property
    def parent_field(self) -> Domain:
        """The coefficient field ``K``."""
        return self.generator_derivative.domain

    @property
    def field(self) -> Domain:
        """The rational function field ``K(t)``."""
        return self._field

    def require_preserved_constants(self, algorithm: str) -> None:
        r"""Require the no-new-constants hypothesis for a structure theorem."""
        if self.constant_field_status is not ConstantFieldStatus.PRESERVED:
            raise ValueError(
                f"{algorithm} requires Const(K(t)) = Const(K); "
                f"constant-field status is {self.constant_field_status.value}"
            )

    def require_no_special_polynomials(self, algorithm: str) -> None:
        r"""Require :math:`S^{\mathrm{irr}}=\varnothing`."""
        if self.special_polynomial_status is not SpecialPolynomialStatus.ABSENT:
            raise ValueError(
                f"{algorithm} requires no nonconstant special polynomials; "
                f"special-polynomial status is "
                f"{self.special_polynomial_status.value}"
            )

    @property
    def family(self) -> MonomialFamily:
        r"""Classify the exact shape of ``Dt`` in the chosen generator.

        The primitive, hyperexponential, and hypertangent forms are
        respectively ``Dt = a``, ``Dt = a*t``, and ``Dt = a*(1 + t**2)``
        for nonzero ``a`` in the parent field.  Other polynomial derivative
        laws remain general polynomial monomials.
        """
        derivative = self.generator_derivative
        degree = derivative.degree()

        if degree == 0:
            return MonomialFamily.PRIMITIVE
        if degree == 1 and derivative.nth(0) == self.parent_field.zero:
            return MonomialFamily.HYPEREXPONENTIAL
        if (
            degree == 2
            and derivative.nth(1) == self.parent_field.zero
            and derivative.nth(0) == derivative.nth(2)
        ):
            return MonomialFamily.HYPERTANGENT
        return MonomialFamily.POLYNOMIAL

    def _validate_polynomial(self, polynomial: Poly) -> None:
        if polynomial.gens != (self.generator,):
            raise ValueError("polynomial generator does not match the tower level")
        if polynomial.domain != self.parent_field:
            raise ValueError("polynomial coefficients are not in the parent field")

    def differentiate_coefficients(
        self,
        polynomial: Poly,
        coefficient_derivation: CoefficientDerivation,
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
        coefficient_derivation: CoefficientDerivation,
    ) -> Poly:
        """Differentiate a polynomial in the extension generator."""
        coefficient_part = self.differentiate_coefficients(
            polynomial,
            coefficient_derivation,
        )
        return coefficient_part + polynomial.diff()*self.generator_derivative

    def differentiate(
        self,
        element: FieldElement,
        coefficient_derivation: CoefficientDerivation,
    ) -> FieldElement:
        """Differentiate an element of ``K(t)`` by the quotient rule."""
        expression = self.field.to_sympy(element)
        numerator_expression, denominator_expression = fraction(expression)
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

        return cast(
            FieldElement,
            self.field.from_sympy(
                result_numerator.as_expr()/result_denominator.as_expr()
            ),
        )


@dataclass(frozen=True, slots=True)
class AlgebraicExtension:
    r"""A finite algebraic extension ``K[y]/(m(y))``.

    The minimal polynomial must be monic and irreducible over the parent
    field.  The derivative of the generator is computed by implicit
    differentiation whenever the extension derivation is evaluated.
    """

    minimal_polynomial: Poly
    _field: FiniteExtension = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

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
        if modulus.domain.convert(modulus.LC()) != modulus.domain.one:
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
        return cast(Symbol, self.minimal_polynomial.gens[0])

    @property
    def parent_field(self) -> Domain:
        """The coefficient field ``K``."""
        return self.minimal_polynomial.domain

    @property
    def field(self) -> FiniteExtension:
        """The native SymPy finite-extension field."""
        return self._field

    def _differentiate_coefficients(
        self,
        polynomial: Poly,
        coefficient_derivation: CoefficientDerivation,
    ) -> Poly:
        return Poly.from_dict(
            {
                monomial: coefficient_derivation(coefficient)
                for monomial, coefficient in polynomial.terms()
            },
            self.generator,
            domain=self.parent_field,
        )

    def _from_polynomial(self, polynomial: Poly) -> FieldElement:
        """Embed a native parent-domain polynomial without an Expr round trip."""
        return cast(
            FieldElement,
            self.field.dtype(polynomial.rep.rem(self.field.mod), self.field),
        )

    def generator_derivative(
        self,
        coefficient_derivation: CoefficientDerivation,
    ) -> FieldElement:
        r"""Return ``Dy = -D_K(m)/partial_y(m)``."""
        coefficient_part = self._differentiate_coefficients(
            self.minimal_polynomial,
            coefficient_derivation,
        )
        partial_generator = self.minimal_polynomial.diff()
        return -self._from_polynomial(coefficient_part) / self._from_polynomial(
            partial_generator,
        )

    def differentiate(
        self,
        element: FieldElement,
        coefficient_derivation: CoefficientDerivation,
        generator_derivative: FieldElement | None = None,
    ) -> FieldElement:
        """Differentiate a reduced representative modulo the minimal polynomial."""
        representative = Poly(
            self.field.to_sympy(element),
            self.generator,
            domain=self.parent_field,
        )
        coefficient_part = self._differentiate_coefficients(
            representative,
            coefficient_derivation,
        )
        if generator_derivative is None:
            generator_derivative = self.generator_derivative(
                coefficient_derivation,
            )
        generator_part = (
            self._from_polynomial(representative.diff())
            * generator_derivative
        )
        return self._from_polynomial(coefficient_part) + generator_part


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
                _differentiate = level.differentiate
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
        return DifferentialTower(self.constants, (*self.levels, extension))

    def derivation(self, level: int | None = None) -> Derivation:
        """Return an immutable derivation bound to ``level``."""
        if level is None:
            level = self.top
        return Derivation(self, level)


def _rebase_monomial_derivation(
    derivation: Derivation,
    constants: Domain,
) -> Derivation:
    """Copy a monomial derivation over a larger exact constant field.

    Generator symbols, derivative laws, and proof metadata are preserved.
    Algebraic levels require a separate constant-extension construction and
    are intentionally rejected.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("rebasing requires a Derivation")
    if not constants.is_Field or constants.characteristic() != 0:
        raise ValueError(
            "rebasing requires an exact characteristic-zero constant field"
        )

    tower = DifferentialTower(constants)
    for level_index in range(derivation.level + 1):
        source = derivation.tower.levels[level_index]
        if not isinstance(source, MonomialExtension):
            raise NotImplementedError(
                "constant-field rebasing does not yet support algebraic levels"
            )
        derivative = Poly(
            source.generator_derivative.as_expr(),
            source.generator,
            domain=tower.field(),
        )
        tower = tower.extend(
            MonomialExtension(
                derivative,
                source.constant_field_status,
                source.special_polynomial_status,
            )
        )
    return tower.derivation()


@dataclass(frozen=True, slots=True)
class Derivation:
    """The derivation on one explicit level of a differential tower."""

    tower: DifferentialTower
    level: int
    _lower_derivation: Derivation | None = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )
    _algebraic_generator_derivative: FieldElement | None = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not -1 <= self.level <= self.tower.top:
            raise IndexError(f"tower level {self.level} is out of range")

        if self.level == -1:
            object.__setattr__(self, "_lower_derivation", None)
            object.__setattr__(self, "_algebraic_generator_derivative", None)
            return

        lower_derivation = Derivation(self.tower, self.level - 1)
        object.__setattr__(self, "_lower_derivation", lower_derivation)

        extension = self.tower.levels[self.level]
        generator_derivative = None
        if isinstance(extension, AlgebraicExtension):
            generator_derivative = extension.generator_derivative(
                lower_derivation,
            )
        object.__setattr__(
            self,
            "_algebraic_generator_derivative",
            generator_derivative,
        )

    @property
    def field(self) -> Domain:
        """Field on which this derivation operates."""
        return self.tower.field(self.level)

    def lower(self) -> Derivation:
        """Return the derivation on the next lower field."""
        if self._lower_derivation is None:
            raise ValueError("the constant field has no lower derivation")
        return self._lower_derivation

    def __call__(self, element: object) -> FieldElement:
        """Differentiate ``element`` and return a native domain element."""
        element = _coerce(self.field, element)

        if self.level == -1:
            return cast(FieldElement, self.field.zero)

        extension = self.tower.levels[self.level]
        if isinstance(extension, AlgebraicExtension):
            return extension.differentiate(
                element,
                self.lower(),
                self._algebraic_generator_derivative,
            )
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

    def fraction(self, numerator: Poly, denominator: Poly) -> FieldElement:
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
