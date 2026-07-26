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
from sympy.polys.agca.extensions import ExtensionElement, FiniteExtension
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
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
    if isinstance(element, ExtensionElement) and element.ext != domain:
        # FiniteExtension.convert currently accepts some foreign elements by
        # reinterpreting their printed generator.  Different defining
        # polynomials have no implicit field embedding here.
        raise ValueError(
            "cannot coerce an element from a different finite algebraic extension"
        )
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


def _require_exact_characteristic_zero_field(
    domain: Domain,
    *,
    role: str,
) -> None:
    """Validate the common coefficient-field contract of tower objects."""
    if not domain.is_Field:
        raise ValueError(f"{role} must be a field")
    if not domain.is_Exact:
        raise ValueError(f"{role} must be exact")
    if domain.characteristic() != 0:
        raise ValueError("differential towers require characteristic zero")


def _contains_finite_extension(domain: Domain) -> bool:
    """Whether a nested SymPy domain contains a function-field extension.

    ``FiniteExtension`` represents an algebraic extension of a function
    field in this tower model.  SymPy currently cannot faithfully form a
    further rational-function field over one: its conversion layer loses the
    algebraic generator.  Keep the capability check separate from
    ``_domain_symbols`` so proposal objects can still be inspected by the
    conservative dependency factories.
    """
    current: Domain | None = domain
    while current is not None:
        if isinstance(current, FiniteExtension):
            return True
        current = getattr(current, "domain", None)
    return False


_ALGEBRAIC_MONOMIAL_BACKEND_LIMITATION = (
    "monomial extensions over algebraic parent fields are not yet "
    "representable by the SymPy domain backend"
)


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


class TowerRepresentationOutcome(Enum):
    r"""Whether a conceptual algebraic/monomial prefix has a native field.

    ``REPRESENTABLE`` means that the current SymPy domain backend can build
    the prefix and perform one exact coercion and derivation at its top level.
    ``BACKEND_LIMITED`` is deliberately distinct from ``UNSUPPORTED``: the
    former is a mathematically meaningful ordering for which the present
    native-domain representation is unavailable, whereas the latter is not a
    well-formed algebraic/monomial prefix for this small validation service.
    """

    REPRESENTABLE = "representable"
    BACKEND_LIMITED = "backend_limited"
    UNSUPPORTED = "unsupported"


class TowerRepresentationReason(Enum):
    """Stable explanation for a non-representable tower-prefix proposal."""

    MONOMIAL_OVER_ALGEBRAIC = "monomial_over_algebraic"
    NATIVE_OPERATION = "native_operation"
    PARENT_FIELD_MISMATCH = "parent_field_mismatch"
    DUPLICATE_GENERATOR = "duplicate_generator"
    UNSUPPORTED_EXTENSION = "unsupported_extension"


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
    special_polynomial_status: SpecialPolynomialStatus = SpecialPolynomialStatus.UNKNOWN
    _field: Domain = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        derivative = self.generator_derivative

        if not isinstance(derivative, Poly):
            raise TypeError("generator_derivative must be a SymPy Poly")
        if len(derivative.gens) != 1:
            raise ValueError("a monomial extension needs exactly one generator")
        _require_exact_characteristic_zero_field(
            derivative.domain,
            role="the parent domain",
        )
        if derivative.is_zero:
            raise ValueError("a zero-derivative generator belongs in the constants")
        if not isinstance(self.constant_field_status, ConstantFieldStatus):
            raise TypeError("constant_field_status must be a ConstantFieldStatus")
        if not isinstance(self.special_polynomial_status, SpecialPolynomialStatus):
            raise TypeError(
                "special_polynomial_status must be a SpecialPolynomialStatus"
            )
        if self.generator in _domain_symbols(self.parent_field):
            raise ValueError(
                f"generator {self.generator} already occurs in the parent field"
            )

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
        if _contains_finite_extension(self.parent_field):
            raise NotImplementedError(_ALGEBRAIC_MONOMIAL_BACKEND_LIMITATION)
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
        constant = self.parent_field.convert(derivative.nth(0))
        if degree == 1 and constant == self.parent_field.zero:
            return MonomialFamily.HYPEREXPONENTIAL
        if (
            degree == 2
            and self.parent_field.convert(derivative.nth(1)) == self.parent_field.zero
            and constant == self.parent_field.convert(derivative.nth(2))
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
        return coefficient_part + polynomial.diff() * self.generator_derivative

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
            numerator_derivative * denominator - numerator * denominator_derivative
        )
        result_denominator = denominator**2

        return cast(
            FieldElement,
            self.field.from_sympy(
                result_numerator.as_expr() / result_denominator.as_expr()
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
        _require_exact_characteristic_zero_field(
            modulus.domain,
            role="the parent domain",
        )
        if _contains_finite_extension(modulus.domain):
            raise NotImplementedError(
                "nested algebraic extensions are not yet representable by "
                "the SymPy domain backend"
            )
        if modulus.degree() < 2:
            raise ValueError("an algebraic extension must have degree at least two")
        if modulus.domain.convert(modulus.LC()) != modulus.domain.one:
            raise ValueError("the minimal polynomial must be monic")
        if modulus.gcd(modulus.diff()).degree() != 0:
            raise ValueError("the minimal polynomial must be square-free")
        if modulus.is_irreducible is not True:
            raise ValueError("the minimal polynomial must be irreducible")
        if self.generator in _domain_symbols(self.parent_field):
            raise ValueError(
                f"generator {self.generator} already occurs in the parent field"
            )

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

    def _element(self, value: object) -> ExtensionElement:
        """Coerce ``value`` to this finite extension."""
        return cast(ExtensionElement, _coerce(self.field, value))

    def multiplication_matrix(self, element: object) -> DomainMatrix:
        r"""Return the parent-linear matrix of multiplication by ``element``.

        Columns are the power-basis coordinates of ``element*y**j``.  Keeping
        the matrix over the native parent domain lets trace, norm, and
        characteristic-polynomial computations avoid expression matrices.
        """
        value = self._element(element)
        rank = self.field.rank
        columns = [
            tuple((value * basis_element).rep.nth(row) for row in range(rank))
            for basis_element in self.field.basis
        ]
        return DomainMatrix(
            [[columns[column][row] for column in range(rank)] for row in range(rank)],
            (rank, rank),
            self.parent_field,
        )

    def trace(self, element: object) -> FieldElement:
        r"""Return the field trace ``Tr_{L/K}(element)``."""
        diagonal = self.multiplication_matrix(element).diagonal()
        return cast(
            FieldElement,
            sum(diagonal, self.parent_field.zero),
        )

    def norm(self, element: object) -> FieldElement:
        r"""Return the field norm ``N_{L/K}(element)``."""
        return cast(
            FieldElement,
            self.multiplication_matrix(element).det(),
        )

    def characteristic_polynomial(
        self,
        element: object,
        indeterminate: Symbol,
    ) -> Poly:
        r"""Return the characteristic polynomial of multiplication by ``element``."""
        if not isinstance(indeterminate, Symbol):
            raise TypeError("the characteristic indeterminate must be a Symbol")
        if indeterminate == self.generator or indeterminate in _domain_symbols(
            self.field
        ):
            raise ValueError(
                "the characteristic indeterminate must be new over the extension"
            )
        return Poly.from_list(
            self.multiplication_matrix(element).charpoly(),
            indeterminate,
            domain=self.parent_field,
        )

    def element_minimal_polynomial(
        self,
        element: object,
        indeterminate: Symbol,
    ) -> Poly:
        r"""Return the minimal polynomial of ``element`` over the parent field.

        In characteristic zero the extension is separable.  The
        characteristic polynomial is therefore a positive power of the
        element's minimal polynomial, so its monic square-free part is exact.
        """
        return self.characteristic_polynomial(element, indeterminate).sqf_part().monic()

    def trace_pairing_matrix(
        self,
        basis: tuple[object, ...] | None = None,
    ) -> DomainMatrix:
        r"""Return ``(Tr(b_i*b_j))`` for a parent-field basis candidate."""
        if basis is None:
            basis = cast(tuple[object, ...], self.field.basis)
        if len(basis) != self.field.rank:
            raise ValueError(
                f"a trace-pairing basis must contain {self.field.rank} elements"
            )
        elements = tuple(self._element(element) for element in basis)
        return DomainMatrix(
            [
                [
                    self.parent_field.convert(self.trace(left * right))
                    for right in elements
                ]
                for left in elements
            ],
            (self.field.rank, self.field.rank),
            self.parent_field,
        )

    def basis_discriminant(
        self,
        basis: tuple[object, ...] | None = None,
    ) -> FieldElement:
        r"""Return the determinant of the trace pairing on ``basis``."""
        return cast(
            FieldElement,
            self.trace_pairing_matrix(basis).det(),
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
        # Although ``m_y`` is invertible because ``m`` is separable, SymPy's
        # native ``FiniteExtension`` inverse can spuriously report a zero
        # divisor when the constant field is a general AlgebraicField.  The
        # extended Euclidean identity is the same field operation, computed
        # over the parent rational-function field rather than through that
        # backend path.
        inverse, _, gcd = partial_generator.gcdex(self.minimal_polynomial)
        if gcd.degree() != 0:
            raise ValueError(
                "the algebraic generator derivative requires a separable "
                "minimal polynomial"
            )
        inverse = inverse.mul_ground(
            self.parent_field.one / self.parent_field.convert(gcd.nth(0))
        )
        return -self._from_polynomial(coefficient_part * inverse)

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
            self._from_polynomial(representative.diff()) * generator_derivative
        )
        return self._from_polynomial(coefficient_part) + generator_part


@dataclass(frozen=True, slots=True)
class DifferentialTower:
    """An immutable sequence of differential-field extensions."""

    constants: Domain
    levels: tuple[DifferentialExtension, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_characteristic_zero_field(
            self.constants,
            role="the constant domain",
        )

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


def rebase_derivation_constants(
    derivation: Derivation,
    constants: Domain,
) -> Derivation:
    """Copy a derivation over a larger exact constant field.

    Generator symbols, derivative laws, algebraic defining polynomials, and
    monomial proof metadata are preserved.  The returned derivation is bound
    to a newly constructed immutable tower; source domain elements can be
    embedded through their exact SymPy expressions.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("rebasing requires a Derivation")
    _require_exact_characteristic_zero_field(
        constants,
        role="the rebased constant domain",
    )
    try:
        unified_constants = derivation.tower.constants.unify(constants)
    except UnificationFailed as exc:
        raise ValueError(
            "the new constants must contain the existing constant field"
        ) from exc
    if unified_constants != constants:
        raise ValueError("the new constants must contain the existing constant field")

    tower = DifferentialTower(constants)
    for level_index in range(derivation.level + 1):
        source = derivation.tower.levels[level_index]
        if isinstance(source, MonomialExtension):
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
            continue
        if isinstance(source, AlgebraicExtension):
            minimal_polynomial = Poly(
                source.minimal_polynomial.as_expr(),
                source.generator,
                domain=tower.field(),
            )
            tower = tower.extend(AlgebraicExtension(minimal_polynomial))
            continue
        raise TypeError("tower levels must be supported differential extensions")
    return tower.derivation()


def _rebase_monomial_derivation(
    derivation: Derivation,
    constants: Domain,
) -> Derivation:
    """Preserve the original internal name for monomial-only callers."""
    return rebase_derivation_constants(derivation, constants)


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
            numerator.as_expr() / denominator.as_expr()
        )
        return extension.differentiate(element, self.lower())


@dataclass(frozen=True, slots=True)
class TowerRepresentationProposal:
    r"""One immutable conceptual prefix to check against SymPy's backend.

    This is intentionally only tower metadata: it neither stores custom
    elements nor tries to emulate a field that SymPy cannot represent.  In
    particular, it can describe a proposed monomial level over an algebraic
    one, so that :func:`analyze_tower_representation` can return a structured
    backend limitation instead of requiring callers to interpret an exception
    from :meth:`DifferentialTower.extend`.
    """

    constants: Domain
    levels: tuple[DifferentialExtension, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_characteristic_zero_field(
            self.constants,
            role="the constant domain",
        )
        try:
            levels = tuple(self.levels)
        except TypeError as exc:
            raise TypeError("proposal levels must be an iterable") from exc
        object.__setattr__(self, "levels", levels)

    @classmethod
    def from_tower(
        cls,
        tower: DifferentialTower,
        level: int | None = None,
    ) -> TowerRepresentationProposal:
        """Copy one active prefix of an existing immutable tower.

        The returned proposal may be analyzed independently of levels above
        ``level``.  It is useful to later solvers that only need a lower
        representable prefix while a broader conceptual tower is still being
        assembled.
        """
        if not isinstance(tower, DifferentialTower):
            raise TypeError(
                "a tower representation proposal requires a DifferentialTower"
            )
        if level is None:
            level = tower.top
        if not isinstance(level, int):
            raise TypeError("a tower representation level must be an integer")
        if not -1 <= level <= tower.top:
            raise IndexError(f"tower level {level} is out of range")
        return cls(tower.constants, tower.levels[: level + 1])


@dataclass(frozen=True, slots=True)
class TowerRepresentationAnalysis:
    r"""Checked native-SymPy representation result for a tower prefix.

    A representable result contains the reconstructed :attr:`native_tower`.
    A non-representable result never contains a partial tower.  This keeps a
    backend limitation separate from mathematical integration decisions and
    prevents callers from accidentally dispatching against a partially built
    native field.
    """

    proposal: TowerRepresentationProposal
    outcome: TowerRepresentationOutcome
    reason: TowerRepresentationReason | None
    detail: str
    failed_level: int | None
    native_tower: DifferentialTower | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, TowerRepresentationProposal):
            raise TypeError(
                "a representation analysis needs a TowerRepresentationProposal"
            )
        if not isinstance(self.outcome, TowerRepresentationOutcome):
            raise TypeError(
                "a representation analysis needs a TowerRepresentationOutcome"
            )
        if self.reason is not None and not isinstance(
            self.reason,
            TowerRepresentationReason,
        ):
            raise TypeError(
                "a representation analysis reason must be a TowerRepresentationReason"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a representation analysis needs nonempty detail")
        if self.failed_level is not None and not isinstance(self.failed_level, int):
            raise TypeError("a failed representation level must be an integer")

        if self.outcome is TowerRepresentationOutcome.REPRESENTABLE:
            if self.reason is not None:
                raise ValueError(
                    "a representable tower must not carry a failure reason"
                )
            if self.failed_level is not None:
                raise ValueError("a representable tower must not carry a failed level")
            if not isinstance(self.native_tower, DifferentialTower):
                raise ValueError("a representable tower needs its native tower")
            if (
                self.native_tower.constants != self.proposal.constants
                or self.native_tower.levels != self.proposal.levels
            ):
                raise ValueError(
                    "a native tower must exactly represent the proposed prefix"
                )
            _exercise_native_tower(self.native_tower)
            return

        if self.reason is None:
            raise ValueError("a non-representable tower needs a failure reason")
        if self.native_tower is not None:
            raise ValueError("a non-representable tower must not carry a partial tower")
        if self.failed_level is None:
            raise ValueError("a non-representable tower needs a failed level")
        if not 0 <= self.failed_level < len(self.proposal.levels):
            raise IndexError(f"tower level {self.failed_level} is out of range")

        backend_reasons = {
            TowerRepresentationReason.MONOMIAL_OVER_ALGEBRAIC,
            TowerRepresentationReason.NATIVE_OPERATION,
        }
        if self.outcome is TowerRepresentationOutcome.BACKEND_LIMITED:
            if self.reason not in backend_reasons:
                raise ValueError(
                    "a backend-limited tower needs a backend limitation reason"
                )
            return
        if self.outcome is TowerRepresentationOutcome.UNSUPPORTED:
            if self.reason in backend_reasons:
                raise ValueError(
                    "a backend limitation must use the backend-limited outcome"
                )
            return
        raise AssertionError("unhandled tower representation outcome")

    @property
    def representable(self) -> bool:
        """Whether a complete native tower is available."""
        return self.outcome is TowerRepresentationOutcome.REPRESENTABLE

    @property
    def backend_limited(self) -> bool:
        """Whether SymPy's native domain backend blocks this valid ordering."""
        return self.outcome is TowerRepresentationOutcome.BACKEND_LIMITED

    @property
    def unsupported(self) -> bool:
        """Whether the proposal is outside this algebraic/monomial shape."""
        return self.outcome is TowerRepresentationOutcome.UNSUPPORTED


def _nonrepresentable_representation(
    proposal: TowerRepresentationProposal,
    outcome: TowerRepresentationOutcome,
    reason: TowerRepresentationReason,
    failed_level: int,
    detail: str,
) -> TowerRepresentationAnalysis:
    """Build a checked non-representable result without a partial tower."""
    return TowerRepresentationAnalysis(
        proposal,
        outcome,
        reason,
        detail,
        failed_level,
    )


def _exercise_native_tower(tower: DifferentialTower) -> None:
    """Exercise the top native coercion and derivation without integration.

    Building a ``FiniteExtension`` alone is not enough evidence for the
    algebraic-over-monomial ordering: the generator must also embed into the
    top field and its implicit derivative must traverse the lower tower.  One
    generator derivative is a small, exact backend smoke test; it deliberately
    performs no Hermite reduction or integration dispatch.
    """
    if tower.top < 0:
        return
    extension = tower.levels[tower.top]
    generator = _coerce(tower.field(), extension.generator)
    _coerce(tower.field(), tower.derivation()(generator))


def analyze_tower_representation(
    proposal: TowerRepresentationProposal,
) -> TowerRepresentationAnalysis:
    r"""Classify the current native representation of one mixed prefix.

    The presently admitted physical orderings are any finite monomial prefix
    and an optional single terminal algebraic extension over that prefix.
    SymPy cannot yet form a rational-function field over a ``FiniteExtension``
    or nest finite extensions.  Those mathematically legitimate cases return
    :attr:`TowerRepresentationOutcome.BACKEND_LIMITED`, while malformed or
    unknown tower shapes return ``UNSUPPORTED``.  Neither non-representable
    outcome is an integration or elementary-function decision.
    """
    if not isinstance(proposal, TowerRepresentationProposal):
        raise TypeError(
            "tower representation analysis requires a TowerRepresentationProposal"
        )

    expected_parent = proposal.constants
    generators: set[Symbol] = set()

    for level, extension in enumerate(proposal.levels):
        if not isinstance(extension, (MonomialExtension, AlgebraicExtension)):
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.UNSUPPORTED_EXTENSION,
                level,
                f"level {level} is not a native monomial or algebraic extension",
            )
        if extension.parent_field != expected_parent:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.PARENT_FIELD_MISMATCH,
                level,
                f"level {level} does not use the preceding native field as parent",
            )
        if extension.generator in generators:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.UNSUPPORTED,
                TowerRepresentationReason.DUPLICATE_GENERATOR,
                level,
                f"level {level} duplicates generator {extension.generator}",
            )

        generators.add(extension.generator)
        if isinstance(extension, MonomialExtension) and _contains_finite_extension(
            expected_parent
        ):
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.BACKEND_LIMITED,
                TowerRepresentationReason.MONOMIAL_OVER_ALGEBRAIC,
                level,
                "SymPy cannot form a native monomial field over an "
                "algebraic finite-extension parent",
            )
        try:
            expected_parent = extension.field
        except (CoercionFailed, NotImplementedError) as exc:
            return _nonrepresentable_representation(
                proposal,
                TowerRepresentationOutcome.BACKEND_LIMITED,
                TowerRepresentationReason.NATIVE_OPERATION,
                level,
                f"SymPy could not construct native level {level}: {exc}",
            )

    try:
        native_tower = DifferentialTower(proposal.constants, proposal.levels)
        return TowerRepresentationAnalysis(
            proposal,
            TowerRepresentationOutcome.REPRESENTABLE,
            None,
            "SymPy represents the complete prefix and its top generator coerces "
            "and differentiates exactly",
            None,
            native_tower,
        )
    except (CoercionFailed, NotImplementedError) as exc:
        if not proposal.levels:
            raise
        return _nonrepresentable_representation(
            proposal,
            TowerRepresentationOutcome.BACKEND_LIMITED,
            TowerRepresentationReason.NATIVE_OPERATION,
            len(proposal.levels) - 1,
            f"SymPy could not validate the native tower: {exc}",
        )
