r"""Exact infrastructure for finite algebraic function fields."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from math import gcd
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.agca.extensions import ExtensionElement, FiniteExtension
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.matrices.normalforms import smith_normal_decomp
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, invert, resultant

from ._typing import SupportsAsExpr, is_field_zero
from .tower import AlgebraicExtension, Derivation, MonomialExtension


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


@dataclass(frozen=True, slots=True)
class AlgebraicFraction:
    r"""Normalized coordinates ``(A_1, ..., A_n)/D`` in an integral basis."""

    numerators: tuple[Poly, ...]
    denominator: Poly

    @property
    def is_square_free(self) -> bool:
        """Whether the common denominator is square-free."""
        return self.denominator.gcd(self.denominator.diff()).degree() == 0


@dataclass(frozen=True, slots=True)
class TragerAlgebraicResidueResult:
    r"""Certified finite-residue polynomial for an algebraic differential.

    For a monic polynomial model ``F(x, y) = 0`` and

    ``integrand*dx = G(x, y)/D(x)*dx``,

    where ``D`` is square-free and no root of ``D`` lies below a branch
    place, Trager's construction computes

    ``R(Z) = Res_x(Res_y(Z*D'(x) - G(x, y), F(x, y)), D(x))``.

    The roots of the monic ``residue_polynomial`` are exactly the finite
    residues, counted with their conjugate-place multiplicities.
    ``square_free_residue_polynomial`` removes those multiplicities without
    changing the splitting field required for the logarithmic coefficients.
    """

    extension: AlgebraicExtension
    derivation: Derivation
    integrand: ExtensionElement
    basis: IntegralBasis
    residue_symbol: Symbol
    defining_polynomial: Poly
    numerator: Poly
    denominator: Poly
    discriminant: Poly
    branch_gcd: Poly
    inner_resultant: Poly | None
    raw_residue_polynomial: Poly
    residue_polynomial: Poly

    @property
    def has_finite_poles(self) -> bool:
        """Whether the integral-basis denominator has a finite zero."""
        return self.denominator.degree() > 0

    @property
    def square_free_residue_polynomial(self) -> Poly:
        """Return the monic polynomial with the distinct finite residues."""
        if self.residue_polynomial.degree() <= 0:
            return self.residue_polynomial
        return self.residue_polynomial.sqf_part().monic()

    @property
    def normalizes_resultant(self) -> bool:
        """Whether the public polynomial is the monic outer resultant."""
        return (
            not self.raw_residue_polynomial.is_zero
            and self.raw_residue_polynomial.monic() == self.residue_polynomial
        )

    def place_residues(self) -> tuple[AlgebraicPlaceResidue, ...]:
        """Factor the unramified pole fibers and return exact local residues."""
        return trager_algebraic_place_residues(self)

    @property
    def reconstructs_input(self) -> bool:
        """Whether ``G/D`` is exactly the supplied algebraic element."""
        field = self.extension.field
        return field.from_sympy(
            self.numerator.as_expr() / self.denominator.as_expr()
        ) == _extension_element(self.extension, self.integrand)


@dataclass(frozen=True, slots=True)
class IntegralBasis:
    r"""A caller-supplied integral basis of a monogenic function field.

    SymPy currently provides finite-extension arithmetic but does not verify
    that arbitrary function-field elements form the full integral closure over
    ``K[x]``.  Construction therefore validates that ``elements`` are a field
    basis; the caller is responsible for their integrality and completeness.
    """

    extension: AlgebraicExtension
    elements: tuple[ExtensionElement, ...]
    _power_inverse: DomainMatrix = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        elements = tuple(
            _extension_element(self.extension, element) for element in self.elements
        )
        object.__setattr__(self, "elements", elements)

        rank = self.extension.field.rank
        if len(elements) != rank:
            raise ValueError(f"an integral basis must contain {rank} elements")

        columns = [self._power_coordinates(element) for element in elements]
        parent = self.extension.parent_field
        # Keep the change-of-basis matrix over the native parent domain.  An
        # expression Matrix would repeatedly convert QQ(x) elements to SymPy
        # expressions and call cancel during every coordinate transformation.
        matrix = DomainMatrix(
            [[columns[column][row] for column in range(rank)] for row in range(rank)],
            (rank, rank),
            parent,
        )
        try:
            inverse = matrix.inv()
        except DMNonInvertibleMatrixError as exc:
            raise ValueError("integral basis elements are linearly dependent") from exc
        object.__setattr__(self, "_power_inverse", inverse)

    @classmethod
    def power_basis(cls, extension: AlgebraicExtension) -> IntegralBasis:
        """Use the power basis when it is known to be an integral basis."""
        return cls(extension, extension.field.basis)

    def _power_coordinates(
        self,
        element: object,
    ) -> tuple[FracElement, ...]:
        representative = _extension_element(self.extension, element).rep
        return cast(
            tuple[FracElement, ...],
            tuple(
                representative.nth(degree)
                for degree in range(self.extension.field.rank)
            ),
        )

    def coordinates(self, element: object) -> tuple[FracElement, ...]:
        """Return the coordinates of ``element`` in this basis."""
        parent = self.extension.parent_field
        power_coordinates = self._power_coordinates(element)
        rank = len(power_coordinates)
        vector = DomainMatrix(
            [[coefficient] for coefficient in power_coordinates],
            (rank, 1),
            parent,
        )
        result = self._power_inverse * vector
        return cast(
            tuple[FracElement, ...],
            tuple(row[0] for row in result.to_list()),
        )

    def element(
        self,
        coordinates: tuple[object, ...],
    ) -> ExtensionElement:
        """Construct a field element from integral-basis coordinates."""
        if len(coordinates) != len(self.elements):
            raise ValueError("coordinate vector has the wrong length")

        parent = self.extension.parent_field
        field = self.extension.field
        result = field.zero.rep
        for coefficient, basis_element in zip(coordinates, self.elements, strict=True):
            coefficient = parent.convert(coefficient)
            result += basis_element.rep.mul_ground(coefficient)
        return field.dtype(result, field)

    def fraction(self, element: object) -> AlgebraicFraction:
        """Return the normalized common-denominator representation."""
        parent, constants, variable = _base_polynomial_ring(self.extension)
        coordinates = self.coordinates(element)
        denominator = _common_denominator(coordinates, constants, variable)
        denominator_element = _polynomial_parent_element(denominator, parent)
        numerators = tuple(
            _parent_element_polynomial(
                coefficient * denominator_element,
                parent,
                constants,
                variable,
            )
            for coefficient in coordinates
        )
        return AlgebraicFraction(numerators, denominator)


def _polynomial_plane_model(
    extension: AlgebraicExtension,
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> Poly:
    r"""Return the monic model ``F(y, x)`` over ``constants``."""
    expression = 0
    try:
        for (degree,), coefficient in extension.minimal_polynomial.terms():
            coefficient_polynomial = _parent_element_polynomial(
                coefficient,
                parent,
                constants,
                variable,
            )
            expression += coefficient_polynomial.as_expr() * extension.generator**degree
    except ValueError as exc:
        raise NotImplementedError(
            "Trager residues currently require a defining polynomial in C[x, y]"
        ) from exc

    coefficient_ring = constants.poly_ring(variable)
    model_in_generator = Poly(
        expression,
        extension.generator,
        domain=coefficient_ring,
    )
    if coefficient_ring.convert(model_in_generator.LC()) != coefficient_ring.one:
        raise NotImplementedError(
            "Trager residues currently require a model monic in the "
            "algebraic generator over C[x]"
        )
    return Poly(
        expression,
        extension.generator,
        variable,
        domain=constants,
    )


def _polynomial_power_numerator(
    element: ExtensionElement,
    fraction: AlgebraicFraction,
    basis: IntegralBasis,
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> Poly:
    r"""Return ``G(y, x)`` when ``element = G/D`` in the power basis."""
    extension = basis.extension
    numerator_element = basis.element(
        tuple(
            _polynomial_parent_element(numerator, parent)
            for numerator in fraction.numerators
        )
    )
    expression = 0
    try:
        for degree in range(extension.field.rank):
            coefficient = numerator_element.rep.nth(degree)
            coefficient_polynomial = _parent_element_polynomial(
                coefficient,
                parent,
                constants,
                variable,
            )
            expression += coefficient_polynomial.as_expr() * extension.generator**degree
    except ValueError as exc:
        raise NotImplementedError(
            "Trager residues currently require the integral numerator "
            "to be polynomial in the selected plane model"
        ) from exc

    numerator = Poly(
        expression,
        extension.generator,
        variable,
        domain=constants,
    )
    if (
        extension.field.from_sympy(numerator.as_expr() / fraction.denominator.as_expr())
        != element
    ):
        raise ValueError(
            "the polynomial plane numerator does not reconstruct the input"
        )
    return numerator


def trager_algebraic_residue_polynomial(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    residue: Symbol,
) -> TragerAlgebraicResidueResult:
    r"""Compute Trager's finite-residue polynomial over an unbranched model.

    The current implementation is the first algebraic logarithmic-part
    checkpoint. It requires one algebraic level over ``C(x)``, a supplied
    integral basis, a square-free common denominator, and a monic polynomial
    plane model whose discriminant is coprime to that denominator. The last
    condition certifies that every finite pole is unbranched.

    The nested resultant is equation (7) in Trager, *Integration of
    Algebraic Functions*, Chapter 5, section 5.2. The explicit branch-locus
    rejection preserves the hypothesis discussed in section 5.4 instead of
    silently applying the unbranched formula to a ramified pole.

    Branch places, poles at infinity, construction of the algebraic constant
    extension, and the associated principal-divisor problem are deliberately
    separate later algorithms.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("Trager residues require a Derivation")
    if not isinstance(basis, IntegralBasis):
        raise TypeError("Trager residues require an IntegralBasis")
    if not isinstance(residue, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")

    extension = basis.extension
    if derivation.field != extension.field:
        raise ValueError(
            "the derivation and integral basis must describe the same algebraic field"
        )

    parent, constants, variable = _base_polynomial_ring(extension)
    lower_derivation = derivation.lower()
    if lower_derivation.field != parent or not is_field_zero(
        lower_derivation(parent.from_sympy(variable)) - parent.one
    ):
        raise NotImplementedError(
            "Trager residues currently require the standard rational derivation Dx = 1"
        )
    constant_symbols = set(getattr(constants, "symbols", ()))
    if (
        residue
        in {
            variable,
            extension.generator,
        }
        | constant_symbols
    ):
        raise ValueError("the residue indeterminate must be new over the plane model")

    element = _extension_element(extension, integrand)
    fraction = basis.fraction(element)
    denominator = fraction.denominator
    if not fraction.is_square_free:
        raise ValueError(
            "Trager residues require a Hermite-reduced square-free denominator"
        )

    defining_polynomial = _polynomial_plane_model(
        extension,
        parent,
        constants,
        variable,
    )
    if residue in defining_polynomial.as_expr().free_symbols:
        raise ValueError("the residue indeterminate must be new over the plane model")
    numerator = _polynomial_power_numerator(
        element,
        fraction,
        basis,
        parent,
        constants,
        variable,
    )

    model_in_generator = Poly(
        defining_polynomial.as_expr(),
        extension.generator,
        domain=constants.poly_ring(variable),
    )
    discriminant = Poly(
        model_in_generator.discriminant(),
        variable,
        domain=constants,
    )
    branch_gcd = denominator.gcd(discriminant)
    if branch_gcd.degree() > 0:
        raise NotImplementedError(
            "Trager's unbranched residue formula does not apply because "
            f"the denominator meets the branch locus {branch_gcd.as_expr()}"
        )

    if denominator.degree() == 0:
        residue_polynomial = Poly(1, residue, domain=constants)
        result = TragerAlgebraicResidueResult(
            extension,
            derivation,
            element,
            basis,
            residue,
            defining_polynomial,
            numerator,
            denominator,
            discriminant,
            branch_gcd,
            None,
            residue_polynomial,
            residue_polynomial,
        )
        if not result.reconstructs_input:
            raise ValueError("Trager residue data did not reconstruct its input")
        return result

    denominator_derivative = denominator.diff()
    inner_expression = resultant(
        (residue * denominator_derivative.as_expr() - numerator.as_expr()),
        defining_polynomial.as_expr(),
        extension.generator,
    )
    inner_resultant = Poly(
        inner_expression,
        variable,
        residue,
        domain=constants,
    )
    outer_expression = resultant(
        inner_expression,
        denominator.as_expr(),
        variable,
    )
    raw_residue_polynomial = Poly(
        outer_expression,
        residue,
        domain=constants,
    )
    if raw_residue_polynomial.is_zero:
        raise ValueError("Trager's nested resultant unexpectedly vanished")
    residue_polynomial = raw_residue_polynomial.monic()

    result = TragerAlgebraicResidueResult(
        extension,
        derivation,
        element,
        basis,
        residue,
        defining_polynomial,
        numerator,
        denominator,
        discriminant,
        branch_gcd,
        inner_resultant,
        raw_residue_polynomial,
        residue_polynomial,
    )
    if not result.reconstructs_input:
        raise ValueError("Trager residue data did not reconstruct its input")
    return result


@dataclass(frozen=True, slots=True)
class UnramifiedFinitePlace:
    r"""A closed finite place on a monic plane model.

    ``base_prime`` is an irreducible polynomial in the separating variable
    ``x``.  After adjoining one root of that polynomial to the constants,
    ``fiber_factor`` is one irreducible factor of the specialized defining
    polynomial in ``y``.  The pair identifies a closed place.  This initial
    representation deliberately covers only fibers disjoint from the
    discriminant, where ``base_prime`` is a uniformizer and the ramification
    index is one.
    """

    extension: AlgebraicExtension
    defining_polynomial: Poly
    base_prime: Poly
    base_residue_field: Domain
    base_root: Expr
    specialized_polynomial: Poly
    fiber_factor: Poly
    residue_field: FiniteExtension

    @property
    def base_degree(self) -> int:
        """Degree of the place below over the original constants."""
        return self.base_prime.degree()

    @property
    def relative_degree(self) -> int:
        """Residue degree over the base place."""
        return self.fiber_factor.degree()

    @property
    def degree(self) -> int:
        """Absolute residue-field degree over the original constants."""
        return self.base_degree * self.relative_degree

    @property
    def ramification_index(self) -> int:
        """Ramification index, fixed to one by this representation."""
        return 1

    @property
    def uniformizer(self) -> ExtensionElement:
        """Return the base prime embedded as a local uniformizer."""
        return _extension_element(
            self.extension,
            self.base_prime.as_expr(),
        )

    def reduce_polynomial(self, polynomial: Poly) -> ExtensionElement:
        r"""Reduce a polynomial in ``x`` and ``y`` into the residue field."""
        if not isinstance(polynomial, Poly):
            raise TypeError("place reduction requires a SymPy Poly")
        variable = self.base_prime.gens[0]
        expression = polynomial.as_expr().subs(variable, self.base_root)
        return cast(
            ExtensionElement,
            self.residue_field.from_sympy(expression),
        )

    def residue(
        self,
        result: TragerAlgebraicResidueResult,
    ) -> ExtensionElement:
        r"""Return the differential residue represented by ``result``.

        At an unramified simple pole over ``p(x)``, the local parameter is
        ``p`` and

        ``res_P(G/D dx) = (G/D')(P)``.
        """
        if not isinstance(result, TragerAlgebraicResidueResult):
            raise TypeError("place residues require Trager residue data")
        if result.extension != self.extension:
            raise ValueError(
                "the place and residue data must describe the same extension"
            )
        if result.defining_polynomial != self.defining_polynomial:
            raise ValueError("the place and residue data must use the same plane model")
        _, remainder = result.denominator.div(self.base_prime)
        if not remainder.is_zero:
            raise ValueError("the place does not lie over a pole of the differential")

        denominator_derivative = self.reduce_polynomial(result.denominator.diff())
        if denominator_derivative == self.residue_field.zero:
            raise ValueError(
                "the finite pole is not simple in the selected base parameter"
            )
        return self.reduce_polynomial(result.numerator) / denominator_derivative


@dataclass(frozen=True, slots=True)
class AlgebraicPlaceResidue:
    """One exact finite residue together with its closed place."""

    place: UnramifiedFinitePlace
    value: ExtensionElement


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


def unramified_finite_places(
    extension: AlgebraicExtension,
    base_prime: Poly,
) -> tuple[UnramifiedFinitePlace, ...]:
    r"""Return the closed places above an unramified finite base prime.

    The plane model must be polynomial and monic in the algebraic generator.
    ``base_prime`` is normalized to monic form and must be irreducible over
    the constant field.  A common factor with the model discriminant is
    rejected because ramified places require integral local models and
    nontrivial uniformizers.
    """
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("finite places require an AlgebraicExtension")
    if not isinstance(base_prime, Poly):
        raise TypeError("the base prime must be a SymPy Poly")

    parent, constants, variable = _base_polynomial_ring(extension)
    if base_prime.gens != (variable,):
        raise ValueError(
            "the base prime generator must match the rational parent variable"
        )
    prime = Poly(
        base_prime.as_expr(),
        variable,
        domain=constants,
    )
    if prime.degree() <= 0:
        raise ValueError("a finite base prime must be nonconstant")
    prime = prime.monic()
    if prime.is_irreducible is not True:
        raise ValueError("a finite base prime must be irreducible")

    defining_polynomial = _polynomial_plane_model(
        extension,
        parent,
        constants,
        variable,
    )
    model_in_generator = Poly(
        defining_polynomial.as_expr(),
        extension.generator,
        domain=constants.poly_ring(variable),
    )
    discriminant = Poly(
        model_in_generator.discriminant(),
        variable,
        domain=constants,
    )
    branch_gcd = prime.gcd(discriminant)
    if branch_gcd.degree() > 0:
        raise NotImplementedError(
            "finite-place construction requires an unramified fiber; "
            f"the base prime meets the branch locus {branch_gcd.as_expr()}"
        )

    algebraic_constants = _algebraic_constant_domain(constants)
    algebraic_prime = Poly(
        prime.as_expr(),
        variable,
        domain=algebraic_constants,
    )
    if algebraic_prime.degree() == 1:
        base_residue_field = algebraic_constants
        base_root = cast(
            Expr,
            algebraic_constants.to_sympy(-algebraic_prime.nth(0)),
        )
    else:
        base_residue_field = cast(
            AlgebraicField,
            algebraic_constants.alg_field_from_poly(algebraic_prime),
        )
        base_root = cast(Expr, base_residue_field.orig_ext[-1])
    specialized_polynomial = Poly(
        defining_polynomial.as_expr().subs(variable, base_root),
        extension.generator,
        domain=base_residue_field,
    )
    try:
        _, factors = specialized_polynomial.factor_list()
    except NotImplementedError as exc:
        raise NotImplementedError(
            "the specialized algebraic fiber could not be factored exactly"
        ) from exc
    if any(multiplicity != 1 for _, multiplicity in factors):
        raise ValueError("an unramified fiber unexpectedly has a repeated factor")

    return tuple(
        UnramifiedFinitePlace(
            extension,
            defining_polynomial,
            prime,
            base_residue_field,
            base_root,
            specialized_polynomial,
            factor.monic(),
            FiniteExtension(factor.monic()),
        )
        for factor, _ in factors
    )


def trager_algebraic_place_residues(
    result: TragerAlgebraicResidueResult,
) -> tuple[AlgebraicPlaceResidue, ...]:
    r"""Refine a Trager residue polynomial into exact closed-place residues."""
    if not isinstance(result, TragerAlgebraicResidueResult):
        raise TypeError("place residues require Trager residue data")

    residues: list[AlgebraicPlaceResidue] = []
    _, denominator_factors = result.denominator.factor_list()
    for base_prime, multiplicity in denominator_factors:
        if multiplicity != 1:
            raise ValueError("place-wise residues require a square-free denominator")
        for place in unramified_finite_places(result.extension, base_prime):
            residues.append(AlgebraicPlaceResidue(place, place.residue(result)))
    return tuple(residues)


@dataclass(frozen=True, slots=True)
class SimpleRadicalData:
    r"""Normalized data for a simple radical extension ``y**n = a``.

    ``normalized_generator`` satisfies ``z**n = normalized_radicand``.  The
    latter is ``n``-th-power-free as a polynomial: every factor in its
    square-free factorization has multiplicity strictly smaller than ``n``.
    """

    extension: AlgebraicExtension
    degree: int
    radicand: FracElement
    numerator: Poly
    denominator: Poly
    extracted_factor: Poly
    normalized_radicand: Poly
    normalized_generator: ExtensionElement


class AlgebraicConstantFieldStatus(Enum):
    r"""Certified status of the constants of one algebraic function field.

    ``REBASING_REQUIRED`` says that this analysis has exhibited a new
    algebraic constant and an exact embedding of its generated constant field
    into the presented function field.  It deliberately does *not* claim
    that the current tower backend can already rewrite the full extension over
    that larger field.
    """

    PRESERVED = "preserved"
    REBASING_REQUIRED = "rebasing_required"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicConstantEmbedding:
    r"""An exact embedding of a proposed algebraic constant field into ``L``.

    The embedding sends the primitive generator selected by SymPy's
    ``constant_domain`` to ``generator_image`` in ``source_extension``.  It
    is a certificate for the *constant subfield* only; turning it into a
    re-based function-field presentation can lower the algebraic degree and
    is a separate operation.
    """

    source_extension: AlgebraicExtension
    constant_domain: AlgebraicField
    minimal_polynomial: Poly
    generator_image: ExtensionElement

    def __post_init__(self) -> None:
        if not isinstance(self.source_extension, AlgebraicExtension):
            raise TypeError("a constant embedding needs an algebraic extension")
        if not isinstance(self.constant_domain, AlgebraicField):
            raise TypeError("a constant embedding needs an algebraic number field")
        if not isinstance(self.minimal_polynomial, Poly):
            raise TypeError("a constant embedding needs a SymPy Poly")
        _, constants, _ = _base_polynomial_ring(self.source_extension)
        if self.minimal_polynomial.domain != constants:
            raise ValueError(
                "the constant polynomial must be over the displayed constants"
            )
        if self.minimal_polynomial.degree() < 2:
            raise ValueError("a rebasing certificate needs a nontrivial polynomial")
        if self.minimal_polynomial.is_irreducible is not True:
            raise ValueError("the constant polynomial must be irreducible")
        target_polynomial = self.constant_domain.ext.minpoly
        if not _same_univariate_polynomial_coefficients(
            target_polynomial,
            self.minimal_polynomial,
        ):
            raise ValueError(
                "the algebraic-number domain must realize the stated polynomial"
            )
        image = _extension_element(self.source_extension, self.generator_image)
        object.__setattr__(
            self,
            "generator_image",
            image,
        )
        if (
            _evaluate_constant_polynomial(
                self.source_extension,
                self.minimal_polynomial,
                image,
            )
            != self.source_extension.field.zero
        ):
            raise ValueError(
                "the proposed constant image does not satisfy its minimal polynomial"
            )

    def embed(self, value: object) -> ExtensionElement:
        r"""Map an exact proposed constant into the source function field."""
        constant = self.constant_domain.convert(value)
        root = cast(Expr, self.constant_domain.ext.root)
        expression = cast(Expr, self.constant_domain.to_sympy(constant))
        image = cast(
            Expr,
            self.source_extension.field.to_sympy(self.generator_image),
        )
        return _extension_element(
            self.source_extension,
            expression.xreplace({root: image}),
        )


@dataclass(frozen=True, slots=True)
class AlgebraicConstantFieldAnalysis:
    r"""Sound constant-field analysis for one supported algebraic extension.

    The current theorem-backed scope is a relation already over ``C`` or a
    simple radical over the standard rational differential field ``C(x)``,
    ``Dx = 1``. An unsupported result is a boundary of this analyzer, not
    evidence that the displayed constants are correct.
    """

    extension: AlgebraicExtension
    derivation: Derivation
    status: AlgebraicConstantFieldStatus
    radical: SimpleRadicalData | None
    exponent_gcd: int | None
    embedding: AlgebraicConstantEmbedding | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.extension, AlgebraicExtension):
            raise TypeError("an algebraic constant analysis needs an extension")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an algebraic constant analysis needs a derivation")
        if not isinstance(self.status, AlgebraicConstantFieldStatus):
            raise TypeError("the constant-field status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic constant analysis needs a detail")
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level] is not self.extension
        ):
            raise ValueError("the derivation must be bound to the extension")
        if self.radical is not None and self.radical.extension is not self.extension:
            raise ValueError("the radical data belongs to another extension")
        if self.exponent_gcd is not None and self.exponent_gcd < 1:
            raise ValueError("a radical exponent gcd must be positive")
        if self.embedding is not None and (
            self.embedding.source_extension is not self.extension
        ):
            raise ValueError("the constant embedding belongs to another extension")

        base_detail = _algebraic_constant_base_detail(
            self.extension,
            self.derivation,
        )
        if self.status is AlgebraicConstantFieldStatus.PRESERVED:
            if base_detail is not None:
                raise ValueError(base_detail)
            recognized = recognize_simple_radical(self.extension)
            if recognized != self.radical:
                raise ValueError("a preserved result needs recognized radical data")
            if recognized is None:
                raise ValueError("a preserved result needs a geometric radical proof")
            exponent_gcd, _witness, _polynomial = _radical_constant_certificate(
                recognized,
            )
            if self.exponent_gcd != exponent_gcd or exponent_gcd != 1:
                raise ValueError("a preserved result needs a geometric radical proof")
            if self.embedding is not None:
                raise ValueError("a preserved result cannot carry a new constant")
        elif self.status is AlgebraicConstantFieldStatus.REBASING_REQUIRED:
            if base_detail is not None:
                raise ValueError(base_detail)
            if self.embedding is None:
                raise ValueError("a rebasing result needs a constant embedding")
            if self.radical is None:
                constant_relation = _constant_minimal_polynomial(self.extension)
                if constant_relation is None:
                    raise ValueError(
                        "a non-radical rebasing result needs a constant relation"
                    )
                if self.exponent_gcd is not None:
                    raise ValueError(
                        "a non-radical rebasing result cannot carry a radical gcd"
                    )
                if self.embedding.generator_image != self.extension.field.generator:
                    raise ValueError(
                        "a constant-relation certificate must map its generator"
                    )
                if not _same_univariate_polynomial_coefficients(
                    self.embedding.minimal_polynomial,
                    constant_relation,
                ):
                    raise ValueError(
                        "the constant relation and embedding polynomial disagree"
                    )
            else:
                recognized = recognize_simple_radical(self.extension)
                if recognized != self.radical:
                    raise ValueError("a rebasing result needs recognized radical data")
                if recognized is None:
                    raise ValueError("a rebasing result needs radical data")
                exponent_gcd, witness, polynomial = _radical_constant_certificate(
                    recognized,
                )
                if self.exponent_gcd != exponent_gcd or exponent_gcd == 1:
                    raise ValueError("a radical rebasing result needs a nontrivial gcd")
                if self.embedding.generator_image != witness:
                    raise ValueError("the radical constant witness is inconsistent")
                if not _same_univariate_polynomial_coefficients(
                    self.embedding.minimal_polynomial,
                    polynomial,
                ):
                    raise ValueError("the radical constant polynomial is inconsistent")
        elif self.embedding is not None:
            raise ValueError("an unsupported result cannot carry a certificate")

    @property
    def preserved(self) -> bool:
        """Whether the declared constants are proved exact in this scope."""
        return self.status is AlgebraicConstantFieldStatus.PRESERVED

    @property
    def rebasing_required(self) -> bool:
        """Whether a new algebraic constant has been exhibited exactly."""
        return self.status is AlgebraicConstantFieldStatus.REBASING_REQUIRED

    @property
    def unsupported(self) -> bool:
        """Whether this analyzer cannot decide the presented extension."""
        return self.status is AlgebraicConstantFieldStatus.UNSUPPORTED

    @property
    def constant_witness(self) -> ExtensionElement | None:
        """Return the displayed new constant when rebasing is required."""
        if self.embedding is None:
            return None
        return self.embedding.generator_image

    @property
    def constant_minimal_polynomial(self) -> Poly | None:
        """Return the minimal polynomial of the displayed new constant."""
        if self.embedding is None:
            return None
        return self.embedding.minimal_polynomial

    @property
    def proposed_constant_domain(self) -> AlgebraicField | None:
        """Return the exact constant field embedded by this analysis."""
        if self.embedding is None:
            return None
        return self.embedding.constant_domain


def recognize_simple_radical(
    extension: AlgebraicExtension,
) -> SimpleRadicalData | None:
    r"""Recognize and normalize ``K(x)[y]/(y**n - a)``.

    Return ``None`` when the minimal polynomial is not a pure radical.  This
    first implementation uses a one-variable rational parent field, matching
    the current scope of algebraic Hermite reduction.
    """
    modulus = extension.minimal_polynomial
    degree = modulus.degree()
    expected = extension.generator**degree + extension.parent_field.to_sympy(
        modulus.TC(),
    )
    if modulus.as_expr() != expected:
        return None

    parent, constants, variable = _base_polynomial_ring(extension)
    radicand = parent.convert(-modulus.nth(0))
    numerator = Poly(radicand.numer.as_expr(), variable, domain=constants)
    denominator = Poly(radicand.denom.as_expr(), variable, domain=constants)
    # If y**n = A/D, then (y*D)**n = A*D**(n-1).  Split the latter as
    # F**n*H by writing each factor multiplicity as n*q + r, 0 <= r < n.
    powered_radicand = numerator * denominator ** (degree - 1)

    unit, factors = powered_radicand.sqf_list()
    extracted_factor = Poly(1, variable, domain=constants)
    normalized_radicand = Poly(unit, variable, domain=constants)
    for factor, multiplicity in factors:
        quotient, remainder = divmod(multiplicity, degree)
        extracted_factor *= factor**quotient
        normalized_radicand *= factor**remainder

    normalized_generator = (
        extension.field.generator
        * _embed_parent(extension, denominator)
        / _embed_parent(extension, extracted_factor)
    )
    return SimpleRadicalData(
        extension=extension,
        degree=degree,
        radicand=radicand,
        numerator=numerator,
        denominator=denominator,
        extracted_factor=extracted_factor,
        normalized_radicand=normalized_radicand,
        normalized_generator=normalized_generator,
    )


def _radical_constant_certificate(
    radical: SimpleRadicalData,
) -> tuple[int, ExtensionElement, Poly]:
    """Return the exponent gcd, constant witness, and its polynomial."""
    extension = radical.extension
    _, constants, variable = _base_polynomial_ring(extension)
    unit, factors = radical.normalized_radicand.sqf_list()
    exponent_gcd = radical.degree
    for _factor, multiplicity in factors:
        exponent_gcd = gcd(exponent_gcd, multiplicity)

    factor_root = Poly(1, variable, domain=constants)
    for factor, multiplicity in factors:
        factor_root *= factor ** (multiplicity // exponent_gcd)
    witness = cast(
        ExtensionElement,
        radical.normalized_generator ** (radical.degree // exponent_gcd)
        / _embed_parent(extension, factor_root),
    )
    constant = constants.convert(unit)
    indeterminate = Dummy("constant")
    constant_minimal_polynomial = Poly(
        indeterminate**exponent_gcd - constants.to_sympy(constant),
        indeterminate,
        domain=constants,
    )
    return exponent_gcd, witness, constant_minimal_polynomial


def _algebraic_constant_base_detail(
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> str | None:
    """Return why the present constant theorem does not apply, if any."""
    if derivation.level != 1:
        return (
            "constant analysis currently supports one algebraic extension "
            "directly over C(x)"
        )

    base = derivation.tower.levels[0]
    if not isinstance(base, MonomialExtension):
        return "the lower level is not a rational monomial extension"

    try:
        parent, constants, variable = _base_polynomial_ring(extension)
    except NotImplementedError as exc:
        return str(exc)

    if base.field != parent or base.parent_field != constants:
        return "the algebraic parent is not the active rational lower field"
    if base.generator != variable:
        return "the rational base generator does not match the algebraic parent"
    if base.generator_derivative != Poly(1, variable, domain=constants):
        return "constant analysis currently requires the standard derivative Dx = 1"
    return None


def _constant_minimal_polynomial(
    extension: AlgebraicExtension,
) -> Poly | None:
    """Return the defining polynomial over ``C`` when it has no ``x`` terms."""
    parent, constants, _variable = _base_polynomial_ring(extension)
    coefficients: dict[tuple[int, ...], object] = {}
    for monomial, coefficient in extension.minimal_polynomial.terms():
        parent_coefficient = parent.convert(coefficient)
        numerator_terms = parent_coefficient.numer.to_dict()
        denominator_terms = parent_coefficient.denom.to_dict()
        if any(monomial != (0,) for monomial in numerator_terms) or any(
            monomial != (0,) for monomial in denominator_terms
        ):
            return None
        numerator = numerator_terms.get((0,), constants.zero)
        denominator = denominator_terms.get((0,), constants.one)
        coefficients[monomial] = constants.convert(numerator) / constants.convert(
            denominator
        )
    return Poly.from_dict(coefficients, extension.generator, domain=constants)


def _build_algebraic_constant_embedding(
    extension: AlgebraicExtension,
    minimal_polynomial: Poly,
    witness: ExtensionElement,
) -> tuple[AlgebraicConstantEmbedding | None, str | None]:
    """Build the exact embedded constant field, or name the backend boundary."""
    if minimal_polynomial.is_irreducible is not True:
        return (
            None,
            "the exhibited constant has no supported irreducible minimal "
            "polynomial over the displayed constants",
        )
    try:
        proposed_candidate = minimal_polynomial.domain.alg_field_from_poly(
            minimal_polynomial,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        return (
            None,
            "the new constant is certified, but SymPy cannot construct its "
            f"exact embedding: {exc}",
        )
    if not isinstance(proposed_candidate, AlgebraicField):
        return (
            None,
            "the new constant is certified, but SymPy did not return an "
            "algebraic-number constant domain",
        )

    try:
        embedding = AlgebraicConstantEmbedding(
            extension,
            proposed_candidate,
            minimal_polynomial,
            witness,
        )
        if embedding.embed(proposed_candidate.unit) != witness:
            return None, "SymPy could not realize the proposed constant embedding"
        if embedding.embed(proposed_candidate.one) != extension.field.one:
            return None, "SymPy could not realize the proposed constant embedding"
    except (NotImplementedError, TypeError, ValueError) as exc:
        return (
            None,
            "the new constant is certified, but SymPy cannot realize its "
            f"exact embedding: {exc}",
        )
    return embedding, None


def _unsupported_algebraic_constant_analysis(
    extension: AlgebraicExtension,
    derivation: Derivation,
    detail: str,
    *,
    radical: SimpleRadicalData | None = None,
    exponent_gcd: int | None = None,
) -> AlgebraicConstantFieldAnalysis:
    """Build the explicit non-decision result used by this narrow analyzer."""
    return AlgebraicConstantFieldAnalysis(
        extension,
        derivation,
        AlgebraicConstantFieldStatus.UNSUPPORTED,
        radical,
        exponent_gcd,
        None,
        detail,
    )


def analyze_algebraic_constant_field(
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> AlgebraicConstantFieldAnalysis:
    r"""Analyze constants in one supported algebraic function field.

    For a normalized radical presentation

    ``L = C(x)(z),  z**n = c * product(f_i**e_i)``,

    the exponent gcd ``d = gcd(n, e_1, ...)`` controls geometricity.  If
    ``d = 1``, the binomial remains irreducible over an algebraic closure of
    ``C`` and ``Const(L) = C``.  If ``d > 1``,

    ``z**(n/d) / product(f_i**(e_i/d))``

    is a constant whose ``d``-th power is ``c``.  The latter case returns an
    exact embedded constant subfield and requires a later presentation rebase;
    it does not claim the current finite-extension level can simply be copied
    over the larger constants.

    A defining polynomial already over ``C`` gives an immediate rebasing
    certificate with its generator as witness. A result of
    :attr:`AlgebraicConstantFieldStatus.UNSUPPORTED` is never a preservation
    claim. The first implementation intentionally covers only a single
    algebraic level over the standard rational base ``C(x)``, ``Dx = 1``.
    """
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("constant analysis requires an algebraic extension")
    if not isinstance(derivation, Derivation):
        raise TypeError("constant analysis requires a Derivation")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not extension
    ):
        raise ValueError("the derivation must be bound to the supplied extension")

    base_detail = _algebraic_constant_base_detail(extension, derivation)
    if base_detail is not None:
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            base_detail,
        )

    constant_relation = _constant_minimal_polynomial(extension)
    if constant_relation is not None:
        witness = extension.field.generator
        if derivation(witness) != extension.field.zero:
            raise ValueError("a displayed algebraic constant is not annihilated by D")
        embedding, detail = _build_algebraic_constant_embedding(
            extension,
            constant_relation,
            witness,
        )
        if embedding is None:
            if detail is None:
                raise ValueError("a failed constant embedding needs a reason")
            return _unsupported_algebraic_constant_analysis(
                extension,
                derivation,
                detail,
            )
        return AlgebraicConstantFieldAnalysis(
            extension,
            derivation,
            AlgebraicConstantFieldStatus.REBASING_REQUIRED,
            None,
            None,
            embedding,
            "the defining polynomial is constant, so its algebraic generator "
            "belongs in the exact constant field",
        )

    radical = recognize_simple_radical(extension)
    if radical is None:
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            "the extension is not a recognized simple radical",
        )

    exponent_gcd, witness, constant_minimal_polynomial = _radical_constant_certificate(
        radical
    )

    if exponent_gcd == 1:
        return AlgebraicConstantFieldAnalysis(
            extension,
            derivation,
            AlgebraicConstantFieldStatus.PRESERVED,
            radical,
            exponent_gcd,
            None,
            "the normalized radical is geometrically irreducible by the "
            "binomial exponent-gcd criterion",
        )

    constant = constant_minimal_polynomial.domain.convert(
        -constant_minimal_polynomial.nth(0),
    )
    embedded_constant = _extension_element(
        extension,
        constant_minimal_polynomial.domain.to_sympy(constant),
    )
    if witness**exponent_gcd != embedded_constant:
        raise ValueError("the radical constant witness failed its power identity")
    if derivation(witness) != extension.field.zero:
        raise ValueError("the radical constant witness is not annihilated by D")

    embedding, detail = _build_algebraic_constant_embedding(
        extension,
        constant_minimal_polynomial,
        witness,
    )
    if embedding is None:
        if detail is None:
            raise ValueError("a failed constant embedding needs a reason")
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            detail,
            radical=radical,
            exponent_gcd=exponent_gcd,
        )

    return AlgebraicConstantFieldAnalysis(
        extension,
        derivation,
        AlgebraicConstantFieldStatus.REBASING_REQUIRED,
        radical,
        exponent_gcd,
        embedding,
        "a nontrivial algebraic constant was certified; the function-field "
        "presentation must be rebased before global algebraic decisions",
    )


def simple_radical_integral_basis(
    extension: AlgebraicExtension,
) -> IntegralBasis:
    r"""Construct Bronstein's explicit integral basis for ``y**n = a``.

    For normalized ``z**n = product(H_j**j)``, section 2.2, equation (11) of
    Bronstein's ISSAC'98 tutorial gives
    ``w_(r+1) = z**r/product(H_j**floor(r*j/n))``.
    """
    radical = recognize_simple_radical(extension)
    if radical is None:
        raise ValueError("the algebraic extension is not a simple radical")

    _, factors = radical.normalized_radicand.sqf_list()
    elements = []
    for power in range(radical.degree):
        basis_denominator = radical.normalized_radicand.one
        for factor, multiplicity in factors:
            basis_denominator *= factor ** ((power * multiplicity) // radical.degree)
        elements.append(
            radical.normalized_generator**power
            / _embed_parent(extension, basis_denominator),
        )
    return IntegralBasis(extension, tuple(elements))


def _coordinate_matrix(
    basis: IntegralBasis,
    elements: tuple[ExtensionElement, ...],
) -> DomainMatrix:
    parent = basis.extension.parent_field
    rank = len(basis.elements)
    columns = [basis.coordinates(element) for element in elements]
    return DomainMatrix(
        [[columns[column][row] for column in range(rank)] for row in range(rank)],
        (rank, rank),
        parent,
    )


def _primitive_polynomial_vector(
    elements: tuple[FracElement, ...],
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> tuple[Poly, ...]:
    """Clear denominators and remove the polynomial gcd from a vector."""
    denominator = _common_denominator(elements, constants, variable)
    denominator_element = _polynomial_parent_element(denominator, parent)
    numerators = tuple(
        _parent_element_polynomial(
            element * denominator_element,
            parent,
            constants,
            variable,
        )
        for element in elements
    )
    common_factor = Poly(0, variable, domain=constants)
    for numerator in numerators:
        common_factor = common_factor.gcd(numerator)
    if common_factor.is_zero:
        raise ValueError("expected a nonzero polynomial relation")
    common_factor = common_factor.monic()
    return tuple(numerator.exquo(common_factor) for numerator in numerators)


def _denominator_excess(denominator: Poly) -> int:
    return sum(
        (multiplicity - 1) * factor.degree()
        for factor, multiplicity in denominator.sqf_list()[1]
    )


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


def _adjoin_integral_element(
    basis: IntegralBasis,
    numerators: tuple[Poly, ...],
    denominator: Poly,
) -> IntegralBasis:
    r"""Adjoin ``w0 = (C_1*w_1 + ... + C_n*w_n)/F`` to the basis module.

    Relative to ``(w_1/F, ..., w_n/F)``, the old basis and ``w0`` are the
    rows of ``[F*I; C]``.  Bronstein reduces this matrix by Hermitian row
    reduction after Theorem 3 in section 2.1 of his ISSAC'98 tutorial.  A
    Smith decomposition performs equivalent unimodular row operations over
    the PID ``K[x]``; the nonzero rows of ``left*[F*I; C]`` are therefore a
    basis of the enlarged module.
    """
    extension = basis.extension
    parent, constants, variable = _base_polynomial_ring(extension)
    polynomial_ring = constants.poly_ring(variable)
    rank = len(basis.elements)
    denominator_coefficient = polynomial_ring.from_sympy(denominator.as_expr())
    zero = polynomial_ring.zero
    rows = [
        [denominator_coefficient if row == column else zero for column in range(rank)]
        for row in range(rank)
    ]
    rows.append(
        [polynomial_ring.from_sympy(numerator.as_expr()) for numerator in numerators]
    )
    generators = DomainMatrix(rows, (rank + 1, rank), polynomial_ring)
    _, left, _ = smith_normal_decomp(generators)
    reduced_rows = [
        row
        for row in (left * generators).to_list()
        if any(coefficient != zero for coefficient in row)
    ]
    if len(reduced_rows) != rank:
        raise ValueError("integral-element adjunction changed the field rank")

    denominator_element = _polynomial_parent_element(denominator, parent)
    elements = tuple(
        basis.element(
            tuple(
                parent.from_sympy(polynomial_ring.to_sympy(coefficient))
                / denominator_element
                for coefficient in row
            )
        )
        for row in reduced_rows
    )
    return IntegralBasis(extension, elements)


def _hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool,
) -> tuple[ExtensionElement, ExtensionElement, IntegralBasis]:
    r"""Implement equations (7)--(9) of Bronstein's ISSAC'98 tutorial.

    For denominator ``D = U*V**m``, the algorithm seeks ``B`` such that
    subtracting ``derivation(B/V**(m-1))`` removes the poles of order ``m``.
    The lazy branch implements Bronstein's three missing-integral-element
    certificates: dependent ``S_i``, a solution denominator intersecting
    ``V``, and a nonsquarefree basis-derivative denominator.
    """
    extension = basis.extension
    if derivation.level < 0 or derivation.tower.levels[derivation.level] != extension:
        raise ValueError("the derivation must be bound to the algebraic extension")

    parent, constants, variable = _base_polynomial_ring(extension)
    lower_derivation = derivation.lower()
    if (
        lower_derivation.field != parent
        or lower_derivation(parent.from_sympy(variable)) != parent.one
    ):
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires the standard "
            "rational derivation Dx = 1",
        )
    remainder = _extension_element(extension, integrand)
    exact_part = extension.field.zero
    differentiated_basis: IntegralBasis | None = None
    basis_derivatives: tuple[ExtensionElement, ...] | None = None
    derivative_matrix: DomainMatrix | None = None

    while True:
        representation = basis.fraction(remainder)
        repeated_factors = [
            (factor, multiplicity)
            for factor, multiplicity in representation.denominator.sqf_list()[1]
            if multiplicity > 1
        ]
        if not repeated_factors:
            return exact_part, remainder, basis

        repeated_factor, multiplicity = max(
            repeated_factors,
            key=lambda item: item[1],
        )
        denominator = representation.denominator
        cofactor = denominator.exquo(repeated_factor**multiplicity)
        scale = cofactor * repeated_factor**multiplicity
        candidate_denominator = repeated_factor ** (multiplicity - 1)

        if basis is not differentiated_basis:
            # D(w_i) depends only on the basis and derivation, not on the pole
            # currently being reduced.  Cache its coordinate matrix until a
            # lazy refinement replaces the basis.
            basis_derivatives = cast(
                tuple[ExtensionElement, ...],
                tuple(derivation(element) for element in basis.elements),
            )
            derivative_matrix = _coordinate_matrix(
                basis,
                basis_derivatives,
            )
            differentiated_basis = basis
        assert basis_derivatives is not None
        assert derivative_matrix is not None
        derivative_rows = derivative_matrix.to_list()

        if lazy:
            refined_from_derivative = False
            for column, basis_derivative in enumerate(basis_derivatives):
                derivative_denominator = _common_denominator(
                    tuple(
                        derivative_rows[row][column]
                        for row in range(len(basis.elements))
                    ),
                    constants,
                    variable,
                )
                if (
                    derivative_denominator.gcd(
                        derivative_denominator.diff(),
                    ).degree()
                    == 0
                ):
                    continue

                # Bronstein, Theorem 3: if F = product(F_j**j) is the
                # nonsquarefree denominator of D(w_i), then
                # product(F_j)*D(w_i) is integral but missing from the current
                # K[x]-module.
                squarefree_part = Poly(1, variable, domain=constants)
                for factor, _ in derivative_denominator.sqf_list()[1]:
                    squarefree_part *= factor
                integral_element = (
                    _embed_parent(extension, squarefree_part) * basis_derivative
                )
                integral_fraction = basis.fraction(integral_element)
                if integral_fraction.denominator.degree() == 0:
                    raise ValueError(
                        "nonsquarefree derivative did not enlarge the basis module",
                    )
                basis = _adjoin_integral_element(
                    basis,
                    integral_fraction.numerators,
                    integral_fraction.denominator,
                )
                refined_from_derivative = True
                break
            if refined_from_derivative:
                continue

        scale_parent = _polynomial_parent_element(scale, parent)
        candidate_denominator_parent = _polynomial_parent_element(
            candidate_denominator,
            parent,
        )
        candidate_denominator_derivative = derivation.lower()(
            candidate_denominator_parent,
        )
        candidate_denominator_element = _embed_parent(
            extension,
            candidate_denominator,
        )
        # Bronstein (8)--(9): S_i = U*V**m * D(w_i/V**(m-1)).
        # Solving sum(f_i*S_i) = numerator determines the correction modulo V.
        derivative_scale = scale_parent / candidate_denominator_parent
        diagonal_scale = (
            scale_parent
            * candidate_denominator_derivative
            / candidate_denominator_parent**2
        )
        rank = len(basis.elements)
        system = DomainMatrix(
            [
                [
                    derivative_scale * derivative_rows[row][column]
                    - (diagonal_scale if row == column else parent.zero)
                    for column in range(rank)
                ]
                for row in range(rank)
            ],
            (rank, rank),
            parent,
        )
        numerator_element = remainder * _embed_parent(extension, denominator)
        numerator_coordinates = basis.coordinates(numerator_element)
        right_hand_side = DomainMatrix(
            [[coordinate] for coordinate in numerator_coordinates],
            (rank, 1),
            parent,
        )

        try:
            # Fraction-free solving retains one common denominator.  Besides
            # avoiding expression swell, that denominator is exactly what the
            # lazy-refinement test below must compare with V.
            solution_numerators_matrix, solution_denominator_element = system.solve_den(
                right_hand_side
            )
        except DMNonInvertibleMatrixError as exc:
            if lazy:
                # Bronstein, Theorem 1: a primitive polynomial relation
                # sum(T_i*S_i) = 0 exposes the missing integral element
                # U/V * sum(T_i*w_i).
                relations = system.nullspace().to_list()
                if not relations:
                    raise ValueError(
                        "the singular Hermite system has no nullspace",
                    ) from exc
                relation = _primitive_polynomial_vector(
                    tuple(relations[0]),
                    parent,
                    constants,
                    variable,
                )
                basis = _adjoin_integral_element(
                    basis,
                    tuple(cofactor * coefficient for coefficient in relation),
                    repeated_factor,
                )
                continue
            raise ValueError(
                "the supplied integral basis does not support Hermite reduction",
            ) from exc
        solution = tuple(
            row[0] / solution_denominator_element
            for row in solution_numerators_matrix.to_list()
        )

        solution_denominator = _common_denominator(
            solution,
            constants,
            variable,
        )
        solution_denominator_element = _polynomial_parent_element(
            solution_denominator,
            parent,
        )
        solution_numerators = tuple(
            _parent_element_polynomial(
                coefficient * solution_denominator_element,
                parent,
                constants,
                variable,
            )
            for coefficient in solution
        )
        shared_factor = solution_denominator.gcd(repeated_factor)
        if shared_factor.degree() != 0:
            if lazy:
                # Bronstein, Theorem 2: for f_i = T_i/Q and G = gcd(V, Q),
                #
                #     w0 = U*(V/G)/G * sum(T_i*w_i)
                #
                # is integral but absent from the current K[x]-module.  Its
                # adjunction strictly enlarges that module before we retry V.
                multiplier = cofactor * repeated_factor.exquo(shared_factor)
                basis = _adjoin_integral_element(
                    basis,
                    tuple(multiplier * numerator for numerator in solution_numerators),
                    shared_factor,
                )
                continue
            raise ValueError(
                "the supplied basis is not an integral basis at the repeated factor",
            )
        # When gcd(Q, V) = 1, Q has an inverse modulo V.  Taking
        # B_i = Q**(-1)*T_i mod V gives B/V**(m-1), the exact part in (7).
        inverse = invert(solution_denominator, repeated_factor)
        correction_coordinates = tuple(
            _polynomial_parent_element(
                (inverse * numerator).rem(repeated_factor),
                parent,
            )
            for numerator in solution_numerators
        )
        correction = (
            basis.element(correction_coordinates) / candidate_denominator_element
        )

        previous_excess = _denominator_excess(denominator)
        new_remainder = remainder - derivation(correction)
        new_denominator = basis.fraction(new_remainder).denominator
        if _denominator_excess(new_denominator) >= previous_excess:
            raise ValueError(
                "algebraic Hermite reduction did not lower the repeated pole; "
                "the supplied basis may not be integral",
            )

        exact_part += correction
        remainder = new_remainder


def hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[ExtensionElement, ExtensionElement]:
    r"""Write ``integrand = D(g) + h`` using a supplied integral basis."""
    exact, remainder, _ = _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=False,
    )
    return exact, remainder


def lazy_hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[ExtensionElement, ExtensionElement, IntegralBasis]:
    r"""Hermite-reduce while refining denominator-obstructed integral bases.

    This implements the three lazy-refinement theorems in Manuel Bronstein,
    *Symbolic Integration Tutorial*, ISSAC 1998, section 2.1, which summarizes
    the fuller treatment in *The Lazy Hermite Reduction*, INRIA RR-3562.
    """
    return _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=True,
    )
