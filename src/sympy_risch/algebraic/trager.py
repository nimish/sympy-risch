r"""Trager finite residues and unramified finite places."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement, FiniteExtension
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.polytools import Poly, resultant

from .._typing import is_field_zero
from ..tower import AlgebraicExtension, Derivation
from ._field import (
    _algebraic_constant_domain,
    _base_polynomial_ring,
    _extension_element,
    _parent_element_polynomial,
    _polynomial_parent_element,
)
from .basis import AlgebraicFraction, IntegralBasis


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
