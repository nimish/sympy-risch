r"""Hermite reduction in finite algebraic function fields."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from sympy.core.symbol import Symbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp
from sympy.polys.polytools import Poly, cancel, invert

from .tower import AlgebraicExtension, Derivation


def _extension_element(extension: AlgebraicExtension, value: Any) -> Any:
    if getattr(value, "ext", None) == extension.field:
        return value
    if hasattr(value, "as_expr"):
        value = value.as_expr()
    return extension.field.from_sympy(value)


def _base_polynomial_ring(
    extension: AlgebraicExtension,
) -> tuple[FractionField, Domain, Symbol]:
    parent = extension.parent_field
    if not isinstance(parent, FractionField) or len(parent.symbols) != 1:
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires a one-variable "
            "rational parent field",
        )
    if not parent.domain.is_Field:
        raise ValueError("the rational function field must be over a field")
    return parent, parent.domain, parent.symbols[0]


def _parent_element_polynomial(
    element: Any,
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> Poly:
    element = parent.convert(element)
    denominator = Poly(element.denom.as_expr(), variable, domain=constants)
    if denominator.degree() > 0:
        raise ValueError("expected a polynomial element of the parent field")
    return Poly(element.as_expr(), variable, domain=constants)


def _common_denominator(
    elements: tuple[Any, ...],
    constants: Domain,
    variable: Symbol,
) -> Poly:
    denominator = Poly(1, variable, domain=constants)
    for element in elements:
        element_denominator = Poly(
            element.denom.as_expr(),
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
class IntegralBasis:
    r"""A caller-supplied integral basis of a monogenic function field.

    SymPy currently provides finite-extension arithmetic but does not verify
    that arbitrary function-field elements form the full integral closure over
    ``K[x]``.  Construction therefore validates that ``elements`` are a field
    basis; the caller is responsible for their integrality and completeness.
    """

    extension: AlgebraicExtension
    elements: tuple[Any, ...]
    _power_inverse: ImmutableMatrix = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        elements = tuple(
            _extension_element(self.extension, element)
            for element in self.elements
        )
        object.__setattr__(self, "elements", elements)

        rank = self.extension.field.rank
        if len(elements) != rank:
            raise ValueError(f"an integral basis must contain {rank} elements")

        columns = [self._power_coordinates(element) for element in elements]
        parent = self.extension.parent_field
        matrix = ImmutableMatrix([
            [parent.to_sympy(columns[column][row]) for column in range(rank)]
            for row in range(rank)
        ])
        try:
            inverse = matrix.inv()
        except NonInvertibleMatrixError as exc:
            raise ValueError("integral basis elements are linearly dependent") from exc
        object.__setattr__(self, "_power_inverse", inverse)

    @classmethod
    def power_basis(cls, extension: AlgebraicExtension) -> IntegralBasis:
        """Use the power basis when it is known to be an integral basis."""
        return cls(extension, extension.field.basis)

    def _power_coordinates(self, element: Any) -> tuple[Any, ...]:
        representative = Poly(
            _extension_element(self.extension, element).as_expr(),
            self.extension.generator,
            domain=self.extension.parent_field,
        )
        return tuple(
            representative.nth(degree)
            for degree in range(self.extension.field.rank)
        )

    def coordinates(self, element: Any) -> tuple[Any, ...]:
        """Return the coordinates of ``element`` in this basis."""
        parent = self.extension.parent_field
        power_coordinates = self._power_coordinates(element)
        vector = ImmutableMatrix([
            parent.to_sympy(coefficient)
            for coefficient in power_coordinates
        ])
        result = self._power_inverse*vector
        return tuple(parent.from_sympy(cancel(coefficient)) for coefficient in result)

    def element(self, coordinates: tuple[Any, ...]) -> Any:
        """Construct a field element from integral-basis coordinates."""
        if len(coordinates) != len(self.elements):
            raise ValueError("coordinate vector has the wrong length")

        parent = self.extension.parent_field
        result = self.extension.field.zero
        for coefficient, basis_element in zip(coordinates, self.elements, strict=True):
            coefficient = parent.convert(coefficient)
            embedded = self.extension.field.from_sympy(
                parent.to_sympy(coefficient),
            )
            result += embedded*basis_element
        return result

    def fraction(self, element: Any) -> AlgebraicFraction:
        """Return the normalized common-denominator representation."""
        parent, constants, variable = _base_polynomial_ring(self.extension)
        coordinates = self.coordinates(element)
        denominator = _common_denominator(coordinates, constants, variable)
        denominator_element = parent.from_sympy(denominator.as_expr())
        numerators = tuple(
            _parent_element_polynomial(
                coefficient*denominator_element,
                parent,
                constants,
                variable,
            )
            for coefficient in coordinates
        )
        return AlgebraicFraction(numerators, denominator)


@dataclass(frozen=True, slots=True)
class SimpleRadicalData:
    r"""Normalized data for a simple radical extension ``y**n = a``.

    ``normalized_generator`` satisfies ``z**n = normalized_radicand``.  The
    latter is ``n``-th-power-free as a polynomial: every factor in its
    square-free factorization has multiplicity strictly smaller than ``n``.
    """

    extension: AlgebraicExtension
    degree: int
    radicand: Any
    numerator: Poly
    denominator: Poly
    extracted_factor: Poly
    normalized_radicand: Poly
    normalized_generator: Any


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
    powered_radicand = numerator*denominator**(degree - 1)

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


def simple_radical_integral_basis(
    extension: AlgebraicExtension,
) -> IntegralBasis:
    r"""Construct Bronstein's explicit integral basis for ``y**n = a``."""
    radical = recognize_simple_radical(extension)
    if radical is None:
        raise ValueError("the algebraic extension is not a simple radical")

    _, factors = radical.normalized_radicand.sqf_list()
    elements = []
    for power in range(radical.degree):
        basis_denominator = radical.normalized_radicand.one
        for factor, multiplicity in factors:
            basis_denominator *= factor**(
                (power*multiplicity) // radical.degree
            )
        elements.append(
            radical.normalized_generator**power
            / _embed_parent(extension, basis_denominator),
        )
    return IntegralBasis(extension, tuple(elements))


def _coordinate_matrix(
    basis: IntegralBasis,
    elements: tuple[Any, ...],
) -> ImmutableMatrix:
    parent = basis.extension.parent_field
    rank = len(basis.elements)
    columns = [basis.coordinates(element) for element in elements]
    return ImmutableMatrix([
        [parent.to_sympy(columns[column][row]) for column in range(rank)]
        for row in range(rank)
    ])


def _denominator_excess(denominator: Poly) -> int:
    return sum(
        (multiplicity - 1)*factor.degree()
        for factor, multiplicity in denominator.sqf_list()[1]
    )


def _embed_parent(extension: AlgebraicExtension, element: Any) -> Any:
    parent = extension.parent_field
    if isinstance(element, Poly):
        element = parent.from_sympy(element.as_expr())
    else:
        element = parent.convert(element)
    return extension.field.from_sympy(parent.to_sympy(element))


def _adjoin_integral_element(
    basis: IntegralBasis,
    numerators: tuple[Poly, ...],
    denominator: Poly,
) -> IntegralBasis:
    """Return a basis for ``K[x] basis + K[x] (numerators*basis)/denominator``."""
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
    rows.append([
        polynomial_ring.from_sympy(numerator.as_expr())
        for numerator in numerators
    ])
    generators = DomainMatrix(rows, (rank + 1, rank), polynomial_ring)
    _, left, _ = smith_normal_decomp(generators)
    reduced_rows = [
        row
        for row in (left*generators).to_list()
        if any(coefficient != zero for coefficient in row)
    ]
    if len(reduced_rows) != rank:
        raise ValueError("integral-element adjunction changed the field rank")

    denominator_element = parent.from_sympy(denominator.as_expr())
    elements = tuple(
        basis.element(tuple(
            parent.from_sympy(polynomial_ring.to_sympy(coefficient))
            / denominator_element
            for coefficient in row
        ))
        for row in reduced_rows
    )
    return IntegralBasis(extension, elements)


def _hermite_reduce_algebraic(
    integrand: Any,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool,
) -> tuple[Any, Any, IntegralBasis]:
    extension = basis.extension
    if derivation.level < 0 or derivation.tower.levels[derivation.level] != extension:
        raise ValueError("the derivation must be bound to the algebraic extension")

    parent, constants, variable = _base_polynomial_ring(extension)
    remainder = _extension_element(extension, integrand)
    exact_part = extension.field.zero

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
        scale = cofactor*repeated_factor**multiplicity
        candidate_denominator = repeated_factor**(multiplicity - 1)

        scale_element = _embed_parent(extension, scale)
        candidate_denominator_element = _embed_parent(
            extension,
            candidate_denominator,
        )
        scaled_derivatives = tuple(
            scale_element*derivation(
                basis_element/candidate_denominator_element,
            )
            for basis_element in basis.elements
        )
        system = _coordinate_matrix(basis, scaled_derivatives)
        numerator_element = remainder*_embed_parent(extension, denominator)
        numerator_coordinates = basis.coordinates(numerator_element)
        right_hand_side = ImmutableMatrix([
            parent.to_sympy(coordinate)
            for coordinate in numerator_coordinates
        ])

        try:
            solution_expressions = system.inv()*right_hand_side
        except NonInvertibleMatrixError as exc:
            raise ValueError(
                "the supplied integral basis does not support Hermite reduction",
            ) from exc
        solution = tuple(
            parent.from_sympy(cancel(coefficient))
            for coefficient in solution_expressions
        )

        solution_denominator = _common_denominator(
            solution,
            constants,
            variable,
        )
        solution_denominator_element = parent.from_sympy(
            solution_denominator.as_expr(),
        )
        solution_numerators = tuple(
            _parent_element_polynomial(
                coefficient*solution_denominator_element,
                parent,
                constants,
                variable,
            )
            for coefficient in solution
        )
        shared_factor = solution_denominator.gcd(repeated_factor)
        if shared_factor.degree() != 0:
            if lazy:
                multiplier = cofactor*repeated_factor.exquo(shared_factor)
                basis = _adjoin_integral_element(
                    basis,
                    tuple(multiplier*numerator for numerator in solution_numerators),
                    shared_factor,
                )
                continue
            raise ValueError(
                "the supplied basis is not an integral basis at the repeated factor",
            )
        inverse = invert(solution_denominator, repeated_factor)
        correction_coordinates = tuple(
            parent.from_sympy(
                (inverse*numerator).rem(repeated_factor).as_expr(),
            )
            for numerator in solution_numerators
        )
        correction = basis.element(correction_coordinates) / candidate_denominator_element

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
    integrand: Any,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[Any, Any]:
    r"""Write ``integrand = D(g) + h`` using a supplied integral basis."""
    exact, remainder, _ = _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=False,
    )
    return exact, remainder


def lazy_hermite_reduce_algebraic(
    integrand: Any,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[Any, Any, IntegralBasis]:
    r"""Hermite-reduce while refining denominator-obstructed integral bases."""
    return _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=True,
    )
