r"""Integral bases and simple-radical normalization."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from ..tower import AlgebraicExtension
from ._field import (
    _base_polynomial_ring,
    _common_denominator,
    _embed_parent,
    _extension_element,
    _parent_element_polynomial,
    _polynomial_parent_element,
)


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
