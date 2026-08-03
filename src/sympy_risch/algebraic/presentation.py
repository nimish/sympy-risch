r"""Primitive-element normalization for algebraic presentations.

This module combines several finite extensions of the same exact base field
without constructing nested :class:`~sympy.polys.agca.extensions.FiniteExtension`
domains.  The tensor-product basis is used only to prove that a displayed
linear combination is primitive and to recover each source generator in the
resulting monogenic field.

The current factory admits the useful cell where the displayed generator sum
has an irreducible full-degree characteristic polynomial.  Failure is reported
as unsupported: it can mean either that another primitive linear combination
is needed or that the tensor product is reducible and branch information must
select a component.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import cast

from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower import AlgebraicExtension


def _embed_parent(
    target: AlgebraicExtension,
    value: object,
) -> ExtensionElement:
    """Embed a parent-field value through the target power basis."""
    parent = target.parent_field
    return target.from_power_basis_coordinates(
        (
            parent.convert(value),
            *(parent.zero for _ in range(target.field.rank - 1)),
        )
    )


def _evaluate_polynomial(
    polynomial: Poly,
    value: ExtensionElement,
    target: AlgebraicExtension,
) -> ExtensionElement:
    """Evaluate a parent-field polynomial in ``target`` by Horner's rule."""
    result = target.field.zero
    for coefficient in polynomial.all_coeffs():
        result = result * value + _embed_parent(target, coefficient)
    return cast(ExtensionElement, result)


@dataclass(frozen=True, slots=True)
class PrimitiveElementPresentation:
    r"""Checked maps from several source extensions to one primitive field.

    If the source generators are ``a_i`` and the target generator is ``z``,
    the certificate records both

    ``z = sum(c_i*a_i)`` and ``a_i = r_i(z)``.

    The target degree equals the product of the source degrees.  This is the
    executable certificate that the admitted tensor product is a field rather
    than a reducible algebra.
    """

    sources: tuple[AlgebraicExtension, ...]
    primitive_coefficients: tuple[FieldElement, ...]
    target: AlgebraicExtension
    source_images: tuple[ExtensionElement, ...]

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        coefficients = tuple(self.primitive_coefficients)
        images = tuple(self.source_images)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "primitive_coefficients", coefficients)
        object.__setattr__(self, "source_images", images)

        if len(sources) < 2:
            raise ValueError("a compositum presentation needs at least two sources")
        if any(not isinstance(source, AlgebraicExtension) for source in sources):
            raise TypeError("presentation sources must be algebraic extensions")
        if len(coefficients) != len(sources) or len(images) != len(sources):
            raise ValueError("source extensions, coefficients, and images must align")
        parent = sources[0].parent_field
        if any(source.parent_field != parent for source in sources):
            raise ValueError("all source extensions must have the same parent field")
        if self.target.parent_field != parent:
            raise ValueError("the primitive target must use the common parent field")
        if len({source.generator for source in sources}) != len(sources):
            raise ValueError("source extension generators must be distinct")
        if self.target.generator in {source.generator for source in sources}:
            raise ValueError("the primitive target generator must be fresh")
        expected_rank = prod(source.field.rank for source in sources)
        if self.target.field.rank != expected_rank:
            raise ValueError("the primitive target does not have full tensor degree")

        normalized_coefficients = tuple(parent.convert(value) for value in coefficients)
        normalized_images = tuple(
            cast(ExtensionElement, self.target.field.convert(value)) for value in images
        )
        object.__setattr__(self, "primitive_coefficients", normalized_coefficients)
        object.__setattr__(self, "source_images", normalized_images)

        for source, image in zip(sources, normalized_images, strict=True):
            if (
                _evaluate_polynomial(source.minimal_polynomial, image, self.target)
                != self.target.field.zero
            ):
                raise ValueError(
                    "a source image does not satisfy its defining relation"
                )
        primitive_image = sum(
            (
                _embed_parent(self.target, coefficient) * image
                for coefficient, image in zip(
                    normalized_coefficients,
                    normalized_images,
                    strict=True,
                )
            ),
            self.target.field.zero,
        )
        if primitive_image != self.target.field.generator:
            raise ValueError(
                "the source images do not reconstruct the primitive element"
            )

    @property
    def tensor_rank(self) -> int:
        """Dimension of the certified compositum over its parent field."""
        return self.target.field.rank


def _primitive_sum_matrix(
    sources: tuple[AlgebraicExtension, ...],
    basis: tuple[tuple[int, ...], ...],
    basis_indices: dict[tuple[int, ...], int],
    parent: Domain,
) -> DomainMatrix:
    """Return multiplication by the source-generator sum in the tensor basis."""
    rank = len(basis)
    rows = [[parent.zero for _ in range(rank)] for _ in range(rank)]
    for column, exponents in enumerate(basis):
        for source_index, source in enumerate(sources):
            degree = source.field.rank
            next_power = exponents[source_index] + 1
            if next_power < degree:
                target = list(exponents)
                target[source_index] = next_power
                rows[basis_indices[tuple(target)]][column] += parent.one
                continue
            for power in range(degree):
                coefficient = -parent.convert(source.minimal_polynomial.nth(power))
                if coefficient == parent.zero:
                    continue
                target = list(exponents)
                target[source_index] = power
                row = basis_indices[tuple(target)]
                rows[row][column] += coefficient
    return DomainMatrix(rows, (rank, rank), parent)


def normalize_primitive_element(
    sources: tuple[AlgebraicExtension, ...],
    generator: Symbol,
) -> PrimitiveElementPresentation:
    r"""Combine linearly-disjoint source fields using a primitive element.

    The characteristic polynomial of multiplication by
    ``a_1 + ... + a_r`` is computed in the tensor-product basis.  If it is
    irreducible at the full product degree, it simultaneously proves that the
    tensor product is a field and that this sum is primitive.  One exact
    change-of-basis solve then recovers every ``a_i`` as a polynomial in the
    target generator.

    A reducible characteristic polynomial is an unsupported presentation, not
    a claim that no compositum exists.
    """
    sources = tuple(sources)
    if len(sources) < 2:
        raise ValueError("primitive-element normalization needs at least two sources")
    if any(not isinstance(source, AlgebraicExtension) for source in sources):
        raise TypeError("primitive-element sources must be algebraic extensions")
    if not isinstance(generator, Symbol):
        raise TypeError("the primitive generator must be a SymPy Symbol")
    parent = sources[0].parent_field
    if parent.characteristic() != 0:
        raise NotImplementedError(
            "primitive-element normalization currently requires characteristic zero"
        )
    if any(source.parent_field != parent for source in sources):
        raise ValueError("all source extensions must have the same parent field")
    if generator in {source.generator for source in sources}:
        raise ValueError("the primitive generator must be fresh")

    degrees = tuple(source.field.rank for source in sources)
    basis = tuple(product(*(range(degree) for degree in degrees)))
    basis_indices = {monomial: index for index, monomial in enumerate(basis)}
    rank = len(basis)
    primitive_coefficients = tuple(cast(FieldElement, parent.one) for _ in sources)
    primitive_matrix = _primitive_sum_matrix(
        sources,
        basis,
        basis_indices,
        parent,
    )
    modulus = Poly.from_list(
        primitive_matrix.charpoly(),
        generator,
        domain=parent,
    ).monic()
    if modulus.degree() != rank:
        raise NotImplementedError(
            "the displayed generator sum does not certify a primitive compositum"
        )
    try:
        target = AlgebraicExtension(modulus)
    except ValueError as exc:
        raise NotImplementedError(
            "the displayed generator sum does not certify a primitive compositum"
        ) from exc

    unit = [parent.zero for _ in range(rank)]
    unit[basis_indices[(0,) * len(sources)]] = parent.one
    power_columns: list[list[object]] = []
    power = DomainMatrix([[entry] for entry in unit], (rank, 1), parent)
    for _ in range(rank):
        power_columns.append([row[0] for row in power.to_list()])
        power = primitive_matrix * power
    power_matrix = DomainMatrix(
        [[power_columns[column][row] for column in range(rank)] for row in range(rank)],
        (rank, rank),
        parent,
    )
    try:
        power_inverse = power_matrix.inv()
    except DMNonInvertibleMatrixError as exc:
        raise RuntimeError(
            "a full-degree primitive characteristic polynomial had dependent powers"
        ) from exc

    images: list[ExtensionElement] = []
    for source_index in range(len(sources)):
        monomial = [0 for _ in sources]
        monomial[source_index] = 1
        source_vector = [parent.zero for _ in range(rank)]
        source_vector[basis_indices[tuple(monomial)]] = parent.one
        coordinates = power_inverse * DomainMatrix(
            [[entry] for entry in source_vector],
            (rank, 1),
            parent,
        )
        images.append(
            target.from_power_basis_coordinates(
                tuple(row[0] for row in coordinates.to_list())
            )
        )

    return PrimitiveElementPresentation(
        sources,
        primitive_coefficients,
        target,
        tuple(images),
    )


__all__ = ["PrimitiveElementPresentation", "normalize_primitive_element"]
