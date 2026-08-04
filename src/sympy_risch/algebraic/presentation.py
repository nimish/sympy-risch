r"""Checked presentations of finite algebraic composita.

The tensor product of separable source fields need not itself be a field.  A
primitive-element computation must therefore distinguish a single compositum
field from the finite etale algebra containing every source-compatible field
component.  The data objects in this module validate the maps after the
algorithm in :mod:`sympy_risch.algebraic.compositum` constructs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement

from .._typing import FieldElement
from ..tower import AlgebraicExtension


def _embed_parent(
    target: AlgebraicExtension,
    value: object,
) -> ExtensionElement:
    """Embed one parent-field value in ``target``."""
    parent = target.parent_field
    return target.from_power_basis_coordinates(
        (
            parent.convert(value),
            *(parent.zero for _ in range(target.field.rank - 1)),
        )
    )


def _evaluate_source_polynomial(
    source: AlgebraicExtension,
    value: ExtensionElement,
    target: AlgebraicExtension,
) -> ExtensionElement:
    """Evaluate a source defining polynomial at an element of ``target``."""
    result = target.field.zero
    for coefficient in source.minimal_polynomial.all_coeffs():
        result = result * value + _embed_parent(target, coefficient)
    return cast(ExtensionElement, result)


@dataclass(frozen=True, slots=True)
class PrimitiveElementPresentation:
    r"""Checked maps from source extensions into one compositum component.

    If the source generators are ``a_i`` and the target generator is ``z``,
    the certificate records

    ``z = sum(c_i*a_i)`` and ``a_i = r_i(z)``.

    The target degree can be smaller than the product of the source degrees:
    that is exactly what happens when the source fields are dependent and this
    object describes one field component of their tensor product.
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

        normalized_coefficients = tuple(parent.convert(value) for value in coefficients)
        normalized_images = tuple(
            cast(ExtensionElement, self.target.field.convert(value)) for value in images
        )
        object.__setattr__(self, "primitive_coefficients", normalized_coefficients)
        object.__setattr__(self, "source_images", normalized_images)

        for source, image in zip(sources, normalized_images, strict=True):
            if (
                _evaluate_source_polynomial(source, image, self.target)
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
        """Dimension of this compositum component over the parent field."""
        return self.target.field.rank

    @property
    def has_full_tensor_degree(self) -> bool:
        """Whether the source tensor product is this single field."""
        return self.tensor_rank == prod(source.field.rank for source in self.sources)


@dataclass(frozen=True, slots=True)
class EtaleCompositumPresentation:
    r"""Complete field-component decomposition of a separable tensor product.

    In characteristic zero every admitted source extension is separable.  The
    sum of the component degrees must therefore equal the product of the source
    degrees.  Each component carries its own exact source embeddings; equal
    abstract target polynomials can still represent different source maps.
    """

    sources: tuple[AlgebraicExtension, ...]
    components: tuple[PrimitiveElementPresentation, ...]

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        components = tuple(self.components)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "components", components)
        if len(sources) < 2:
            raise ValueError("an etale compositum needs at least two sources")
        if not components:
            raise ValueError("an etale compositum needs at least one component")
        if any(component.sources != sources for component in components):
            raise ValueError("every compositum component must describe all sources")
        expected_rank = prod(source.field.rank for source in sources)
        actual_rank = sum(component.target.field.rank for component in components)
        if actual_rank != expected_rank:
            raise ValueError(
                "compositum component degrees do not exhaust the source tensor product"
            )

    @property
    def tensor_rank(self) -> int:
        """Dimension of the complete tensor product over the parent field."""
        return sum(component.target.field.rank for component in self.components)

    @property
    def is_field(self) -> bool:
        """Whether the tensor product has exactly one field component."""
        return len(self.components) == 1


class AmbiguousCompositumError(NotImplementedError):
    """The tensor product has several fields and no component was selected."""

    decomposition: EtaleCompositumPresentation

    def __init__(self, decomposition: EtaleCompositumPresentation) -> None:
        self.decomposition = decomposition
        super().__init__(
            "the source tensor product has several field components; exact "
            "source-relation evidence is required to select one"
        )


__all__ = [
    "AmbiguousCompositumError",
    "EtaleCompositumPresentation",
    "PrimitiveElementPresentation",
]
