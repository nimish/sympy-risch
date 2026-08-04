r"""Trager composita for finite separable extensions of one exact field.

For source fields ``K(a)`` and ``K(b)``, choose ``z = a + s*b`` and compute

``Res_a(f(a), s**deg(g) * g((z - a)/s))``.

When this norm is square-free, ``z`` separates every geometric point of the
finite etale tensor product.  Factoring the norm over ``K`` gives exactly its
field components.  Reducing the resultant's subresultant sequence modulo each
factor gives a linear relation that recovers ``a`` and then ``b``.  This is
Trager's square-free-norm primitive-element algorithm; unlike an
irreducibility search it terminates immediately on ordinary dependent
composita and exposes their components rather than rejecting them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.symbol import Dummy, Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import DomainError, OperationNotSupported
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower import AlgebraicExtension
from .presentation import (
    AmbiguousCompositumError,
    EtaleCompositumPresentation,
    PrimitiveElementPresentation,
)


@dataclass(frozen=True, slots=True)
class _PartialComponent:
    """One component while sources are merged from left to right."""

    target: AlgebraicExtension
    coefficients: tuple[FieldElement, ...]
    source_images: tuple[ExtensionElement, ...]


@dataclass(frozen=True, slots=True)
class _TragerNormData:
    """One norm and its already-computed subresultant recovery sequence."""

    modulus: Poly
    subresultants: tuple[Poly, ...]


def _embed_parent(
    target: AlgebraicExtension,
    value: object,
) -> ExtensionElement:
    """Embed a parent value without generic ``FiniteExtension.convert``."""
    parent = target.parent_field
    return target.from_power_basis_coordinates(
        (
            parent.convert(value),
            *(parent.zero for _ in range(target.field.rank - 1)),
        )
    )


def _validate_sources(
    sources: tuple[AlgebraicExtension, ...],
    generator: Symbol,
) -> Domain:
    """Validate the common exact characteristic-zero compositum boundary."""
    if len(sources) < 2:
        raise ValueError("primitive-element normalization needs at least two sources")
    if any(not isinstance(source, AlgebraicExtension) for source in sources):
        raise TypeError("primitive-element sources must be algebraic extensions")
    if not isinstance(generator, Symbol):
        raise TypeError("the primitive generator must be a SymPy Symbol")
    parent = sources[0].parent_field
    if parent.characteristic() != 0:
        raise NotImplementedError(
            "Trager compositum normalization currently requires characteristic zero"
        )
    if any(source.parent_field != parent for source in sources):
        raise ValueError("all source extensions must have the same parent field")
    if len({source.generator for source in sources}) != len(sources):
        raise ValueError("source extension generators must be distinct")
    if generator in {source.generator for source in sources}:
        raise ValueError("the primitive generator must be fresh")
    if generator in (getattr(parent, "symbols", ()) or ()):
        raise ValueError(
            f"primitive generator {generator} already occurs in the parent field"
        )
    return parent


def _scaled_shifted_relation(
    polynomial: Poly,
    eliminated_generator: Symbol,
    primitive_value: object,
    scalar: object,
    coefficient_domain: Domain,
) -> Poly:
    r"""Return ``s**n*f((z-a)/s)`` as a polynomial in ``a``.

    The scaled Horner recurrence avoids division in the temporary polynomial
    ring used for both the resultant and its subresultant recovery sequence.
    """
    linear = Poly.from_dict(
        {
            (1,): -coefficient_domain.one,
            (0,): coefficient_domain.convert(primitive_value),
        },
        eliminated_generator,
        domain=coefficient_domain,
    )
    scalar_value = coefficient_domain.convert(scalar)
    coefficients = polynomial.all_coeffs()
    result = Poly(
        coefficient_domain.convert(coefficients[0]),
        eliminated_generator,
        domain=coefficient_domain,
    )
    scalar_power = coefficient_domain.one
    for coefficient in coefficients[1:]:
        scalar_power *= scalar_value
        result = result * linear + Poly(
            coefficient_domain.convert(coefficient) * scalar_power,
            eliminated_generator,
            domain=coefficient_domain,
        )
    return result


def _trager_norm(
    left: AlgebraicExtension,
    right: AlgebraicExtension,
    generator: Symbol,
    scalar: FieldElement,
) -> _TragerNormData:
    """Compute a pairwise norm and recovery PRS in one resultant pass."""
    parent = left.parent_field
    coefficient_ring = parent.poly_ring(generator)
    left_polynomial = Poly.from_dict(
        {
            monomial: coefficient_ring.convert(coefficient)
            for monomial, coefficient in left.minimal_polynomial.terms()
        },
        left.generator,
        domain=coefficient_ring,
    )
    shifted_right = _scaled_shifted_relation(
        right.minimal_polynomial,
        left.generator,
        coefficient_ring.gens[0],
        scalar,
        coefficient_ring,
    )
    # Poly.resultant converts a constant-in-a result through SymPy.  The DMP
    # operation returns the native K[z] element directly, which matters for
    # rational-function and later effective tower coefficient fields.
    try:
        native_resultant, subresultant_reps = left_polynomial.rep.resultant(
            shifted_right.rep,
            includePRS=True,
        )
    except (DomainError, NotImplementedError, OperationNotSupported) as exc:
        raise NotImplementedError(
            "Trager compositum normalization needs exact resultants over the "
            "parent field"
        ) from exc
    ring_resultant = coefficient_ring.convert(native_resultant)
    modulus = Poly.from_dict(ring_resultant.to_dict(), generator, domain=parent).monic()
    subresultants = tuple(
        Poly.new(representation, left.generator) for representation in subresultant_reps
    )
    return _TragerNormData(modulus, subresultants)


def _factor_square_free_norm(modulus: Poly) -> tuple[Poly, ...]:
    """Factor a square-free Trager norm over its exact parent field."""
    try:
        _unit, factor_list = modulus.factor_list()
    except (DomainError, NotImplementedError, OperationNotSupported) as exc:
        raise NotImplementedError(
            "Trager compositum normalization needs univariate factorization "
            "over the parent field"
        ) from exc
    if any(multiplicity != 1 for _factor, multiplicity in factor_list):
        raise RuntimeError("a square-free Trager norm factored with multiplicity")
    factors = tuple(factor.monic() for factor, _multiplicity in factor_list)
    if sum(factor.degree() for factor in factors) != modulus.degree():
        raise RuntimeError("Trager factorization did not preserve the norm degree")
    return factors


def _source_images_in_factor(
    target: AlgebraicExtension,
    scalar: FieldElement,
    subresultants: tuple[Poly, ...],
) -> tuple[ExtensionElement, ExtensionElement]:
    r"""Recover ``a`` and ``b`` from one factor by a linear subresultant.

    SymPy's ``FiniteExtension`` is not yet a fully composable field domain:
    polynomial gcd over it can invoke quotient-ring exact division. Trager's
    factor-recovery theorem needs no nested-domain gcd. Reduce the
    already-computed subresultant sequence modulo the selected factor and use
    its last nonzero (linear) member instead.
    """
    parent = target.parent_field
    factor = target.minimal_polynomial
    constant: Poly | None = None
    leading: Poly | None = None
    for subresultant in reversed(subresultants):
        reduced_coefficients = {
            power: Poly.from_dict(
                subresultant.rep.nth(power).to_dict(),
                target.generator,
                domain=parent,
            ).rem(factor)
            for power in range(subresultant.degree() + 1)
        }
        nonzero_powers = tuple(
            power
            for power, coefficient in reduced_coefficients.items()
            if not coefficient.is_zero
        )
        if not nonzero_powers:
            continue
        if max(nonzero_powers) != 1:
            raise RuntimeError(
                "a square-free Trager norm did not recover a linear source relation"
            )
        constant = reduced_coefficients[0]
        leading = reduced_coefficients[1]
        break
    if constant is None or leading is None:
        raise RuntimeError(
            "a Trager factor annihilated the complete subresultant sequence"
        )

    left_polynomial = (-constant * leading.invert(factor)).rem(factor)
    left_image = target.from_power_basis_coordinates(
        tuple(left_polynomial.nth(power) for power in range(target.field.rank))
    )
    inverse_scalar = parent.quo(parent.one, scalar)
    right_image = cast(
        ExtensionElement,
        (target.field.generator - left_image) * _embed_parent(target, inverse_scalar),
    )
    return left_image, right_image


def _pairwise_components(
    left: AlgebraicExtension,
    right: AlgebraicExtension,
    generator: Symbol,
) -> tuple[
    tuple[FieldElement, AlgebraicExtension, ExtensionElement, ExtensionElement], ...
]:
    """Return every field component of ``left tensor right``.

    For tensor rank ``n`` there are at most ``binomial(n, 2)`` bad scalars:
    each equates one pair of geometric points.  The final deterministic trial
    makes the characteristic-zero search complete.  Crucially, reducibility
    is not failure: the first square-free norm is factored into all components.
    """
    parent = left.parent_field
    tensor_rank = left.field.rank * right.field.rank
    bad_scalar_bound = tensor_rank * (tensor_rank - 1) // 2
    for integer_scalar in range(1, bad_scalar_bound + 2):
        scalar = cast(FieldElement, parent.convert(integer_scalar))
        norm_data = _trager_norm(left, right, generator, scalar)
        modulus = norm_data.modulus
        if modulus.degree() != tensor_rank:
            raise RuntimeError("the Trager norm lost tensor-product degree")
        if modulus.gcd(modulus.diff()).degree() != 0:
            continue
        components = []
        for factor in _factor_square_free_norm(modulus):
            target = AlgebraicExtension(factor)
            left_image, right_image = _source_images_in_factor(
                target,
                scalar,
                norm_data.subresultants,
            )
            components.append((scalar, target, left_image, right_image))
        return tuple(components)
    raise RuntimeError("the complete Trager separator bound was exhausted")


def _transport_element(
    source: AlgebraicExtension,
    target: AlgebraicExtension,
    source_generator_image: ExtensionElement,
    value: ExtensionElement,
) -> ExtensionElement:
    """Apply a checked source-generator map to one extension element."""
    result = target.field.zero
    for coefficient in reversed(source.power_basis_coordinates(value)):
        result = result * source_generator_image + _embed_parent(target, coefficient)
    return cast(ExtensionElement, result)


def trager_compositum(
    sources: tuple[AlgebraicExtension, ...],
    generator: Symbol,
) -> EtaleCompositumPresentation:
    r"""Decompose the finite etale tensor product of ``sources``.

    Sources are merged pairwise.  This keeps each resultant at the degree of
    the current field component times the next source degree, rather than
    constructing one dense matrix at the product of every source degree.
    Multiple components are retained because choosing one is source/branch
    semantics, not a field-arithmetic decision.
    """
    sources = tuple(sources)
    parent = _validate_sources(sources, generator)
    first = sources[0]
    partials = (
        _PartialComponent(
            first,
            (cast(FieldElement, parent.one),),
            (cast(ExtensionElement, first.field.generator),),
        ),
    )
    for source_index, source in enumerate(sources[1:], start=1):
        is_last = source_index == len(sources) - 1
        merge_generator = generator if is_last else Dummy("algebraic_composite")
        next_partials: list[_PartialComponent] = []
        for partial in partials:
            for (
                scalar,
                target,
                old_generator_image,
                new_source_image,
            ) in _pairwise_components(partial.target, source, merge_generator):
                transported_images = tuple(
                    _transport_element(
                        partial.target,
                        target,
                        old_generator_image,
                        image,
                    )
                    for image in partial.source_images
                )
                next_partials.append(
                    _PartialComponent(
                        target,
                        (*partial.coefficients, scalar),
                        (*transported_images, new_source_image),
                    )
                )
        partials = tuple(next_partials)

    components = tuple(
        PrimitiveElementPresentation(
            sources,
            partial.coefficients,
            partial.target,
            partial.source_images,
        )
        for partial in partials
    )
    return EtaleCompositumPresentation(sources, components)


def normalize_primitive_element(
    sources: tuple[AlgebraicExtension, ...],
    generator: Symbol,
) -> PrimitiveElementPresentation:
    r"""Return the unique Trager compositum field or report exact ambiguity.

    Call :func:`trager_compositum` when all field components are useful.  This
    compatibility-shaped convenience boundary accepts only a tensor product
    that is itself a field; its exception carries the complete decomposition.
    """
    decomposition = trager_compositum(sources, generator)
    if not decomposition.is_field:
        raise AmbiguousCompositumError(decomposition)
    return decomposition.components[0]


__all__ = ["normalize_primitive_element", "trager_compositum"]
