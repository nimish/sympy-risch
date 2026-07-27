r"""Exact embeddings between supported algebraic constant fields.

This module contains no algebraic-function-field or Risch policy.  It builds
the compositum of ``QQ`` (or a SymPy algebraic number field over ``QQ``) with
the root field of an irreducible polynomial over ``QQ`` and records the exact
embedding of the old constants into that compositum.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.numbers import AlgebraicNumber
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.numberfields import field_isomorphism
from sympy.polys.polytools import Poly


def _field_degree(field: Domain) -> int:
    """Return ``[field:QQ]`` for the supported exact constant domains."""
    if field == QQ:
        return 1
    if isinstance(field, AlgebraicField):
        return field.ext.minpoly.degree()
    raise NotImplementedError(
        "constant composita currently require QQ or an algebraic number field"
    )


def _evaluate_in_target(
    value: object,
    source: AlgebraicField,
    target: AlgebraicField,
    generator_image: object,
) -> object:
    """Evaluate a native source ANP at its exact image in ``target``."""
    source_value = source.convert(value)
    result = target.zero
    for coefficient in source_value.to_list():
        result = result * generator_image + target.convert(coefficient)
    return result


@dataclass(frozen=True, slots=True)
class AlgebraicConstantFieldExtension:
    r"""A certified embedding ``source -> target`` of algebraic constants.

    ``factor`` is the rational irreducible polynomial whose root field was
    joined to ``source``.  ``source_generator_image`` is the exact image of
    the primitive generator when ``source`` is already algebraic; it is
    absent for ``QQ``.
    """

    source: Domain
    factor: Poly
    target: AlgebraicField
    source_generator_image: object | None

    def __post_init__(self) -> None:
        _field_degree(self.source)
        if not isinstance(self.factor, Poly):
            raise TypeError("an algebraic constant extension needs a SymPy Poly")
        if len(self.factor.gens) != 1 or self.factor.domain != QQ:
            raise ValueError("the adjoining polynomial must be univariate over QQ")
        if not self.factor.is_monic or self.factor.degree() < 2:
            raise ValueError("the adjoining polynomial must be monic and non-linear")
        if self.factor.is_irreducible is not True:
            raise ValueError("the adjoining polynomial must be irreducible over QQ")
        if not isinstance(self.target, AlgebraicField):
            raise TypeError("the target constants must be an algebraic number field")
        if _field_degree(self.target) <= _field_degree(self.source):
            raise ValueError("the target must strictly enlarge the source constants")

        if self.source == QQ:
            if self.source_generator_image is not None:
                raise ValueError("the rational field has no primitive generator")
        else:
            if not isinstance(self.source, AlgebraicField):
                raise NotImplementedError(
                    "constant composita currently require an algebraic number field"
                )
            if self.source_generator_image is None:
                raise ValueError(
                    "an algebraic source needs its primitive-generator image"
                )
            source_minpoly = self.source.ext.minpoly
            image = self.target.convert(self.source_generator_image)
            value = self.target.zero
            for coefficient in source_minpoly.all_coeffs():
                value = value * image + self.target.convert(coefficient)
            if value != self.target.zero:
                raise ValueError(
                    "the proposed source-generator image violates its minimal polynomial"
                )
            object.__setattr__(self, "source_generator_image", image)

        target_factor = Poly(
            self.factor.as_expr(),
            self.factor.gens[0],
            domain=self.target,
        )
        _unit, factors = target_factor.factor_list()
        if len(factors) != self.factor.degree() or any(
            component.degree() != 1 or multiplicity != 1
            for component, multiplicity in factors
        ):
            raise ValueError(
                "the adjoining factor does not split completely in the target field"
            )

    def embed(self, value: object) -> object:
        """Map a native source constant into the target field."""
        if self.source == QQ:
            return self.target.convert(QQ.convert(value))
        if not isinstance(self.source, AlgebraicField):
            raise NotImplementedError(
                "constant composita currently require an algebraic number field"
            )
        image = self.source_generator_image
        if image is None:
            raise ValueError("the algebraic source embedding has no generator image")
        return _evaluate_in_target(value, self.source, self.target, image)


def extend_algebraic_constants(
    source: Domain,
    factor: Poly,
) -> AlgebraicConstantFieldExtension:
    r"""Adjoin the splitting root field of ``factor`` to ``source``.

    The present constructive cell accepts an irreducible polynomial over
    ``QQ``.  If ``source`` is already algebraic, SymPy constructs a primitive
    element for the compositum and ``field_isomorphism`` supplies the exact
    image of the old primitive generator.
    """
    _field_degree(source)
    if not isinstance(factor, Poly):
        raise TypeError("an algebraic constant extension needs a SymPy Poly")
    if len(factor.gens) != 1:
        raise ValueError("the adjoining polynomial must be univariate")
    normalized = Poly(factor.as_expr(), factor.gens[0], domain=QQ).monic()
    if normalized.degree() < 2 or normalized.is_irreducible is not True:
        raise ValueError("the adjoining polynomial must be irreducible over QQ")

    root_field = QQ.alg_field_from_poly(normalized)
    if not isinstance(root_field, AlgebraicField):
        raise TypeError("SymPy did not construct an algebraic root field")
    if source == QQ:
        target = root_field
        source_image = None
    elif isinstance(source, AlgebraicField):
        target = QQ.algebraic_field(*source.orig_ext, root_field.ext.root)
        coefficients = field_isomorphism(source.ext, target.ext)
        if coefficients is None:
            raise ValueError("the compositum lost the source constant field")
        source_image = target(AlgebraicNumber(target.ext, coefficients).native_coeffs())
    else:
        raise NotImplementedError(
            "constant composita currently require QQ or an algebraic number field"
        )

    return AlgebraicConstantFieldExtension(
        source,
        normalized,
        target,
        source_image,
    )
