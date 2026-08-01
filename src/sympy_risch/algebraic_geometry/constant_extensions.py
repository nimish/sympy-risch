r"""Exact embeddings between supported algebraic constant fields.

This module contains no algebraic-function-field or Risch policy.  It builds
the compositum of ``QQ`` or a SymPy algebraic number field with splitting root
fields of polynomials over the displayed source field, and records the exact
embedding of the old constants into that compositum.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from sympy.core.numbers import AlgebraicNumber
from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.numberfields import field_isomorphism
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly, resultant


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


def _normalized_requested_factor(source: Domain, factor: Poly) -> Poly:
    """Return a monic square-free factor over the displayed source field."""
    _field_degree(source)
    if not isinstance(factor, Poly):
        raise TypeError("an algebraic constant extension needs a SymPy Poly")
    if len(factor.gens) != 1:
        raise ValueError("the adjoining polynomial must be univariate")
    try:
        normalized = Poly.from_dict(
            {
                monomial: source.convert(coefficient)
                for monomial, coefficient in factor.rep.to_dict().items()
            },
            factor.gens[0],
            domain=source,
        ).monic()
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "the adjoining polynomial coefficients do not belong to the source field"
        ) from exc
    if normalized.degree() < 2:
        raise ValueError("the adjoining polynomial must be non-linear")
    if normalized.sqf_part() != normalized:
        raise ValueError("the adjoining polynomial must be square-free")
    return normalized


def _normalized_irreducible_factor(source: Domain, factor: Poly) -> Poly:
    """Return an admitted irreducible one-root extension polynomial."""
    normalized = _normalized_requested_factor(source, factor)
    if normalized.is_irreducible is not True:
        raise ValueError(
            "a one-root constant extension needs a polynomial irreducible "
            "over its source field"
        )
    return normalized


def _rational_descent(factor: Poly) -> Poly | None:
    """Return the same factor over ``QQ`` when every coefficient descends."""
    try:
        descended = Poly.from_dict(
            {
                monomial: QQ.convert(coefficient)
                for monomial, coefficient in factor.rep.to_dict().items()
            },
            factor.gens[0],
            domain=QQ,
        ).monic()
    except (BasePolynomialError, NotImplementedError, TypeError, ValueError):
        return None
    if descended.is_irreducible is not True:
        return None
    return descended


def _splits_completely(factor: Poly) -> bool:
    """Whether a square-free factor splits linearly over its own domain."""
    _unit, factors = factor.factor_list()
    return len(factors) == factor.degree() and all(
        component.degree() == 1 and multiplicity == 1
        for component, multiplicity in factors
    )


def _source_element_expression(
    source: AlgebraicField,
    value: object,
    generator: Dummy,
) -> object:
    """Represent a native source constant as a rational polynomial."""
    coefficients = source.convert(value).to_list()
    degree = len(coefficients) - 1
    return sum(
        QQ.to_sympy(coefficient) * generator ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )


def _relative_root_field(
    source: AlgebraicField,
    factor: Poly,
) -> tuple[AlgebraicField, object]:
    r"""Return an absolute field containing ``source`` and one factor root.

    For a source generator ``alpha`` and a root ``beta`` of ``factor``, the
    primitive-element theorem guarantees that ``theta = alpha + c*beta`` is
    primitive for all but finitely many positive integers ``c``.  A resultant
    supplies the absolute polynomial of ``theta``.  A linear gcd over the
    resulting field then recovers the exact image of ``alpha``.
    """
    alpha = Dummy("source_generator")
    theta = Dummy("absolute_generator")
    factor_generator = factor.gens[0]
    factor_expression = sum(
        _source_element_expression(source, coefficient, alpha)
        * factor_generator ** monomial[0]
        for monomial, coefficient in factor.rep.to_dict().items()
    )
    source_minpoly = source.ext.minpoly
    source_relation = source_minpoly.as_expr().subs(
        source_minpoly.gens[0],
        alpha,
    )
    expected_degree = source_minpoly.degree() * factor.degree()

    for primitive_coefficient in count(1):
        relative_relation = Poly(
            primitive_coefficient ** factor.degree()
            * factor_expression.subs(
                factor_generator,
                (theta - alpha) / primitive_coefficient,
            ),
            alpha,
            theta,
            domain=QQ,
        )
        absolute_relation = Poly(
            resultant(source_relation, relative_relation.as_expr(), alpha),
            theta,
            domain=QQ,
        ).monic()
        if (
            absolute_relation.degree() != expected_degree
            or absolute_relation.is_irreducible is not True
        ):
            continue

        target = QQ.alg_field_from_poly(absolute_relation)
        if not isinstance(target, AlgebraicField):
            raise TypeError("SymPy did not construct an absolute algebraic field")

        relation_by_alpha: dict[tuple[int], object] = {}
        for (
            alpha_degree,
            theta_degree,
        ), coefficient in relative_relation.rep.to_dict().items():
            monomial = (alpha_degree,)
            relation_by_alpha[monomial] = (
                relation_by_alpha.get(monomial, target.zero)
                + target.convert(coefficient) * target.unit**theta_degree
            )
        relative_over_target = Poly.from_dict(
            relation_by_alpha,
            alpha,
            domain=target,
        )
        source_over_target = Poly(source_relation, alpha, domain=target)
        common = source_over_target.gcd(relative_over_target).monic()
        if common.degree() != 1:
            continue
        leading, constant = common.rep.to_list()
        source_image = target.quo(-constant, leading)
        return target, source_image

    raise RuntimeError("the primitive-element search did not terminate")


@dataclass(frozen=True, slots=True)
class AlgebraicConstantFieldExtension:
    r"""A certified embedding ``source -> target`` of algebraic constants.

    ``factor`` is irreducible over ``source`` and its one-root field contains
    all of its roots. ``source_generator_image`` is the exact image of the
    primitive generator when ``source`` is already algebraic; it is absent
    for ``QQ``.
    """

    source: Domain
    factor: Poly
    target: AlgebraicField
    source_generator_image: object | None

    def __post_init__(self) -> None:
        _field_degree(self.source)
        if _normalized_irreducible_factor(self.source, self.factor) != self.factor:
            raise ValueError(
                "the adjoining polynomial must be normalized over its source field"
            )
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

        if not _splits_completely(self.embed_polynomial(self.factor)):
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

    def embed_polynomial(self, factor: Poly) -> Poly:
        """Map a polynomial over ``source`` coefficientwise into ``target``."""
        if not isinstance(factor, Poly) or factor.domain != self.source:
            raise ValueError("the polynomial does not belong to the extension source")
        return Poly.from_dict(
            {
                monomial: self.embed(coefficient)
                for monomial, coefficient in factor.rep.to_dict().items()
            },
            factor.gens[0],
            domain=self.target,
        )


def extend_algebraic_constants(
    source: Domain,
    factor: Poly,
) -> AlgebraicConstantFieldExtension:
    r"""Adjoin the splitting root field of ``factor`` to ``source``.

    The factor is irreducible over ``source`` and must split in its one-root
    field. Rational factors use SymPy's native root-field constructor.
    Algebraic-coefficient factors use a resultant primitive element and retain
    the exact image of the old generator.
    """
    _field_degree(source)
    normalized = _normalized_irreducible_factor(source, factor)
    rational_factor = _rational_descent(normalized)
    if rational_factor is not None:
        root_field = QQ.alg_field_from_poly(rational_factor)
        if not isinstance(root_field, AlgebraicField):
            raise TypeError("SymPy did not construct an algebraic root field")
        result = _extend_algebraic_constants_with_root_field(
            source,
            normalized,
            root_field,
        )
    elif isinstance(source, AlgebraicField):
        target, source_image = _relative_root_field(source, normalized)
        result = AlgebraicConstantFieldExtension(
            source,
            normalized,
            target,
            source_image,
        )
    else:
        raise NotImplementedError(
            "constant composita currently require QQ or an algebraic number field"
        )
    return result


def _extend_algebraic_constants_with_root_field(
    source: Domain,
    factor: Poly,
    root_field: AlgebraicField,
) -> AlgebraicConstantFieldExtension:
    """Join a precomputed root field without rebuilding it."""
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
        factor,
        target,
        source_image,
    )


@dataclass(frozen=True, slots=True)
class AlgebraicConstantFieldCompositum:
    r"""A certified common field for several exact splitting factors.

    ``steps`` is an exact tower of primitive-element embeddings.  The final
    target is therefore one number field even though it may have been built
    from several independent residue fields.  A requested factor already
    split over ``source`` or an earlier step remains in ``factors`` but does
    not create a redundant extension.

    The empty chain is the identity compositum.  This is useful to number-
    field clients, although a function-field base change will reject it
    because there is then no field change to perform.
    """

    source: Domain
    factors: tuple[Poly, ...]
    steps: tuple[AlgebraicConstantFieldExtension, ...]

    def __post_init__(self) -> None:
        _field_degree(self.source)
        normalized = tuple(
            _normalized_requested_factor(self.source, factor) for factor in self.factors
        )
        if normalized != self.factors:
            raise ValueError("compositum factors must be normalized over the source")
        if len(set(normalized)) != len(normalized):
            raise ValueError("compositum factors must be distinct")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, AlgebraicConstantFieldExtension) for step in self.steps
        ):
            raise TypeError("compositum steps must be exact constant extensions")

        current = self.source
        consumed = 0
        prefix: list[AlgebraicConstantFieldExtension] = []
        for factor in normalized:
            transported = factor
            for step in prefix:
                transported = step.embed_polynomial(transported)
            while not _splits_completely(transported):
                if consumed >= len(self.steps):
                    raise ValueError("the compositum does not split every factor")
                step = self.steps[consumed]
                if step.source is not current:
                    raise ValueError("constant-compositum steps do not form a chain")
                if not transported.rem(step.factor).is_zero:
                    raise ValueError(
                        "a compositum step does not adjoin a requested factor root"
                    )
                transported = step.embed_polynomial(transported)
                current = step.target
                prefix.append(step)
                consumed += 1
            if transported.domain != current:
                raise ValueError(
                    "a compositum factor was not transported to its target"
                )
        if consumed != len(self.steps):
            raise ValueError("a compositum contains an unrequested extension step")
        if any(
            not _splits_completely(self.embed_polynomial(factor))
            for factor in normalized
        ):
            raise ValueError("the compositum does not split every requested factor")

    @property
    def target(self) -> Domain:
        """The final common constant field."""
        if self.steps:
            return self.steps[-1].target
        return self.source

    @property
    def is_identity(self) -> bool:
        """Whether every requested factor already split over ``source``."""
        return not self.steps

    def embed(self, value: object) -> object:
        """Map a native source constant through the complete exact chain."""
        embedded = self.source.convert(value)
        for step in self.steps:
            embedded = step.embed(embedded)
        return embedded

    def embed_polynomial(self, factor: Poly) -> Poly:
        """Map a source polynomial through the complete exact chain."""
        embedded = _normalized_requested_factor(self.source, factor)
        for step in self.steps:
            embedded = step.embed_polynomial(embedded)
        return embedded


def extend_algebraic_constant_compositum(
    source: Domain,
    factors: tuple[Poly, ...],
) -> AlgebraicConstantFieldCompositum:
    r"""Construct one exact field in which every requested factor splits.

    Each irreducible component over the original source must already split in
    its own one-root field. This keeps general splitting-field construction an
    explicit boundary independent of caller order. Components are then joined
    in caller order, and the final certificate rechecks every requested factor.
    """
    _field_degree(source)
    if not isinstance(factors, tuple):
        raise TypeError("constant-compositum factors must be a tuple")
    normalized = tuple(
        _normalized_requested_factor(source, factor) for factor in factors
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("constant-compositum factors must be distinct")

    prechecked: dict[Poly, AlgebraicConstantFieldExtension] = {}
    for factor in normalized:
        _unit, components = factor.factor_list()
        for component, multiplicity in components:
            if multiplicity != 1:
                raise ValueError("constant-compositum factors must be square-free")
            if component.degree() == 1:
                continue
            try:
                prechecked[component] = extend_algebraic_constants(source, component)
            except ValueError as exc:
                raise ValueError(
                    "general splitting fields are unsupported: a requested factor "
                    "does not split in its one-root field"
                ) from exc

    current = source
    steps: list[AlgebraicConstantFieldExtension] = []
    for factor in normalized:
        transported = factor
        for step in steps:
            transported = step.embed_polynomial(transported)
        while not _splits_completely(transported):
            _unit, components = transported.factor_list()
            component = next(
                candidate
                for candidate, _multiplicity in components
                if candidate.degree() > 1
            )
            if current is source and component in prechecked:
                step = prechecked[component]
            else:
                step = extend_algebraic_constants(current, component)
            steps.append(step)
            current = step.target
            transported = step.embed_polynomial(transported)
    return AlgebraicConstantFieldCompositum(source, normalized, tuple(steps))
