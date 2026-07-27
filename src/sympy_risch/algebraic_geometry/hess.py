r"""Hess-style Riemann--Roch spaces for algebraic function fields.

For a divisor ``D`` of ``L/C(x)``, Hess represents the finite and infinite
conditions separately by inverted ideals in the two maximal orders
``O`` and ``O_infinity``.  The Riemann--Roch space is their intersection.
Rather than expanding every candidate at every place, the intersection is
computed from the rational transition matrix between their bases and a weak
Popov reduction over ``C[x]``.

This module is deliberately a function-field service: it knows about
divisors, orders, ideals, and reciprocal charts, but nothing about Risch
derivations or elementary integration.  It follows Algorithm 6.1 of Hess,
``Computing Riemann--Roch Spaces in Algebraic Function Fields and Related
Topics``, JSC 33 (2002), Sections 4 and 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from ..algebraic._field import (
    _base_polynomial_ring,
    _common_denominator,
    _embed_parent,
    _parent_element_polynomial,
    _polynomial_parent_element,
)
from .function_field import AlgebraicDivisor, AlgebraicFunctionField
from .infinity import InfiniteAlgebraicPlace
from .orders import AlgebraicFractionalIdeal


def _row_matrix(
    rows: tuple[tuple[FracElement, ...], ...],
    parent: FractionField,
) -> DomainMatrix:
    """Build a square coordinate matrix with the supplied row vectors."""
    dimension = len(rows)
    if dimension == 0 or any(len(row) != dimension for row in rows):
        raise ValueError("a Hess ideal basis must be a nonempty square matrix")
    return DomainMatrix(
        [list(row) for row in rows],
        (dimension, dimension),
        parent,
    )


def _polynomial_rows(
    matrix: DomainMatrix,
    denominator: Poly,
    model: AlgebraicFunctionField,
) -> list[list[Poly]]:
    """Clear ``denominator`` from a rational matrix exactly."""
    extension = model.extension
    parent, constants, variable = _base_polynomial_ring(extension)
    denominator_element = _polynomial_parent_element(denominator, parent)
    return [
        [
            _parent_element_polynomial(
                cast(FracElement, entry) * denominator_element,
                parent,
                constants,
                variable,
            )
            for entry in row
        ]
        for row in matrix.to_list()
    ]


def _pivot(row: list[Poly]) -> tuple[int, int] | None:
    """Return the rightmost entry of maximal polynomial degree in ``row``."""
    best_column = -1
    best_degree = -1
    for column, entry in enumerate(row):
        if entry.is_zero:
            continue
        degree = entry.degree()
        if degree >= best_degree:
            best_column = column
            best_degree = degree
    if best_column < 0:
        return None
    return best_column, best_degree


def _weak_popov_reduce(
    rows: list[list[Poly]],
    generators: list[ExtensionElement],
    model: AlgebraicFunctionField,
) -> tuple[list[list[Poly]], list[ExtensionElement], list[tuple[int, int]]]:
    r"""Reduce a square polynomial matrix to weak Popov form.

    Row operations are performed simultaneously on ``generators``.  The
    reduced row ``i`` therefore still denotes the same function-field element
    required by Hess's intersection construction.
    """
    extension = model.extension
    _parent, constants, variable = _base_polynomial_ring(extension)
    base_variable = _embed_parent(
        extension,
        Poly(variable, variable, domain=constants),
    )

    while True:
        pivots = tuple(_pivot(row) for row in rows)
        conflict: tuple[int, int, int] | None = None
        for left, left_pivot in enumerate(pivots):
            if left_pivot is None:
                raise ValueError("a Hess transition matrix has a zero row")
            for right in range(left):
                right_pivot = pivots[right]
                if right_pivot is not None and left_pivot[0] == right_pivot[0]:
                    conflict = (left, right, left_pivot[0])
                    break
            if conflict is not None:
                break
        if conflict is None:
            reduced_pivots = [pivot for pivot in pivots if pivot is not None]
            if len(reduced_pivots) != len(rows):
                raise ValueError("a Hess transition matrix has a zero row")
            return rows, generators, reduced_pivots

        left, right, column = conflict
        left_pivot = _pivot(rows[left])
        right_pivot = _pivot(rows[right])
        if left_pivot is None or right_pivot is None:
            raise ValueError("a Hess transition matrix has a zero row")
        left_degree = left_pivot[1]
        right_degree = right_pivot[1]
        if left_degree < right_degree:
            left, right = right, left
            left_degree, right_degree = right_degree, left_degree

        leading = rows[left][column].LC()
        other_leading = rows[right][column].LC()
        shift = left_degree - right_degree
        coefficient = -leading / other_leading
        multiplier = Poly.from_dict({(shift,): coefficient}, variable, domain=constants)
        rows[left] = [
            entry + multiplier * other
            for entry, other in zip(rows[left], rows[right], strict=True)
        ]
        scalar = _embed_parent(
            extension,
            Poly(coefficient, variable, domain=constants),
        )
        generators[left] += scalar * base_variable**shift * generators[right]


@dataclass(frozen=True, slots=True, eq=False)
class HessDivisorIdeals:
    r"""The finite/infinite inverted-ideal representation of one divisor.

    ``finite_ideal`` is a global representative of the finite conditions and
    ``infinite_ideal`` is one for the reciprocal-chart conditions. Their
    transition matrix is used in Hess reduction; ordinary ideal membership in
    the reciprocal chart is intentionally *not* an ``L(D)`` membership test,
    because its coefficients are localized away from the infinity primes.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    finite_ideal: AlgebraicFractionalIdeal
    infinite_ideal: AlgebraicFractionalIdeal

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a Hess representation needs an algebraic function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a Hess representation needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if self.finite_ideal.order is not self.model.order:
            raise ValueError("the finite ideal belongs to another maximal order")
        if self.infinite_ideal.order is not self.model.reciprocal_chart.closure.order:
            raise ValueError("the infinite ideal belongs to another reciprocal order")

    def contains(self, element: object) -> bool:
        """Whether ``element`` satisfies every global inequality in ``L(D)``."""
        try:
            value = self.model.extension.field.convert(element)
            if value == self.model.extension.field.zero:
                return True
            return (self.model.principal_divisor(value) + self.divisor).is_effective
        except (TypeError, ValueError):
            return False


def hess_divisor_ideals(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> HessDivisorIdeals:
    r"""Build Hess's inverted finite/infinite ideal representation of ``D``."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a Hess representation needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("a Hess representation needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")

    finite = model.order.unit_ideal()
    infinite = model.reciprocal_chart.closure.order.unit_ideal()
    for place, coefficient in divisor.terms:
        if isinstance(place, InfiniteAlgebraicPlace):
            infinite = infinite.multiply(
                place.finite_place.prime_ideal.power(-coefficient)
            )
        else:
            finite = finite.multiply(place.prime_ideal.power(-coefficient))
    return HessDivisorIdeals(model, divisor, finite, infinite)


def hess_riemann_roch_basis(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> tuple[ExtensionElement, ...]:
    r"""Compute the complete Riemann--Roch space ``L(D)`` by Hess reduction.

    The input is represented by inverted ideals at finite places and in the
    reciprocal chart.  Weak Popov reduction of their transition matrix yields
    a finite basis for their intersection, without local Laurent expansion.
    """
    representation = hess_divisor_ideals(model, divisor)
    extension = model.extension
    parent, constants, variable = _base_polynomial_ring(extension)
    if not isinstance(parent, FractionField):
        raise TypeError("Hess reduction needs a rational-function base field")

    finite_generators = list(representation.finite_ideal.basis_elements)
    infinite_generators = tuple(
        model.reciprocal_chart.backward(element)
        for element in representation.infinite_ideal.basis_elements
    )
    finite_matrix = _row_matrix(
        tuple(model.order.coordinates(element) for element in finite_generators),
        parent,
    )
    infinite_matrix = _row_matrix(
        tuple(model.order.coordinates(element) for element in infinite_generators),
        parent,
    )
    try:
        transition = finite_matrix * infinite_matrix.inv()
    except DMNonInvertibleMatrixError as exc:
        raise ValueError(
            "the reciprocal ideal has no full-rank coordinate basis"
        ) from exc

    entries = tuple(
        cast(FracElement, entry) for row in transition.to_list() for entry in row
    )
    denominator = _common_denominator(entries, constants, variable)
    polynomial_rows = _polynomial_rows(transition, denominator, model)
    _reduced_rows, reduced_generators, pivots = _weak_popov_reduce(
        polynomial_rows,
        finite_generators,
        model,
    )
    if len({column for column, _degree in pivots}) != extension.field.rank:
        raise ValueError("Hess reduction did not produce distinct leading positions")

    base_variable = _embed_parent(
        extension,
        Poly(variable, variable, domain=constants),
    )
    basis: list[ExtensionElement] = []
    denominator_degree = denominator.degree()
    for generator, (_column, degree) in zip(
        reduced_generators,
        pivots,
        strict=True,
    ):
        basis.extend(
            base_variable**power * generator
            for power in range(denominator_degree - degree + 1)
        )
    result = tuple(basis)
    if not all(representation.contains(element) for element in result):
        raise ValueError("Hess reduction produced an element outside L(D)")
    return result


__all__ = [
    "HessDivisorIdeals",
    "hess_divisor_ideals",
    "hess_riemann_roch_basis",
]
