r"""Hess-style Riemann--Roch spaces for algebraic function fields.

For a divisor ``D`` of ``L/C(x)``, Hess represents the finite and infinite
conditions separately by inverted ideals in the two maximal orders
``O`` and ``O_infinity``.  The Riemann--Roch space is their intersection.
Rather than expanding every candidate at every place, the intersection is
computed from the rational transition matrix between their bases and a weak
Popov reduction over ``C[x]``.

This module is deliberately a function-field service: it knows about
divisors, orders, ideals, and reciprocal charts, but nothing about Risch
derivations or elementary integration.  It implements Algorithm 13 and the
reusable parametric representation of Hess's Theorem 7 in
``Computing Riemann--Roch Spaces in Algebraic Function Fields and Related
Topics``, JSC 33 (2002).  Algorithms 21 and 24 remain separate future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index
from typing import SupportsIndex, cast

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

        leading = constants.convert(rows[left][column].LC())
        other_leading = constants.convert(rows[right][column].LC())
        shift = left_degree - right_degree
        coefficient = -constants.exquo(leading, other_leading)
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


def _shift_index(shift: SupportsIndex) -> int:
    """Return an exact integer shift, rejecting booleans and inexact values."""
    if isinstance(shift, bool):
        raise TypeError("a Hess parameter shift must be an integer")
    try:
        return int(integer_index(shift))
    except TypeError as exc:
        raise TypeError("a Hess parameter shift must be an integer") from exc


@dataclass(frozen=True, slots=True, eq=False)
class HessParametricBasis:
    r"""The short reusable form of a family of Riemann--Roch spaces.

    If ``v_i`` are :attr:`generators` and ``d_i`` are
    :attr:`invariant_degrees`, Hess's Theorem 7 gives

    ``L(D + r*(x)_infinity) = <x**j*v_i : 0 <= j <= d_i + r>_C``.

    The weak-Popov rows and their common denominator are retained as the exact
    lattice certificate.  Expanding a shifted space performs only field
    multiplications; it does not repeat ideal construction, Popov reduction,
    or principal-divisor computations.
    """

    representation: HessDivisorIdeals
    parameter: ExtensionElement
    parameter_pole_divisor: AlgebraicDivisor
    generators: tuple[ExtensionElement, ...]
    invariant_degrees: tuple[int, ...]
    denominator: Poly
    reduced_rows: tuple[tuple[Poly, ...], ...]
    pivots: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.representation, HessDivisorIdeals):
            raise TypeError("a parametric Hess basis needs its ideal representation")
        model = self.representation.model
        extension = model.extension
        parent, constants, variable = _base_polynomial_ring(extension)
        if not isinstance(parent, FractionField):
            raise TypeError("a parametric Hess basis needs a rational-function base")

        parameter = cast(ExtensionElement, extension.field.convert(self.parameter))
        expected_parameter = _embed_parent(
            extension,
            Poly(variable, variable, domain=constants),
        )
        if parameter != expected_parameter:
            raise ValueError("the Hess family parameter must be the base variable")
        object.__setattr__(self, "parameter", parameter)

        pole_divisor = self.parameter_pole_divisor
        if not isinstance(pole_divisor, AlgebraicDivisor):
            raise TypeError("a Hess family needs the parameter pole divisor")
        if pole_divisor.model is not model:
            raise ValueError("the parameter pole divisor belongs to another model")
        expected_poles = AlgebraicDivisor(
            model,
            tuple(
                (place, place.ramification_index) for place in model.infinite_places()
            ),
        )
        if pole_divisor != expected_poles:
            raise ValueError("the parameter pole divisor is not (x)_infinity")

        rank = extension.field.rank
        generators = tuple(
            cast(ExtensionElement, extension.field.convert(generator))
            for generator in self.generators
        )
        degrees = tuple(self.invariant_degrees)
        rows = tuple(tuple(row) for row in self.reduced_rows)
        pivots = tuple(self.pivots)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "invariant_degrees", degrees)
        object.__setattr__(self, "reduced_rows", rows)
        object.__setattr__(self, "pivots", pivots)

        if not (len(generators) == len(degrees) == len(rows) == len(pivots) == rank):
            raise ValueError("parametric Hess data must have one row per field rank")
        if any(
            isinstance(degree, bool) or not isinstance(degree, int)
            for degree in degrees
        ):
            raise TypeError("Hess invariant degrees must be integers")
        if any(len(row) != rank for row in rows):
            raise ValueError("the reduced Hess matrix must be square")
        if not isinstance(self.denominator, Poly):
            raise TypeError("a parametric Hess basis needs a polynomial denominator")
        if (
            self.denominator.gens != (variable,)
            or self.denominator.domain != constants
            or self.denominator.is_zero
        ):
            raise ValueError(
                "the Hess denominator belongs to the wrong polynomial ring"
            )

        denominator_degree = self.denominator.degree()
        for row, pivot, invariant_degree in zip(
            rows,
            pivots,
            degrees,
            strict=True,
        ):
            if any(
                entry.gens != (variable,) or entry.domain != constants for entry in row
            ):
                raise ValueError("a reduced Hess row belongs to the wrong ring")
            if _pivot(list(row)) != pivot:
                raise ValueError("stored Hess pivot data does not match its row")
            if invariant_degree != denominator_degree - pivot[1]:
                raise ValueError("a Hess invariant degree does not match its pivot")
        if len({column for column, _degree in pivots}) != rank:
            raise ValueError("reduced Hess rows need distinct leading positions")

    @property
    def model(self) -> AlgebraicFunctionField:
        """Return the underlying algebraic function field."""
        return self.representation.model

    @property
    def divisor(self) -> AlgebraicDivisor:
        """Return the base divisor ``D`` of the parametric family."""
        return self.representation.divisor

    def shifted_divisor(self, shift: SupportsIndex = 0) -> AlgebraicDivisor:
        r"""Return ``D + shift*(x)_infinity``."""
        exponent = _shift_index(shift)
        return self.divisor + self.parameter_pole_divisor.scale(exponent)

    def dimension(self, shift: SupportsIndex = 0) -> int:
        """Return the constant-field dimension without expanding a basis."""
        exponent = _shift_index(shift)
        return sum(
            max(invariant_degree + exponent + 1, 0)
            for invariant_degree in self.invariant_degrees
        )

    def expand(self, shift: SupportsIndex = 0) -> tuple[ExtensionElement, ...]:
        r"""Expand a concrete basis of ``L(D + shift*(x)_infinity)``."""
        exponent = _shift_index(shift)
        maximum_power = max(
            (
                invariant_degree + exponent
                for invariant_degree in self.invariant_degrees
            ),
            default=-1,
        )
        if maximum_power < 0:
            return ()
        powers = [self.model.extension.field.one]
        for _power in range(maximum_power):
            powers.append(powers[-1] * self.parameter)
        return tuple(
            powers[power] * generator
            for generator, invariant_degree in zip(
                self.generators,
                self.invariant_degrees,
                strict=True,
            )
            for power in range(invariant_degree + exponent + 1)
        )


def hess_parametric_basis(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> HessParametricBasis:
    r"""Compute Hess's reusable short basis for ``D + r*(x)_infinity``.

    The input is represented by inverted ideals at finite places and in the
    reciprocal chart.  Weak Popov reduction of their transition matrix yields
    generators ``v_i`` and invariant degrees ``d_i``.  The concrete basis for
    any integer shift ``r`` is expanded only on request.
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
        infinite_inverse = infinite_matrix.inv()
    except DMNonInvertibleMatrixError as exc:
        raise ValueError(
            "the reciprocal ideal has no full-rank coordinate basis"
        ) from exc
    transition = finite_matrix * infinite_inverse

    entries = tuple(
        cast(FracElement, entry) for row in transition.to_list() for entry in row
    )
    denominator = _common_denominator(entries, constants, variable)
    polynomial_rows = _polynomial_rows(transition, denominator, model)
    reduced_rows, reduced_generators, pivots = _weak_popov_reduce(
        polynomial_rows,
        finite_generators,
        model,
    )
    if len({column for column, _degree in pivots}) != extension.field.rank:
        raise ValueError("Hess reduction did not produce distinct leading positions")

    reduced_generator_matrix = _row_matrix(
        tuple(model.order.coordinates(element) for element in reduced_generators),
        parent,
    )
    reconstructed_rows = _polynomial_rows(
        reduced_generator_matrix * infinite_inverse,
        denominator,
        model,
    )
    if reconstructed_rows != reduced_rows:
        raise ValueError("reduced Hess generators do not match the lattice rows")

    parameter = _embed_parent(
        extension,
        Poly(variable, variable, domain=constants),
    )
    denominator_degree = denominator.degree()
    invariant_degrees = tuple(denominator_degree - degree for _column, degree in pivots)
    parameter_poles = AlgebraicDivisor(
        model,
        tuple((place, place.ramification_index) for place in model.infinite_places()),
    )
    return HessParametricBasis(
        representation,
        parameter,
        parameter_poles,
        tuple(reduced_generators),
        invariant_degrees,
        denominator,
        tuple(tuple(row) for row in reduced_rows),
        tuple(pivots),
    )


def hess_riemann_roch_basis(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
) -> tuple[ExtensionElement, ...]:
    r"""Expand and independently check the Hess basis of ``L(D)``.

    New callers that need several infinity shifts should construct one
    :class:`HessParametricBasis` with :func:`hess_parametric_basis` instead.
    """
    parametric = hess_parametric_basis(model, divisor)
    result = parametric.expand()
    representation = parametric.representation
    if not all(representation.contains(element) for element in result):
        raise ValueError("Hess reduction produced an element outside L(D)")
    return result


__all__ = [
    "HessDivisorIdeals",
    "HessParametricBasis",
    "hess_divisor_ideals",
    "hess_parametric_basis",
    "hess_riemann_roch_basis",
]
