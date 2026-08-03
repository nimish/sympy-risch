r"""Exact constant-linear algebra used by recursive parametric RDEs.

Every implemented transcendental tower is represented by a flat SymPy
fraction field ``C(t_0, ..., t_n)`` while the tower object retains the
recursive differential structure.  Consequently, constant-linear relations
among finitely many field elements are exactly the coefficient relations
obtained after clearing their common denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..tower import _coerce


@dataclass(frozen=True, slots=True)
class ParametricRDESolution:
    """One vector ``(parameters, solution)`` over a constant field."""

    parameters: tuple[FieldElement, ...]
    solution: FieldElement


def _fraction_elements(
    elements: Sequence[object],
    field: Domain,
) -> tuple[FracElement, ...]:
    """Coerce elements into the flat transcendental fraction field."""
    if not isinstance(field, FractionField):
        raise TypeError("parametric tower linear algebra needs a fraction field")
    converted = tuple(
        cast(FracElement, _coerce(field, element)) for element in elements
    )
    if any(not isinstance(element, FracElement) for element in converted):
        raise TypeError("a fraction-field conversion did not return a FracElement")
    return converted


def _cleared_solution_coordinates(
    elements: Sequence[object],
    field: Domain,
    constants: Domain,
) -> tuple[
    tuple[dict[tuple[int, ...], FieldElement], ...],
    tuple[Symbol, ...],
    Expr,
]:
    """Clear tower denominators and expose coefficient dictionaries over ``C``."""
    if not isinstance(field, FractionField):
        raise TypeError("parametric tower linear algebra needs a fraction field")
    converted = _fraction_elements(elements, field)
    if isinstance(constants, FractionField):
        constant_symbols = set(constants.symbols or ())
        generators = tuple(
            symbol for symbol in (field.symbols or ()) if symbol not in constant_symbols
        )
        fractions = tuple(
            tuple(field.to_sympy(element).as_numer_denom()) for element in converted
        )
        polynomial_fractions = tuple(
            (
                Poly(numerator, *generators, domain=constants),
                Poly(denominator, *generators, domain=constants),
            )
            for numerator, denominator in fractions
        )
        common_denominator = polynomial_fractions[0][1]
        for _, denominator in polynomial_fractions[1:]:
            common_denominator = common_denominator.lcm(denominator)
        numerators = tuple(
            numerator * common_denominator.exquo(denominator)
            for numerator, denominator in polynomial_fractions
        )
        return (
            tuple(
                cast(dict[tuple[int, ...], FieldElement], numerator.as_dict())
                for numerator in numerators
            ),
            cast(tuple[Symbol, ...], generators),
            common_denominator.as_expr(),
        )

    denominator = converted[0].denom.ring.one
    for element in converted:
        denominator = denominator.lcm(element.denom)
    numerators = tuple(
        element.numer * denominator.exquo(element.denom) for element in converted
    )
    return (
        tuple(
            cast(dict[tuple[int, ...], FieldElement], numerator.to_dict())
            for numerator in numerators
        ),
        cast(tuple[Symbol, ...], tuple(field.symbols or ())),
        denominator.as_expr(),
    )


def constant_linear_kernel(
    elements: Sequence[object],
    field: Domain,
    constants: Domain,
) -> tuple[tuple[FieldElement, ...], ...]:
    r"""Return a basis of ``c`` satisfying ``sum(c[i]*elements[i]) = 0``."""
    if not elements:
        return ()
    columns, _, _ = _cleared_solution_coordinates(elements, field, constants)
    monomials = sorted(
        {monomial for column in columns for monomial in column},
        reverse=True,
    )
    matrix = DomainMatrix(
        [
            [
                constants.convert(column.get(monomial, constants.zero))
                for column in columns
            ]
            for monomial in monomials
        ],
        (len(monomials), len(columns)),
        constants,
    )
    return tuple(
        tuple(cast(FieldElement, constants.convert(entry)) for entry in row)
        for row in matrix.nullspace(divide_last=True).to_list()
    )


def canonicalize_directions(
    directions: Sequence[ParametricRDESolution],
    field: Domain,
    constants: Domain,
) -> tuple[ParametricRDESolution, ...]:
    r"""Return a parameter-first row basis for the span of ``directions``."""
    if not directions:
        return ()
    parameter_count = len(directions[0].parameters)
    if any(len(direction.parameters) != parameter_count for direction in directions):
        raise ValueError("parametric directions have inconsistent parameter counts")

    numerators, generators, denominator = _cleared_solution_coordinates(
        [direction.solution for direction in directions],
        field,
        constants,
    )
    monomials = sorted(
        {monomial for numerator in numerators for monomial in numerator},
        reverse=True,
    )
    rows = [
        [
            *(constants.convert(parameter) for parameter in direction.parameters),
            *(
                constants.convert(numerator.get(monomial, constants.zero))
                for monomial in monomials
            ),
        ]
        for direction, numerator in zip(directions, numerators, strict=True)
    ]
    coordinates = DomainMatrix(
        rows,
        (len(rows), parameter_count + len(monomials)),
        constants,
    )
    reduced, _ = coordinates.rref()

    result = []
    for row in reduced.to_list():
        if all(constants.convert(entry) == constants.zero for entry in row):
            continue
        parameters = tuple(
            cast(FieldElement, constants.convert(entry))
            for entry in row[:parameter_count]
        )
        numerator = Poly.from_dict(
            {
                monomial: constants.convert(entry)
                for monomial, entry in zip(
                    monomials,
                    row[parameter_count:],
                    strict=True,
                )
                if constants.convert(entry) != constants.zero
            },
            *generators,
            domain=constants,
        )
        solution = _coerce(
            field,
            numerator.as_expr() / denominator,
        )
        result.append(ParametricRDESolution(parameters, solution))
    return tuple(result)
