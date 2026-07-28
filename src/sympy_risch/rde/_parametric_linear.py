r"""Finite constant-linear algebra used by recursive parametric RDEs.

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

from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix

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


def constant_linear_kernel(
    elements: Sequence[object],
    field: Domain,
    constants: Domain,
) -> tuple[tuple[FieldElement, ...], ...]:
    r"""Return a basis of ``c`` satisfying ``sum(c[i]*elements[i]) = 0``."""
    if not elements:
        return ()
    converted = _fraction_elements(elements, field)
    denominator = converted[0].denom.ring.one
    for element in converted:
        denominator = denominator.lcm(element.denom)

    columns = tuple(
        element.numer * denominator.exquo(element.denom) for element in converted
    )
    monomials = sorted(
        {monomial for column in columns for monomial in column.monoms()},
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

    solutions = _fraction_elements(
        [direction.solution for direction in directions],
        field,
    )
    denominator = solutions[0].denom.ring.one
    for solution in solutions:
        denominator = denominator.lcm(solution.denom)
    numerators = tuple(
        solution.numer * denominator.exquo(solution.denom) for solution in solutions
    )
    monomials = sorted(
        {monomial for numerator in numerators for monomial in numerator.monoms()},
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

    ring = denominator.ring
    result = []
    for row in reduced.to_list():
        if all(constants.convert(entry) == constants.zero for entry in row):
            continue
        parameters = tuple(
            cast(FieldElement, constants.convert(entry))
            for entry in row[:parameter_count]
        )
        numerator = ring.from_dict(
            {
                monomial: constants.convert(entry)
                for monomial, entry in zip(
                    monomials,
                    row[parameter_count:],
                    strict=True,
                )
                if constants.convert(entry) != constants.zero
            }
        )
        solution = _coerce(
            field,
            numerator.as_expr() / denominator.as_expr(),
        )
        result.append(ParametricRDESolution(parameters, solution))
    return tuple(result)
