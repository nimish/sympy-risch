r"""Parametric Risch differential equations over a rational base.

This module solves

``D(y) + a*y = c_1*b_1 + ... + c_m*b_m``

in ``C(x)``, where the ``c_i`` belong to a supported exact rational constant
field ``C``. Bronstein's denominator and degree theorems reduce the
problem to one exact finite-dimensional nullspace.  The matrix is therefore
the complete terminal algorithm here, not a bounded heuristic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly, cancel

from .._typing import ExactInput, FieldElement
from ..constant_fields import is_supported_rational_constant_field
from ._common import _integer_constant
from .constant_field import (
    _as_domain_rational,
    _weakly_normalize_over_constant_field,
)


def _exact_expression(value: ExactInput) -> Expr:
    """Convert one declared exact input without losing native domain values."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, int):
        return Integer(value)
    return value.as_expr()


@dataclass(frozen=True, slots=True)
class RationalParametricRDESolution:
    """One normalized basis vector ``(c_1, ..., c_m, y)``."""

    parameters: tuple[FieldElement, ...]
    numerator: Poly
    solution: Expr


@dataclass(frozen=True, slots=True)
class RationalParametricRischDifferentialEquationResult:
    r"""The full solution space of a parametric RDE in ``C(x)``."""

    coefficient: Expr
    right_hand_sides: tuple[Expr, ...]
    variable: Symbol
    constant_field: Domain
    normalization_factor: Poly
    normalized_coefficient: Expr
    denominator_bound: Poly
    degree_bound: int
    constraint_matrix: DomainMatrix
    basis: tuple[RationalParametricRDESolution, ...]

    def __post_init__(self) -> None:
        """Validate reconstruction, independence, and nullspace dimension."""
        if not self.right_hand_sides:
            raise ValueError("a parametric RDE needs at least one right-hand side")
        parameter_count = len(self.right_hand_sides)
        unknown_count = self.degree_bound + 1 + parameter_count
        if self.constraint_matrix.domain != self.constant_field:
            raise ValueError("the constraint matrix uses the wrong constant field")
        if self.constraint_matrix.shape[1] != unknown_count:
            raise ValueError("the constraint matrix has the wrong column count")
        if len(self.basis) != unknown_count - self.constraint_matrix.rank():
            raise ValueError("the stored vectors do not form a full nullspace basis")

        coordinate_rows = []
        rational_field = self.constant_field.frac_field(self.variable)
        for vector in self.basis:
            if len(vector.parameters) != parameter_count:
                raise ValueError(
                    "a parametric RDE vector has the wrong parameter count"
                )
            if (
                vector.numerator.gens != (self.variable,)
                or vector.numerator.domain != self.constant_field
                or (
                    not vector.numerator.is_zero
                    and vector.numerator.degree() > self.degree_bound
                )
            ):
                raise ValueError("a parametric RDE vector has an invalid numerator")
            coordinates = [
                self.constant_field.convert(vector.numerator.nth(degree))
                for degree in range(self.degree_bound + 1)
            ]
            coordinates.extend(
                self.constant_field.convert(parameter)
                for parameter in vector.parameters
            )
            coordinate_rows.append(coordinates)

            equation = (
                vector.solution.diff(self.variable)
                + self.coefficient * vector.solution
                - sum(
                    self.constant_field.to_sympy(self.constant_field.convert(parameter))
                    * right_hand_side
                    for parameter, right_hand_side in zip(
                        vector.parameters,
                        self.right_hand_sides,
                        strict=True,
                    )
                )
            )
            if rational_field.from_sympy(cancel(equation)) != rational_field.zero:
                raise ValueError("a parametric RDE basis vector does not reconstruct")

        coordinate_matrix = DomainMatrix(
            coordinate_rows,
            (len(coordinate_rows), unknown_count),
            self.constant_field,
        )
        if coordinate_matrix.rank() != len(self.basis):
            raise ValueError("the parametric RDE basis vectors are dependent")
        if self.constraint_matrix * coordinate_matrix.transpose() != DomainMatrix(
            [
                [self.constant_field.zero for _ in self.basis]
                for _ in range(self.constraint_matrix.shape[0])
            ],
            (self.constraint_matrix.shape[0], len(self.basis)),
            self.constant_field,
        ):
            raise ValueError("a parametric RDE basis vector violates its constraints")

    @property
    def parameter_basis(self) -> tuple[RationalParametricRDESolution, ...]:
        """Return basis vectors with a nonzero parameter direction."""
        return tuple(
            vector
            for vector in self.basis
            if any(
                self.constant_field.convert(parameter) != self.constant_field.zero
                for parameter in vector.parameters
            )
        )

    @property
    def homogeneous_basis(self) -> tuple[RationalParametricRDESolution, ...]:
        """Return basis vectors solving the homogeneous scalar equation."""
        return tuple(
            vector
            for vector in self.basis
            if all(
                self.constant_field.convert(parameter) == self.constant_field.zero
                for parameter in vector.parameters
            )
        )

    @property
    def has_nontrivial_parameter_solution(self) -> bool:
        """Whether some nonzero right-hand-side combination is solvable."""
        return bool(self.parameter_basis)


def _parametric_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
) -> int:
    """Return the complete polynomial numerator bound."""
    maximum_rhs_degree = max(
        (
            right_hand_side.degree()
            for right_hand_side in right_hand_sides
            if not right_hand_side.is_zero
        ),
        default=-1,
    )
    derivative_degree = derivative_coefficient.degree()
    if solution_coefficient.is_zero:
        dominant_degree = derivative_degree - 1
    else:
        dominant_degree = max(
            solution_coefficient.degree(),
            derivative_degree - 1,
        )
    bound = max(0, maximum_rhs_degree - dominant_degree)

    if (
        not solution_coefficient.is_zero
        and solution_coefficient.degree() == derivative_degree - 1
    ):
        domain = derivative_coefficient.domain
        cancellation_degree = _integer_constant(
            domain.exquo(
                -domain.convert(solution_coefficient.LC()),
                domain.convert(derivative_coefficient.LC()),
            ),
            domain,
        )
        if cancellation_degree is not None and cancellation_degree >= 0:
            bound = max(bound, cancellation_degree)
    return bound


def _constraint_matrix(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_sides: tuple[Poly, ...],
    degree_bound: int,
) -> DomainMatrix:
    """Build the coefficient matrix of the complete polynomial problem."""
    variable = cast(Symbol, derivative_coefficient.gens[0])
    domain = derivative_coefficient.domain
    solution_columns = []
    for degree in range(degree_bound + 1):
        monomial = Poly(variable**degree, variable, domain=domain)
        solution_columns.append(
            derivative_coefficient * monomial.diff() + solution_coefficient * monomial
        )
    columns = (*solution_columns, *(-rhs for rhs in right_hand_sides))
    maximum_degree = max(
        (column.degree() for column in columns if not column.is_zero),
        default=-1,
    )
    rows = [
        [domain.convert(column.nth(degree)) for column in columns]
        for degree in range(maximum_degree + 1)
    ]
    return DomainMatrix(
        rows,
        (len(rows), len(columns)),
        domain,
    )


def _split_solution_rows(
    nullspace: DomainMatrix,
    solution_coefficient_count: int,
    parameter_count: int,
) -> list[list[FieldElement]]:
    """Choose parameter lifts followed by a homogeneous-kernel basis.

    Row-reducing with parameter columns first makes the split intrinsic:
    pivots in the parameter block give independent solvable combinations,
    while later pivots have an identically zero parameter block and span the
    homogeneous scalar solutions.
    """
    if nullspace.shape[0] == 0:
        return []
    reordered = DomainMatrix(
        [
            [
                *row[solution_coefficient_count:],
                *row[:solution_coefficient_count],
            ]
            for row in nullspace.to_list()
        ],
        nullspace.shape,
        nullspace.domain,
    )
    reduced, _ = reordered.rref()
    return [
        [
            *row[parameter_count:],
            *row[:parameter_count],
        ]
        for row in reduced.to_list()
    ]


def solve_rational_parametric_risch_differential_equation(
    coefficient: ExactInput,
    right_hand_sides: Iterable[ExactInput],
    x: Symbol,
    *,
    domain: Domain = QQ,
) -> RationalParametricRischDifferentialEquationResult:
    r"""Return every solution of a parametric RDE over ``domain(x)``.

    This solves

    ``Dy + coefficient*y = sum(parameters[i] * right_hand_sides[i])``

    as a finite-dimensional vector-space problem over ``domain``.  The common
    denominator and degree bounds are consequences of the scalar rational RDE
    reductions, so the returned nullspace is the full solution space rather
    than a heuristic bounded search.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    if not is_supported_rational_constant_field(domain):
        raise ValueError(
            "a rational parametric RDE requires a supported exact rational "
            "constant field"
        )
    right_hand_side_inputs = tuple(right_hand_sides)
    if not right_hand_side_inputs:
        raise ValueError("a parametric RDE needs at least one right-hand side")

    coefficient_expression, _, _ = _as_domain_rational(
        _exact_expression(coefficient),
        x,
        domain,
    )
    right_hand_side_data = tuple(
        _as_domain_rational(
            _exact_expression(right_hand_side),
            x,
            domain,
        )
        for right_hand_side in right_hand_side_inputs
    )
    right_hand_side_expressions = tuple(item[0] for item in right_hand_side_data)
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_over_constant_field(
            coefficient_expression,
            x,
            domain,
        )
    )
    normalized_right_hand_sides = tuple(
        _as_domain_rational(
            cancel(normalization_factor.as_expr() * right_hand_side),
            x,
            domain,
        )
        for right_hand_side in right_hand_side_expressions
    )
    _, coefficient_numerator, coefficient_denominator = _as_domain_rational(
        normalized_coefficient,
        x,
        domain,
    )

    # A reduced denominator of any parameter combination divides this LCM.
    # The scalar normal-denominator reduction then proves that the denominator
    # of every solution divides it as well.
    denominator_bound = Poly(1, x, domain=domain)
    for _, _, denominator in normalized_right_hand_sides:
        denominator_bound = denominator_bound.lcm(denominator).monic()

    derivative_coefficient = coefficient_denominator * denominator_bound
    solution_coefficient = (
        denominator_bound * coefficient_numerator
        - coefficient_denominator * denominator_bound.diff()
    )
    multiplier = coefficient_denominator * denominator_bound * denominator_bound
    transformed_right_hand_sides = []
    for _, numerator, denominator in normalized_right_hand_sides:
        quotient, remainder = multiplier.div(denominator)
        if not remainder.is_zero:
            raise ValueError(
                "the common parametric denominator failed to clear a right-hand side"
            )
        transformed_right_hand_sides.append(quotient * numerator)
    transformed_tuple = tuple(transformed_right_hand_sides)

    degree_bound = _parametric_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_tuple,
    )
    constraints = _constraint_matrix(
        derivative_coefficient,
        solution_coefficient,
        transformed_tuple,
        degree_bound,
    )
    nullspace_rows = _split_solution_rows(
        constraints.nullspace(divide_last=True),
        degree_bound + 1,
        len(right_hand_side_expressions),
    )
    basis = []
    denominator_expression = (
        normalization_factor.as_expr() * denominator_bound.as_expr()
    )
    for row in nullspace_rows:
        numerator = Poly.from_dict(
            {
                (degree,): domain.convert(row[degree])
                for degree in range(degree_bound + 1)
                if domain.convert(row[degree]) != domain.zero
            },
            x,
            domain=domain,
        )
        parameters = tuple(
            cast(FieldElement, domain.convert(value))
            for value in row[degree_bound + 1 :]
        )
        basis.append(
            RationalParametricRDESolution(
                parameters,
                numerator,
                cancel(numerator.as_expr() / denominator_expression),
            )
        )

    return RationalParametricRischDifferentialEquationResult(
        coefficient_expression,
        right_hand_side_expressions,
        x,
        domain,
        normalization_factor,
        normalized_coefficient,
        denominator_bound,
        degree_bound,
        constraints,
        tuple(basis),
    )
