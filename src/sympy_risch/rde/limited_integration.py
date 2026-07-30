r"""Limited integration as an affine section of a parametric RDE space.

Given ``f, g_1, ..., g_m`` in a differential field ``K``, the limited
integration problem asks for ``v`` in ``K`` and constants ``c_j`` such that

``D(v) = f + c_1*g_1 + ... + c_m*g_m``.

The complete parametric RDE solver computes a vector space with a parameter
coordinate for ``f``.  Intersecting that space with the affine hyperplane
where this coordinate is one gives the complete limited-integration family.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass

from .._typing import FieldElement, FieldInput
from ..admission import RischOperation, assess_risch_operation
from ..tower import Derivation, _coerce
from ._common import _field_elements_equal
from ._parametric_linear import ParametricRDESolution, canonicalize_directions
from .parametric_dispatch import (
    ParametricRischDifferentialEquationResult,
    _solve_admitted_parametric_risch_differential_equation,
)

_RESULT_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class LimitedIntegrationSolution:
    r"""One pair ``(c, v)`` in a limited-integration affine space."""

    coefficients: tuple[FieldElement, ...]
    antiderivative: FieldElement


@dataclass(frozen=True, slots=True)
class LimitedIntegrationResult:
    r"""The complete affine solution family of a limited-integration problem.

    If :attr:`particular` exists, every solution is that pair plus an arbitrary
    constant-linear combination of :attr:`direction_basis`.  If it does not,
    the complete underlying parametric result proves that no solution exists
    in the current differential field.
    """

    integrand: FieldElement
    correction_terms: tuple[FieldElement, ...]
    derivation: Derivation
    particular: LimitedIntegrationSolution | None
    direction_basis: tuple[LimitedIntegrationSolution, ...]
    parametric_result: ParametricRischDifferentialEquationResult
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        """Check the affine section against its complete parametric result."""
        if _construction_token is not _RESULT_CONSTRUCTION_TOKEN:
            raise TypeError(
                "limited-integration results must be constructed by the solver"
            )

        field = self.derivation.field
        constants = self.derivation.tower.constants
        parametric = self.parametric_result
        if not parametric.derivation.has_same_position(self.derivation):
            raise ValueError(
                "the limited-integration and parametric results have "
                "different tower positions"
            )
        if not _field_elements_equal(parametric.coefficient, field.zero, field):
            raise ValueError("limited integration requires a zero RDE coefficient")

        expected_right_hand_sides = (self.integrand, *self.correction_terms)
        if len(parametric.right_hand_sides) != len(expected_right_hand_sides) or any(
            not _field_elements_equal(actual, expected, field)
            for actual, expected in zip(
                parametric.right_hand_sides,
                expected_right_hand_sides,
                strict=True,
            )
        ):
            raise ValueError(
                "the limited-integration and parametric right-hand sides differ"
            )

        has_affine_direction = any(
            constants.convert(vector.parameters[0]) != constants.zero
            for vector in parametric.basis
        )
        if has_affine_direction != (self.particular is not None):
            raise ValueError(
                "the limited-integration solution disagrees with the "
                "complete parametric basis"
            )
        if self.particular is None:
            if self.direction_basis:
                raise ValueError("an empty affine space cannot have directions")
            return

        expected_coefficient_count = len(self.correction_terms)
        if len(self.particular.coefficients) != expected_coefficient_count:
            raise ValueError(
                "the limited-integration particular solution has the wrong "
                "coefficient count"
            )
        for coefficient in self.particular.coefficients:
            constants.convert(coefficient)
        if len(self.direction_basis) != len(parametric.basis) - 1:
            raise ValueError(
                "the limited-integration direction basis has the wrong dimension"
            )
        for direction in self.direction_basis:
            if len(direction.coefficients) != expected_coefficient_count:
                raise ValueError(
                    "a limited-integration direction has the wrong coefficient count"
                )
            for coefficient in direction.coefficients:
                constants.convert(coefficient)

    @property
    def has_solution(self) -> bool:
        """Whether the affine limited-integration problem has a solution."""
        return self.particular is not None

    @property
    def proves_no_solution(self) -> bool:
        """Whether completeness of the parametric basis proves no in-field solution."""
        return self.particular is None


def solve_limited_integration(
    integrand: FieldInput,
    correction_terms: Iterable[FieldInput],
    derivation: Derivation,
) -> LimitedIntegrationResult:
    r"""Solve ``D(v) = integrand + sum(c_j*correction_terms[j])`` completely."""
    admission = assess_risch_operation(
        derivation,
        RischOperation.LIMITED_INTEGRATION,
    )
    if not admission.accepted:
        if admission.reason is None:
            raise RuntimeError(
                "unsupported limited-integration admission omitted its reason"
            )
        raise NotImplementedError(
            "limited integration is unsupported for this tower prefix: "
            f"{admission.reason.value}: {admission.detail}"
        )

    field = derivation.field
    constants = derivation.tower.constants
    correction_inputs = tuple(correction_terms)
    parametric = _solve_admitted_parametric_risch_differential_equation(
        0,
        (integrand, *correction_inputs),
        derivation,
    )
    integrand_element = _coerce(field, integrand)
    correction_elements = tuple(_coerce(field, term) for term in correction_inputs)

    pivot = next(
        (
            vector
            for vector in parametric.basis
            if constants.convert(vector.parameters[0]) != constants.zero
        ),
        None,
    )
    if pivot is None:
        return LimitedIntegrationResult(
            integrand_element,
            correction_elements,
            derivation,
            None,
            (),
            parametric,
            _construction_token=_RESULT_CONSTRUCTION_TOKEN,
        )

    pivot_coordinate = constants.convert(pivot.parameters[0])
    inverse_pivot = constants.one / pivot_coordinate
    inverse_pivot_in_field = field.from_sympy(constants.to_sympy(inverse_pivot))
    particular = LimitedIntegrationSolution(
        tuple(
            constants.convert(parameter) * inverse_pivot
            for parameter in pivot.parameters[1:]
        ),
        pivot.solution * inverse_pivot_in_field,
    )

    directions = []
    for vector in parametric.basis:
        first_coordinate = constants.convert(vector.parameters[0])
        scalar_in_field = field.from_sympy(constants.to_sympy(first_coordinate))
        directions.append(
            ParametricRDESolution(
                tuple(
                    constants.convert(parameter)
                    - first_coordinate * particular_coefficient
                    for parameter, particular_coefficient in zip(
                        vector.parameters[1:],
                        particular.coefficients,
                        strict=True,
                    )
                ),
                vector.solution - scalar_in_field * particular.antiderivative,
            )
        )

    canonical_directions = canonicalize_directions(
        directions,
        field,
        constants,
    )
    direction_basis = tuple(
        LimitedIntegrationSolution(vector.parameters, vector.solution)
        for vector in canonical_directions
    )
    return LimitedIntegrationResult(
        integrand_element,
        correction_elements,
        derivation,
        particular,
        direction_basis,
        parametric,
        _construction_token=_RESULT_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "LimitedIntegrationResult",
    "LimitedIntegrationSolution",
    "solve_limited_integration",
]
