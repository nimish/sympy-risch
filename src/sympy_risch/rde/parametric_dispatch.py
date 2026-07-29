r"""Recursive dispatch for parametric Risch differential equations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass

from .._typing import ExactInput, FieldElement
from ..admission import assess_scalar_rde_admission
from ..constant_fields import finite_constant_field_profile
from ..tower import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._common import _field_elements_equal
from ._parametric_linear import (
    ParametricRDESolution,
    canonicalize_directions,
)
from ._tower_polynomials import _is_standard_rational_derivation_over
from .parametric import solve_rational_parametric_risch_differential_equation

_RESULT_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ParametricRischDifferentialEquationResult:
    r"""The solver-certified full solution space of a parametric RDE.

    Instances are returned by
    :func:`solve_parametric_risch_differential_equation`; direct construction
    is intentionally rejected because reconstructing the stored vectors can
    prove soundness, but not that the basis is complete.
    """

    coefficient: FieldElement
    right_hand_sides: tuple[FieldElement, ...]
    derivation: Derivation
    basis: tuple[ParametricRDESolution, ...]
    _construction_token: InitVar[object]

    def __post_init__(self, _construction_token: object) -> None:
        """Check dimensions, constant parameters, and exact reconstruction."""
        if _construction_token is not _RESULT_CONSTRUCTION_TOKEN:
            raise TypeError("parametric RDE results must be constructed by the solver")
        if not self.right_hand_sides:
            raise ValueError("a parametric RDE needs at least one right-hand side")
        field = self.derivation.field
        constants = self.derivation.tower.constants
        for vector in self.basis:
            if len(vector.parameters) != len(self.right_hand_sides):
                raise ValueError(
                    "a parametric RDE vector has the wrong parameter count"
                )
            parameters = tuple(constants.convert(value) for value in vector.parameters)
            right_hand_side = sum(
                (
                    field.from_sympy(constants.to_sympy(parameter)) * right_hand_side
                    for parameter, right_hand_side in zip(
                        parameters,
                        self.right_hand_sides,
                        strict=True,
                    )
                ),
                field.zero,
            )
            if not _field_elements_equal(
                self.derivation(vector.solution) + self.coefficient * vector.solution,
                right_hand_side,
                field,
            ):
                raise ValueError("a parametric RDE basis vector does not reconstruct")

    @property
    def parameter_basis(self) -> tuple[ParametricRDESolution, ...]:
        """Return basis vectors having a nonzero parameter direction."""
        constants = self.derivation.tower.constants
        return tuple(
            vector
            for vector in self.basis
            if any(
                constants.convert(value) != constants.zero
                for value in vector.parameters
            )
        )

    @property
    def homogeneous_basis(self) -> tuple[ParametricRDESolution, ...]:
        """Return basis vectors of the homogeneous scalar equation."""
        constants = self.derivation.tower.constants
        return tuple(
            vector
            for vector in self.basis
            if all(
                constants.convert(value) == constants.zero
                for value in vector.parameters
            )
        )

    @property
    def has_nontrivial_parameter_solution(self) -> bool:
        """Whether some nonzero right-hand-side combination is solvable."""
        return bool(self.parameter_basis)


def _build_parametric_rde_result(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
    basis: tuple[ParametricRDESolution, ...],
) -> ParametricRischDifferentialEquationResult:
    """Construct a solver-owned complete PRDE result."""
    return ParametricRischDifferentialEquationResult(
        coefficient,
        right_hand_sides,
        derivation,
        basis,
        _construction_token=_RESULT_CONSTRUCTION_TOKEN,
    )


def _solve_nonempty_parametric_rde(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> ParametricRischDifferentialEquationResult:
    """Dispatch one admitted nonempty family."""
    profile = finite_constant_field_profile(derivation.tower.constants)
    if profile is not None and _is_standard_rational_derivation_over(
        derivation,
        profile.constants,
    ):
        field = derivation.field
        extension = derivation.tower.levels[0]
        rational_result = solve_rational_parametric_risch_differential_equation(
            field.to_sympy(coefficient),
            tuple(field.to_sympy(value) for value in right_hand_sides),
            extension.generator,
            domain=profile.constants,
        )
        directions = canonicalize_directions(
            tuple(
                ParametricRDESolution(
                    vector.parameters,
                    _coerce(field, vector.solution),
                )
                for vector in rational_result.basis
            ),
            field,
            profile.constants,
        )
        return _build_parametric_rde_result(
            coefficient,
            right_hand_sides,
            derivation,
            directions,
        )

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "parametric RDE dispatch does not yet support algebraic tower levels"
        )
    if extension.family is MonomialFamily.PRIMITIVE:
        from .primitive_parametric import solve_primitive_parametric_rde

        return solve_primitive_parametric_rde(
            coefficient,
            right_hand_sides,
            derivation,
        )
    raise NotImplementedError(
        "parametric RDE dispatch currently supports rational and primitive levels"
    )


def _solve_parametric_rde_basis(
    coefficient: FieldElement,
    right_hand_sides: tuple[FieldElement, ...],
    derivation: Derivation,
) -> tuple[ParametricRDESolution, ...]:
    """Internal recursive basis operation, including the homogeneous-only case."""
    field = derivation.field
    coefficient_element = _coerce(field, coefficient)
    if right_hand_sides:
        elements = tuple(_coerce(field, value) for value in right_hand_sides)
        return _solve_nonempty_parametric_rde(
            coefficient_element,
            elements,
            derivation,
        ).basis

    # The public problem requires at least one parameter.  Internally, a
    # zero right-hand side exposes the homogeneous basis without creating a
    # second scalar-homogeneous algorithm.
    augmented = _solve_nonempty_parametric_rde(
        coefficient_element,
        (field.zero,),
        derivation,
    )
    return canonicalize_directions(
        tuple(
            ParametricRDESolution((), vector.solution)
            for vector in augmented.basis
            if vector.parameters[0] == derivation.tower.constants.zero
        ),
        field,
        derivation.tower.constants,
    )


def solve_parametric_risch_differential_equation(
    coefficient: ExactInput,
    right_hand_sides: Iterable[ExactInput],
    derivation: Derivation,
) -> ParametricRischDifferentialEquationResult:
    r"""Return the complete PRDE solution space over an implemented tower."""
    right_hand_side_inputs = tuple(right_hand_sides)
    if not right_hand_side_inputs:
        raise ValueError("a parametric RDE needs at least one right-hand side")
    admission = assess_scalar_rde_admission(derivation)
    if not admission.accepted:
        if admission.reason is None:
            raise RuntimeError(
                "unsupported parametric RDE admission omitted its reason"
            )
        raise NotImplementedError(
            "parametric RDE dispatch is unsupported for this tower prefix: "
            f"{admission.reason.value}: {admission.detail}"
        )

    field = derivation.field
    coefficient_element = _coerce(field, coefficient)
    right_hand_side_elements = tuple(
        _coerce(field, value) for value in right_hand_side_inputs
    )
    return _solve_nonempty_parametric_rde(
        coefficient_element,
        right_hand_side_elements,
        derivation,
    )
