r"""Algebraic RDEs in a certified finite-dimensional function space.

Given an effective divisor ``E`` and a certified basis ``v_i`` of ``L(E)``,
the equation

``D(u) + a*u = b``

for ``u`` in ``L(E)`` is an exact constant-linear system whose columns are
``D(v_i) + a*v_i``.  This module implements that complete finite-dimensional
decision.  It does not claim that a caller-supplied ``E`` contains every
solution of the unrestricted algebraic RDE; deriving such a global pole bound
is the next local-valuation step of the general algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed

from .._typing import FieldElement
from ..algebraic_geometry.function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
)
from ..algebraic_geometry.riemann_roch import (
    CertifiedRiemannRochSpace,
    RiemannRochUnsupported,
    compute_riemann_roch_space,
)
from ..tower import Derivation
from ._field import _extension_element
from .constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from .linear import constant_linear_combination, solve_constant_linear_span


@dataclass(frozen=True, slots=True)
class AlgebraicRDEInSpaceProblem:
    r"""A validated request to solve an algebraic RDE inside ``L(E)``."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_side: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    pole_divisor: AlgebraicDivisor

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an algebraic RDE needs a function field")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an algebraic RDE needs a derivation")
        if self.derivation.field != self.model.extension.field:
            raise ValueError("the algebraic RDE derivation belongs to another field")
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level]
            is not self.model.extension
        ):
            raise ValueError("the algebraic RDE derivation is not bound to the model")
        analysis = self.constant_field_analysis
        if not isinstance(analysis, AlgebraicConstantFieldAnalysis):
            raise TypeError("an algebraic RDE needs constant-field analysis")
        if analysis.extension is not self.model.extension:
            raise ValueError("the constant analysis belongs to another function field")
        if not analysis.derivation.has_same_position(self.derivation):
            raise ValueError("the constant analysis belongs to another derivation")
        if not isinstance(self.pole_divisor, AlgebraicDivisor):
            raise TypeError("an algebraic RDE pole bound must be a divisor")
        if self.pole_divisor.model is not self.model:
            raise ValueError("the algebraic RDE pole bound belongs to another model")
        if not self.pole_divisor.is_effective:
            raise ValueError("an algebraic RDE pole bound must be effective")
        object.__setattr__(
            self,
            "coefficient",
            _extension_element(self.model.extension, self.coefficient),
        )
        object.__setattr__(
            self,
            "right_hand_side",
            _extension_element(self.model.extension, self.right_hand_side),
        )


def _validate_space(
    problem: AlgebraicRDEInSpaceProblem,
    space: CertifiedRiemannRochSpace,
) -> None:
    """Require the complete space for the problem's exact pole bound."""
    if not isinstance(space, CertifiedRiemannRochSpace):
        raise TypeError("an algebraic RDE decision needs a certified space")
    if space.model is not problem.model or space.divisor != problem.pole_divisor:
        raise ValueError("the certified space belongs to another RDE problem")


@dataclass(frozen=True, slots=True)
class AlgebraicRDEZeroSolution:
    r"""The unconditional solution ``u = 0`` of a homogeneous RDE.

    This positive witness needs neither a full-constant-field proof nor a
    constructed Riemann--Roch space: zero belongs to ``L(E)`` for every
    effective divisor ``E``.
    """

    problem: AlgebraicRDEInSpaceProblem
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AlgebraicRDEInSpaceProblem):
            raise TypeError("a zero algebraic RDE solution needs a problem")
        if self.problem.right_hand_side != self.problem.model.extension.field.zero:
            raise ValueError("the zero function solves only a homogeneous RDE")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a zero algebraic RDE solution needs a nonempty detail")

    @property
    def solution(self) -> ExtensionElement:
        """Return the zero element of the problem's function field."""
        return self.problem.model.extension.field.zero

    @property
    def reconstructs(self) -> bool:
        """Whether zero satisfies the requested homogeneous RDE."""
        return (
            self.problem.derivation(self.solution)
            + self.problem.coefficient * self.solution
            == self.problem.right_hand_side
        )


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicRDEInSpaceSolution:
    """A verified solution of the RDE belonging to the requested ``L(E)``."""

    problem: AlgebraicRDEInSpaceProblem
    space: CertifiedRiemannRochSpace
    coordinates: tuple[FieldElement, ...]
    solution: ExtensionElement
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("algebraic RDE solutions must be created by the solver")

    @classmethod
    def _create(
        cls,
        problem: AlgebraicRDEInSpaceProblem,
        space: CertifiedRiemannRochSpace,
        coordinates: tuple[FieldElement, ...],
        detail: str,
    ) -> AlgebraicRDEInSpaceSolution:
        result = object.__new__(cls)
        object.__setattr__(result, "problem", problem)
        object.__setattr__(result, "space", space)
        object.__setattr__(result, "coordinates", coordinates)
        object.__setattr__(
            result,
            "solution",
            constant_linear_combination(problem.model, space.basis, coordinates),
        )
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        _validate_space(self.problem, self.space)
        if len(self.coordinates) != len(self.space.basis):
            raise ValueError("the algebraic RDE coordinates have the wrong dimension")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic RDE solution needs a nonempty detail")
        if not self.reconstructs:
            raise ValueError("the algebraic RDE solution does not reconstruct")

    @property
    def reconstructs(self) -> bool:
        """Whether the stored function satisfies the requested RDE."""
        return (
            self.problem.derivation(self.solution)
            + self.problem.coefficient * self.solution
            == self.problem.right_hand_side
        )


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicRDENoSolutionInSpace:
    r"""A complete proof that the requested ``L(E)`` contains no solution.

    This is deliberately not a global no-solution result.  It becomes one only
    after a separate local theorem proves that every unrestricted solution must
    lie in this exact space.
    """

    problem: AlgebraicRDEInSpaceProblem
    space: CertifiedRiemannRochSpace
    operator_rank: int
    augmented_rank: int
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "negative algebraic RDE decisions must be created by the solver"
        )

    @classmethod
    def _create(
        cls,
        problem: AlgebraicRDEInSpaceProblem,
        space: CertifiedRiemannRochSpace,
        operator_rank: int,
        augmented_rank: int,
        detail: str,
    ) -> AlgebraicRDENoSolutionInSpace:
        result = object.__new__(cls)
        object.__setattr__(result, "problem", problem)
        object.__setattr__(result, "space", space)
        object.__setattr__(result, "operator_rank", operator_rank)
        object.__setattr__(result, "augmented_rank", augmented_rank)
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        _validate_space(self.problem, self.space)
        if not 0 <= self.operator_rank <= len(self.space.basis):
            raise ValueError("the algebraic RDE operator rank is inconsistent")
        if self.augmented_rank != self.operator_rank + 1:
            raise ValueError("the algebraic RDE augmented rank is inconsistent")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a negative algebraic RDE decision needs a detail")

    @property
    def complete_within_space(self) -> bool:
        """The negative is complete exactly for the certified ``L(E)``."""
        return True


@dataclass(frozen=True, slots=True)
class AlgebraicRDEInSpaceUnsupported:
    """A boundary where no complete decision in the requested space was made."""

    problem: AlgebraicRDEInSpaceProblem
    riemann_roch: RiemannRochUnsupported | None
    detail: str

    def __post_init__(self) -> None:
        if self.riemann_roch is not None and (
            self.riemann_roch.model is not self.problem.model
            or self.riemann_roch.divisor != self.problem.pole_divisor
        ):
            raise ValueError("the unsupported space belongs to another RDE problem")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an unsupported algebraic RDE decision needs a detail")


AlgebraicRDEInSpaceResult: TypeAlias = (
    AlgebraicRDEZeroSolution
    | AlgebraicRDEInSpaceSolution
    | AlgebraicRDENoSolutionInSpace
    | AlgebraicRDEInSpaceUnsupported
)


def solve_algebraic_rde_in_space(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    pole_divisor: AlgebraicDivisor,
) -> AlgebraicRDEInSpaceResult:
    r"""Decide ``D(u) + coefficient*u = right_hand_side`` for ``u in L(E)``.

    ``pole_divisor`` is an explicit, effective ``E``.  The result is complete
    within the certified Riemann--Roch space ``L(E)``; this function does not
    infer that ``E`` bounds every solution of the unrestricted equation.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("an algebraic RDE needs a function field")
    problem = AlgebraicRDEInSpaceProblem(
        model,
        derivation,
        _extension_element(model.extension, coefficient),
        _extension_element(model.extension, right_hand_side),
        constant_field_analysis,
        pole_divisor,
    )
    if problem.right_hand_side == model.extension.field.zero:
        return AlgebraicRDEZeroSolution(
            problem,
            "the zero function solves the homogeneous algebraic RDE",
        )
    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return AlgebraicRDEInSpaceUnsupported(
            problem,
            None,
            "constant-linear RDE solving needs a certified full constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    try:
        space_result = compute_riemann_roch_space(model, pole_divisor)
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return AlgebraicRDEInSpaceUnsupported(
            problem,
            None,
            "the complete Riemann--Roch space is unavailable: " + str(exc),
        )
    if isinstance(space_result, RiemannRochUnsupported):
        return AlgebraicRDEInSpaceUnsupported(
            problem,
            space_result,
            "the complete Riemann--Roch space is unavailable: " + space_result.detail,
        )
    space = space_result

    try:
        operator_images = tuple(
            _extension_element(model.extension, derivation(element))
            + problem.coefficient * element
            for element in space.basis
        )
        solve = solve_constant_linear_span(
            model,
            operator_images,
            problem.right_hand_side,
        )
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return AlgebraicRDEInSpaceUnsupported(
            problem,
            None,
            "the exact constant-linear RDE solve is unavailable: " + str(exc),
        )

    if solve.coordinates is None:
        return AlgebraicRDENoSolutionInSpace._create(
            problem,
            space,
            solve.operator_rank,
            solve.augmented_rank,
            "the augmented constant-linear system proves that this complete "
            "Riemann--Roch space contains no solution",
        )
    return AlgebraicRDEInSpaceSolution._create(
        problem,
        space,
        solve.coordinates,
        "an exact constant-linear solve found a solution in the certified "
        "Riemann--Roch space",
    )


__all__ = [
    "AlgebraicRDEInSpaceProblem",
    "AlgebraicRDEInSpaceResult",
    "AlgebraicRDEInSpaceSolution",
    "AlgebraicRDEInSpaceUnsupported",
    "AlgebraicRDENoSolutionInSpace",
    "AlgebraicRDEZeroSolution",
    "solve_algebraic_rde_in_space",
]
