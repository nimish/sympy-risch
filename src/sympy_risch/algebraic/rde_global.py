r"""Unrestricted scalar Risch equations on direct algebraic curves.

The pole-bound factory proves that every solution of ``D(u) + a*u = b`` lies
in one explicit Riemann--Roch space.  This module composes that proof with the
finite-dimensional solver.  Consequently, an inconsistent linear system can
be promoted from "no solution in this supplied space" to a genuine global
no-solution result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed

from ..algebraic_geometry.function_field import AlgebraicFunctionField
from ..tower import Derivation
from ._field import _extension_element
from .constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from .rde_bounds import (
    AlgebraicRDEPoleBoundCertificate,
    certify_algebraic_rde_pole_bound,
)
from .rde_space import (
    AlgebraicRDEInSpaceSolution,
    AlgebraicRDEInSpaceUnsupported,
    AlgebraicRDENoSolutionInSpace,
    AlgebraicRDEZeroSolution,
    solve_algebraic_rde_in_space,
)


@dataclass(frozen=True, slots=True)
class AlgebraicScalarRDEProblem:
    """A validated unrestricted scalar RDE in one algebraic function field."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_side: ExtensionElement
    constant_field_analysis: AlgebraicConstantFieldAnalysis

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an algebraic scalar RDE needs a function field")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an algebraic scalar RDE needs a derivation")
        if self.derivation.field != self.model.extension.field:
            raise ValueError("the algebraic scalar RDE derivation belongs elsewhere")
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level]
            is not self.model.extension
        ):
            raise ValueError("the algebraic scalar RDE derivation is not model-bound")
        analysis = self.constant_field_analysis
        if not isinstance(analysis, AlgebraicConstantFieldAnalysis):
            raise TypeError("an algebraic scalar RDE needs constant-field analysis")
        if analysis.extension is not self.model.extension:
            raise ValueError("the constant analysis belongs to another function field")
        if not analysis.derivation.has_same_position(self.derivation):
            raise ValueError("the constant analysis belongs to another derivation")
        if not isinstance(self.coefficient, ExtensionElement) or (
            self.coefficient.ext is not self.model.extension.field
        ):
            raise TypeError("an algebraic scalar RDE coefficient must be native")
        if not isinstance(self.right_hand_side, ExtensionElement) or (
            self.right_hand_side.ext is not self.model.extension.field
        ):
            raise TypeError("an algebraic scalar RDE right-hand side must be native")


def _validate_pole_bound(
    problem: AlgebraicScalarRDEProblem,
    pole_bound: AlgebraicRDEPoleBoundCertificate,
) -> None:
    """Bind a pole certificate to the exact scalar problem it proves."""
    if not isinstance(pole_bound, AlgebraicRDEPoleBoundCertificate):
        raise TypeError("an algebraic scalar RDE result needs a pole-bound proof")
    if pole_bound.model is not problem.model or not (
        pole_bound.derivation.has_same_position(problem.derivation)
    ):
        raise ValueError("the pole-bound certificate belongs to another RDE")
    if (
        pole_bound.coefficient != problem.coefficient
        or pole_bound.right_hand_side != problem.right_hand_side
    ):
        raise ValueError("the pole-bound certificate proves another equation")


def _validate_bounded_result(
    problem: AlgebraicScalarRDEProblem,
    pole_bound: AlgebraicRDEPoleBoundCertificate,
    bounded: AlgebraicRDEInSpaceSolution
    | AlgebraicRDENoSolutionInSpace
    | AlgebraicRDEInSpaceUnsupported,
) -> None:
    """Require a bounded decision for the exact certified global space."""
    bounded_problem = bounded.problem
    if (
        bounded_problem.model is not problem.model
        or not bounded_problem.derivation.has_same_position(problem.derivation)
        or bounded_problem.coefficient != problem.coefficient
        or bounded_problem.right_hand_side != problem.right_hand_side
        or bounded_problem.constant_field_analysis
        is not problem.constant_field_analysis
        or bounded_problem.pole_divisor != pole_bound.pole_divisor
    ):
        raise ValueError("the bounded decision belongs to another scalar RDE")


@dataclass(frozen=True, slots=True)
class AlgebraicScalarRDEZeroSolution:
    """The unconditional zero solution of a homogeneous scalar RDE."""

    problem: AlgebraicScalarRDEProblem
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AlgebraicScalarRDEProblem):
            raise TypeError("a zero scalar RDE result needs a problem")
        if self.problem.right_hand_side != self.problem.model.extension.field.zero:
            raise ValueError("the zero function solves only a homogeneous RDE")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a zero scalar RDE result needs a nonempty detail")

    @property
    def solution(self) -> ExtensionElement:
        """Return the zero element of the problem field."""
        return self.problem.model.extension.field.zero

    @property
    def reconstructs(self) -> bool:
        """Whether zero satisfies the original homogeneous equation."""
        return (
            self.problem.derivation(self.solution)
            + self.problem.coefficient * self.solution
            == self.problem.right_hand_side
        )


@dataclass(frozen=True, slots=True)
class AlgebraicScalarRDESolution:
    """A reconstructed global scalar RDE solution with its complete pole bound."""

    problem: AlgebraicScalarRDEProblem
    pole_bound: AlgebraicRDEPoleBoundCertificate
    bounded_solution: AlgebraicRDEInSpaceSolution
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AlgebraicScalarRDEProblem):
            raise TypeError("an algebraic scalar RDE solution needs a problem")
        _validate_pole_bound(self.problem, self.pole_bound)
        if not isinstance(self.bounded_solution, AlgebraicRDEInSpaceSolution):
            raise TypeError("an algebraic scalar RDE solution needs a bounded solution")
        _validate_bounded_result(self.problem, self.pole_bound, self.bounded_solution)
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic scalar RDE solution needs a nonempty detail")
        if not self.reconstructs:
            raise ValueError("the algebraic scalar RDE solution does not reconstruct")

    @property
    def solution(self) -> ExtensionElement:
        """Return the solution reconstructed by the complete-space solver."""
        return self.bounded_solution.solution

    @property
    def reconstructs(self) -> bool:
        """Whether the stored solution satisfies the unrestricted equation."""
        return self.bounded_solution.reconstructs


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicScalarRDENoSolution:
    """A complete proof that the unrestricted scalar RDE has no solution."""

    problem: AlgebraicScalarRDEProblem
    pole_bound: AlgebraicRDEPoleBoundCertificate
    bounded_obstruction: AlgebraicRDENoSolutionInSpace
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("global algebraic RDE negatives must be created by the solver")

    @classmethod
    def _create(
        cls,
        problem: AlgebraicScalarRDEProblem,
        pole_bound: AlgebraicRDEPoleBoundCertificate,
        bounded_obstruction: AlgebraicRDENoSolutionInSpace,
        detail: str,
    ) -> AlgebraicScalarRDENoSolution:
        result = object.__new__(cls)
        object.__setattr__(result, "problem", problem)
        object.__setattr__(result, "pole_bound", pole_bound)
        object.__setattr__(result, "bounded_obstruction", bounded_obstruction)
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        if not isinstance(self.problem, AlgebraicScalarRDEProblem):
            raise TypeError("a global algebraic RDE negative needs a problem")
        _validate_pole_bound(self.problem, self.pole_bound)
        if not isinstance(self.bounded_obstruction, AlgebraicRDENoSolutionInSpace):
            raise TypeError("a global algebraic RDE negative needs a bounded proof")
        _validate_bounded_result(
            self.problem,
            self.pole_bound,
            self.bounded_obstruction,
        )
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a global algebraic RDE negative needs a nonempty detail")

    @property
    def complete_negative(self) -> bool:
        """The pole proof makes the bounded inconsistency globally complete."""
        return True


@dataclass(frozen=True, slots=True)
class AlgebraicScalarRDEUnsupported:
    """A boundary where no complete global scalar RDE decision was made."""

    problem: AlgebraicScalarRDEProblem
    pole_bound: AlgebraicRDEPoleBoundCertificate | None
    bounded_result: AlgebraicRDEInSpaceUnsupported | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AlgebraicScalarRDEProblem):
            raise TypeError("an unsupported scalar RDE result needs a problem")
        if self.pole_bound is None:
            if self.bounded_result is not None:
                raise ValueError("a bounded RDE boundary needs its pole certificate")
        else:
            _validate_pole_bound(self.problem, self.pole_bound)
            if self.bounded_result is not None:
                if not isinstance(
                    self.bounded_result,
                    AlgebraicRDEInSpaceUnsupported,
                ):
                    raise TypeError(
                        "an unsupported scalar RDE has a wrong bounded result"
                    )
                _validate_bounded_result(
                    self.problem,
                    self.pole_bound,
                    self.bounded_result,
                )
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an unsupported scalar RDE result needs a nonempty detail")


AlgebraicScalarRDEResult: TypeAlias = (
    AlgebraicScalarRDEZeroSolution
    | AlgebraicScalarRDESolution
    | AlgebraicScalarRDENoSolution
    | AlgebraicScalarRDEUnsupported
)


def solve_algebraic_rde(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
) -> AlgebraicScalarRDEResult:
    r"""Decide a scalar RDE on the admitted direct algebraic-curve cell."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("an algebraic scalar RDE needs a function field")
    problem = AlgebraicScalarRDEProblem(
        model,
        derivation,
        _extension_element(model.extension, coefficient),
        _extension_element(model.extension, right_hand_side),
        constant_field_analysis,
    )
    if problem.right_hand_side == model.extension.field.zero:
        return AlgebraicScalarRDEZeroSolution(
            problem,
            "the zero function solves the homogeneous algebraic scalar RDE",
        )
    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return AlgebraicScalarRDEUnsupported(
            problem,
            None,
            None,
            "global algebraic RDE solving needs a certified full constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    try:
        pole_bound = certify_algebraic_rde_pole_bound(
            problem.coefficient,
            problem.right_hand_side,
            derivation,
            model,
        )
    except (HeuristicGCDFailed, NotImplementedError) as exc:
        return AlgebraicScalarRDEUnsupported(
            problem,
            None,
            None,
            "the complete algebraic RDE pole bound is unavailable: " + str(exc),
        )

    bounded = solve_algebraic_rde_in_space(
        problem.coefficient,
        problem.right_hand_side,
        derivation,
        model,
        constant_field_analysis,
        pole_bound.pole_divisor,
    )
    match bounded:
        case AlgebraicRDEInSpaceSolution():
            return AlgebraicScalarRDESolution(
                problem,
                pole_bound,
                bounded,
                "the certified global pole bound reduces the scalar RDE to one "
                "complete Riemann--Roch space",
            )
        case AlgebraicRDENoSolutionInSpace():
            return AlgebraicScalarRDENoSolution._create(
                problem,
                pole_bound,
                bounded,
                "every global solution would lie in the certified pole space, "
                "whose exact constant-linear system is inconsistent",
            )
        case AlgebraicRDEInSpaceUnsupported():
            return AlgebraicScalarRDEUnsupported(
                problem,
                pole_bound,
                bounded,
                "the pole bound is complete, but the bounded solver is unavailable: "
                + bounded.detail,
            )
        case AlgebraicRDEZeroSolution():
            raise ValueError("a nonhomogeneous scalar RDE returned a zero-only result")


__all__ = [
    "AlgebraicScalarRDENoSolution",
    "AlgebraicScalarRDEProblem",
    "AlgebraicScalarRDEResult",
    "AlgebraicScalarRDESolution",
    "AlgebraicScalarRDEUnsupported",
    "AlgebraicScalarRDEZeroSolution",
    "solve_algebraic_rde",
]
