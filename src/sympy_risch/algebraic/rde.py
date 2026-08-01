r"""A sound integrating-factor cell for algebraic Risch differential equations.

This module handles only coefficients ``a`` for which the preceding
logarithmic-derivative recognizer finds ``a = D(h)/h`` in the current
algebraic function field.  Then

``D(u) + a*u = b`` iff ``D(h*u) = h*b``.

The right-hand problem is the certified exactness problem.  General algebraic
RDEs need local resonance and a global pole bound for ``u``; they deliberately
remain outside this small endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.agca.extensions import ExtensionElement

from ..algebraic_geometry.function_field import AlgebraicFunctionField
from ..tower import Derivation
from ._field import _extension_element, _quotient_extension_elements
from .constant_analysis import AlgebraicConstantFieldAnalysis
from .exactness import (
    AlgebraicExactnessResult,
    AlgebraicExactnessUnsupported,
    AlgebraicExactPrimitive,
    AlgebraicNoPrimitive,
    solve_algebraic_exactness,
)
from .log_derivative import (
    AlgebraicIntegratingFactorResult,
    find_algebraic_integrating_factor,
)


class AlgebraicIntegratingFactorRDEStatus(Enum):
    """Outcome of the integrating-factor algebraic RDE cell."""

    SOLVED = "solved"
    NO_SOLUTION = "no_solution"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicIntegratingFactorRDEResult:
    r"""A solution or structured boundary for ``D(u) + a*u = b``.

    ``NO_SOLUTION`` is complete only when a verified integrating factor
    reduces the problem to a complete in-field no-primitive exactness result.
    ``UNSUPPORTED`` includes coefficients that are not logarithmic derivatives:
    that fact alone says nothing about the general RDE.
    """

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_side: ExtensionElement
    status: AlgebraicIntegratingFactorRDEStatus
    solution: ExtensionElement | None
    integrating_factor: AlgebraicIntegratingFactorResult | None
    exactness: AlgebraicExactnessResult | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an algebraic RDE result needs a function field")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an algebraic RDE result needs a derivation")
        if self.derivation.field != self.model.extension.field:
            raise ValueError("the algebraic RDE derivation belongs to another field")
        if not isinstance(self.status, AlgebraicIntegratingFactorRDEStatus):
            raise TypeError("the algebraic RDE status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic RDE result needs a detail")

        coefficient = _extension_element(self.model.extension, self.coefficient)
        right_hand_side = _extension_element(self.model.extension, self.right_hand_side)
        solution = (
            None
            if self.solution is None
            else _extension_element(self.model.extension, self.solution)
        )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "right_hand_side", right_hand_side)
        object.__setattr__(self, "solution", solution)
        if self.integrating_factor is not None and (
            self.integrating_factor.model is not self.model
            or not self.integrating_factor.derivation.has_same_position(self.derivation)
        ):
            raise ValueError("the integrating factor belongs to another RDE")
        if self.exactness is not None and (
            self.exactness.model is not self.model
            or not self.exactness.derivation.has_same_position(self.derivation)
        ):
            raise ValueError("the exactness result belongs to another RDE")
        if solution is not None and (
            self.derivation(solution) + coefficient * solution != right_hand_side
        ):
            raise ValueError("the algebraic RDE solution does not reconstruct")
        if self.status is AlgebraicIntegratingFactorRDEStatus.SOLVED:
            if solution is None:
                raise ValueError("a solved algebraic RDE needs a solution")
        elif solution is not None:
            raise ValueError("only a solved algebraic RDE carries a solution")

    @property
    def has_solution(self) -> bool:
        """Whether the result carries a verified solution in the current field."""
        return self.status is AlgebraicIntegratingFactorRDEStatus.SOLVED

    @property
    def complete_negative(self) -> bool:
        """Whether the transformed exactness problem proves no solution."""
        return self.status is AlgebraicIntegratingFactorRDEStatus.NO_SOLUTION

    @property
    def reconstructs(self) -> bool:
        """Whether a stored solution satisfies the original RDE."""
        return self.solution is not None and (
            self.derivation(self.solution) + self.coefficient * self.solution
            == self.right_hand_side
        )


def solve_algebraic_integrating_factor_rde(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
) -> AlgebraicIntegratingFactorRDEResult:
    r"""Solve an algebraic RDE when its integrating factor is in the field."""
    if not isinstance(derivation, Derivation):
        raise TypeError("an algebraic RDE needs a derivation")
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("an algebraic RDE needs a function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("an algebraic RDE needs constant-field analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant analysis belongs to another function field")
    if not constant_field_analysis.derivation.has_same_position(derivation):
        raise ValueError("the constant analysis belongs to another derivation")
    if derivation.field != model.extension.field:
        raise ValueError("the algebraic RDE derivation belongs to another field")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not model.extension
    ):
        raise ValueError("the algebraic RDE derivation is not bound to the model")
    a = _extension_element(model.extension, coefficient)
    b = _extension_element(model.extension, right_hand_side)
    zero = model.extension.field.zero
    if b == zero:
        return AlgebraicIntegratingFactorRDEResult(
            model,
            derivation,
            a,
            b,
            AlgebraicIntegratingFactorRDEStatus.SOLVED,
            zero,
            None,
            None,
            "the zero function solves the homogeneous RDE",
        )

    factor = find_algebraic_integrating_factor(
        a,
        derivation,
        model,
        constant_field_analysis,
    )
    if not factor.found:
        return AlgebraicIntegratingFactorRDEResult(
            model,
            derivation,
            a,
            b,
            AlgebraicIntegratingFactorRDEStatus.UNSUPPORTED,
            None,
            factor,
            None,
            "the integrating-factor RDE cell cannot continue: " + factor.detail,
        )
    if factor.factor is None:
        raise ValueError("a found integrating factor lacks its witness")

    match solve_algebraic_exactness(
        factor.factor * b,
        derivation,
        model,
        constant_field_analysis,
    ):
        case AlgebraicExactPrimitive(primitive=primitive) as exactness:
            solution = _quotient_extension_elements(
                model.extension,
                primitive,
                factor.factor,
            )
            return AlgebraicIntegratingFactorRDEResult(
                model,
                derivation,
                a,
                b,
                AlgebraicIntegratingFactorRDEStatus.SOLVED,
                solution,
                factor,
                exactness,
                "a verified integrating factor reduces the RDE to an exact "
                "differential",
            )
        case AlgebraicNoPrimitive() as exactness:
            return AlgebraicIntegratingFactorRDEResult(
                model,
                derivation,
                a,
                b,
                AlgebraicIntegratingFactorRDEStatus.NO_SOLUTION,
                None,
                factor,
                exactness,
                "the transformed differential has no primitive in the current "
                "algebraic field",
            )
        case AlgebraicExactnessUnsupported(detail=detail) as exactness:
            return AlgebraicIntegratingFactorRDEResult(
                model,
                derivation,
                a,
                b,
                AlgebraicIntegratingFactorRDEStatus.UNSUPPORTED,
                None,
                factor,
                exactness,
                "the transformed exactness problem is not completely available: "
                + detail,
            )
