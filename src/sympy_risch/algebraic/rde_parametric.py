r"""Parametric first-order Risch equations on direct algebraic curves.

For constants ``c_i``, this module computes the complete solution space of

``D(u) + a*u = sum(c_i*b_i)``.

At each place, the least order among the differentials ``b_i*dx`` is a lower
bound for every nonzero constant-linear combination.  The scalar dominant-
balance theorem therefore gives one pole divisor containing every possible
``u``.  One Riemann--Roch space and one constant-linear kernel then decide the
whole parametric problem at once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed

from .._typing import FieldElement, FieldInput
from ..algebraic_geometry.function_field import (
    AlgebraicDifferential,
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
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
from .linear import constant_linear_combination, constant_linear_kernel
from .rde_bounds import (
    AlgebraicRDELocalBound,
    _local_bound,
    _validate_rde_context,
)


def _right_hand_side_order(
    divisors: tuple[AlgebraicDivisor, ...] | list[AlgebraicDivisor],
    place: AlgebraicPlace,
) -> int:
    """Least local order in a parameter span, or zero outside its support."""
    return min((divisor.coefficient(place) for divisor in divisors), default=0)


@dataclass(frozen=True, slots=True)
class AlgebraicParametricRDEProblem:
    """A validated parametric RDE over one direct algebraic curve."""

    model: AlgebraicFunctionField
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_sides: tuple[ExtensionElement, ...]
    constant_field_analysis: AlgebraicConstantFieldAnalysis

    def __post_init__(self) -> None:
        _validate_rde_context(self.model, self.derivation)
        if not self.right_hand_sides:
            raise ValueError("a parametric algebraic RDE needs at least one parameter")
        analysis = self.constant_field_analysis
        if not isinstance(analysis, AlgebraicConstantFieldAnalysis):
            raise TypeError("a parametric algebraic RDE needs constant-field analysis")
        if analysis.extension is not self.model.extension or not (
            analysis.derivation.has_same_position(self.derivation)
        ):
            raise ValueError("the constant analysis belongs to another function field")
        if not isinstance(self.coefficient, ExtensionElement) or (
            self.coefficient.ext is not self.model.extension.field
        ):
            raise TypeError("a parametric RDE coefficient must be native")
        if any(
            not isinstance(value, ExtensionElement)
            or value.ext is not self.model.extension.field
            for value in self.right_hand_sides
        ):
            raise TypeError("parametric RDE right-hand sides must be native")


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicParametricRDEPoleBoundCertificate:
    """Factory-owned common pole bound for every parameter combination."""

    problem: AlgebraicParametricRDEProblem
    coefficient_differential: AlgebraicDifferential | None
    right_hand_side_differentials: tuple[AlgebraicDifferential | None, ...]
    local_bounds: tuple[AlgebraicRDELocalBound, ...]
    pole_divisor: AlgebraicDivisor
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("parametric RDE pole bounds must be created by the factory")

    @classmethod
    def _create(
        cls,
        problem: AlgebraicParametricRDEProblem,
        coefficient_differential: AlgebraicDifferential | None,
        right_hand_side_differentials: tuple[AlgebraicDifferential | None, ...],
        local_bounds: tuple[AlgebraicRDELocalBound, ...],
        pole_divisor: AlgebraicDivisor,
        detail: str,
    ) -> AlgebraicParametricRDEPoleBoundCertificate:
        result = object.__new__(cls)
        object.__setattr__(result, "problem", problem)
        object.__setattr__(result, "coefficient_differential", coefficient_differential)
        object.__setattr__(
            result,
            "right_hand_side_differentials",
            right_hand_side_differentials,
        )
        object.__setattr__(result, "local_bounds", local_bounds)
        object.__setattr__(result, "pole_divisor", pole_divisor)
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        problem = self.problem
        if not isinstance(problem, AlgebraicParametricRDEProblem):
            raise TypeError("a parametric pole bound needs its exact problem")
        model = problem.model
        zero = model.extension.field.zero
        if problem.coefficient == zero:
            if self.coefficient_differential is not None:
                raise ValueError("a zero coefficient cannot carry a differential")
            coefficient_divisor = None
        else:
            differential = self.coefficient_differential
            if (
                differential is None
                or differential.model is not model
                or (differential.coefficient != problem.coefficient)
            ):
                raise ValueError("the coefficient differential is inconsistent")
            coefficient_divisor = differential.divisor()

        if len(self.right_hand_side_differentials) != len(problem.right_hand_sides):
            raise ValueError("the parametric differentials have the wrong dimension")
        right_divisors: list[AlgebraicDivisor] = []
        for value, differential in zip(
            problem.right_hand_sides,
            self.right_hand_side_differentials,
            strict=True,
        ):
            if value == zero:
                if differential is not None:
                    raise ValueError(
                        "a zero right-hand side has no differential divisor"
                    )
            elif (
                differential is None
                or differential.model is not model
                or (differential.coefficient != value)
            ):
                raise ValueError("a right-hand-side differential is inconsistent")
            else:
                right_divisors.append(differential.divisor())

        support = dict.fromkeys(
            (() if coefficient_divisor is None else coefficient_divisor.support)
            + tuple(place for divisor in right_divisors for place in divisor.support)
        )
        if tuple(bound.place for bound in self.local_bounds) != tuple(support):
            raise ValueError("the parametric local bounds do not cover exact support")
        for bound in self.local_bounds:
            expected = _local_bound(
                model,
                bound.place,
                self.coefficient_differential,
                (
                    None
                    if coefficient_divisor is None
                    else coefficient_divisor.coefficient(bound.place)
                ),
                _right_hand_side_order(right_divisors, bound.place),
            )
            if bound != expected:
                raise ValueError("a parametric local bound is inconsistent")

        expected_divisor = AlgebraicDivisor(
            model,
            tuple(
                (bound.place, bound.pole_order)
                for bound in self.local_bounds
                if bound.pole_order
            ),
        )
        if self.pole_divisor != expected_divisor:
            raise ValueError("the parametric global pole divisor is inconsistent")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a parametric pole bound needs a nonempty detail")


def _certify_parametric_pole_bound(
    problem: AlgebraicParametricRDEProblem,
) -> AlgebraicParametricRDEPoleBoundCertificate:
    """Construct the common dominant-balance bound without repeated divisors."""
    model = problem.model
    zero = model.extension.field.zero
    coefficient_differential = (
        None if problem.coefficient == zero else model.differential(problem.coefficient)
    )
    right_differentials = tuple(
        None if value == zero else model.differential(value)
        for value in problem.right_hand_sides
    )
    coefficient_divisor = (
        None if coefficient_differential is None else coefficient_differential.divisor()
    )
    right_divisors = tuple(
        differential.divisor()
        for differential in right_differentials
        if differential is not None
    )
    places = tuple(
        dict.fromkeys(
            (() if coefficient_divisor is None else coefficient_divisor.support)
            + tuple(place for divisor in right_divisors for place in divisor.support)
        )
    )
    local_bounds = tuple(
        _local_bound(
            model,
            place,
            coefficient_differential,
            (
                None
                if coefficient_divisor is None
                else coefficient_divisor.coefficient(place)
            ),
            _right_hand_side_order(right_divisors, place),
        )
        for place in places
    )
    pole_divisor = AlgebraicDivisor(
        model,
        tuple(
            (bound.place, bound.pole_order)
            for bound in local_bounds
            if bound.pole_order
        ),
    )
    return AlgebraicParametricRDEPoleBoundCertificate._create(
        problem,
        coefficient_differential,
        right_differentials,
        local_bounds,
        pole_divisor,
        "the least local order in the right-hand-side span and exact scalar "
        "dominant balance bound every parameter combination",
    )


def certify_algebraic_parametric_rde_pole_bound(
    coefficient: FieldInput,
    right_hand_sides: Iterable[FieldInput],
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
) -> AlgebraicParametricRDEPoleBoundCertificate:
    """Certify one pole divisor for every constant parameter combination."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a parametric algebraic RDE needs a function field")
    problem = AlgebraicParametricRDEProblem(
        model,
        derivation,
        _extension_element(model.extension, coefficient),
        tuple(_extension_element(model.extension, value) for value in right_hand_sides),
        constant_field_analysis,
    )
    return _certify_parametric_pole_bound(problem)


@dataclass(frozen=True, slots=True)
class AlgebraicParametricRDESolutionVector:
    """One basis vector ``(u, c_1, ..., c_r)`` of the solution space."""

    function_coordinates: tuple[FieldElement, ...]
    parameters: tuple[FieldElement, ...]
    solution: ExtensionElement


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicParametricRDESolutionSpace:
    """The complete constant-vector space of solutions to a parametric RDE."""

    problem: AlgebraicParametricRDEProblem
    pole_bound: AlgebraicParametricRDEPoleBoundCertificate
    space: CertifiedRiemannRochSpace
    basis: tuple[AlgebraicParametricRDESolutionVector, ...]
    system_rank: int
    detail: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("parametric RDE solution spaces must be created by the solver")

    @classmethod
    def _create(
        cls,
        problem: AlgebraicParametricRDEProblem,
        pole_bound: AlgebraicParametricRDEPoleBoundCertificate,
        space: CertifiedRiemannRochSpace,
        kernel: tuple[tuple[FieldElement, ...], ...],
        detail: str,
    ) -> AlgebraicParametricRDESolutionSpace:
        dimension = len(space.basis)
        vectors = tuple(
            AlgebraicParametricRDESolutionVector(
                row[:dimension],
                row[dimension:],
                constant_linear_combination(
                    model=problem.model, basis=space.basis, coordinates=row[:dimension]
                ),
            )
            for row in kernel
        )
        result = object.__new__(cls)
        object.__setattr__(result, "problem", problem)
        object.__setattr__(result, "pole_bound", pole_bound)
        object.__setattr__(result, "space", space)
        object.__setattr__(result, "basis", vectors)
        object.__setattr__(
            result,
            "system_rank",
            dimension + len(problem.right_hand_sides) - len(vectors),
        )
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        if self.pole_bound.problem is not self.problem:
            raise ValueError("the parametric pole bound proves another problem")
        if self.space.model is not self.problem.model or (
            self.space.divisor != self.pole_bound.pole_divisor
        ):
            raise ValueError("the Riemann--Roch space belongs to another problem")
        columns = len(self.space.basis) + len(self.problem.right_hand_sides)
        if self.system_rank + len(self.basis) != columns:
            raise ValueError("the parametric RDE rank-nullity data is inconsistent")
        for vector in self.basis:
            if len(vector.function_coordinates) != len(self.space.basis) or len(
                vector.parameters
            ) != len(self.problem.right_hand_sides):
                raise ValueError("a parametric solution vector has the wrong dimension")
            if vector.solution != constant_linear_combination(
                self.problem.model,
                self.space.basis,
                vector.function_coordinates,
            ):
                raise ValueError(
                    "a parametric solution vector is reconstructed wrongly"
                )
            right_hand_side = constant_linear_combination(
                self.problem.model,
                self.problem.right_hand_sides,
                vector.parameters,
            )
            if (
                self.problem.derivation(vector.solution)
                + (self.problem.coefficient * vector.solution)
                != right_hand_side
            ):
                raise ValueError("a parametric solution vector does not reconstruct")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a parametric solution space needs a nonempty detail")

    @property
    def complete(self) -> bool:
        """Whether the basis spans every global solution and parameter relation."""
        return True


@dataclass(frozen=True, slots=True)
class AlgebraicParametricRDEUnsupported:
    """A boundary where the complete parametric solution space is unavailable."""

    problem: AlgebraicParametricRDEProblem
    pole_bound: AlgebraicParametricRDEPoleBoundCertificate | None
    riemann_roch: RiemannRochUnsupported | None
    detail: str

    def __post_init__(self) -> None:
        if self.pole_bound is None:
            if self.riemann_roch is not None:
                raise ValueError("an unsupported Riemann--Roch result needs its bound")
        elif self.pole_bound.problem is not self.problem:
            raise ValueError("the parametric pole bound proves another problem")
        if self.riemann_roch is not None and (
            self.riemann_roch.model is not self.problem.model
            or self.pole_bound is None
            or self.riemann_roch.divisor != self.pole_bound.pole_divisor
        ):
            raise ValueError("the unsupported space belongs to another problem")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an unsupported parametric RDE needs a nonempty detail")


AlgebraicParametricRDEResult: TypeAlias = (
    AlgebraicParametricRDESolutionSpace | AlgebraicParametricRDEUnsupported
)


def solve_algebraic_parametric_rde(
    coefficient: FieldInput,
    right_hand_sides: Iterable[FieldInput],
    derivation: Derivation,
    model: AlgebraicFunctionField,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
) -> AlgebraicParametricRDEResult:
    """Compute all ``(u, c)`` with ``D(u) + coefficient*u = sum(c_i*b_i)``."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a parametric algebraic RDE needs a function field")
    problem = AlgebraicParametricRDEProblem(
        model,
        derivation,
        _extension_element(model.extension, coefficient),
        tuple(_extension_element(model.extension, value) for value in right_hand_sides),
        constant_field_analysis,
    )
    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return AlgebraicParametricRDEUnsupported(
            problem,
            None,
            None,
            "parametric constant-linear solving needs a certified full constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    try:
        pole_bound = _certify_parametric_pole_bound(problem)
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return AlgebraicParametricRDEUnsupported(
            problem,
            None,
            None,
            "the complete parametric pole bound is unavailable: " + str(exc),
        )
    try:
        space_result = compute_riemann_roch_space(model, pole_bound.pole_divisor)
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return AlgebraicParametricRDEUnsupported(
            problem,
            pole_bound,
            None,
            "the complete Riemann--Roch space is unavailable: " + str(exc),
        )
    if isinstance(space_result, RiemannRochUnsupported):
        return AlgebraicParametricRDEUnsupported(
            problem,
            pole_bound,
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
        kernel = constant_linear_kernel(
            model,
            (*operator_images, *(-value for value in problem.right_hand_sides)),
        )
    except (HeuristicGCDFailed, NotImplementedError, TypeError, ValueError) as exc:
        return AlgebraicParametricRDEUnsupported(
            problem,
            pole_bound,
            None,
            "the exact parametric constant-linear solve is unavailable: " + str(exc),
        )
    return AlgebraicParametricRDESolutionSpace._create(
        problem,
        pole_bound,
        space,
        kernel,
        "one exact kernel gives every global function and parameter relation",
    )


__all__ = [
    "AlgebraicParametricRDEPoleBoundCertificate",
    "AlgebraicParametricRDEProblem",
    "AlgebraicParametricRDEResult",
    "AlgebraicParametricRDESolutionSpace",
    "AlgebraicParametricRDESolutionVector",
    "AlgebraicParametricRDEUnsupported",
    "certify_algebraic_parametric_rde_pole_bound",
    "solve_algebraic_parametric_rde",
]
