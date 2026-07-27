r"""Relative simple-radical descent over an implemented lower tower.

For a terminal radical extension

``L = K(theta),   theta**n = u in K``,

write ``v = sum(c_j*theta**j)``.  If ``a`` belongs to ``K`` and
``D(theta) = eta*theta``, then the algebraic Risch differential equation

``D(v) + a*v = b``

splits into the scalar equations

``D(c_j) + (a + j*eta)*c_j = b_j``

over ``K``.  This is a genuine mixed algebraic-over-monomial cell: it reuses
the existing lower scalar RDE solver without treating a relative curve as a
curve over the constant field.  General algebraic extensions instead produce
coupled differential systems and remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from sympy.polys.agca.extensions import ExtensionElement

from .._typing import FieldElement
from ..tower import AlgebraicExtension, Derivation
from ._field import _extension_element

if TYPE_CHECKING:
    from ..rde import RischDifferentialEquationResult


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalData:
    """The pure-radical presentation ``theta**n = radicand`` over ``K``."""

    extension: AlgebraicExtension
    degree: int
    radicand: FieldElement


def recognize_relative_simple_radical(
    extension: AlgebraicExtension,
) -> RelativeSimpleRadicalData | None:
    r"""Recognize ``K[theta]/(theta**n - u)`` without a ``C(x)`` assumption."""
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("relative radical recognition needs an algebraic extension")
    modulus = extension.minimal_polynomial
    degree = modulus.degree()
    parent = extension.parent_field
    if any(
        parent.convert(modulus.nth(power)) != parent.zero for power in range(1, degree)
    ):
        return None
    radicand = -parent.convert(modulus.nth(0))
    if radicand == parent.zero:
        return None
    return RelativeSimpleRadicalData(extension, degree, radicand)


class RelativeSimpleRadicalRDEStatus(Enum):
    """Outcome of diagonal RDE descent through a relative simple radical."""

    SOLVED = "solved"
    NO_SOLUTION = "no_solution"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalRDEComponent:
    """One scalar lower-field equation for the ``theta**power`` coordinate."""

    power: int
    coefficient: FieldElement
    right_hand_side: FieldElement
    lower_result: RischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether the lower scalar RDE solved this power coordinate."""
        return self.lower_result.has_solution


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalRDEResult:
    r"""A diagonal relative-radical RDE decision.

    ``NO_SOLUTION`` is complete within the current simple-radical field: one
    lower coordinate equation has a complete scalar RDE obstruction, and the
    power basis is linearly independent over the lower field.  ``UNSUPPORTED``
    never makes a negative solvability claim.
    """

    radical: RelativeSimpleRadicalData | None
    derivation: Derivation
    coefficient: ExtensionElement
    right_hand_side: ExtensionElement
    status: RelativeSimpleRadicalRDEStatus
    solution: ExtensionElement | None
    logarithmic_derivative: FieldElement | None
    components: tuple[RelativeSimpleRadicalRDEComponent, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.derivation, Derivation):
            raise TypeError("a relative radical RDE result needs a derivation")
        extension = self._extension
        coefficient = _extension_element(extension, self.coefficient)
        right_hand_side = _extension_element(extension, self.right_hand_side)
        solution = (
            None
            if self.solution is None
            else _extension_element(extension, self.solution)
        )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "right_hand_side", right_hand_side)
        object.__setattr__(self, "solution", solution)
        if not isinstance(self.status, RelativeSimpleRadicalRDEStatus):
            raise TypeError("the relative radical RDE status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a relative radical RDE result needs a detail")
        if solution is not None and (
            self.derivation(solution) + coefficient * solution != right_hand_side
        ):
            raise ValueError("the relative radical RDE solution does not reconstruct")
        if self.status is RelativeSimpleRadicalRDEStatus.SOLVED:
            if solution is None:
                raise ValueError("a solved relative radical RDE needs a solution")
        elif solution is not None:
            raise ValueError("only a solved relative radical RDE carries a solution")

    @property
    def _extension(self) -> AlgebraicExtension:
        if self.derivation.level < 0:
            raise ValueError("a relative radical RDE needs a nonconstant level")
        extension = self.derivation.tower.levels[self.derivation.level]
        if not isinstance(extension, AlgebraicExtension):
            raise ValueError("the relative radical RDE derivation is not algebraic")
        if self.radical is not None and self.radical.extension is not extension:
            raise ValueError("the relative radical data belongs to another extension")
        return extension

    @property
    def has_solution(self) -> bool:
        """Whether the result contains a verified solution in the top field."""
        return self.status is RelativeSimpleRadicalRDEStatus.SOLVED

    @property
    def complete_negative(self) -> bool:
        """Whether a lower coordinate obstruction proves no top-field solution."""
        return self.status is RelativeSimpleRadicalRDEStatus.NO_SOLUTION

    @property
    def reconstructs(self) -> bool:
        """Whether a stored solution satisfies the original relative RDE."""
        return self.solution is not None and (
            self.derivation(self.solution) + self.coefficient * self.solution
            == self.right_hand_side
        )


def _unsupported(
    radical: RelativeSimpleRadicalData | None,
    derivation: Derivation,
    coefficient: ExtensionElement,
    right_hand_side: ExtensionElement,
    logarithmic_derivative: FieldElement | None,
    components: tuple[RelativeSimpleRadicalRDEComponent, ...],
    detail: str,
) -> RelativeSimpleRadicalRDEResult:
    """Construct an explicit relative-RDE boundary without a fake solution."""
    return RelativeSimpleRadicalRDEResult(
        radical,
        derivation,
        coefficient,
        right_hand_side,
        RelativeSimpleRadicalRDEStatus.UNSUPPORTED,
        None,
        logarithmic_derivative,
        components,
        detail,
    )


def _lower_scalar(
    extension: AlgebraicExtension,
    value: ExtensionElement,
) -> FieldElement | None:
    """Return ``value`` when it belongs to the relative lower field ``K``."""
    coordinates = extension.power_basis_coordinates(value)
    if any(coordinate != extension.parent_field.zero for coordinate in coordinates[1:]):
        return None
    return coordinates[0]


def _diagonal_generator_logarithmic_derivative(
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> FieldElement | None:
    """Return ``eta`` when ``D(theta) = eta*theta`` over the lower field."""
    coordinates = extension.power_basis_coordinates(
        derivation(extension.field.generator),
    )
    if coordinates[0] != extension.parent_field.zero or any(
        coordinate != extension.parent_field.zero
        for power, coordinate in enumerate(coordinates)
        if power != 1
    ):
        return None
    return coordinates[1]


def solve_relative_simple_radical_rde(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> RelativeSimpleRadicalRDEResult:
    r"""Solve a diagonal RDE on a terminal simple radical over an RDE lower field.

    The coefficient must lie in the lower field.  This restriction is what
    makes the power-basis equations scalar; a non-lower coefficient is a
    future coupled-system problem rather than a negative decision.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("relative simple-radical RDE needs a derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "relative simple-radical RDE needs a terminal algebraic extension "
            "over a nonconstant lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("relative simple-radical RDE needs an algebraic top level")
    if extension.parent_field != derivation.lower().field:
        raise ValueError("the algebraic extension does not match its lower derivation")

    a = _extension_element(extension, coefficient)
    b = _extension_element(extension, right_hand_side)
    radical = recognize_relative_simple_radical(extension)
    if radical is None:
        return _unsupported(
            None,
            derivation,
            a,
            b,
            None,
            (),
            "the terminal algebraic extension is not a pure simple radical",
        )
    lower_coefficient = _lower_scalar(extension, a)
    if lower_coefficient is None:
        return _unsupported(
            radical,
            derivation,
            a,
            b,
            None,
            (),
            "a non-lower algebraic RDE coefficient requires a coupled system",
        )

    eta = _diagonal_generator_logarithmic_derivative(extension, derivation)
    if eta is None:
        return _unsupported(
            radical,
            derivation,
            a,
            b,
            None,
            (),
            "the simple-radical generator derivative is not diagonal over the lower field",
        )
    lower_derivation = derivation.lower()
    right_coordinates = extension.power_basis_coordinates(b)
    components: list[RelativeSimpleRadicalRDEComponent] = []

    from ..rde import solve_risch_differential_equation

    for power, right_coordinate in enumerate(right_coordinates):
        shifted_coefficient = (
            lower_coefficient + extension.parent_field.convert(power) * eta
        )
        try:
            lower_result = solve_risch_differential_equation(
                shifted_coefficient,
                right_coordinate,
                lower_derivation,
            )
        except NotImplementedError as exc:
            return _unsupported(
                radical,
                derivation,
                a,
                b,
                eta,
                tuple(components),
                "the lower scalar RDE is outside its implemented contract: " + str(exc),
            )
        component = RelativeSimpleRadicalRDEComponent(
            power,
            shifted_coefficient,
            right_coordinate,
            lower_result,
        )
        components.append(component)
        if not lower_result.has_solution:
            return RelativeSimpleRadicalRDEResult(
                radical,
                derivation,
                a,
                b,
                RelativeSimpleRadicalRDEStatus.NO_SOLUTION,
                None,
                eta,
                tuple(components),
                "one power-basis coordinate has a complete lower scalar RDE obstruction",
            )

    solution_coordinates = tuple(
        component.lower_result.solution for component in components
    )
    if any(solution is None for solution in solution_coordinates):
        raise ValueError("a solved lower scalar RDE lacks its solution")
    solution = extension.from_power_basis_coordinates(solution_coordinates)
    return RelativeSimpleRadicalRDEResult(
        radical,
        derivation,
        a,
        b,
        RelativeSimpleRadicalRDEStatus.SOLVED,
        solution,
        eta,
        tuple(components),
        "the terminal simple-radical RDE splits into verified lower scalar equations",
    )


def solve_relative_simple_radical_exact_derivative(
    integrand: object,
    derivation: Derivation,
) -> RelativeSimpleRadicalRDEResult:
    """Solve the ``a = 0`` exact-derivative slice of relative radical descent."""
    return solve_relative_simple_radical_rde(0, integrand, derivation)
