r"""Elementary composition for a terminal simple radical over a lower tower.

For ``L = K(theta)`` with ``theta**n = u`` and
``D(theta) = eta*theta``, split an integrand into its power-basis coordinates,

``f = f_0 + sum(f_j*theta**j, j > 0)``.

The nonconstant coordinates are exact whenever the lower scalar equations

``D(c_j) + j*eta*c_j = f_j``

are solvable.  What remains is ``f_0`` in ``K``, so the established recursive
elementary integrator can finish it.  This is a constructive mixed
algebraic-over-monomial cell.  A failed coordinate equation is *not* a
non-elementarity proof: logarithmic reconstruction in the algebraic field is
a later relative-residue problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.polys.agca.extensions import ExtensionElement

from .._typing import FieldElement
from ..tower import AlgebraicExtension, Derivation
from ._field import _extension_element
from .relative import (
    RelativeSimpleRadicalData,
    RelativeSimpleRadicalRDEComponent,
    _diagonal_generator_logarithmic_derivative,
    recognize_relative_simple_radical,
)


def _terminal_extension(derivation: Derivation) -> AlgebraicExtension:
    """Return the terminal algebraic extension for this mixed reducer."""
    if not isinstance(derivation, Derivation):
        raise TypeError("relative simple-radical integration needs a derivation")
    if derivation.level < 1:
        raise NotImplementedError(
            "relative simple-radical integration needs a terminal algebraic "
            "extension over a differential lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError(
            "relative simple-radical integration needs an algebraic top level"
        )
    if extension.parent_field != derivation.lower().field:
        raise ValueError("the algebraic extension does not match its lower derivation")
    return extension


@dataclass(frozen=True, slots=True)
class RelativeSimpleRadicalIntegrationCertificate:
    """Verified lower RDEs for the nonconstant power-basis coordinates."""

    radical: RelativeSimpleRadicalData
    logarithmic_derivative: FieldElement
    components: tuple[RelativeSimpleRadicalRDEComponent, ...]


@dataclass(frozen=True, slots=True)
class _RelativeSimpleRadicalIntegrationReduction:
    """One exact relative-radical reduction before lower recursion."""

    derivation: Derivation
    integrand: ExtensionElement
    exact_part: ExtensionElement
    lower_remainder: FieldElement
    certificate: RelativeSimpleRadicalIntegrationCertificate

    @property
    def reconstructs(self) -> bool:
        """Whether the exact part and embedded lower remainder reconstruct."""
        extension = _terminal_extension(self.derivation)
        embedded_remainder = extension.from_power_basis_coordinates(
            (
                self.lower_remainder,
                *(extension.parent_field.zero for _ in range(extension.field.rank - 1)),
            ),
        )
        return self.derivation(self.exact_part) + embedded_remainder == self.integrand


def reduce_relative_simple_radical(
    integrand: object,
    derivation: Derivation,
) -> _RelativeSimpleRadicalIntegrationReduction:
    r"""Reduce a terminal simple radical to one lower elementary integrand.

    It constructs ``f = D(v) + f_0`` when every nonconstant power-basis
    coordinate has an exact lower scalar-RDE solution.  The caller owns the
    recursive integration of ``f_0`` so this algebraic reducer stays
    independent of the elementary dispatcher.

    A nonexact coordinate or unsupported lower algorithm raises
    :class:`NotImplementedError`.  Neither outcome is a claim that the
    original algebraic-field integrand is non-elementary: relative logarithmic
    reconstruction can still introduce algebraic logarithms.
    """
    extension = _terminal_extension(derivation)
    element = _extension_element(extension, integrand)
    radical = recognize_relative_simple_radical(extension)
    if radical is None:
        raise NotImplementedError(
            "relative simple-radical integration needs a terminal pure "
            "simple-radical extension"
        )
    eta = _diagonal_generator_logarithmic_derivative(extension, derivation)
    if eta is None:
        raise NotImplementedError(
            "relative simple-radical integration needs a generator derivative "
            "that is diagonal over the lower field"
        )

    from ..rde import solve_risch_differential_equation

    lower_field = extension.parent_field
    coordinates = extension.power_basis_coordinates(element)
    exact_coordinates: list[FieldElement] = [lower_field.zero] * extension.field.rank
    components: list[RelativeSimpleRadicalRDEComponent] = []
    for power, coordinate in enumerate(coordinates[1:], start=1):
        if coordinate == lower_field.zero:
            continue
        coefficient = lower_field.convert(power) * eta
        try:
            lower_result = solve_risch_differential_equation(
                coefficient,
                coordinate,
                derivation.lower(),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "relative simple-radical integration needs an implemented "
                "lower scalar RDE: " + str(exc)
            ) from exc
        component = RelativeSimpleRadicalRDEComponent(
            power,
            coefficient,
            coordinate,
            lower_result,
        )
        components.append(component)
        if not lower_result.has_solution or lower_result.solution is None:
            raise NotImplementedError(
                "relative simple-radical integration cannot turn a nonexact "
                "power-basis coordinate into a non-elementarity decision before "
                "relative logarithmic reconstruction is implemented"
            )
        exact_coordinates[power] = lower_result.solution

    exact_part = extension.from_power_basis_coordinates(tuple(exact_coordinates))
    lower_coordinate = coordinates[0]

    result = _RelativeSimpleRadicalIntegrationReduction(
        derivation,
        element,
        exact_part,
        lower_coordinate,
        RelativeSimpleRadicalIntegrationCertificate(
            radical,
            eta,
            tuple(components),
        ),
    )
    if not result.reconstructs:
        raise ValueError("relative simple-radical reduction failed to reconstruct")
    return result
