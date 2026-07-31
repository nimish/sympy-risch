r"""Reciprocal charts and algebraic places at infinity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.agca.extensions import ExtensionElement, FiniteExtension
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..algebraic._field import (
    _base_polynomial_ring,
    _embed_parent,
    _extension_element,
    _native_or_exact_quotient,
)
from ..tower import AlgebraicExtension
from .closure import IntegralClosureResult, integral_closure
from .finite_places import FiniteAlgebraicPlace, LocalLaurentExpansion


@dataclass(frozen=True, slots=True)
class ReciprocalFunctionField:
    """A reciprocal chart used to represent every place at infinity."""

    original_extension: AlgebraicExtension
    variable: Symbol
    scale_exponent: int
    extension: AlgebraicExtension
    closure: IntegralClosureResult

    @classmethod
    def create(
        cls,
        extension: AlgebraicExtension,
        variable: Symbol | None = None,
    ) -> ReciprocalFunctionField:
        parent, constants, base_variable = _base_polynomial_ring(extension)
        reciprocal_variable = Dummy("s") if variable is None else variable
        if not isinstance(reciprocal_variable, Symbol):
            raise TypeError("the reciprocal variable must be a Symbol")
        if reciprocal_variable in {
            base_variable,
            extension.generator,
        } | set(getattr(constants, "symbols", ())):
            raise ValueError("the reciprocal variable must be new")
        reciprocal_parent = constants.frac_field(reciprocal_variable)
        degree = extension.field.rank
        scale_exponent = 0
        for power in range(degree):
            coefficient = cast(
                FracElement,
                parent.convert(extension.minimal_polynomial.nth(power)),
            )
            if not coefficient:
                continue
            numerator = Poly.from_dict(
                coefficient.numer.to_dict(),
                base_variable,
                domain=constants,
            )
            denominator = Poly.from_dict(
                coefficient.denom.to_dict(),
                base_variable,
                domain=constants,
            )
            pole_order = max(0, numerator.degree() - denominator.degree())
            exponent_gap = degree - power
            scale_exponent = max(
                scale_exponent,
                (pole_order + exponent_gap - 1) // exponent_gap,
            )

        reciprocal_generator = Dummy("z")
        defining_expression = reciprocal_generator**degree
        for power in range(degree):
            coefficient = parent.to_sympy(extension.minimal_polynomial.nth(power))
            transformed = coefficient.subs(
                base_variable,
                1 / reciprocal_variable,
            ) * reciprocal_variable ** (scale_exponent * (degree - power))
            defining_expression += transformed * reciprocal_generator**power
        reciprocal_extension = AlgebraicExtension(
            Poly(
                defining_expression,
                reciprocal_generator,
                domain=reciprocal_parent,
            )
        )
        closure = integral_closure(reciprocal_extension)
        return cls(
            original_extension=extension,
            variable=reciprocal_variable,
            scale_exponent=scale_exponent,
            extension=reciprocal_extension,
            closure=closure,
        )

    def forward(self, element: object) -> ExtensionElement:
        """Map an original field element into the reciprocal chart."""
        original = _extension_element(self.original_extension, element)
        original_parent, _, original_variable = _base_polynomial_ring(
            self.original_extension
        )
        expression = 0
        for power in range(self.original_extension.field.rank):
            coefficient = original_parent.to_sympy(original.rep.nth(power))
            expression += (
                coefficient.subs(
                    original_variable,
                    1 / self.variable,
                )
                * (self.extension.generator / self.variable**self.scale_exponent)
                ** power
            )
        return cast(
            ExtensionElement,
            self.extension.field.from_sympy(expression),
        )

    def backward(self, element: object) -> ExtensionElement:
        """Map a reciprocal-chart element back to the original field."""
        reciprocal = _extension_element(self.extension, element)
        _, _, original_variable = _base_polynomial_ring(self.original_extension)
        reciprocal_parent = cast(FractionField, self.extension.parent_field)
        expression = 0
        for power in range(self.extension.field.rank):
            coefficient = reciprocal_parent.to_sympy(reciprocal.rep.nth(power))
            expression += (
                coefficient.subs(
                    self.variable,
                    1 / original_variable,
                )
                * (
                    self.original_extension.generator
                    / original_variable**self.scale_exponent
                )
                ** power
            )
        return cast(
            ExtensionElement,
            self.original_extension.field.from_sympy(expression),
        )


@dataclass(frozen=True, slots=True, eq=False)
class InfiniteAlgebraicPlace:
    """A closed place at infinity represented on a reciprocal chart."""

    chart: ReciprocalFunctionField
    finite_place: FiniteAlgebraicPlace

    @property
    def extension(self) -> AlgebraicExtension:
        """The original algebraic function field."""
        return self.chart.original_extension

    @property
    def residue_field(self) -> FiniteExtension:
        """Exact residue field."""
        return self.finite_place.residue_field

    @property
    def degree(self) -> int:
        """Residue-field degree over the constant field."""
        return self.finite_place.degree

    @property
    def relative_degree(self) -> int:
        """Residue degree over the rational point at infinity."""
        return self.finite_place.relative_degree

    @property
    def ramification_index(self) -> int:
        """Ramification index above infinity."""
        return self.finite_place.ramification_index

    @property
    def is_ramified(self) -> bool:
        """Whether infinity is ramified."""
        return self.finite_place.is_ramified

    @property
    def uniformizer(self) -> ExtensionElement:
        """Return a valuation-one element in the original field."""
        return self.chart.backward(self.finite_place.uniformizer)

    @property
    def different_exponent(self) -> int:
        """Valuation of the relative different at infinity."""
        return self.finite_place.different_exponent

    def valuation(self, element: object) -> int | Expr:
        """Normalized valuation in the original field."""
        return self.finite_place.valuation(self.chart.forward(element))

    def reduce_local(self, element: object) -> ExtensionElement:
        """Reduce an original-field element regular at infinity."""
        return self.finite_place.reduce_local(self.chart.forward(element))

    def lift_residue(self, value: object) -> ExtensionElement:
        """Lift a residue value into the original field."""
        return self.chart.backward(self.finite_place.lift_residue(value))

    def constant_coordinates(self, value: object) -> tuple[FieldElement, ...]:
        r"""Return the finite-place coordinates of one local residue.

        The reciprocal chart preserves the residue field, so its closed-point
        coordinate basis is also the canonical coordinate basis at infinity.
        """
        return self.finite_place.constant_coordinates(value)

    def from_constant_coordinates(
        self,
        coordinates: tuple[object, ...],
    ) -> ExtensionElement:
        """Reconstruct a residue-field element from local constant coordinates."""
        return self.finite_place.from_constant_coordinates(coordinates)

    def local_expansion(
        self,
        element: object,
        *,
        stop: int,
    ) -> LocalLaurentExpansion:
        """Compute a Laurent segment at infinity."""
        value = _extension_element(self.extension, element)
        reciprocal_expansion = self.finite_place.local_expansion(
            self.chart.forward(value),
            stop=stop,
        )
        return LocalLaurentExpansion(
            place=self,
            element=value,
            start=reciprocal_expansion.start,
            stop=reciprocal_expansion.stop,
            coefficients=reciprocal_expansion.coefficients,
            remainder=self.chart.backward(reciprocal_expansion.remainder),
            remainder_valuation=reciprocal_expansion.remainder_valuation,
        )

    def differential_residue(self, coefficient: object) -> ExtensionElement:
        r"""Return the residue of ``coefficient*dx`` at infinity."""
        transformed = self.chart.forward(coefficient)
        reciprocal_base = _embed_parent(
            self.chart.extension,
            Poly(
                self.chart.variable,
                self.chart.variable,
                domain=_base_polynomial_ring(self.chart.extension)[1],
            ),
        )
        return self.finite_place.differential_residue(
            _native_or_exact_quotient(
                self.chart.extension,
                -transformed,
                reciprocal_base**2,
            )
        )

    def differential_residues_for_powers(
        self,
        coefficient: object,
        parameter: object,
        count: int,
    ) -> tuple[ExtensionElement, ...]:
        r"""Return residues for a power family after one reciprocal-chart map."""
        transformed = self.chart.forward(coefficient)
        transformed_parameter = self.chart.forward(parameter)
        reciprocal_base = _embed_parent(
            self.chart.extension,
            Poly(
                self.chart.variable,
                self.chart.variable,
                domain=_base_polynomial_ring(self.chart.extension)[1],
            ),
        )
        return self.finite_place.differential_residues_for_powers(
            _native_or_exact_quotient(
                self.chart.extension,
                -transformed,
                reciprocal_base**2,
            ),
            transformed_parameter,
            count,
        )
