r"""A narrow real Möbius normal form for hyperbolic tangent monomials.

For one monomial extension with

.. math::

   D t = \eta (1 - t^2),

the coordinate ``u = (1 + t)/(1 - t)`` is hyperexponential:

.. math::

   D u = 2 \eta u.

The construction below records this as an immutable differential-field
isomorphism.  It is intentionally limited to that exact derivative law; it
does not purport to normalize general quadratic or Riccati monomials.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from ._typing import FieldElement, is_field_zero
from .tower import (
    Derivation,
    DifferentialTower,
    MonomialExtension,
    SpecialPolynomialStatus,
    _coerce,
    _domain_symbols,
)

_HYPERBOLIC_LAW_MESSAGE = "hyperbolic Möbius normal form requires Dt = eta*(1 - t**2)"


def _field_elements_equal(
    left: FieldElement,
    right: FieldElement,
    field: Domain,
) -> bool:
    """Compare exact native field elements without backend representation bias."""
    difference = left - right
    return is_field_zero(difference) or cancel(field.to_sympy(difference)) == 0


@dataclass(frozen=True, slots=True)
class HyperbolicMobiusNormalForm:
    r"""Immutable conjugacy from ``Dt=eta*(1-t**2)`` to ``Du=2*eta*u``.

    The lower tower is shared structurally by the source and target immutable
    towers.  Only the active top level changes: the source generator is
    replaced by the fresh target generator.  ``forward`` and ``inverse``
    accept and return native SymPy-domain elements, not expression wrappers.
    """

    source_derivation: Derivation
    target_derivation: Derivation
    source_generator: Symbol
    target_generator: Symbol

    @property
    def source_field(self) -> Domain:
        """Return the field containing the original hyperbolic generator."""
        return self.source_derivation.field

    @property
    def target_field(self) -> Domain:
        """Return the field containing the hyperexponential coordinate."""
        return self.target_derivation.field

    @property
    def forward_generator_expression(self) -> Expr:
        r"""Return the target expression for ``t = (u-1)/(u+1)``."""
        return cancel((self.target_generator - 1) / (self.target_generator + 1))

    @property
    def inverse_generator_expression(self) -> Expr:
        r"""Return the source expression for ``u = (1+t)/(1-t)``."""
        return cancel((1 + self.source_generator) / (1 - self.source_generator))

    def forward_expression(self, expression: Expr) -> Expr:
        """Rewrite one source expression in the target coordinate."""
        return cancel(
            expression.xreplace(
                {self.source_generator: self.forward_generator_expression},
            )
        )

    def inverse_expression(self, value: object) -> Expr:
        """Rewrite one target native element as a source expression."""
        target_value = _coerce(self.target_field, value)
        return cancel(
            self.target_field.to_sympy(target_value).xreplace(
                {self.target_generator: self.inverse_generator_expression},
            )
        )

    def forward(self, value: object) -> FieldElement:
        """Map a source native field element into the target field."""
        source_value = _coerce(self.source_field, value)
        return _coerce(
            self.target_field,
            self.forward_expression(self.source_field.to_sympy(source_value)),
        )

    def inverse(self, value: object) -> FieldElement:
        """Map a target native field element back into the source field."""
        return _coerce(self.source_field, self.inverse_expression(value))

    def verifies_derivation(self) -> bool:
        r"""Whether ``D_target(forward(v)) = forward(D_source(v))`` on generators.

        A differential-field homomorphism is determined by the tower
        generators.  The lower levels are identical, and the final equality
        checks the Möbius law for the active source generator.
        """
        for extension in self.source_derivation.tower.levels[
            : self.source_derivation.level + 1
        ]:
            source_generator = _coerce(self.source_field, extension.generator)
            expected = self.forward(self.source_derivation(source_generator))
            actual = self.target_derivation(self.forward(source_generator))
            if not _field_elements_equal(actual, expected, self.target_field):
                return False
        return True


def _validate_hyperbolic_source(
    derivation: Derivation,
    target_generator: Symbol,
) -> tuple[MonomialExtension, FieldElement]:
    """Validate the exact monomial law and return its coefficient ``eta``."""
    if not isinstance(derivation, Derivation):
        raise TypeError("hyperbolic Möbius normal form requires a Derivation")
    if derivation.level < 0:
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)
    if not isinstance(target_generator, Symbol):
        raise TypeError("the target generator must be a SymPy Symbol")
    if target_generator in _domain_symbols(derivation.field):
        raise ValueError("the target generator must be fresh over the source field")

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)

    derivative = extension.generator_derivative
    parent = extension.parent_field
    if derivative.degree() != 2:
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)
    eta = parent.convert(derivative.nth(0))
    linear_coefficient = parent.convert(derivative.nth(1))
    quadratic_coefficient = parent.convert(derivative.nth(2))
    if (
        is_field_zero(eta)
        or not is_field_zero(linear_coefficient)
        or not is_field_zero(quadratic_coefficient + eta)
    ):
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)
    return extension, eta


def build_hyperbolic_mobius_normal_form(
    derivation: Derivation,
    target_generator: Symbol,
) -> HyperbolicMobiusNormalForm:
    r"""Build the exact ``Dt=eta*(1-t**2)`` to ``Du=2*eta*u`` normal form.

    ``target_generator`` must be fresh over the source field.  The source and
    target have the same constant-field metadata because a rational Möbius
    change of coordinate cannot add constants.  The target is marked as
    having special polynomials: its generator ``u`` divides ``Du``.
    """
    source_extension, eta = _validate_hyperbolic_source(
        derivation,
        target_generator,
    )
    target_tower = DifferentialTower(
        derivation.tower.constants,
        derivation.tower.levels[: derivation.level],
    )
    target_parent = target_tower.field()
    target_derivative = Poly(
        2 * target_parent.to_sympy(eta) * target_generator,
        target_generator,
        domain=target_parent,
    )
    target_extension = MonomialExtension(
        target_derivative,
        source_extension.constant_field_status,
        SpecialPolynomialStatus.PRESENT,
    )
    target_tower = target_tower.extend(target_extension)
    normal_form = HyperbolicMobiusNormalForm(
        derivation,
        target_tower.derivation(),
        source_extension.generator,
        target_generator,
    )
    if not normal_form.verifies_derivation():
        raise ValueError(
            "hyperbolic Möbius normal form does not intertwine derivations"
        )
    return normal_form
