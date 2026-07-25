r"""Gaussian normal forms for real hypertangent monomial towers.

The Cayley coordinate

``z = (1 + i*t)/(1 - i*t)``

turns a hypertangent generator ``Dt = eta*(1 + t**2)`` into the
hyperexponential generator ``Dz = 2*i*eta*z``.  This module records the
resulting differential-field isomorphism explicitly, rather than replacing a
user's real generator in place.  The immutable map lets a Gaussian scalar
algorithm work in the transformed tower and then recover real coordinates in
the original tower.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from ._typing import FieldElement
from .tower import (
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
    _coerce,
)


def _zero_in_field(value: FieldElement, field: Domain) -> bool:
    """Test zero through expressions to tolerate Gaussian unit choices."""
    return value == field.zero or cancel(field.to_sympy(value)) == 0


def _substitute_once(
    expression: Expr,
    substitutions: dict[Symbol, Expr],
) -> Expr:
    """Apply simultaneous generator substitutions and normalize the fraction."""
    if not substitutions:
        return expression
    return cancel(expression.xreplace(substitutions))


@dataclass(frozen=True, slots=True)
class HypertangentGaussianNormalForm:
    r"""An immutable Cayley isomorphism for a real monomial tower.

    ``target_derivation`` has constants ``QQ_I`` and replaces every active
    hypertangent level by a hyperexponential level in the *same symbol*.  The
    symbol is intentionally reused because source and target live in separate
    SymPy domains.  ``forward_substitutions`` maps the source interpretation
    of each such symbol to its target expression; ``inverse_substitutions``
    maps target expressions back to the original real interpretation.
    """

    source_derivation: Derivation
    target_derivation: Derivation
    forward_substitutions: tuple[tuple[Symbol, Expr], ...]
    inverse_substitutions: tuple[tuple[Symbol, Expr], ...]

    @property
    def source_field(self) -> Domain:
        """Return the real source field."""
        return self.source_derivation.field

    @property
    def target_field(self) -> Domain:
        """Return the Gaussian target field."""
        return self.target_derivation.field

    def forward_expression(self, expression: Expr) -> Expr:
        """Map a source expression to the Gaussian normal form."""
        return _substitute_once(
            expression,
            dict(self.forward_substitutions),
        )

    def inverse_expression(self, value: object) -> Expr:
        """Map a target element to an expression over source generators."""
        target_value = _coerce(self.target_field, value)
        return _substitute_once(
            self.target_field.to_sympy(target_value),
            dict(self.inverse_substitutions),
        )

    def forward(self, value: object) -> FieldElement:
        """Embed one source-field element in the Gaussian normal form."""
        source_value = _coerce(self.source_field, value)
        return _coerce(
            self.target_field,
            self.forward_expression(self.source_field.to_sympy(source_value)),
        )

    def split_inverse(self, value: object) -> tuple[FieldElement, FieldElement]:
        r"""Return ``(p, q)`` when the inverse image is ``p + i*q``.

        Source generators are formal real generators, so after the inverse
        Cayley substitutions the usual coefficient conjugation separates the
        two source-field coordinates exactly.
        """
        target_value = _coerce(self.target_field, value)
        expression = self.inverse_expression(target_value)
        conjugate = expression.xreplace({I: -I})
        real_expression = cancel((expression + conjugate) / 2)
        imaginary_expression = cancel((expression - conjugate) / (2 * I))
        real_part = _coerce(self.source_field, real_expression)
        imaginary_part = _coerce(self.source_field, imaginary_expression)
        reconstructed = self.forward(real_part) + self.target_field.from_sympy(
            I
        ) * self.forward(imaginary_part)
        if not _zero_in_field(reconstructed - target_value, self.target_field):
            raise ValueError("Gaussian normal-form coordinates failed to reconstruct")
        return real_part, imaginary_part

    def verifies_derivation(self) -> bool:
        """Whether the Cayley map intertwines both tower derivations."""
        for extension in self.source_derivation.tower.levels[
            : self.source_derivation.level + 1
        ]:
            if not isinstance(extension, MonomialExtension):
                return False
            generator = _coerce(self.source_field, extension.generator)
            source_derivative = self.forward(self.source_derivation(generator))
            target_derivative = self.target_derivation(self.forward(generator))
            if not _zero_in_field(
                target_derivative - source_derivative,
                self.target_field,
            ):
                return False
        return True


def _validate_source_tower(derivation: Derivation) -> None:
    """Require a certified real monomial tower supported by the transform."""
    if not isinstance(derivation, Derivation):
        raise TypeError("Gaussian normal form requires a Derivation")
    if derivation.level < 1:
        raise ValueError(
            "Gaussian normal form requires a hypertangent extension above QQ(x)"
        )
    if derivation.tower.constants != QQ:
        raise NotImplementedError(
            "Gaussian normal form currently requires a QQ(x) base"
        )
    base = derivation.tower.levels[0]
    if (
        not isinstance(base, MonomialExtension)
        or base.parent_field != QQ
        or base.generator_derivative != Poly(1, base.generator, domain=QQ)
    ):
        raise NotImplementedError(
            "Gaussian normal form currently requires QQ(x) with Dx = 1"
        )

    has_hypertangent = False
    for level, extension in enumerate(
        derivation.tower.levels[: derivation.level + 1],
    ):
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "Gaussian normal form does not support algebraic tower levels"
            )
        if extension.family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            raise NotImplementedError(
                "Gaussian normal form supports primitive, hyperexponential, "
                "and hypertangent monomial levels"
            )
        if (
            level > 0
            and extension.constant_field_status is not ConstantFieldStatus.PRESERVED
        ):
            raise ValueError(
                "Gaussian normal form requires Const(K(t)) = Const(K) at "
                "every monomial level"
            )
        has_hypertangent |= extension.family is MonomialFamily.HYPERTANGENT
    if not has_hypertangent:
        raise ValueError(
            "Gaussian normal form requires at least one hypertangent level"
        )


def build_hypertangent_gaussian_normal_form(
    derivation: Derivation,
) -> HypertangentGaussianNormalForm:
    r"""Build the immutable Cayley normal form of a real monomial tower.

    The construction is level-by-level.  Before each source derivative law is
    copied, all earlier hypertangent generators are rewritten by their Cayley
    inverse.  A current hypertangent generator is then represented directly
    as a hyperexponential one.  Thus later derivative laws are constructed in
    the correct transformed lower field rather than merely post-processed.
    """
    _validate_source_tower(derivation)

    target_tower = DifferentialTower(QQ_I)
    forward_substitutions: dict[Symbol, Expr] = {}
    inverse_substitutions: dict[Symbol, Expr] = {}

    for source_extension in derivation.tower.levels[: derivation.level + 1]:
        assert isinstance(source_extension, MonomialExtension)
        generator = source_extension.generator
        target_parent = target_tower.field()
        if source_extension.family is MonomialFamily.HYPERTANGENT:
            eta = source_extension.parent_field.to_sympy(
                source_extension.generator_derivative.nth(2)
            )
            target_derivative = (
                2 * I * _substitute_once(eta, forward_substitutions) * generator
            )
            target_special_status = SpecialPolynomialStatus.PRESENT
        else:
            target_derivative = _substitute_once(
                source_extension.generator_derivative.as_expr(),
                forward_substitutions,
            )
            target_special_status = source_extension.special_polynomial_status

        target_extension = MonomialExtension(
            Poly(target_derivative, generator, domain=target_parent),
            source_extension.constant_field_status,
            target_special_status,
        )
        target_tower = target_tower.extend(target_extension)

        if source_extension.family is MonomialFamily.HYPERTANGENT:
            forward_substitutions[generator] = cancel(
                -I * (generator - 1) / (generator + 1)
            )
            inverse_substitutions[generator] = cancel(
                (1 + I * generator) / (1 - I * generator)
            )

    normal_form = HypertangentGaussianNormalForm(
        derivation,
        target_tower.derivation(),
        tuple(forward_substitutions.items()),
        tuple(inverse_substitutions.items()),
    )
    if not normal_form.verifies_derivation():
        raise ValueError("Cayley Gaussian normal form does not intertwine derivations")
    return normal_form
