r"""Exact residue fields of closed points on the rational base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.numbers import AlgebraicNumber
from sympy.core.symbol import Dummy
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly

from ..algebraic._field import (
    _algebraic_constant_domain,
    _base_polynomial_ring,
)
from ..tower import AlgebraicExtension


@dataclass(frozen=True, slots=True)
class _BaseResidueContext:
    """Exact reduction and lifting for one closed base point."""

    extension: AlgebraicExtension
    prime: Poly
    field: Domain
    root: Expr

    @classmethod
    def create(
        cls,
        extension: AlgebraicExtension,
        prime: Poly,
    ) -> _BaseResidueContext:
        normalized = _normalize_base_prime(extension, prime)
        _, constants, variable = _base_polynomial_ring(extension)
        algebraic_constants = _algebraic_constant_domain(constants)
        algebraic_prime = Poly(
            normalized.as_expr(),
            variable,
            domain=algebraic_constants,
        )
        if normalized.degree() == 1:
            root = cast(
                Expr,
                algebraic_constants.to_sympy(
                    algebraic_constants.convert(-algebraic_prime.nth(0))
                ),
            )
            return cls(extension, normalized, algebraic_constants, root)

        alias = Dummy(f"{variable.name}_bar")
        residue_field = algebraic_constants.alg_field_from_poly(
            algebraic_prime,
            alias=alias,
        )
        root = cast(AlgebraicNumber, residue_field.orig_ext[-1])
        return cls(extension, normalized, residue_field, root)

    def reduce_polynomial(self, polynomial: Poly) -> object:
        """Evaluate a base polynomial at the selected algebraic root."""
        _, constants, variable = _base_polynomial_ring(self.extension)
        normalized = Poly(
            polynomial.as_expr(),
            variable,
            domain=constants,
        )
        return self.field.from_sympy(normalized.as_expr().subs(variable, self.root))

    def lift(self, value: object) -> Poly:
        """Lift a residue-field element to degree below ``prime``."""
        _, constants, variable = _base_polynomial_ring(self.extension)
        converted = self.field.convert(value)
        expression = self.field.to_sympy(converted)
        if self.prime.degree() > 1:
            expression = expression.xreplace({self.root: variable})
        return Poly(
            expression,
            variable,
            domain=constants,
        ).rem(self.prime)


def _normalize_base_prime(
    extension: AlgebraicExtension,
    prime: Poly,
) -> Poly:
    if not isinstance(prime, Poly):
        raise TypeError("the base prime must be a SymPy Poly")
    _, constants, variable = _base_polynomial_ring(extension)
    if prime.gens != (variable,):
        raise ValueError(
            "the base prime generator must match the rational parent variable"
        )
    normalized = Poly(prime.as_expr(), variable, domain=constants)
    if normalized.degree() <= 0:
        raise ValueError("a finite base prime must be nonconstant")
    normalized = normalized.monic()
    if normalized.is_irreducible is not True:
        raise ValueError("a finite base prime must be irreducible")
    return normalized
