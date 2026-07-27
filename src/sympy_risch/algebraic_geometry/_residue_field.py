r"""Exact residue fields of closed points on the rational base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.expr import Expr
from sympy.core.numbers import AlgebraicNumber
from sympy.core.symbol import Dummy, Symbol
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
    root_value: object

    @classmethod
    def create(
        cls,
        extension: AlgebraicExtension,
        prime: Poly,
    ) -> _BaseResidueContext:
        normalized = _normalize_base_prime(extension, prime)
        _, constants, variable = _base_polynomial_ring(extension)
        algebraic_constants = _algebraic_constant_domain(constants)
        algebraic_prime = _rebase_base_polynomial(
            normalized,
            variable,
            algebraic_constants,
        )
        if normalized.degree() == 1:
            root_value = algebraic_constants.convert(-algebraic_prime.rep.nth(0))
            root = cast(
                Expr,
                algebraic_constants.to_sympy(root_value),
            )
            return cls(
                extension,
                normalized,
                algebraic_constants,
                root,
                root_value,
            )

        alias = Dummy(f"{variable.name}_bar")
        residue_field = algebraic_constants.alg_field_from_poly(
            algebraic_prime,
            alias=alias,
        )
        root = cast(AlgebraicNumber, residue_field.orig_ext[-1])
        return cls(extension, normalized, residue_field, root, residue_field.unit)

    def reduce_polynomial(self, polynomial: Poly) -> object:
        """Evaluate a base polynomial at the selected algebraic root."""
        _, constants, variable = _base_polynomial_ring(self.extension)
        normalized = _rebase_base_polynomial(polynomial, variable, constants)
        if normalized.is_zero:
            return self.field.zero
        result = self.field.zero
        for power in range(normalized.degree(), -1, -1):
            result = result * self.root_value + self.field.convert(
                normalized.rep.nth(power)
            )
        return result

    def lift(self, value: object) -> Poly:
        """Lift a residue-field element to degree below ``prime``."""
        _, constants, variable = _base_polynomial_ring(self.extension)
        converted = self.field.convert(value)
        if self.prime.degree() == 1:
            return Poly.from_dict(
                {(0,): constants.convert(converted)},
                variable,
                domain=constants,
            )
        expression = self.field.to_sympy(converted)
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
    normalized = _rebase_base_polynomial(prime, variable, constants)
    if normalized.degree() <= 0:
        raise ValueError("a finite base prime must be nonconstant")
    normalized = normalized.monic()
    if normalized.is_irreducible is not True:
        raise ValueError("a finite base prime must be irreducible")
    return normalized


def _rebase_base_polynomial(
    polynomial: Poly,
    variable: Symbol,
    domain: Domain,
) -> Poly:
    """Copy a univariate polynomial without round-tripping its coefficients.

    ``Poly.as_expr()`` turns native ``ANP`` coefficients into symbolic
    ``CRootOf`` expressions.  SymPy cannot always coerce those expressions
    back into the original algebraic domain (notably for non-quadratic
    primitive elements), even though the native coefficients are already
    valid elements of that domain.
    """
    return Poly.from_dict(
        {
            monomial: domain.convert(coefficient)
            for monomial, coefficient in polynomial.rep.to_dict().items()
        },
        variable,
        domain=domain,
    )
