r"""Certified residue maps for rational and algebraic number fields.

For ``QQ`` there is one residue map at a rational prime ``p``.  For a
monogenic algebraic number field ``QQ(alpha)``, an irreducible factor ``h`` of
the integral minimal polynomial of ``alpha`` modulo ``p`` defines the residue
map

``alpha -> u in GF(p)[u] / (h)``.

The factory below only returns unramified factors of an integral monogenic
presentation.  Reduction is intentionally partial: a field element whose
denominator vanishes at the selected prime is rejected as bad reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.core.symbol import Dummy
from sympy.ntheory.primetest import isprime
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domains import GF, QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly


class BadConstantFieldReduction(ValueError):
    """The selected prime does not define a supported residue map."""


def _same_polynomial_coefficients(left: Poly, right: Poly) -> bool:
    return (
        left.domain == right.domain
        and left.degree() == right.degree()
        and all(left.nth(i) == right.nth(i) for i in range(left.degree() + 1))
    )


def _integral_primitive_polynomial(source: AlgebraicField, symbol: Dummy) -> Poly:
    rational = Poly.from_list(source.mod.to_list(), symbol, domain=QQ)
    denominator, integral = rational.clear_denoms(convert=True)
    if denominator != 1 or integral.LC() != 1:
        raise NotImplementedError(
            "number-field reduction currently requires an integral primitive generator"
        )
    return integral


@dataclass(frozen=True, slots=True, init=False)
class ConstantFieldPrimeReduction:
    r"""A factory-owned exact map from ``QQ`` or ``QQ(alpha)`` to ``GF(p^f)``."""

    source: Domain
    rational_prime: int
    target: Domain
    residue_factor: Poly | None
    residue_degree: int

    def __init__(self) -> None:
        raise TypeError("constant-field reductions must be created by the factory")

    @classmethod
    def _create(
        cls,
        source: Domain,
        rational_prime: int,
        target: Domain,
        residue_factor: Poly | None,
        residue_degree: int,
    ) -> ConstantFieldPrimeReduction:
        result = object.__new__(cls)
        object.__setattr__(result, "source", source)
        object.__setattr__(result, "rational_prime", rational_prime)
        object.__setattr__(result, "target", target)
        object.__setattr__(result, "residue_factor", residue_factor)
        object.__setattr__(result, "residue_degree", residue_degree)
        result._validate()
        return result

    def _validate(self) -> None:
        prime = self.rational_prime
        if not isinstance(prime, int) or not isprime(prime):
            raise ValueError("a residue map needs a rational prime")
        if self.target.characteristic() != prime:
            raise ValueError("the residue field has the wrong characteristic")
        if self.residue_degree <= 0:
            raise ValueError("the residue degree must be positive")

        if self.source == QQ:
            if (
                self.residue_factor is not None
                or self.residue_degree != 1
                or self.target != GF(prime)
            ):
                raise ValueError("the rational residue map is inconsistent")
            return

        if not isinstance(self.source, AlgebraicField):
            raise TypeError("only rational and algebraic-number constants reduce")
        factor = self.residue_factor
        if factor is None:
            raise ValueError("an algebraic residue map needs its prime factor")
        if factor.degree() != self.residue_degree or not factor.is_irreducible:
            raise ValueError(
                "the residue factor must be irreducible of the stated degree"
            )
        if self.residue_degree == 1:
            if self.target != GF(prime):
                raise ValueError("a degree-one residue field must be GF(p)")
        elif not isinstance(self.target, FiniteExtension) or not (
            _same_polynomial_coefficients(factor, self.target.modulus)
        ):
            raise ValueError("the residue field is not defined by the certified factor")

    @property
    def field_size(self) -> int:
        """Return the cardinality ``p**f`` of the finite residue field."""
        return self.rational_prime**self.residue_degree

    def reduce(self, value: object) -> object:
        """Reduce one source element, rejecting a vanishing denominator."""
        if self.source == QQ:
            return self._reduce_rational(value)

        source = cast(AlgebraicField, self.source)
        try:
            converted = source.convert(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "the coefficient does not belong to the source field"
            ) from exc
        target = self.target
        generator = (
            target.generator
            if isinstance(target, FiniteExtension)
            else target.convert(-cast(Poly, self.residue_factor).nth(0))
        )
        result = target.zero
        for coefficient in converted.rep:
            result = result * generator + target.convert(
                self._reduce_rational(coefficient)
            )
        return result

    def _reduce_rational(self, value: object) -> object:
        rational = QQ.convert(value)
        numerator = int(QQ.numer(rational))
        denominator = int(QQ.denom(rational))
        if denominator % self.rational_prime == 0:
            raise BadConstantFieldReduction(
                "a coefficient denominator vanishes at the selected prime"
            )
        return self.target.exquo(
            self.target.convert(numerator),
            self.target.convert(denominator),
        )


def constant_field_prime_reductions(
    source: Domain,
    rational_prime: int,
) -> tuple[ConstantFieldPrimeReduction, ...]:
    r"""Return every supported unramified residue map above ``rational_prime``."""
    if not isinstance(source, Domain):
        raise TypeError("constant-field reduction needs an exact SymPy domain")
    if not isinstance(rational_prime, int) or not isprime(rational_prime):
        raise ValueError("the reduction characteristic must be a rational prime")

    if source == QQ:
        return (
            ConstantFieldPrimeReduction._create(
                source,
                rational_prime,
                GF(rational_prime),
                None,
                1,
            ),
        )
    if not isinstance(source, AlgebraicField):
        raise NotImplementedError(
            "constant-field reduction currently requires QQ or QQ(alpha)"
        )

    symbol = Dummy("a_bar")
    integral = _integral_primitive_polynomial(source, symbol)
    reduced = integral.set_domain(GF(rational_prime))
    _unit, square_free_factors = reduced.sqf_list()
    if any(multiplicity != 1 for _factor, multiplicity in square_free_factors):
        raise BadConstantFieldReduction(
            "the primitive polynomial is ramified at the selected prime"
        )
    _unit, factors = reduced.factor_list()
    return tuple(
        ConstantFieldPrimeReduction._create(
            source,
            rational_prime,
            (
                GF(rational_prime)
                if factor.degree() == 1
                else FiniteExtension(factor.monic())
            ),
            factor.monic(),
            factor.degree(),
        )
        for factor, multiplicity in factors
        if multiplicity == 1
    )


__all__ = [
    "BadConstantFieldReduction",
    "ConstantFieldPrimeReduction",
    "constant_field_prime_reductions",
]
