r"""Certified maximal-order refinement and integral closure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly

from ..algebraic._field import (
    _base_polynomial_ring,
    _common_denominator,
    _embed_parent,
)
from ..algebraic.basis import IntegralBasis
from ..tower import AlgebraicExtension
from ._residue_field import _normalize_base_prime
from .orders import AlgebraicFractionalIdeal, AlgebraicOrder


@dataclass(frozen=True, slots=True)
class IntegralClosureStep:
    """One certified strict order enlargement."""

    prime: Poly
    previous_discriminant: Poly
    new_discriminant: Poly
    index: Poly
    previous_order: AlgebraicOrder
    new_order: AlgebraicOrder

    @property
    def verifies_discriminant_index_formula(self) -> bool:
        """Whether the monic discriminant ideals satisfy the index formula."""
        return (
            self.previous_discriminant.monic()
            == (self.new_discriminant * self.index**2).monic()
        )

    @property
    def verifies(self) -> bool:
        """Independently verify this trace-radical order enlargement."""
        try:
            prime = _normalize_base_prime(
                self.previous_order.extension,
                self.prime,
            )
            if prime != self.prime:
                return False
            if self.previous_order.extension != self.new_order.extension:
                return False
            if self.previous_discriminant != self.previous_order.discriminant:
                return False
            if self.new_discriminant != self.new_order.discriminant:
                return False
            if not self.new_order.contains_order(self.previous_order):
                return False
            if self.new_order.equivalent(self.previous_order):
                return False
            if self.index != self.new_order.index_of(self.previous_order):
                return False
            expected_idealizer = self.previous_order.idealizer(
                self.previous_order.trace_radical(prime)
            )
            return (
                self.new_order.equivalent(expected_idealizer)
                and self.verifies_discriminant_index_formula
            )
        except (NotImplementedError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class OrderMaximalityCertificate:
    """One final ``prime``-maximality certificate."""

    order: AlgebraicOrder
    prime: Poly
    trace_radical: AlgebraicFractionalIdeal
    idealizer: AlgebraicOrder

    @property
    def verifies(self) -> bool:
        """Independently verify that the order is maximal at ``prime``."""
        try:
            prime = _normalize_base_prime(self.order.extension, self.prime)
            if prime != self.prime or self.trace_radical.order is not self.order:
                return False
            expected_radical = self.order.trace_radical(prime)
            if self.trace_radical != expected_radical:
                return False
            expected_idealizer = self.order.idealizer(expected_radical)
            return self.idealizer.equivalent(
                expected_idealizer
            ) and expected_idealizer.equivalent(self.order)
        except (NotImplementedError, TypeError, ValueError):
            return False


@dataclass(frozen=True, slots=True)
class IntegralClosureResult:
    """Certified maximal order and its trace-radical refinement history."""

    initial_order: AlgebraicOrder
    order: AlgebraicOrder
    steps: tuple[IntegralClosureStep, ...]
    tested_primes: tuple[Poly, ...]
    maximality_certificates: tuple[OrderMaximalityCertificate, ...]

    @property
    def changed(self) -> bool:
        """Whether the initial order required enlargement."""
        return bool(self.steps)

    @property
    def is_certified(self) -> bool:
        """Whether refinement and final prime-maximality checks all verify."""
        try:
            if self.initial_order.extension != self.order.extension:
                return False

            current = self.initial_order
            for step in self.steps:
                if not step.previous_order.equivalent(current) or not step.verifies:
                    return False
                current = step.new_order
            if not current.equivalent(self.order):
                return False

            _, factors = self.order.discriminant.factor_list()
            expected_primes = {
                factor.monic() for factor, multiplicity in factors if multiplicity >= 2
            }
            certified_primes = [
                certificate.prime for certificate in self.maximality_certificates
            ]
            if (
                len(certified_primes) != len(expected_primes)
                or set(certified_primes) != expected_primes
            ):
                return False
            return all(
                certificate.order.equivalent(self.order) and certificate.verifies
                for certificate in self.maximality_certificates
            )
        except (NotImplementedError, TypeError, ValueError):
            return False


def polynomial_order(extension: AlgebraicExtension) -> AlgebraicOrder:
    r"""Construct an integral power order from an arbitrary monic model.

    If the defining polynomial has rational-function coefficients, multiplying
    its generator by one common coefficient denominator produces an integral
    generator.  The resulting power basis is therefore a valid starting order
    for maximal-order refinement.
    """
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("a polynomial order requires an AlgebraicExtension")
    parent, constants, variable = _base_polynomial_ring(extension)
    coefficients = cast(
        tuple[FracElement, ...],
        tuple(
            parent.convert(extension.minimal_polynomial.nth(degree))
            for degree in range(extension.minimal_polynomial.degree())
        ),
    )
    denominator = _common_denominator(coefficients, constants, variable)
    denominator_element = _embed_parent(extension, denominator)
    integral_generator = extension.field.generator * denominator_element
    elements = tuple(integral_generator**power for power in range(extension.field.rank))
    return AlgebraicOrder(IntegralBasis(extension, elements))


def integral_closure(
    extension: AlgebraicExtension,
    *,
    initial_order: AlgebraicOrder | None = None,
    max_refinements: int = 256,
) -> IntegralClosureResult:
    r"""Compute the integral closure of ``C[x]`` in ``extension``.

    In characteristic zero, the radical of ``prime*order`` is the radical of
    the trace pairing modulo ``prime``.  The order is ``prime``-maximal exactly
    when the idealizer of that radical is the order itself.  Repeating strict
    idealizer enlargements over every prime whose square divides the
    discriminant terminates at the maximal order.
    """
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("integral closure requires an AlgebraicExtension")
    _, constants, _ = _base_polynomial_ring(extension)
    if constants.characteristic() != 0:
        raise NotImplementedError(
            "trace-radical maximal orders currently require characteristic zero"
        )
    if not isinstance(max_refinements, int) or max_refinements <= 0:
        raise ValueError("max_refinements must be a positive integer")

    current = polynomial_order(extension) if initial_order is None else initial_order
    if current.extension != extension:
        raise ValueError("the initial order belongs to a different extension")
    start = current
    steps: list[IntegralClosureStep] = []
    tested_primes: list[Poly] = []
    final_certificates: tuple[OrderMaximalityCertificate, ...] = ()

    while True:
        enlarged = False
        round_certificates: list[OrderMaximalityCertificate] = []
        _, factors = current.discriminant.factor_list()
        candidates = tuple(
            factor.monic() for factor, multiplicity in factors if multiplicity >= 2
        )
        for prime in candidates:
            tested_primes.append(prime)
            radical = current.trace_radical(prime)
            idealizer = current.idealizer(radical)
            if idealizer.equivalent(current):
                round_certificates.append(
                    OrderMaximalityCertificate(
                        order=current,
                        prime=prime,
                        trace_radical=radical,
                        idealizer=idealizer,
                    )
                )
                continue
            if len(steps) >= max_refinements:
                raise RuntimeError("maximal-order refinement exceeded max_refinements")
            if not idealizer.contains_order(current):
                raise ValueError("an idealizer failed to contain its source order")
            index = idealizer.index_of(current)
            step = IntegralClosureStep(
                prime=prime,
                previous_discriminant=current.discriminant,
                new_discriminant=idealizer.discriminant,
                index=index,
                previous_order=current,
                new_order=idealizer,
            )
            if not step.verifies_discriminant_index_formula:
                raise ValueError(
                    "order enlargement violated the discriminant-index identity"
                )
            steps.append(step)
            current = idealizer
            enlarged = True
            break
        if not enlarged:
            final_certificates = tuple(round_certificates)
            break

    result = IntegralClosureResult(
        initial_order=start,
        order=current,
        steps=tuple(steps),
        tested_primes=tuple(tested_primes),
        maximality_certificates=final_certificates,
    )
    if not result.is_certified:
        raise ValueError("maximal-order refinement certificate is inconsistent")
    return result
