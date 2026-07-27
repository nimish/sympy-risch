r"""Good reduction of algebraic function fields and divisors modulo primes.

This module is neutral algebraic-function-field infrastructure.  It reduces a
``QQ(x)[y]`` presentation coefficientwise to ``GF(p)(x)[y]``, rebuilds both
maximal orders, and verifies that the source order bases reduce to the target
orders.  A divisor is reduced through its finite and reciprocal-chart prime
ideals, so higher-degree places may split without any point-matching
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.ntheory.primetest import isprime
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import GF, QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly

from ..algebraic._field import (
    _base_polynomial_ring,
    _extension_element,
    _polynomial_parent_element,
    _same_univariate_polynomial_coefficients,
)
from ..algebraic.basis import IntegralBasis
from ..tower import AlgebraicExtension
from .finite_places import FiniteAlgebraicPlace
from .function_field import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
)
from .infinity import InfiniteAlgebraicPlace
from .orders import AlgebraicFractionalIdeal, AlgebraicOrder


class BadAlgebraicReduction(ValueError):
    """The selected rational prime does not give a supported good model."""


def _reduce_rational(
    value: object,
    characteristic: int,
    target_constants: Domain,
) -> object:
    rational = QQ.convert(value)
    numerator = int(QQ.numer(rational))
    denominator = int(QQ.denom(rational))
    if denominator % characteristic == 0:
        raise BadAlgebraicReduction(
            "a rational coefficient denominator vanishes modulo the prime"
        )
    return target_constants.exquo(
        target_constants.convert(numerator),
        target_constants.convert(denominator),
    )


def _reduce_parent_element(
    value: object,
    source_parent: FractionField,
    target_parent: FractionField,
    characteristic: int,
) -> FracElement:
    """Reduce one element of ``QQ(x)`` into ``GF(p)(x)``."""
    source = cast(FracElement, source_parent.convert(value))
    symbols = target_parent.symbols
    target_constants = target_parent.domain
    if symbols is None or len(symbols) != 1 or target_constants is None:
        raise BadAlgebraicReduction("the target is not a univariate function field")
    variable = symbols[0]
    checked_constants = cast(Domain, target_constants)

    def reduce_polynomial(terms: dict[tuple[int, ...], object]) -> Poly:
        return Poly.from_dict(
            {
                monomial: _reduce_rational(
                    coefficient,
                    characteristic,
                    checked_constants,
                )
                for monomial, coefficient in terms.items()
            },
            variable,
            domain=checked_constants,
        )

    numerator = reduce_polynomial(source.numer.to_dict())
    denominator = reduce_polynomial(source.denom.to_dict())
    if denominator.is_zero:
        raise BadAlgebraicReduction(
            "a rational-function denominator vanishes modulo the prime"
        )
    return cast(
        FracElement,
        _polynomial_parent_element(numerator, target_parent)
        / _polynomial_parent_element(denominator, target_parent),
    )


def _reduce_extension_polynomial(
    source: AlgebraicExtension,
    target_parent: FractionField,
    characteristic: int,
) -> Poly:
    source_parent, constants, _variable = _base_polynomial_ring(source)
    if constants != QQ:
        raise NotImplementedError(
            "finite reduction currently requires a function field over QQ"
        )
    return Poly.from_dict(
        {
            (power,): _reduce_parent_element(
                source.minimal_polynomial.nth(power),
                source_parent,
                target_parent,
                characteristic,
            )
            for power in range(source.minimal_polynomial.degree() + 1)
        },
        source.generator,
        domain=target_parent,
    )


def _reduce_element_between(
    source: AlgebraicExtension,
    target: AlgebraicExtension,
    value: object,
    characteristic: int,
) -> ExtensionElement:
    source_parent, source_constants, source_variable = _base_polynomial_ring(source)
    target_parent, target_constants, target_variable = _base_polynomial_ring(target)
    if (
        source_constants != QQ
        or source_variable != target_variable
        or target_constants.characteristic() != characteristic
        or source.field.rank != target.field.rank
    ):
        raise BadAlgebraicReduction("the source and target presentations do not align")
    element = _extension_element(source, value)
    return target.from_power_basis_coordinates(
        tuple(
            _reduce_parent_element(
                source_parent.convert(element.rep.nth(power)),
                source_parent,
                target_parent,
                characteristic,
            )
            for power in range(source.field.rank)
        )
    )


def _reduced_order(
    source: AlgebraicOrder,
    target_extension: AlgebraicExtension,
    characteristic: int,
) -> AlgebraicOrder:
    return AlgebraicOrder(
        IntegralBasis(
            target_extension,
            tuple(
                _reduce_element_between(
                    source.extension,
                    target_extension,
                    element,
                    characteristic,
                )
                for element in source.basis.elements
            ),
        )
    )


def _reduce_ideal(
    source: AlgebraicFractionalIdeal,
    target_extension: AlgebraicExtension,
    target_order: AlgebraicOrder,
    characteristic: int,
) -> AlgebraicFractionalIdeal:
    return AlgebraicFractionalIdeal(
        target_order,
        tuple(
            _reduce_element_between(
                source.order.extension,
                target_extension,
                element,
                characteristic,
            )
            for element in source.basis_elements
        ),
    )


@dataclass(frozen=True, slots=True)
class AlgebraicFunctionFieldReduction:
    """A verified good reduction of one ``QQ`` function-field model."""

    source: AlgebraicFunctionField
    prime: int
    target: AlgebraicFunctionField

    def reduce_element(self, value: object) -> ExtensionElement:
        """Reduce one source field element coefficientwise."""
        return _reduce_element_between(
            self.source.extension,
            self.target.extension,
            value,
            self.prime,
        )

    def reduce_divisor(self, divisor: AlgebraicDivisor) -> AlgebraicDivisor:
        """Reduce a divisor through finite and reciprocal prime ideals."""
        if not isinstance(divisor, AlgebraicDivisor):
            raise TypeError("finite reduction needs an algebraic divisor")
        if divisor.model is not self.source:
            raise ValueError("the divisor belongs to another source model")

        terms: list[tuple[AlgebraicPlace, int]] = []
        source_chart = self.source.reciprocal_chart
        target_chart = self.target.reciprocal_chart
        target_infinities = self.target.infinite_places()
        reciprocal_model = AlgebraicFunctionField(
            target_chart.extension,
            closure=target_chart.closure,
        )

        for place, coefficient in divisor.terms:
            if isinstance(place, InfiniteAlgebraicPlace):
                reduced_ideal = _reduce_ideal(
                    place.finite_place.prime_ideal,
                    target_chart.extension,
                    target_chart.closure.order,
                    self.prime,
                )
                reduced_divisor = reciprocal_model.ideal_divisor(reduced_ideal)
                for finite_place, multiplicity in reduced_divisor.terms:
                    if not isinstance(finite_place, FiniteAlgebraicPlace):
                        raise BadAlgebraicReduction(
                            "a reciprocal finite ideal produced an infinity place"
                        )
                    matches = tuple(
                        infinity
                        for infinity in target_infinities
                        if infinity.finite_place.prime_ideal == finite_place.prime_ideal
                    )
                    if len(matches) != 1:
                        raise BadAlgebraicReduction(
                            "an infinity prime did not map uniquely to the target chart"
                        )
                    terms.append((matches[0], coefficient * multiplicity))
                continue

            reduced_ideal = _reduce_ideal(
                place.prime_ideal,
                self.target.extension,
                self.target.order,
                self.prime,
            )
            for target_place, multiplicity in self.target.ideal_divisor(
                reduced_ideal
            ).terms:
                terms.append((target_place, coefficient * multiplicity))

        result = AlgebraicDivisor(self.target, cast(tuple, tuple(terms)))
        if result.degree != divisor.degree:
            raise BadAlgebraicReduction(
                "divisor degree changed under the proposed reduction"
            )
        if source_chart.scale_exponent != target_chart.scale_exponent:
            raise BadAlgebraicReduction(
                "the reciprocal generator scaling changed under reduction"
            )
        return result


def reduce_algebraic_function_field(
    model: AlgebraicFunctionField,
    prime: int,
) -> AlgebraicFunctionFieldReduction:
    """Construct and verify one supported good reduction modulo ``prime``."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("finite reduction needs an algebraic function field")
    if not isinstance(prime, int) or not isprime(prime):
        raise ValueError("the reduction characteristic must be a rational prime")
    _source_parent, constants, variable = _base_polynomial_ring(model.extension)
    if constants != QQ:
        raise NotImplementedError(
            "finite reduction currently requires a function field over QQ"
        )

    finite_constants = GF(prime)
    target_parent = finite_constants.frac_field(variable)
    try:
        target_extension = AlgebraicExtension(
            _reduce_extension_polynomial(
                model.extension,
                target_parent,
                prime,
            )
        )
        target = AlgebraicFunctionField(
            target_extension,
            infinity_variable=model.reciprocal_chart.variable,
        )
        target.infinite_places()
    except (NotImplementedError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise BadAlgebraicReduction(
            f"the curve has unsupported bad reduction modulo {prime}: {exc}"
        ) from exc

    reduced_finite_order = _reduced_order(model.order, target.extension, prime)
    if not reduced_finite_order.equivalent(target.order):
        raise BadAlgebraicReduction(
            "the finite maximal-order basis did not survive reduction"
        )

    source_chart = model.reciprocal_chart
    target_chart = target.reciprocal_chart
    reduced_reciprocal_polynomial = _reduce_extension_polynomial(
        source_chart.extension,
        cast(FractionField, target_chart.extension.parent_field),
        prime,
    )
    if not _same_univariate_polynomial_coefficients(
        reduced_reciprocal_polynomial,
        target_chart.extension.minimal_polynomial,
    ):
        raise BadAlgebraicReduction(
            "the reciprocal presentation did not commute with reduction"
        )
    reduced_infinite_order = _reduced_order(
        source_chart.closure.order,
        target_chart.extension,
        prime,
    )
    if not reduced_infinite_order.equivalent(target_chart.closure.order):
        raise BadAlgebraicReduction(
            "the reciprocal maximal-order basis did not survive reduction"
        )
    if target.genus != model.genus:
        raise BadAlgebraicReduction("the genus changed under reduction")

    return AlgebraicFunctionFieldReduction(model, prime, target)


__all__ = [
    "AlgebraicFunctionFieldReduction",
    "BadAlgebraicReduction",
    "reduce_algebraic_function_field",
]
