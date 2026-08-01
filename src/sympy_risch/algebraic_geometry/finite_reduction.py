r"""Good reduction of algebraic function fields and divisors modulo primes.

This module is neutral algebraic-function-field infrastructure.  It reduces a
``C(x)[y]`` presentation coefficientwise at a certified finite prime of
``C = QQ`` or ``QQ(alpha)``, rebuilds both maximal orders, and verifies that
the source order bases reduce to the target orders.  A divisor is reduced
through its finite and reciprocal-chart prime ideals, so higher-degree places
may split without any point-matching assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
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
from .constant_reduction import (
    ConstantFieldPrimeReduction,
    constant_field_prime_reductions,
)
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


def _reduce_parent_element(
    value: object,
    source_parent: FractionField,
    target_parent: FractionField,
    constant_reduction: ConstantFieldPrimeReduction,
) -> FracElement:
    """Reduce one rational-function element coefficientwise."""
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
                monomial: checked_constants.convert(
                    constant_reduction.reduce(coefficient)
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
    constant_reduction: ConstantFieldPrimeReduction,
) -> Poly:
    source_parent, constants, _variable = _base_polynomial_ring(source)
    if constants is not constant_reduction.source:
        raise BadAlgebraicReduction(
            "the constant-field reduction belongs to another source domain"
        )
    return Poly.from_dict(
        {
            (power,): _reduce_parent_element(
                source.minimal_polynomial.nth(power),
                source_parent,
                target_parent,
                constant_reduction,
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
    constant_reduction: ConstantFieldPrimeReduction,
) -> ExtensionElement:
    source_parent, source_constants, source_variable = _base_polynomial_ring(source)
    target_parent, target_constants, target_variable = _base_polynomial_ring(target)
    if (
        source_constants is not constant_reduction.source
        or source_variable != target_variable
        or target_constants != constant_reduction.target
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
                constant_reduction,
            )
            for power in range(source.field.rank)
        )
    )


def _reduced_order(
    source: AlgebraicOrder,
    target_extension: AlgebraicExtension,
    constant_reduction: ConstantFieldPrimeReduction,
) -> AlgebraicOrder:
    return AlgebraicOrder(
        IntegralBasis(
            target_extension,
            tuple(
                _reduce_element_between(
                    source.extension,
                    target_extension,
                    element,
                    constant_reduction,
                )
                for element in source.basis.elements
            ),
        )
    )


def _reduce_ideal(
    source: AlgebraicFractionalIdeal,
    target_extension: AlgebraicExtension,
    target_order: AlgebraicOrder,
    constant_reduction: ConstantFieldPrimeReduction,
) -> AlgebraicFractionalIdeal:
    return AlgebraicFractionalIdeal(
        target_order,
        tuple(
            _reduce_element_between(
                source.order.extension,
                target_extension,
                element,
                constant_reduction,
            )
            for element in source.basis_elements
        ),
    )


@dataclass(frozen=True, slots=True)
class AlgebraicFunctionFieldReduction:
    """A verified good reduction of one characteristic-zero field model."""

    source: AlgebraicFunctionField
    constant_reduction: ConstantFieldPrimeReduction
    target: AlgebraicFunctionField

    def __post_init__(self) -> None:
        if not isinstance(self.source, AlgebraicFunctionField):
            raise TypeError("a function-field reduction needs a source model")
        if not isinstance(self.constant_reduction, ConstantFieldPrimeReduction):
            raise TypeError("a function-field reduction needs a constant residue map")
        if not isinstance(self.target, AlgebraicFunctionField):
            raise TypeError("a function-field reduction needs a target model")
        _source_parent, constants, _variable = _base_polynomial_ring(
            self.source.extension
        )
        if self.constant_reduction.source is not constants:
            raise ValueError("the finite prime belongs to another constant field")
        if self.constant_reduction.residue_degree != 1:
            raise NotImplementedError(
                "curve reduction over non-prime finite residue fields awaits a "
                "GF(p^f)(x) factorization backend"
            )

        target_parent = cast(FractionField, self.target.extension.parent_field)
        reduced_polynomial = _reduce_extension_polynomial(
            self.source.extension,
            target_parent,
            self.constant_reduction,
        )
        if not _same_univariate_polynomial_coefficients(
            reduced_polynomial,
            self.target.extension.minimal_polynomial,
        ):
            raise BadAlgebraicReduction(
                "the target equation is not the reduction of the source equation"
            )
        reduced_finite_order = _reduced_order(
            self.source.order,
            self.target.extension,
            self.constant_reduction,
        )
        if not reduced_finite_order.equivalent(self.target.order):
            raise BadAlgebraicReduction(
                "the finite maximal-order basis did not survive reduction"
            )

        source_chart = self.source.reciprocal_chart
        target_chart = self.target.reciprocal_chart
        reduced_reciprocal_polynomial = _reduce_extension_polynomial(
            source_chart.extension,
            cast(FractionField, target_chart.extension.parent_field),
            self.constant_reduction,
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
            self.constant_reduction,
        )
        if not reduced_infinite_order.equivalent(target_chart.closure.order):
            raise BadAlgebraicReduction(
                "the reciprocal maximal-order basis did not survive reduction"
            )
        if self.target.genus != self.source.genus:
            raise BadAlgebraicReduction("the genus changed under reduction")

    @property
    def prime(self) -> int:
        """Return the rational residue characteristic."""
        return self.constant_reduction.rational_prime

    @property
    def field_size(self) -> int:
        """Return the cardinality of the target constant field."""
        return self.constant_reduction.field_size

    def reduce_element(self, value: object) -> ExtensionElement:
        """Reduce one source field element coefficientwise."""
        return _reduce_element_between(
            self.source.extension,
            self.target.extension,
            value,
            self.constant_reduction,
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
                    self.constant_reduction,
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
                self.constant_reduction,
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
    prime: int | ConstantFieldPrimeReduction,
) -> AlgebraicFunctionFieldReduction:
    """Construct and verify one supported good reduction at a finite prime."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("finite reduction needs an algebraic function field")
    _source_parent, constants, variable = _base_polynomial_ring(model.extension)
    if isinstance(prime, int):
        candidates = constant_field_prime_reductions(constants, prime)
        if not candidates:
            raise BadAlgebraicReduction("no supported prime lies above the input")
        if len(candidates) == 1:
            return reduce_algebraic_function_field(model, candidates[0])
        failures: list[str] = []
        for candidate in candidates:
            try:
                return reduce_algebraic_function_field(model, candidate)
            except (BadAlgebraicReduction, NotImplementedError) as exc:
                failures.append(str(exc))
        raise BadAlgebraicReduction(
            "no prime above the selected characteristic gave good reduction: "
            + "; ".join(failures)
        )
    elif isinstance(prime, ConstantFieldPrimeReduction):
        constant_reduction = prime
        if constant_reduction.source is not constants:
            raise ValueError("the finite prime belongs to another constant field")
    else:
        raise TypeError("finite reduction needs a rational prime or residue map")

    if constant_reduction.residue_degree != 1:
        raise NotImplementedError(
            "curve reduction over non-prime finite residue fields awaits a "
            "GF(p^f)(x) factorization backend"
        )

    target_parent = constant_reduction.target.frac_field(variable)
    try:
        target_extension = AlgebraicExtension(
            _reduce_extension_polynomial(
                model.extension,
                target_parent,
                constant_reduction,
            )
        )
        target = AlgebraicFunctionField(
            target_extension,
            infinity_variable=model.reciprocal_chart.variable,
        )
        target.infinite_places()
    except (NotImplementedError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise BadAlgebraicReduction(
            "the curve has unsupported bad reduction at characteristic "
            f"{constant_reduction.rational_prime}: {exc}"
        ) from exc

    return AlgebraicFunctionFieldReduction(
        model,
        constant_reduction,
        target,
    )


__all__ = [
    "AlgebraicFunctionFieldReduction",
    "BadAlgebraicReduction",
    "reduce_algebraic_function_field",
]
