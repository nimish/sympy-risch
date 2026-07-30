r"""Global algebraic function fields, divisors, and differentials."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import cached_property
from typing import cast

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polytools import Poly

from ..algebraic._field import _base_polynomial_ring, _extension_element
from ..tower import AlgebraicExtension
from ._residue_field import _normalize_base_prime
from .closure import IntegralClosureResult, integral_closure
from .finite_places import (
    FiniteAlgebraicPlace,
    _decompose_finite_places,
    _element_base_prime_support,
    _fractional_ideal_base_prime_support,
    _integral_ideal_scale,
)
from .infinity import InfiniteAlgebraicPlace, ReciprocalFunctionField
from .orders import AlgebraicFractionalIdeal


class AlgebraicFunctionField:
    r"""Certified global model for one finite extension of ``C(x)``."""

    def __init__(
        self,
        extension: AlgebraicExtension,
        *,
        closure: IntegralClosureResult | None = None,
        infinity_variable: Symbol | None = None,
    ) -> None:
        if not isinstance(extension, AlgebraicExtension):
            raise TypeError("an algebraic function field needs an extension")
        computed_closure = integral_closure(extension) if closure is None else closure
        if computed_closure.order.extension != extension:
            raise ValueError("the maximal-order certificate is for another field")
        if not computed_closure.is_certified:
            raise ValueError("the finite maximal order is not certified")
        self.extension = extension
        self.closure = computed_closure
        self.order = computed_closure.order
        self._infinity_variable = infinity_variable
        self._finite_cache: dict[Poly, tuple[FiniteAlgebraicPlace, ...]] = {}
        self._reciprocal_chart: ReciprocalFunctionField | None = None
        self._infinite_cache: tuple[InfiniteAlgebraicPlace, ...] | None = None

    def finite_places(
        self,
        base_prime: Poly,
    ) -> tuple[FiniteAlgebraicPlace, ...]:
        """Return every closed place above an irreducible finite base prime."""
        prime = _normalize_base_prime(self.extension, base_prime)
        places = self._finite_cache.get(prime)
        if places is None:
            places = _decompose_finite_places(self.order, prime)
            self._finite_cache[prime] = places
        return places

    @property
    def reciprocal_chart(self) -> ReciprocalFunctionField:
        """Reciprocal model used for the points at infinity."""
        if self._reciprocal_chart is None:
            self._reciprocal_chart = ReciprocalFunctionField.create(
                self.extension,
                self._infinity_variable,
            )
        return self._reciprocal_chart

    def infinite_places(self) -> tuple[InfiniteAlgebraicPlace, ...]:
        """Return every closed place over the point at infinity."""
        if self._infinite_cache is None:
            chart = self.reciprocal_chart
            _, constants, variable = _base_polynomial_ring(chart.extension)
            reciprocal_prime = Poly(variable, variable, domain=constants)
            finite_places = _decompose_finite_places(
                chart.closure.order,
                reciprocal_prime,
            )
            self._infinite_cache = tuple(
                InfiniteAlgebraicPlace(chart, place) for place in finite_places
            )
        return self._infinite_cache

    def base_parameter_pole_divisor(self) -> AlgebraicDivisor:
        r"""Return the full pole divisor ``(x)_infinity``."""
        return AlgebraicDivisor(
            self,
            tuple(
                (place, place.ramification_index) for place in self.infinite_places()
            ),
        )

    def principal_divisor(self, element: object) -> AlgebraicDivisor:
        """Return the complete divisor of a nonzero field element."""
        value = _extension_element(self.extension, element)
        if value == self.extension.field.zero:
            raise ValueError("the zero element has no principal divisor")

        terms: list[tuple[AlgebraicPlace, int]] = []
        for prime in _element_base_prime_support(self.order, value):
            for place in self.finite_places(prime):
                valuation = place.valuation(value)
                if valuation is S.Infinity:
                    raise ValueError("a nonzero element had infinite valuation")
                if valuation:
                    terms.append((place, cast(int, valuation)))
        for place in self.infinite_places():
            valuation = place.valuation(value)
            if valuation is S.Infinity:
                raise ValueError("a nonzero element had infinite valuation")
            if valuation:
                terms.append((place, cast(int, valuation)))

        divisor = AlgebraicDivisor(self, tuple(terms))
        if divisor.degree != 0:
            raise ValueError("a principal divisor failed the degree-zero identity")
        return divisor

    def ideal_divisor(
        self,
        ideal: AlgebraicFractionalIdeal,
    ) -> AlgebraicDivisor:
        """Factor a fractional ideal into its finite-place divisor."""
        if ideal.order is not self.order:
            raise ValueError("ideal factorization requires the model's maximal order")
        denominator, scaled = _integral_ideal_scale(ideal)
        terms: list[tuple[AlgebraicPlace, int]] = []
        for prime in _fractional_ideal_base_prime_support(
            ideal,
            denominator,
            scaled,
        ):
            for place in self.finite_places(prime):
                valuation = place._scaled_ideal_valuation(
                    scaled,
                    denominator,
                )
                if valuation:
                    terms.append((place, valuation))
        return AlgebraicDivisor(self, tuple(terms))

    def finite_ideal(
        self,
        divisor: AlgebraicDivisor,
    ) -> AlgebraicFractionalIdeal:
        """Convert a divisor with finite support into a fractional ideal."""
        if not isinstance(divisor, AlgebraicDivisor):
            raise TypeError("finite ideal construction requires a divisor")
        if divisor.model is not self:
            raise ValueError("the divisor belongs to another global model")
        if any(isinstance(place, InfiniteAlgebraicPlace) for place, _ in divisor.terms):
            raise ValueError("a finite-order ideal cannot encode infinity support")

        result = self.order.unit_ideal()
        for place, coefficient in divisor.terms:
            if not isinstance(place, FiniteAlgebraicPlace):
                raise TypeError("finite divisor support has an invalid place")
            result = result.multiply(place.prime_ideal.power(coefficient))
        if self.ideal_divisor(result) != divisor:
            raise ValueError("finite divisor-to-ideal conversion failed to round-trip")
        return result

    def canonical_divisor(self) -> AlgebraicDivisor:
        r"""Return the divisor of ``dx`` via the relative different."""
        terms: list[tuple[AlgebraicPlace, int]] = []
        _, factors = self.order.discriminant.factor_list()
        for prime, _ in factors:
            for place in self.finite_places(prime.monic()):
                exponent = place.different_exponent
                if exponent:
                    terms.append((place, exponent))
        for place in self.infinite_places():
            exponent = place.different_exponent - 2 * place.ramification_index
            if exponent:
                terms.append((place, exponent))
        return AlgebraicDivisor(self, tuple(terms))

    @cached_property
    def genus(self) -> int:
        """Genus obtained from the degree of a canonical divisor."""
        canonical_degree = self.canonical_divisor().degree
        if (canonical_degree + 2) % 2:
            raise ValueError("the canonical divisor has impossible odd degree")
        genus = (canonical_degree + 2) // 2
        if genus < 0:
            raise ValueError("the canonical divisor gives a negative genus")
        return genus

    def differential(self, coefficient: object) -> AlgebraicDifferential:
        """Construct the differential ``coefficient*dx``."""
        return AlgebraicDifferential(self, coefficient)


AlgebraicPlace = FiniteAlgebraicPlace | InfiniteAlgebraicPlace


@dataclass(frozen=True, slots=True, eq=False)
class AlgebraicDivisor:
    """An immutable formal integer sum of model-owned closed places."""

    model: AlgebraicFunctionField
    terms: tuple[tuple[AlgebraicPlace, int], ...] = ()
    _coefficients: dict[AlgebraicPlace, int] = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        combined: dict[AlgebraicPlace, int] = {}
        order: list[AlgebraicPlace] = []
        for place, coefficient in self.terms:
            if not isinstance(
                place,
                (FiniteAlgebraicPlace, InfiniteAlgebraicPlace),
            ):
                raise TypeError("divisor support must consist of algebraic places")
            if place.extension != self.model.extension:
                raise ValueError("a divisor place belongs to another function field")
            if (
                isinstance(place, FiniteAlgebraicPlace)
                and place.order is not self.model.order
            ):
                raise ValueError("a finite place belongs to another global model")
            if (
                isinstance(place, InfiniteAlgebraicPlace)
                and place.chart is not self.model.reciprocal_chart
            ):
                raise ValueError("an infinite place belongs to another global model")
            if not isinstance(coefficient, int):
                raise TypeError("divisor coefficients must be integers")
            if place not in combined:
                order.append(place)
                combined[place] = 0
            combined[place] += coefficient
        coefficients = {place: combined[place] for place in order if combined[place]}
        object.__setattr__(self, "terms", tuple(coefficients.items()))
        object.__setattr__(self, "_coefficients", coefficients)

    @property
    def support(self) -> tuple[AlgebraicPlace, ...]:
        """Places with nonzero coefficient."""
        return tuple(place for place, _ in self.terms)

    @property
    def degree(self) -> int:
        """Weighted divisor degree over the constant field."""
        return sum(coefficient * place.degree for place, coefficient in self.terms)

    @property
    def is_zero(self) -> bool:
        """Whether the divisor has empty support."""
        return not self.terms

    @property
    def is_effective(self) -> bool:
        """Whether every coefficient is nonnegative."""
        return all(coefficient >= 0 for _, coefficient in self.terms)

    def coefficient(self, place: AlgebraicPlace) -> int:
        """Coefficient at one place."""
        return self._coefficients.get(place, 0)

    def add(self, other: AlgebraicDivisor) -> AlgebraicDivisor:
        """Divisor addition."""
        self._require_same_model(other)
        return AlgebraicDivisor(self.model, self.terms + other.terms)

    def __add__(self, other: AlgebraicDivisor) -> AlgebraicDivisor:
        return self.add(other)

    def __neg__(self) -> AlgebraicDivisor:
        return AlgebraicDivisor(
            self.model,
            tuple((place, -coefficient) for place, coefficient in self.terms),
        )

    def __sub__(self, other: AlgebraicDivisor) -> AlgebraicDivisor:
        return self.add(-other)

    def scale(self, coefficient: int) -> AlgebraicDivisor:
        """Multiply every divisor coefficient by an integer."""
        if not isinstance(coefficient, int):
            raise TypeError("a divisor scalar must be an integer")
        return AlgebraicDivisor(
            self.model,
            tuple(
                (place, coefficient * place_coefficient)
                for place, place_coefficient in self.terms
            ),
        )

    @property
    def positive_part(self) -> AlgebraicDivisor:
        """Effective positive part."""
        return AlgebraicDivisor(
            self.model,
            tuple(
                (place, coefficient)
                for place, coefficient in self.terms
                if coefficient > 0
            ),
        )

    @property
    def negative_part(self) -> AlgebraicDivisor:
        """Effective pole part, with signs reversed."""
        return AlgebraicDivisor(
            self.model,
            tuple(
                (place, -coefficient)
                for place, coefficient in self.terms
                if coefficient < 0
            ),
        )

    def _require_same_model(self, other: AlgebraicDivisor) -> None:
        if not isinstance(other, AlgebraicDivisor):
            raise TypeError("divisor arithmetic needs another divisor")
        if other.model is not self.model:
            raise ValueError("divisors must belong to the same global model")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AlgebraicDivisor):
            return NotImplemented
        if other.model is not self.model:
            return False
        return self._coefficients == other._coefficients


@dataclass(frozen=True, slots=True)
class AlgebraicDifferential:
    r"""A differential represented globally as ``coefficient*dx``.

    The private caches memoize exact divisor and local-residue computations.
    They are not part of the mathematical value or equality; they let a
    collector validate the data it just computed without repeating expensive
    global and local algebra.
    """

    model: AlgebraicFunctionField
    coefficient: object
    _divisor_cache: AlgebraicDivisor | None = dataclass_field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _residue_cache: dict[AlgebraicPlace, ExtensionElement] = dataclass_field(
        init=False,
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient",
            _extension_element(self.model.extension, self.coefficient),
        )

    def order_at(self, place: AlgebraicPlace) -> int | Expr:
        """Order of the differential at one finite or infinite place."""
        if place.extension != self.model.extension:
            raise ValueError("the place belongs to another function field")
        valuation = place.valuation(self.coefficient)
        if valuation is S.Infinity:
            return S.Infinity
        if isinstance(place, FiniteAlgebraicPlace):
            return cast(int, valuation) + place.different_exponent
        return (
            cast(int, valuation)
            + place.different_exponent
            - 2 * place.ramification_index
        )

    def residue(self, place: AlgebraicPlace) -> ExtensionElement:
        """Exact local residue in the place's residue field."""
        if place.extension != self.model.extension:
            raise ValueError("the place belongs to another function field")
        cached = self._residue_cache.get(place)
        if cached is not None:
            return cached
        value = place.differential_residue(self.coefficient)
        self._residue_cache[place] = value
        return value

    def divisor(self) -> AlgebraicDivisor:
        """Complete divisor of this nonzero differential."""
        if self.coefficient == self.model.extension.field.zero:
            raise ValueError("the zero differential has no finite divisor")
        if self._divisor_cache is not None:
            return self._divisor_cache
        divisor = (
            self.model.principal_divisor(self.coefficient)
            + self.model.canonical_divisor()
        )
        object.__setattr__(self, "_divisor_cache", divisor)
        return divisor
