r"""Finite places, valuations, residue maps, and Laurent expansions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import count, product
from typing import Protocol, cast

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.agca.extensions import ExtensionElement, FiniteExtension
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..algebraic._field import (
    _base_polynomial_ring,
    _common_denominator,
    _embed_parent,
    _extension_element,
    _polynomial_parent_element,
)
from ..tower import AlgebraicExtension
from ._residue_field import _BaseResidueContext
from .orders import (
    AlgebraicFractionalIdeal,
    AlgebraicOrder,
    _matrix_from_columns,
)


class _LaurentPlace(Protocol):
    """Structural interface shared by finite and infinite local places."""

    @property
    def residue_field(self) -> FiniteExtension: ...

    @property
    def uniformizer(self) -> ExtensionElement: ...

    def lift_residue(self, value: object) -> ExtensionElement: ...


def _reduce_order_coordinates(
    order: AlgebraicOrder,
    element: object,
    context: _BaseResidueContext,
) -> tuple[object, ...]:
    return tuple(
        context.reduce_polynomial(coordinate)
        for coordinate in order.polynomial_coordinates(element)
    )


def _reduce_polynomial_matrix(
    matrix: DomainMatrix,
    context: _BaseResidueContext,
) -> DomainMatrix:
    variable = context.prime.gens[0]
    constants = context.prime.domain
    return DomainMatrix(
        [
            [
                context.reduce_polynomial(
                    Poly(
                        matrix.domain.to_sympy(entry),
                        variable,
                        domain=constants,
                    )
                )
                for entry in row
            ]
            for row in matrix.to_list()
        ],
        matrix.shape,
        context.field,
    )


@dataclass(frozen=True, slots=True)
class _SeparatorData:
    """A certified primitive element of the reduced residue algebra."""

    element: ExtensionElement
    symbol: Symbol
    characteristic_polynomial: Poly
    factors: tuple[tuple[Poly, int], ...]
    nilradical_rows: tuple[tuple[object, ...], ...]
    reduced_rank: int
    power_nilradical_inverse: DomainMatrix


def _separating_residue_element(
    order: AlgebraicOrder,
    context: _BaseResidueContext,
) -> _SeparatorData:
    trace_reduction = _reduce_polynomial_matrix(
        order.trace_pairing_matrix(),
        context,
    )
    nilradical_rows = tuple(tuple(row) for row in trace_reduction.nullspace().to_list())
    reduced_rank = order.rank - len(nilradical_rows)
    if reduced_rank <= 0:
        raise ValueError("the reduced residue algebra has zero dimension")

    symbol = Dummy("residue_generator")
    parent = order.extension.parent_field
    integer_vectors = (
        coefficients
        for bound in count(1)
        for coefficients in product(range(-bound, bound + 1), repeat=order.rank)
        if max(abs(coefficient) for coefficient in coefficients) == bound
    )
    for coefficients in integer_vectors:
        candidate = order.basis.element(
            tuple(parent.convert(coefficient) for coefficient in coefficients)
        )
        multiplication = _reduce_polynomial_matrix(
            order.multiplication_matrix(candidate),
            context,
        )
        characteristic_polynomial = Poly.from_list(
            multiplication.charpoly(),
            symbol,
            domain=context.field,
        )
        try:
            _, factor_list = characteristic_polynomial.factor_list()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "SymPy could not factor the exact residue-algebra polynomial"
            ) from exc
        factors = tuple(
            (factor.monic(), multiplicity) for factor, multiplicity in factor_list
        )
        if sum(factor.degree() for factor, _ in factors) != reduced_rank:
            continue

        power_columns = tuple(
            _reduce_order_coordinates(
                order,
                candidate**power,
                context,
            )
            for power in range(reduced_rank)
        )
        nilradical_columns = tuple(tuple(row) for row in nilradical_rows)
        power_nilradical = _matrix_from_columns(
            power_columns + nilradical_columns,
            context.field,
        )
        try:
            power_nilradical_inverse = power_nilradical.inv()
        except DMNonInvertibleMatrixError:
            continue

        return _SeparatorData(
            element=candidate,
            symbol=symbol,
            characteristic_polynomial=characteristic_polynomial,
            factors=factors,
            nilradical_rows=nilradical_rows,
            reduced_rank=reduced_rank,
            power_nilradical_inverse=power_nilradical_inverse,
        )

    raise RuntimeError("failed to find a separating residue-algebra element")


def _residue_basis_images(
    order: AlgebraicOrder,
    context: _BaseResidueContext,
    separator: _SeparatorData,
    residue_polynomial: Poly,
) -> tuple[ExtensionElement, ...]:
    residue_field = FiniteExtension(residue_polynomial)
    inverse_rows = separator.power_nilradical_inverse.to_list()
    images = []
    for basis_index in range(order.rank):
        expression = sum(
            context.field.to_sympy(inverse_rows[power][basis_index])
            * separator.symbol**power
            for power in range(separator.reduced_rank)
        )
        images.append(
            cast(
                ExtensionElement,
                residue_field.from_sympy(expression),
            )
        )
    return tuple(images)


def _prime_ideal_from_residue_map(
    order: AlgebraicOrder,
    context: _BaseResidueContext,
    residue_polynomial: Poly,
    basis_images: tuple[ExtensionElement, ...],
) -> AlgebraicFractionalIdeal:
    relative_degree = residue_polynomial.degree()
    residue_field = basis_images[0].ext
    map_matrix = DomainMatrix(
        [
            [
                residue_field.domain.convert(image.rep.nth(power))
                for image in basis_images
            ]
            for power in range(relative_degree)
        ],
        (relative_degree, order.rank),
        context.field,
    )
    kernel_rows = map_matrix.nullspace().to_list()
    parent = cast(FractionField, order.extension.parent_field)
    prime_element = _embed_parent(order.extension, context.prime)
    generators: list[ExtensionElement] = [
        prime_element * basis_element for basis_element in order.basis.elements
    ]
    for kernel_row in kernel_rows:
        generators.append(
            order.basis.element(
                tuple(
                    _polynomial_parent_element(
                        context.lift(coefficient),
                        parent,
                    )
                    for coefficient in kernel_row
                )
            )
        )
    return AlgebraicFractionalIdeal(order, tuple(generators))


@dataclass(frozen=True, slots=True, eq=False)
class FiniteAlgebraicPlace:
    r"""A closed finite place represented in a certified maximal order."""

    order: AlgebraicOrder
    base_prime: Poly
    base_residue_field: Domain
    base_root: Expr
    separator: ExtensionElement
    residue_polynomial: Poly
    residue_field: FiniteExtension
    ramification_index: int
    prime_ideal: AlgebraicFractionalIdeal
    _context: _BaseResidueContext = dataclass_field(
        repr=False,
        compare=False,
    )
    _basis_images: tuple[ExtensionElement, ...] = dataclass_field(
        repr=False,
        compare=False,
    )
    _uniformizer_cache: ExtensionElement | None = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _different_exponent_cache: int | None = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _prime_power_cache: dict[int, AlgebraicFractionalIdeal] = dataclass_field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def extension(self) -> AlgebraicExtension:
        """The algebraic function field containing this place."""
        return self.order.extension

    @property
    def base_degree(self) -> int:
        """Degree of the base point over the constant field."""
        return self.base_prime.degree()

    @property
    def relative_degree(self) -> int:
        """Residue degree over the selected base point."""
        return self.residue_polynomial.degree()

    @property
    def degree(self) -> int:
        """Residue-field degree over the original constant field."""
        return self.base_degree * self.relative_degree

    @property
    def is_ramified(self) -> bool:
        """Whether the place is ramified over the rational base."""
        return self.ramification_index > 1

    @property
    def uniformizer(self) -> ExtensionElement:
        """Return an exact element of valuation one."""
        cached = self._uniformizer_cache
        if cached is not None:
            return cached

        square = self._prime_power(2)
        for basis_element in self.prime_ideal.basis_elements:
            if not square.contains(basis_element):
                if self.valuation(basis_element) != 1:
                    raise ValueError("a prime-ideal generator failed valuation one")
                object.__setattr__(
                    self,
                    "_uniformizer_cache",
                    basis_element,
                )
                return basis_element
        raise ValueError("the prime ideal is indistinguishable from its square")

    @property
    def different_exponent(self) -> int:
        """Valuation of the relative different at this place."""
        cached = self._different_exponent_cache
        if cached is None:
            cached = self.ideal_valuation(self.order.different)
            object.__setattr__(self, "_different_exponent_cache", cached)
        return cached

    def _prime_power(self, exponent: int) -> AlgebraicFractionalIdeal:
        """Return a cached nonnegative power of this place's prime ideal."""
        if exponent < 0:
            raise ValueError("cached prime powers require a nonnegative exponent")
        cache = self._prime_power_cache
        if 0 not in cache:
            cache[0] = self.order.unit_ideal()
            cache[1] = self.prime_ideal
        if exponent not in cache:
            largest = max(power for power in cache if power < exponent)
            ideal = cache[largest]
            for power in range(largest + 1, exponent + 1):
                ideal = ideal.multiply(self.prime_ideal)
                cache[power] = ideal
        return cache[exponent]

    def reduce_integral(self, element: object) -> ExtensionElement:
        """Reduce an element of the maximal order into the residue field."""
        coordinates = self.order.polynomial_coordinates(element)
        result = self.residue_field.zero
        for coordinate, image in zip(
            coordinates,
            self._basis_images,
            strict=True,
        ):
            base_value = self._context.reduce_polynomial(coordinate)
            embedded_base_value = self.residue_field.from_sympy(
                self.base_residue_field.to_sympy(base_value)
            )
            result += embedded_base_value * image
        return cast(ExtensionElement, result)

    def lift_residue(self, value: object) -> ExtensionElement:
        """Lift a residue-field element to an element of the maximal order."""
        converted = self.residue_field.convert(value)
        parent = cast(FractionField, self.extension.parent_field)
        result = self.extension.field.zero
        for power in range(self.relative_degree - 1, -1, -1):
            result *= self.separator
            base_coefficient = converted.rep.nth(power)
            result += _embed_parent(
                self.extension,
                _polynomial_parent_element(
                    self._context.lift(base_coefficient),
                    parent,
                ),
            )
        if self.reduce_integral(result) != converted:
            raise ValueError("residue lifting failed to round-trip")
        return result

    def valuation(self, element: object) -> int | Expr:
        """Return the normalized discrete valuation, with ``v(0) = oo``."""
        value = _extension_element(self.extension, element)
        if value == self.extension.field.zero:
            return S.Infinity

        fraction = self.order.basis.fraction(value)
        parent = cast(FractionField, self.extension.parent_field)
        numerator = self.order.basis.element(
            tuple(
                _polynomial_parent_element(polynomial, parent)
                for polynomial in fraction.numerators
            )
        )
        denominator_order = _polynomial_multiplicity(
            fraction.denominator,
            self.base_prime,
        )
        uniformizer = self._uniformizer_cache
        if uniformizer is None:
            numerator_valuation = self._integral_element_valuation(numerator)
        else:
            numerator_valuation = self._regular_element_valuation(
                numerator,
                uniformizer,
            )
        return numerator_valuation - self.ramification_index * denominator_order

    def _integral_element_valuation(self, element: ExtensionElement) -> int:
        if not self.order.contains(element):
            raise ValueError("integral valuation requires an order element")
        if element == self.extension.field.zero:
            raise ValueError("the zero element has infinite valuation")
        exponent = 0
        while self._prime_power(exponent + 1).contains(element):
            exponent += 1
        return exponent

    def ideal_valuation(self, ideal: AlgebraicFractionalIdeal) -> int:
        """Return the exponent of this prime in a fractional ideal."""
        if ideal.order is not self.order:
            raise ValueError("ideal valuation requires the place's order")
        denominator, scaled = _integral_ideal_scale(ideal)
        return self._scaled_ideal_valuation(scaled, denominator)

    def _scaled_ideal_valuation(
        self,
        scaled: AlgebraicFractionalIdeal,
        denominator: Poly,
    ) -> int:
        """Value an ideal from a verified integral scalar multiple."""
        if scaled.order is not self.order or not scaled.is_integral:
            raise ValueError("ideal valuation requires an integral scaled ideal")
        denominator_order = _polynomial_multiplicity(
            denominator,
            self.base_prime,
        )
        exponent = 0
        while scaled.is_subset_of(self._prime_power(exponent + 1)):
            exponent += 1
        return exponent - self.ramification_index * denominator_order

    def reduce_local(self, element: object) -> ExtensionElement:
        """Reduce any element regular at this place."""
        value = _extension_element(self.extension, element)
        valuation = self.valuation(value)
        if valuation is S.Infinity:
            return self.residue_field.zero
        if cast(int, valuation) < 0:
            raise ValueError("an element with a pole has no residue-class value")
        return self._reduce_regular(value)

    def _reduce_regular(
        self,
        value: ExtensionElement,
    ) -> ExtensionElement:
        """Reduce an element already known to be regular at this place."""
        if value == self.extension.field.zero:
            return self.residue_field.zero
        if self.order.contains(value):
            return self.reduce_integral(value)

        denominator_ideal = self.order.unit_ideal().intersection(
            self.order.principal_ideal(self.extension.field.one / value)
        )
        for denominator in denominator_ideal.basis_elements:
            denominator_residue = self.reduce_integral(denominator)
            if denominator_residue == self.residue_field.zero:
                continue
            numerator = denominator * value
            if not self.order.contains(numerator):
                raise ValueError("a local denominator failed to clear the element")
            return self.reduce_integral(numerator) / denominator_residue
        raise ValueError("no denominator outside the place was found")

    def _regular_element_valuation(
        self,
        value: ExtensionElement,
        uniformizer: ExtensionElement,
    ) -> int:
        """Value a regular nonzero element by repeated residue extraction."""
        if value == self.extension.field.zero:
            raise ValueError("the zero element has infinite valuation")
        exponent = 0
        quotient = value
        while self._reduce_regular(quotient) == self.residue_field.zero:
            exponent += 1
            quotient /= uniformizer
        return exponent

    def local_expansion(
        self,
        element: object,
        *,
        stop: int,
    ) -> LocalLaurentExpansion:
        """Expand through powers strictly below ``stop`` in a uniformizer."""
        if not isinstance(stop, int):
            raise TypeError("the local expansion stop must be an integer")
        value = _extension_element(self.extension, element)
        if value == self.extension.field.zero:
            return LocalLaurentExpansion(
                place=self,
                element=value,
                start=stop,
                stop=stop,
                coefficients=(),
                remainder=value,
                remainder_valuation=S.Infinity,
            )
        uniformizer = self.uniformizer
        initial_valuation = self.valuation(value)
        start = cast(int, initial_valuation)
        if stop <= start:
            return LocalLaurentExpansion(
                place=self,
                element=value,
                start=start,
                stop=stop,
                coefficients=(),
                remainder=value,
                remainder_valuation=start,
            )

        uniformizer_power = uniformizer**start
        remainder = value
        coefficients: list[ExtensionElement] = []
        for _ in range(start, stop):
            coefficient = self._reduce_regular(remainder / uniformizer_power)
            if coefficient != self.residue_field.zero:
                remainder -= self.lift_residue(coefficient) * uniformizer_power
            coefficients.append(coefficient)
            uniformizer_power *= uniformizer

        final_valuation = self.valuation(remainder)
        expansion = LocalLaurentExpansion(
            place=self,
            element=value,
            start=start,
            stop=stop,
            coefficients=tuple(coefficients),
            remainder=remainder,
            remainder_valuation=final_valuation,
        )
        if not expansion.reconstructs or (
            final_valuation is not S.Infinity and cast(int, final_valuation) < stop
        ):
            raise ValueError("local expansion failed its exact remainder invariant")
        return expansion

    def differential_residue(self, coefficient: object) -> ExtensionElement:
        r"""Return the residue of ``coefficient*dx`` at this place."""
        value = _extension_element(self.extension, coefficient)
        if value == self.extension.field.zero:
            return self.residue_field.zero
        uniformizer_derivative = _standard_derivative(
            self.extension,
            self.uniformizer,
        )
        local_coefficient = value / uniformizer_derivative
        valuation = self.valuation(local_coefficient)
        if valuation is S.Infinity or cast(int, valuation) > -1:
            return self.residue_field.zero
        expansion = self.local_expansion(local_coefficient, stop=0)
        return expansion.coefficient(-1)


def _polynomial_multiplicity(polynomial: Poly, prime: Poly) -> int:
    exponent = 0
    remainder = polynomial
    while remainder.degree() >= prime.degree():
        quotient, residue = remainder.div(prime)
        if not residue.is_zero:
            break
        exponent += 1
        remainder = quotient
    return exponent


def _parent_prime_support(
    element: FracElement,
    extension: AlgebraicExtension,
) -> tuple[Poly, ...]:
    _, constants, variable = _base_polynomial_ring(extension)
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=constants,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=constants,
    )
    primes: dict[Poly, None] = {}
    for polynomial in (numerator, denominator):
        _, factors = polynomial.factor_list()
        for factor, _ in factors:
            primes[factor.monic()] = None
    return tuple(primes)


def _polynomial_prime_support(polynomial: Poly) -> tuple[Poly, ...]:
    """Return the monic irreducible factors of a nonzero polynomial."""
    if polynomial.is_zero:
        raise ValueError("zero has every base prime in its support")
    _, factors = polynomial.factor_list()
    return tuple(factor.monic() for factor, _ in factors)


def _merge_prime_supports(
    *supports: tuple[Poly, ...],
) -> tuple[Poly, ...]:
    """Merge prime-support tuples without losing deterministic order."""
    primes: dict[Poly, None] = {}
    for support in supports:
        for prime in support:
            primes[prime] = None
    return tuple(primes)


def _element_base_prime_support(
    order: AlgebraicOrder,
    element: ExtensionElement,
) -> tuple[Poly, ...]:
    """Find finite base support without norm cancellation across places."""
    fraction = order.basis.fraction(element)
    parent = cast(FractionField, order.extension.parent_field)
    integral_numerator = order.basis.element(
        tuple(
            _polynomial_parent_element(polynomial, parent)
            for polynomial in fraction.numerators
        )
    )
    numerator_norm = cast(
        FracElement,
        order.extension.norm(integral_numerator),
    )
    return _merge_prime_supports(
        _parent_prime_support(numerator_norm, order.extension),
        _polynomial_prime_support(fraction.denominator),
    )


def _integral_ideal_scale(
    ideal: AlgebraicFractionalIdeal,
) -> tuple[Poly, AlgebraicFractionalIdeal]:
    """Clear all coordinate denominators of a fractional ideal."""
    _, constants, variable = _base_polynomial_ring(ideal.order.extension)
    entries = tuple(
        cast(FracElement, entry)
        for row in ideal.coordinate_matrix.to_list()
        for entry in row
    )
    denominator = _common_denominator(entries, constants, variable)
    scaled = ideal.scale(_embed_parent(ideal.order.extension, denominator))
    if not scaled.is_integral:
        raise ValueError("common-denominator scaling did not make an ideal integral")
    return denominator, scaled


def _fractional_ideal_base_prime_support(
    ideal: AlgebraicFractionalIdeal,
    denominator: Poly,
    scaled: AlgebraicFractionalIdeal,
) -> tuple[Poly, ...]:
    """Find finite support from an integral scalar multiple of an ideal."""
    determinant = cast(FracElement, scaled.coordinate_matrix.det())
    return _merge_prime_supports(
        _parent_prime_support(determinant, ideal.order.extension),
        _polynomial_prime_support(denominator),
    )


def _standard_derivative(
    extension: AlgebraicExtension,
    element: object,
) -> ExtensionElement:
    parent, _, variable = _base_polynomial_ring(extension)

    def differentiate_parent(value: object) -> FieldElement:
        converted = parent.convert(value)
        return cast(
            FieldElement,
            parent.from_sympy(parent.to_sympy(converted).diff(variable)),
        )

    return cast(
        ExtensionElement,
        extension.differentiate(
            _extension_element(extension, element),
            differentiate_parent,
        ),
    )


@dataclass(frozen=True, slots=True)
class LocalLaurentExpansion:
    """A certified finite segment of a local Laurent expansion."""

    place: _LaurentPlace
    element: ExtensionElement
    start: int
    stop: int
    coefficients: tuple[ExtensionElement, ...]
    remainder: ExtensionElement
    remainder_valuation: int | Expr

    def coefficient(self, exponent: int) -> ExtensionElement:
        """Return a known coefficient, rejecting uncomputed tail terms."""
        if not isinstance(exponent, int):
            raise TypeError("a Laurent exponent must be an integer")
        if exponent < self.start:
            return self.place.residue_field.zero
        if self.start <= exponent < self.stop:
            return self.coefficients[exponent - self.start]
        if self.remainder_valuation is S.Infinity:
            return self.place.residue_field.zero
        raise IndexError(
            f"coefficient {exponent} was not computed; "
            f"the expansion stops before {self.stop}"
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the truncated terms plus remainder reproduce the input."""
        reconstructed = self.remainder
        uniformizer = self.place.uniformizer
        for offset, coefficient in enumerate(self.coefficients):
            exponent = self.start + offset
            reconstructed += (
                self.place.lift_residue(coefficient) * uniformizer**exponent
            )
        return reconstructed == self.element


def _decompose_finite_places(
    order: AlgebraicOrder,
    prime: Poly,
) -> tuple[FiniteAlgebraicPlace, ...]:
    context = _BaseResidueContext.create(order.extension, prime)
    separator = _separating_residue_element(order, context)
    places = []
    for residue_polynomial, ramification_index in separator.factors:
        basis_images = _residue_basis_images(
            order,
            context,
            separator,
            residue_polynomial,
        )
        prime_ideal = _prime_ideal_from_residue_map(
            order,
            context,
            residue_polynomial,
            basis_images,
        )
        residue_field = basis_images[0].ext
        place = FiniteAlgebraicPlace(
            order=order,
            base_prime=context.prime,
            base_residue_field=context.field,
            base_root=context.root,
            separator=separator.element,
            residue_polynomial=residue_polynomial,
            residue_field=residue_field,
            ramification_index=ramification_index,
            prime_ideal=prime_ideal,
            _context=context,
            _basis_images=basis_images,
        )
        if place.valuation(_embed_parent(order.extension, context.prime)) != (
            ramification_index
        ):
            raise ValueError("base-prime valuation disagrees with ramification")
        places.append(place)

    if (
        sum(place.ramification_index * place.relative_degree for place in places)
        != order.rank
    ):
        raise ValueError("finite-place degrees violate the fundamental identity")
    return tuple(places)
