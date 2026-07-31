r"""Finite places, valuations, residue maps, and Laurent expansions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from typing import Protocol, cast

from flint import (
    fmpz_mod_poly_ctx,
    fq_default,
    fq_default_ctx,
    fq_default_poly_ctx,
)
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
    _native_or_exact_quotient,
    _parent_element_polynomial,
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
    def extension(self) -> AlgebraicExtension: ...

    @property
    def residue_field(self) -> FiniteExtension: ...

    @property
    def uniformizer(self) -> ExtensionElement: ...

    def lift_residue(self, value: object) -> ExtensionElement: ...

    def constant_coordinates(self, value: object) -> tuple[FieldElement, ...]: ...

    def from_constant_coordinates(
        self,
        coordinates: tuple[object, ...],
    ) -> ExtensionElement: ...


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
    variable = cast(Symbol, context.prime.gens[0])
    constants = context.prime.domain
    return DomainMatrix(
        [
            [
                context.reduce_polynomial(
                    _matrix_entry_polynomial(
                        matrix,
                        entry,
                        variable,
                        constants,
                    )
                )
                for entry in row
            ]
            for row in matrix.to_list()
        ],
        matrix.shape,
        context.field,
    )


def _matrix_entry_polynomial(
    matrix: DomainMatrix,
    entry: object,
    variable: Symbol,
    constants: Domain,
) -> Poly:
    """Return a matrix entry as a base polynomial without symbolic coercion."""
    if isinstance(matrix.domain, FractionField):
        return _parent_element_polynomial(
            cast(FracElement, entry),
            matrix.domain,
            constants,
            variable,
        )
    to_dict = getattr(entry, "to_dict", None)
    if to_dict is None:
        return Poly.from_dict(
            {(0,): constants.convert(entry)}, variable, domain=constants
        )
    return Poly.from_dict(
        {
            monomial: constants.convert(coefficient)
            for monomial, coefficient in to_dict().items()
        },
        variable,
        domain=constants,
    )


def _embed_base_residue_value(
    residue_field: FiniteExtension,
    value: object,
) -> ExtensionElement:
    """Embed a native base-residue-field value into a finite extension."""
    coefficient = residue_field.domain.convert(value)
    return residue_field.dtype(residue_field.ring.new(coefficient), residue_field)


def _finite_extension_to_flint(
    field: FiniteExtension,
) -> tuple[fq_default_ctx, fq_default_poly_ctx]:
    """Build the matching python-flint finite field and polynomial context."""
    constants = field.domain
    if not constants.is_FiniteField:
        raise NotImplementedError("the residue field is not finite")
    characteristic = int(constants.characteristic())
    polynomial_context = fmpz_mod_poly_ctx(characteristic)
    modulus = polynomial_context(
        [int(coefficient) for coefficient in reversed(field.mod.to_list())]
    )
    finite_context = fq_default_ctx(
        modulus=modulus,
        var=str(field.modulus.gens[0]),
    )
    return finite_context, fq_default_poly_ctx(finite_context)


def _extension_element_to_flint(
    value: ExtensionElement,
    context: fq_default_ctx,
) -> fq_default:
    """Convert a SymPy finite-extension value to its flint counterpart."""
    generator = context.gen()
    result = context(0)
    for power in range(value.ext.rank - 1, -1, -1):
        result = result * generator + context(int(value.rep.nth(power)))
    return result


def _flint_to_extension_element(
    field: FiniteExtension,
    value: fq_default,
) -> ExtensionElement:
    """Convert one flint finite-field value back to SymPy's native field."""
    representative = Poly.from_dict(
        {
            (power,): field.domain.convert(int(coefficient))
            for power, coefficient in enumerate(value.to_list())
            if int(coefficient)
        },
        field.modulus.gens[0],
        domain=field.domain,
    )
    return field.dtype(representative.rep, field)


def _factor_residue_polynomial(
    polynomial: Poly,
) -> tuple[tuple[Poly, int], ...]:
    """Factor over an exact residue field, using flint for ``GF(p^d)``."""
    try:
        _, factor_list = polynomial.factor_list()
    except NotImplementedError:
        field = polynomial.domain
        if not isinstance(field, FiniteExtension) or not field.domain.is_FiniteField:
            raise
        finite_context, polynomial_context = _finite_extension_to_flint(field)
        coefficients = [
            _extension_element_to_flint(
                cast(ExtensionElement, polynomial.rep.nth(power)),
                finite_context,
            )
            for power in range(polynomial.degree() + 1)
        ]
        _unit, flint_factors = polynomial_context(coefficients).factor()
        factors: list[tuple[Poly, int]] = []
        for factor, multiplicity in flint_factors:
            factors.append(
                (
                    Poly.from_dict(
                        {
                            (power,): _flint_to_extension_element(field, coefficient)
                            for power, coefficient in enumerate(factor.coeffs())
                            if coefficient
                        },
                        polynomial.gens[0],
                        domain=field,
                    ).monic(),
                    int(multiplicity),
                )
            )
        return tuple(factors)
    return tuple((factor.monic(), multiplicity) for factor, multiplicity in factor_list)


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


def _separator_candidate_data(
    order: AlgebraicOrder,
    context: _BaseResidueContext,
    symbol: Symbol,
    candidate: ExtensionElement,
) -> tuple[Poly, tuple[tuple[Poly, int], ...], int]:
    """Return the characteristic polynomial, factors, and generated rank."""
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
        factors = _factor_residue_polynomial(characteristic_polynomial)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "the exact residue-algebra polynomial could not be factored"
        ) from exc
    return (
        characteristic_polynomial,
        factors,
        sum(factor.degree() for factor, _multiplicity in factors),
    )


def _separator_scalars(
    context: _BaseResidueContext,
    count: int,
) -> tuple[tuple[object, ...], bool]:
    """Return deterministic base-residue scalars and whether they are exhaustive."""
    field = context.field
    characteristic = int(field.characteristic())
    if characteristic == 0:
        return tuple(field.convert(value) for value in range(count)), False
    if field.is_FiniteField:
        size = characteristic
        return (
            tuple(field.convert(value) for value in range(min(size, count))),
            size <= count,
        )
    if isinstance(field, FiniteExtension) and field.domain.is_FiniteField:
        base_size = int(field.domain.characteristic())
        size = base_size**field.rank
        values: list[ExtensionElement] = []
        for coefficients in product(range(base_size), repeat=field.rank):
            value = field.zero
            for coefficient in reversed(coefficients):
                value = value * field.generator + _embed_base_residue_value(
                    field, coefficient
                )
            values.append(value)
            if len(values) == min(size, count):
                break
        return tuple(values), size <= count
    raise NotImplementedError(
        "separator search needs an enumerable finite or characteristic-zero "
        "base residue field"
    )


def _build_separator_data(
    order: AlgebraicOrder,
    context: _BaseResidueContext,
    candidate: ExtensionElement,
    symbol: Symbol,
    characteristic_polynomial: Poly,
    factors: tuple[tuple[Poly, int], ...],
    nilradical_rows: tuple[tuple[object, ...], ...],
    reduced_rank: int,
) -> _SeparatorData:
    """Build and verify the power-plus-nilradical coordinate isomorphism."""
    power_columns = tuple(
        _reduce_order_coordinates(
            order,
            candidate**power,
            context,
        )
        for power in range(reduced_rank)
    )
    power_nilradical = _matrix_from_columns(
        power_columns + nilradical_rows,
        context.field,
    )
    try:
        power_nilradical_inverse = power_nilradical.inv()
    except DMNonInvertibleMatrixError as exc:
        raise ValueError(
            "a separating residue element failed the power-basis check"
        ) from exc
    return _SeparatorData(
        element=candidate,
        symbol=symbol,
        characteristic_polynomial=characteristic_polynomial,
        factors=factors,
        nilradical_rows=nilradical_rows,
        reduced_rank=reduced_rank,
        power_nilradical_inverse=power_nilradical_inverse,
    )


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
    parent = cast(FractionField, order.extension.parent_field)
    # For a reduced algebra of rank n, at most n(n-1)/2 scalars can cause
    # collisions between conjugates of a + c*b.  Trying one more distinct
    # scalar therefore finds the element generating K[a, b] over an infinite
    # field.  Refining one basis direction at a time changes the old
    # exponential vector enumeration into a polynomial number of trials.
    bad_scalar_bound = reduced_rank * (reduced_rank - 1) // 2
    scalars, scalars_exhaustive = _separator_scalars(
        context,
        bad_scalar_bound + 1,
    )
    lifted_scalars = tuple(
        _polynomial_parent_element(context.lift(scalar), parent) for scalar in scalars
    )
    # Preserve the former search's first candidate.  Besides keeping stable
    # residue-coordinate presentations, -sum(b_i) is already primitive for
    # many ordinary power bases and avoids all further trials.
    candidate = order.basis.element(
        tuple(parent.convert(-1) for _direction in order.basis.elements)
    )
    characteristic_polynomial, factors, generated_rank = _separator_candidate_data(
        order,
        context,
        symbol,
        candidate,
    )
    for direction in order.basis.elements:
        best = (candidate, characteristic_polynomial, factors, generated_rank)
        for scalar in lifted_scalars[1:]:
            scaled_direction = order.extension.field.dtype(
                direction.rep.mul_ground(scalar),
                order.extension.field,
            )
            trial = candidate + scaled_direction
            trial_polynomial, trial_factors, trial_rank = _separator_candidate_data(
                order,
                context,
                symbol,
                trial,
            )
            if trial_rank > best[3]:
                best = (trial, trial_polynomial, trial_factors, trial_rank)
                if trial_rank == reduced_rank:
                    break
        candidate, characteristic_polynomial, factors, generated_rank = best
        if generated_rank == reduced_rank:
            return _build_separator_data(
                order,
                context,
                candidate,
                symbol,
                characteristic_polynomial,
                factors,
                nilradical_rows,
                reduced_rank,
            )

    if scalars_exhaustive:
        # Small finite fields can have too few scalars for the generic-line
        # bound.  Exhaust the finite residue algebra once, rather than looping
        # forever over integer vectors that repeat modulo the characteristic.
        for coefficients in product(lifted_scalars, repeat=order.rank):
            candidate = order.basis.element(coefficients)
            characteristic_polynomial, factors, generated_rank = (
                _separator_candidate_data(
                    order,
                    context,
                    symbol,
                    candidate,
                )
            )
            if generated_rank == reduced_rank:
                return _build_separator_data(
                    order,
                    context,
                    candidate,
                    symbol,
                    characteristic_polynomial,
                    factors,
                    nilradical_rows,
                    reduced_rank,
                )

    raise NotImplementedError(
        "the reduced residue algebra has no primitive element over the "
        "available base residue field"
    )


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
        image = residue_field.zero
        for power in range(separator.reduced_rank - 1, -1, -1):
            image *= residue_field.generator
            image += _embed_base_residue_value(
                residue_field,
                inverse_rows[power][basis_index],
            )
        images.append(cast(ExtensionElement, image))
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

        # At an unramified finite place, the rational-base prime has
        # valuation one by definition.  Prefer it to an arbitrary generator
        # of the maximal ideal: the former has a canonical quotient-field
        # representation and avoids needless inversions of split-fibre
        # generators over algebraic constant fields.
        if self.ramification_index == 1:
            base_uniformizer = _embed_parent(self.extension, self.base_prime)
            if self.valuation(base_uniformizer) != 1:
                raise ValueError("an unramified base prime failed valuation one")
            object.__setattr__(
                self,
                "_uniformizer_cache",
                base_uniformizer,
            )
            return base_uniformizer

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
            # Powers are filled without gaps, so this is the largest cached
            # exponent without a linear scan.
            largest = len(cache) - 1
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
            embedded_base_value = _embed_base_residue_value(
                self.residue_field,
                base_value,
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

    def constant_coordinates(self, value: object) -> tuple[FieldElement, ...]:
        r"""Return exact coordinates of a residue-field element over ``C``.

        If ``alpha`` is the root of ``base_prime`` and ``beta`` is this
        place's separating residue generator, the coordinate order is

        ``(1, alpha, ..., alpha**(d - 1), beta, alpha*beta, ...,
        alpha**(d - 1)*beta, ..., alpha**(d - 1)*beta**(f - 1))``,

        where ``d = base_degree`` and ``f = relative_degree``.  Thus the
        returned tuple has length :attr:`degree` and gives the native
        ``C``-vector-space presentation of the closed residue field.  It is
        deliberately local to this place: it does not choose an embedding
        into a common algebraic closure for different places.
        """
        if isinstance(value, ExtensionElement) and value.ext is not self.residue_field:
            raise ValueError("residue coordinates require this place's residue field")
        converted = self.residue_field.convert(value)
        constants = self.base_prime.domain
        coordinates: list[FieldElement] = []
        for relative_power in range(self.relative_degree):
            base_coefficient = self._context.field.convert(
                converted.rep.nth(relative_power)
            )
            lifted = self._context.lift(base_coefficient)
            coordinates.extend(
                constants.convert(lifted.nth(base_power))
                for base_power in range(self.base_degree)
            )
        result = tuple(coordinates)
        if len(result) != self.degree:
            raise ValueError("residue coordinates have the wrong dimension")
        return result

    def from_constant_coordinates(
        self,
        coordinates: tuple[object, ...],
    ) -> ExtensionElement:
        r"""Reconstruct a residue-field element from :meth:`constant_coordinates`.

        This inverse is useful to consumers that need to verify a local
        coordinate computation before turning it into constant-field linear
        equations.
        """
        if not isinstance(coordinates, tuple):
            raise TypeError("residue coordinates must be an immutable tuple")
        if len(coordinates) != self.degree:
            raise ValueError("residue coordinates have the wrong dimension")

        constants = self.base_prime.domain
        variable = self.base_prime.gens[0]
        normalized = tuple(constants.convert(coordinate) for coordinate in coordinates)
        result = self.residue_field.zero
        generator = self.residue_field.generator
        for relative_power in range(self.relative_degree):
            coefficients = {
                (base_power,): normalized[
                    relative_power * self.base_degree + base_power
                ]
                for base_power in range(self.base_degree)
                if normalized[relative_power * self.base_degree + base_power]
                != constants.zero
            }
            polynomial = Poly.from_dict(coefficients, variable, domain=constants)
            base_coefficient = self._context.reduce_polynomial(polynomial)
            embedded = _embed_base_residue_value(
                self.residue_field,
                base_coefficient,
            )
            result += embedded * generator**relative_power
        reconstructed = cast(ExtensionElement, result)
        if self.constant_coordinates(reconstructed) != normalized:
            raise ValueError("residue-coordinate reconstruction failed to round-trip")
        return reconstructed

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

        unit_ideal = self.order.unit_ideal()
        try:
            inverse_ideal = self.order.principal_ideal(self.extension.field.one / value)
        except ZeroDivisionError:
            # A regular local element can retain globally uncancelled factors
            # after multiplying by a uniformizer.  At a split fibre over an
            # algebraic constant field, SymPy can then reject its otherwise
            # nonzero quotient-field inverse as a zero divisor.  Compute the
            # same ``value**-1 O`` as ``(O : value O)`` directly from order
            # multiplication matrices.
            inverse_ideal = unit_ideal.colon(self.order.principal_ideal(value))
        denominator_ideal = unit_ideal.intersection(inverse_ideal)
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
            quotient = _native_or_exact_quotient(
                self.extension,
                quotient,
                uniformizer,
            )
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

        uniformizer_power = _extension_power(
            self.extension,
            uniformizer,
            start,
        )
        remainder = value
        coefficients: list[ExtensionElement] = []
        for _ in range(start, stop):
            coefficient = self._reduce_regular(
                _native_or_exact_quotient(
                    self.extension,
                    remainder,
                    uniformizer_power,
                )
            )
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
        local_coefficient = _native_or_exact_quotient(
            self.extension,
            value,
            uniformizer_derivative,
        )
        valuation = self.valuation(local_coefficient)
        if valuation is S.Infinity or cast(int, valuation) > -1:
            return self.residue_field.zero
        expansion = self.local_expansion(local_coefficient, stop=0)
        return expansion.coefficient(-1)

    def differential_residues_for_powers(
        self,
        coefficient: object,
        parameter: object,
        count: int,
    ) -> tuple[ExtensionElement, ...]:
        r"""Return residues of ``coefficient*parameter**i*dx`` in one expansion.

        The direct residue operation expands every differential separately.
        Hess bases contain long runs of powers of one parameter, so expanding
        the coefficient and parameter once and multiplying their truncated
        Laurent series avoids repeating the local algebra for every power.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("the residue power count must be a nonnegative integer")
        if count == 0:
            return ()

        value = _extension_element(self.extension, coefficient)
        local_parameter = _extension_element(self.extension, parameter)
        if local_parameter == self.extension.field.zero:
            raise ValueError("a residue expansion parameter must be nonzero")
        if value == self.extension.field.zero:
            return tuple(self.residue_field.zero for _ in range(count))

        uniformizer_derivative = _standard_derivative(
            self.extension,
            self.uniformizer,
        )
        local_coefficient = _native_or_exact_quotient(
            self.extension,
            value,
            uniformizer_derivative,
        )
        coefficient_start = cast(int, self.valuation(local_coefficient))
        parameter_start = cast(int, self.valuation(local_parameter))
        maximum_power = count - 1

        coefficient_stop = max(0, -maximum_power * parameter_start)
        coefficient_expansion = self.local_expansion(
            local_coefficient,
            stop=coefficient_stop,
        )
        coefficient_series = {
            exponent: coefficient_expansion.coefficient(exponent)
            for exponent in range(
                coefficient_expansion.start,
                coefficient_expansion.stop,
            )
            if coefficient_expansion.coefficient(exponent) != self.residue_field.zero
        }

        maximum_series_exponent = -1 - coefficient_start
        parameter_series: dict[int, ExtensionElement] = {}
        if maximum_power:
            parameter_stop = -coefficient_start - min(
                0,
                (maximum_power - 1) * parameter_start,
            )
            parameter_expansion = self.local_expansion(
                local_parameter,
                stop=parameter_stop,
            )
            parameter_series = {
                exponent: parameter_expansion.coefficient(exponent)
                for exponent in range(
                    parameter_expansion.start,
                    parameter_expansion.stop,
                )
                if parameter_expansion.coefficient(exponent) != self.residue_field.zero
            }

        power_series = {0: self.residue_field.one}
        residues: list[ExtensionElement] = []
        for power in range(count):
            residue = self.residue_field.zero
            for exponent, coefficient_value in coefficient_series.items():
                residue += coefficient_value * power_series.get(
                    -1 - exponent,
                    self.residue_field.zero,
                )
            residues.append(cast(ExtensionElement, residue))
            if power == maximum_power:
                break
            next_power: dict[int, ExtensionElement] = {}
            for left_exponent, left_value in power_series.items():
                for right_exponent, right_value in parameter_series.items():
                    exponent = left_exponent + right_exponent
                    if exponent > maximum_series_exponent:
                        continue
                    next_power[exponent] = cast(
                        ExtensionElement,
                        next_power.get(exponent, self.residue_field.zero)
                        + left_value * right_value,
                    )
            power_series = next_power
        return tuple(residues)


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


def _extension_power(
    extension: AlgebraicExtension,
    value: ExtensionElement,
    exponent: int,
) -> ExtensionElement:
    """Raise a field element without SymPy's faulty nested-field inverse."""
    if exponent >= 0:
        return cast(ExtensionElement, value**exponent)
    return _native_or_exact_quotient(
        extension,
        extension.field.one,
        value ** (-exponent),
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
            reconstructed += self.place.lift_residue(coefficient) * _extension_power(
                self.place.extension,
                uniformizer,
                exponent,
            )
        # ``ExtensionElement`` equality can retain different unreduced
        # rational-function presentations of the same field element.  Test
        # equality in the field instead, so an exact cancellation is not
        # mistaken for a failed local expansion.
        return reconstructed - self.element == self.element.ext.zero


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
