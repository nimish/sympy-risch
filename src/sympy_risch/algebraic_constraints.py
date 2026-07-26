r"""Certified finite-dimensional valuation constraints on algebraic functions.

This module is the first reusable kernel for the finite/infinite
Riemann--Roch lattice-intersection problem.  It deliberately does *not* claim
to construct an arbitrary Riemann--Roch space.  Instead, for a supplied
finite-dimensional constant subspace ``V`` of an algebraic function field, it
computes exactly

``V intersect L(D)``.

For a basis ``b_1, ..., b_n`` of ``V``, only finitely many places matter:
the support of ``D`` and of the principal divisors of the ``b_i``.  At each
such place ``P``, the inequality

``v_P(sum(c_i*b_i)) >= -D_P``

is equivalent to the vanishing of its local Laurent coefficients below
``-D_P``.  When the residue field of every place that contributes a nontrivial
coefficient constraint is the constant field itself, those are exact linear
equations in ``c_i`` over the certified constant field.  Their exact nullspace
is therefore the desired intersection.

The degree-one-residue restriction is intentional.  A general closed place
requires flattening a residue-field equation into several constant-field
coordinates; returning ``UNSUPPORTED`` here is safer than silently treating a
closed point as rational.

``finite_lattice_slice`` and ``search_nonspecial_lattice_slice`` provide a
small, theorem-backed first consumer.  The former enumerates a finite prefix
of the exact finite ideal lattice.  The latter certifies a full Riemann--Roch
basis only when the kernel reaches the dimension given by Riemann--Roch in
the nonspecial range.  Failure to reach that dimension at a supplied bound is
reported as ``INCOMPLETE`` -- never as a negative answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly

from .algebraic._field import (
    _base_polynomial_ring,
    _embed_parent,
    _extension_element,
)
from .algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from .algebraic_geometry import (
    AlgebraicDivisor,
    AlgebraicFunctionField,
    AlgebraicPlace,
    InfiniteAlgebraicPlace,
)
from .riemann_roch import (
    RiemannRochSpace,
    RiemannRochSpaceStatus,
    _constant_coordinate_matrix,
    compute_riemann_roch_space,
)


class AlgebraicValuationKernelStatus(Enum):
    """Whether a finite-dimensional valuation kernel was certified."""

    CERTIFIED = "certified"
    UNSUPPORTED = "unsupported"


class AlgebraicNonspecialSearchStatus(Enum):
    """Outcome of a bounded finite-lattice Riemann--Roch search."""

    CERTIFIED = "certified"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class _UnsupportedValuationConstraint(Exception):
    """Private control flow for an honest, structured unsupported result."""


def _independent_ambient_basis(
    model: AlgebraicFunctionField,
    ambient_basis: tuple[object, ...],
) -> tuple[ExtensionElement, ...]:
    """Drop zero and constant-linearly dependent ambient generators."""
    zero = model.extension.field.zero
    coerced = tuple(
        element
        for raw_element in ambient_basis
        if (element := _extension_element(model.extension, raw_element)) != zero
    )
    if not coerced:
        return ()
    matrix = _constant_coordinate_matrix(model, coerced)
    _reduced, pivots = matrix.rref()
    return tuple(coerced[index] for index in pivots)


def _relevant_places(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    ambient_basis: tuple[ExtensionElement, ...],
) -> tuple[AlgebraicPlace, ...]:
    """Return every place where the ambient subspace can violate ``D``."""
    places: dict[AlgebraicPlace, None] = {}
    for place in divisor.support:
        places[place] = None
    for element in ambient_basis:
        for place, _coefficient in model.principal_divisor(element).terms:
            places[place] = None
    return tuple(places)


def _residue_scalar(
    place: AlgebraicPlace,
    value: object,
    constants: Domain,
) -> object:
    r"""Return one residue coefficient as an element of the constant field.

    This is called only for a place that contributes a nontrivial local
    equation.  Keeping the degree-one check here makes a future extension to
    higher-degree residue coordinates local and prevents accidental weakening
    of the certificate.
    """
    residue_field = place.residue_field
    if (
        place.degree != 1
        or residue_field.rank != 1
        or residue_field.domain != constants
    ):
        raise _UnsupportedValuationConstraint(
            "valuation constraints currently require every place that needs "
            "a local equation to have residue field equal to the constant field",
        )
    converted = residue_field.convert(value)
    return constants.convert(converted.rep.nth(0))


def _constraint_data(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    ambient_basis: tuple[ExtensionElement, ...],
) -> tuple[DomainMatrix, tuple[AlgebraicPlace, ...]]:
    r"""Build exact local-Laurent equations for ``span(ambient_basis)``.

    A row records one coefficient of one local expansion.  Its nullspace is
    exactly ``span_C(ambient_basis) intersect L(divisor)`` under the
    contract documented in this module for places that contribute equations.
    """
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    places = _relevant_places(model, divisor, ambient_basis)
    rows: list[list[object]] = []

    for place in places:
        threshold = -divisor.coefficient(place)
        valuations = tuple(
            cast(int, place.valuation(element)) for element in ambient_basis
        )
        start = min(valuations, default=threshold)
        if start >= threshold:
            continue

        # Prove that one local coefficient is a scalar equation over C before
        # doing the comparatively expensive expansions.  A place where every
        # ambient vector already meets the bound contributes no equation and
        # therefore needs no residue-coordinate representation.
        _residue_scalar(place, place.residue_field.zero, constants)

        expansions = tuple(
            place.local_expansion(element, stop=threshold) for element in ambient_basis
        )
        for exponent in range(start, threshold):
            row = [
                _residue_scalar(place, expansion.coefficient(exponent), constants)
                for expansion in expansions
            ]
            if any(coefficient != constants.zero for coefficient in row):
                rows.append(row)

    return DomainMatrix(rows, (len(rows), len(ambient_basis)), constants), places


def _linear_combination(
    model: AlgebraicFunctionField,
    ambient_basis: tuple[ExtensionElement, ...],
    coordinates: tuple[object, ...],
) -> ExtensionElement:
    """Reconstruct one constant-linear combination in the native field."""
    if len(ambient_basis) != len(coordinates):
        raise ValueError("kernel coordinates have the wrong ambient dimension")
    _parent, constants, _variable = _base_polynomial_ring(model.extension)
    result = model.extension.field.zero
    for coefficient, element in zip(coordinates, ambient_basis, strict=True):
        converted = constants.convert(coefficient)
        if converted != constants.zero:
            scalar = _extension_element(
                model.extension,
                constants.to_sympy(converted),
            )
            result += scalar * element
    return cast(ExtensionElement, result)


def _coordinate_rows(
    matrix: DomainMatrix,
) -> tuple[tuple[object, ...], ...]:
    """Extract a row-basis matrix as immutable coordinate tuples."""
    return tuple(tuple(row) for row in matrix.to_list())


def _coordinates_span_nullspace(
    constraints: DomainMatrix,
    coordinates: tuple[tuple[object, ...], ...],
) -> bool:
    """Check that coordinate rows are exactly the constraint nullspace."""
    constants = constraints.domain
    width = constraints.shape[1]
    if any(len(row) != width for row in coordinates):
        return False
    coordinate_matrix = DomainMatrix(
        [list(row) for row in coordinates],
        (len(coordinates), width),
        constants,
    )
    if constraints * coordinate_matrix.transpose() != DomainMatrix(
        [[constants.zero for _ in coordinates] for _ in range(constraints.shape[0])],
        (constraints.shape[0], len(coordinates)),
        constants,
    ):
        return False
    expected = constraints.nullspace()
    return (
        coordinate_matrix.rank() == expected.shape[0]
        and len(coordinates) == expected.shape[0]
    )


@dataclass(frozen=True, slots=True)
class AlgebraicValuationKernel:
    r"""A certified exact kernel ``V intersect L(D)`` over the constants.

    ``ambient_basis`` is a normalized constant-linearly independent basis of
    ``V``.  ``coordinates`` gives each output basis element in that ambient
    basis.  Recomputing the local equations in ``__post_init__`` means callers
    cannot forge a complete-kernel result by supplying plausible field
    elements alone.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    status: AlgebraicValuationKernelStatus
    ambient_basis: tuple[ExtensionElement, ...]
    basis: tuple[ExtensionElement, ...]
    coordinates: tuple[tuple[object, ...], ...]
    constraint_places: tuple[AlgebraicPlace, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("an algebraic valuation kernel needs a function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("an algebraic valuation kernel needs a divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError(
                "an algebraic valuation kernel needs constant-field analysis"
            )
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant-field analysis belongs to another field")
        if not isinstance(self.status, AlgebraicValuationKernelStatus):
            raise TypeError("the valuation-kernel status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic valuation kernel needs a nonempty detail")
        if not isinstance(
            self.ambient_basis,
            tuple,
        ) or not isinstance(self.basis, tuple):
            raise TypeError("valuation-kernel bases must be immutable tuples")
        if not isinstance(self.coordinates, tuple) or any(
            not isinstance(coordinates, tuple) for coordinates in self.coordinates
        ):
            raise TypeError("valuation-kernel coordinates must be immutable tuples")
        if not isinstance(self.constraint_places, tuple):
            raise TypeError("valuation-kernel places must be an immutable tuple")

        if self.status is AlgebraicValuationKernelStatus.UNSUPPORTED:
            if (
                self.ambient_basis
                or self.basis
                or self.coordinates
                or self.constraint_places
            ):
                raise ValueError(
                    "an unsupported valuation kernel cannot carry partial data"
                )
            return

        if self.status is not AlgebraicValuationKernelStatus.CERTIFIED:
            raise ValueError("an algebraic valuation kernel has an unknown status")
        if (
            self.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError("a valuation kernel needs a preserved constant field")

        ambient = _independent_ambient_basis(self.model, self.ambient_basis)
        if ambient != self.ambient_basis:
            raise ValueError("the ambient basis is not normalized over the constants")
        if len(self.basis) != len(self.coordinates):
            raise ValueError("kernel basis and coordinate data have different lengths")
        try:
            constraints, places = _constraint_data(self.model, self.divisor, ambient)
        except _UnsupportedValuationConstraint as exc:
            raise ValueError(
                "the valuation kernel has unsupported local constraints"
            ) from exc
        if places != self.constraint_places:
            raise ValueError("the valuation-kernel place support is inconsistent")
        if not _coordinates_span_nullspace(constraints, self.coordinates):
            raise ValueError("kernel coordinates do not span the exact local nullspace")

        reconstructed = tuple(
            _linear_combination(self.model, ambient, coordinates)
            for coordinates in self.coordinates
        )
        if reconstructed != self.basis:
            raise ValueError("kernel basis does not match its constant coordinates")
        if any(not self.contains(element) for element in self.basis):
            raise ValueError("a valuation-kernel basis element lies outside L(D)")
        if _constant_coordinate_matrix(self.model, self.basis).rank() != len(
            self.basis
        ):
            raise ValueError(
                "the valuation-kernel basis is constant-linearly dependent"
            )

    @property
    def dimension(self) -> int | None:
        """The exact dimension of the ambient intersection, if certified."""
        if self.status is AlgebraicValuationKernelStatus.UNSUPPORTED:
            return None
        return len(self.basis)

    @property
    def certified(self) -> bool:
        """Whether this result completely describes its declared ambient kernel."""
        return self.status is AlgebraicValuationKernelStatus.CERTIFIED

    @property
    def unsupported(self) -> bool:
        """Whether a required residue-coordinate system is unavailable."""
        return self.status is AlgebraicValuationKernelStatus.UNSUPPORTED

    def contains(self, element: object) -> bool:
        """Check the global defining inequalities at every nonzero place."""
        value = _extension_element(self.model.extension, element)
        if value == self.model.extension.field.zero:
            return True
        return (self.model.principal_divisor(value) + self.divisor).is_effective


def _unsupported_kernel(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    detail: str,
) -> AlgebraicValuationKernel:
    """Create an intentionally information-free unsupported result."""
    return AlgebraicValuationKernel(
        model,
        divisor,
        constant_field_analysis,
        AlgebraicValuationKernelStatus.UNSUPPORTED,
        (),
        (),
        (),
        (),
        detail,
    )


def solve_algebraic_valuation_constraints(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    ambient_basis: tuple[object, ...],
) -> AlgebraicValuationKernel:
    r"""Compute exactly ``span_C(ambient_basis) intersect L(divisor)``.

    The result is certified only after the existing constant-field analysis
    proves that the displayed constants are preserved, and only when every
    place that contributes a nontrivial local equation has residue degree one.
    It is nevertheless fully general in the supplied finite-dimensional
    ambient subspace within that contract: cancellations between ambient
    generators are solved by the local linear system rather than missed by
    individual valuation tests.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("valuation constraints need an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("valuation constraints need an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("valuation constraints need constant-field analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant-field analysis belongs to another field")
    if not isinstance(ambient_basis, tuple):
        raise TypeError("the ambient basis must be an immutable tuple")

    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_kernel(
            model,
            divisor,
            constant_field_analysis,
            "valuation constraints require a certified preserved constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    ambient = _independent_ambient_basis(model, ambient_basis)
    try:
        constraints, places = _constraint_data(model, divisor, ambient)
    except _UnsupportedValuationConstraint as exc:
        return _unsupported_kernel(model, divisor, constant_field_analysis, str(exc))
    coordinates = _coordinate_rows(constraints.nullspace())
    basis = tuple(
        _linear_combination(model, ambient, coordinate) for coordinate in coordinates
    )
    return AlgebraicValuationKernel(
        model,
        divisor,
        constant_field_analysis,
        AlgebraicValuationKernelStatus.CERTIFIED,
        ambient,
        basis,
        coordinates,
        places,
        "exact local Laurent equations describe the declared ambient intersection",
    )


def finite_lattice_slice(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    max_x_degree: int,
) -> tuple[ExtensionElement, ...]:
    r"""Return a finite prefix of the exact finite ideal lattice for ``L(D)``.

    Let ``D_fin`` be the finite part of ``D``.  The maximal-order fractional
    ideal with divisor ``-D_fin`` consists exactly of the functions satisfying
    all finite valuation inequalities for ``L(D)``.  If ``b_i`` is its
    ``C[x]`` basis, this returns

    ``{x**k * b_i : 0 <= k <= max_x_degree}``.

    It is a finite constant subspace, not a claimed global Riemann--Roch
    space.  Infinite inequalities and possible cancellation are handled by
    :func:`solve_algebraic_valuation_constraints`.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a finite lattice slice needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("a finite lattice slice needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the finite lattice degree bound must be nonnegative")

    finite_divisor = AlgebraicDivisor(
        model,
        tuple(
            (place, coefficient)
            for place, coefficient in divisor.terms
            if not isinstance(place, InfiniteAlgebraicPlace)
        ),
    )
    ideal = model.finite_ideal(-finite_divisor)
    _parent, constants, variable = _base_polynomial_ring(model.extension)
    base_variable = _embed_parent(
        model.extension,
        Poly(variable, variable, domain=constants),
    )
    return tuple(
        base_variable**power * element
        for power in range(max_x_degree + 1)
        for element in ideal.basis_elements
    )


@dataclass(frozen=True, slots=True)
class AlgebraicNonspecialLatticeSearch:
    r"""A bounded enumeration with an unbounded-theorem completion check.

    A certified result is a complete ``L(D)`` basis, not merely a successful
    search prefix: in the nonspecial range Riemann--Roch gives
    ``l(D) = deg(D) + 1 - genus``.  Once the exact ambient kernel reaches that
    dimension, it must equal ``L(D)``.  An ``INCOMPLETE`` result says only that
    the requested degree bound did not reach the target.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    max_x_degree: int
    status: AlgebraicNonspecialSearchStatus
    space: RiemannRochSpace | None
    searched_dimension: int | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a nonspecial search needs an algebraic function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a nonspecial search needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("a nonspecial search needs constant-field analysis")
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant-field analysis belongs to another field")
        if not isinstance(self.max_x_degree, int) or self.max_x_degree < 0:
            raise ValueError("the finite lattice degree bound must be nonnegative")
        if not isinstance(self.status, AlgebraicNonspecialSearchStatus):
            raise TypeError("the nonspecial-search status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a nonspecial search needs a nonempty detail")
        if self.space is not None and not isinstance(self.space, RiemannRochSpace):
            raise TypeError("a nonspecial search space must be a Riemann--Roch space")

        if self.status is AlgebraicNonspecialSearchStatus.UNSUPPORTED:
            if self.space is not None or self.searched_dimension is not None:
                raise ValueError(
                    "an unsupported nonspecial search cannot carry partial data"
                )
            return

        if (
            self.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError("a nonspecial search needs a preserved constant field")
        expected_dimension = self._expected_dimension()
        kernel = solve_algebraic_valuation_constraints(
            self.model,
            self.divisor,
            self.constant_field_analysis,
            finite_lattice_slice(self.model, self.divisor, self.max_x_degree),
        )
        if kernel.status is not AlgebraicValuationKernelStatus.CERTIFIED:
            raise ValueError("the nonspecial search has unsupported local constraints")
        if self.searched_dimension != kernel.dimension:
            raise ValueError(
                "the nonspecial search has an inconsistent kernel dimension"
            )

        if self.status is AlgebraicNonspecialSearchStatus.INCOMPLETE:
            if self.space is not None or (
                kernel.dimension is None or kernel.dimension >= expected_dimension
            ):
                raise ValueError(
                    "an incomplete nonspecial search has inconsistent data"
                )
            return

        if self.status is not AlgebraicNonspecialSearchStatus.CERTIFIED:
            raise ValueError("a nonspecial search has an unknown status")
        if kernel.dimension != expected_dimension or self.space is None:
            raise ValueError(
                "the nonspecial search does not reach the theorem dimension"
            )
        if (
            self.space.model is not self.model
            or self.space.divisor is not self.divisor
            or self.space.constant_field_analysis is not self.constant_field_analysis
            or self.space.status is not RiemannRochSpaceStatus.CERTIFIED
            or self.space.basis != kernel.basis
        ):
            raise ValueError(
                "the nonspecial search does not carry the validated Riemann--Roch space"
            )

    def _expected_dimension(self) -> int:
        """Return the Riemann--Roch dimension in its nonspecial range."""
        genus = self.model.genus
        if self.divisor.degree <= 2 * genus - 2:
            raise ValueError(
                "the divisor is outside the nonspecial Riemann--Roch range"
            )
        return self.divisor.degree + 1 - genus

    @property
    def dimension(self) -> int | None:
        """The complete global-space dimension only when certified."""
        if self.status is not AlgebraicNonspecialSearchStatus.CERTIFIED:
            return None
        if self.space is None:
            raise ValueError(
                "a certified nonspecial search lacks its Riemann--Roch space"
            )
        return self.space.dimension

    @property
    def basis(self) -> tuple[ExtensionElement, ...]:
        """The validated complete basis, or no partial basis on non-success."""
        if self.space is None:
            return ()
        return self.space.basis

    @property
    def expected_dimension(self) -> int | None:
        """The theorem target if this search reached a supported range."""
        if self.status is AlgebraicNonspecialSearchStatus.UNSUPPORTED:
            return None
        return self._expected_dimension()


def _unsupported_nonspecial_search(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    max_x_degree: int,
    detail: str,
) -> AlgebraicNonspecialLatticeSearch:
    """Create an intentionally information-free unsupported search result."""
    return AlgebraicNonspecialLatticeSearch(
        model,
        divisor,
        constant_field_analysis,
        max_x_degree,
        AlgebraicNonspecialSearchStatus.UNSUPPORTED,
        None,
        None,
        detail,
    )


def search_nonspecial_lattice_slice(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    *,
    max_x_degree: int,
) -> AlgebraicNonspecialLatticeSearch:
    r"""Certify ``L(D)`` when a finite ideal slice reaches its theorem target.

    This is a semialgorithm, deliberately bounded by ``max_x_degree``.  On
    success the certificate is complete by the nonspecial Riemann--Roch
    formula; on exhaustion it returns ``INCOMPLETE`` rather than asserting
    that no basis exists.  Higher-degree residue fields and non-preserved
    constants remain structured unsupported cases of the present kernel.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a nonspecial search needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("a nonspecial search needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("a nonspecial search needs constant-field analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant-field analysis belongs to another field")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the finite lattice degree bound must be nonnegative")

    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return _unsupported_nonspecial_search(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            "nonspecial lattice search requires a certified preserved constant field; "
            f"the current presentation is {constant_field_analysis.status.value}",
        )

    try:
        genus = model.genus
    except (NotImplementedError, TypeError, ValueError) as exc:
        return _unsupported_nonspecial_search(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            f"the genus needed for the nonspecial theorem is unavailable: {exc}",
        )
    if divisor.degree <= 2 * genus - 2:
        return _unsupported_nonspecial_search(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            "the divisor is outside the nonspecial Riemann--Roch range",
        )

    kernel = solve_algebraic_valuation_constraints(
        model,
        divisor,
        constant_field_analysis,
        finite_lattice_slice(model, divisor, max_x_degree),
    )
    if kernel.status is AlgebraicValuationKernelStatus.UNSUPPORTED:
        return _unsupported_nonspecial_search(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            kernel.detail,
        )

    expected_dimension = divisor.degree + 1 - genus
    if kernel.dimension is None:
        raise ValueError("a certified valuation kernel has no dimension")
    if kernel.dimension > expected_dimension:
        raise ValueError("a local valuation kernel exceeds Riemann--Roch dimension")
    if kernel.dimension == expected_dimension:
        space = compute_riemann_roch_space(
            model,
            divisor,
            constant_field_analysis,
            candidate_basis=kernel.basis,
        )
        if space.status is not RiemannRochSpaceStatus.CERTIFIED:
            raise ValueError(
                "the existing Riemann--Roch certificate rejected a full lattice kernel"
            )
        return AlgebraicNonspecialLatticeSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicNonspecialSearchStatus.CERTIFIED,
            space,
            kernel.dimension,
            "the finite lattice slice reaches the nonspecial Riemann--Roch dimension",
        )
    return AlgebraicNonspecialLatticeSearch(
        model,
        divisor,
        constant_field_analysis,
        max_x_degree,
        AlgebraicNonspecialSearchStatus.INCOMPLETE,
        None,
        kernel.dimension,
        "the finite lattice slice has not reached the nonspecial Riemann--Roch dimension",
    )
