r"""Closed-place residue profiles for algebraic differentials.

The all-place collector represents a residue at each closed place in that
place's own residue field.  Those fields deliberately do *not* come with
chosen embeddings into a common algebraic closure, so their coordinate vectors
cannot yet be combined into a residue basis.  This module instead records the
canonical invariant available before that base-change step: the characteristic
polynomial of multiplication by each local residue over the displayed
constant field.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum

from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly

from ..tower import AlgebraicExtension
from ._field import _base_polynomial_ring
from .all_places import (
    AlgebraicAllPlaceResidueResult,
    AlgebraicAllPlacesCollected,
    AlgebraicAllPlacesUnsupported,
    AlgebraicHermiteExact,
    AlgebraicLocalDifferentialResidue,
)


class AlgebraicResidueProfileStatus(Enum):
    """Outcome of a closed-place algebraic residue-profile computation."""

    EXACT = "exact"
    PROFILED = "profiled"
    UNSUPPORTED = "unsupported"


def _profile_constant_domain(
    extension: AlgebraicExtension,
    residue_symbol: Symbol,
) -> Domain:
    """Validate ``residue_symbol`` and return the displayed constant field."""
    _, constants, variable = _base_polynomial_ring(extension)
    constant_symbols = set(getattr(constants, "symbols", ()) or ())
    if residue_symbol in {variable, extension.generator} | constant_symbols:
        raise ValueError(
            "the residue indeterminate must be new over the algebraic model"
        )
    return constants


def _local_residue_characteristic_polynomial(
    local_residue: AlgebraicLocalDifferentialResidue,
    residue_symbol: Symbol,
) -> Poly:
    r"""Return ``charpoly_C(m_{res_P})`` in the native closed residue field."""
    if not isinstance(residue_symbol, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")

    place = local_residue.place
    constants = _profile_constant_domain(place.extension, residue_symbol)
    degree = place.degree
    coordinate_basis = tuple(
        place.from_constant_coordinates(
            tuple(
                constants.one if coordinate == basis_index else constants.zero
                for coordinate in range(degree)
            )
        )
        for basis_index in range(degree)
    )
    columns = tuple(
        place.constant_coordinates(local_residue.value * basis_element)
        for basis_element in coordinate_basis
    )
    if any(len(column) != degree for column in columns):
        raise ValueError("local residue coordinates have the wrong dimension")
    multiplication = DomainMatrix(
        [[columns[column][row] for column in range(degree)] for row in range(degree)],
        (degree, degree),
        constants,
    )
    return Poly.from_list(
        multiplication.charpoly(),
        residue_symbol,
        domain=constants,
    ).monic()


@dataclass(frozen=True, slots=True)
class AlgebraicResidueProfileFactor:
    r"""One closed-place factor ``Norm_{k(P)/C}(Z - res_P)``.

    The characteristic polynomial is computed from the local residue field's
    exact constant-coordinate basis.  Its square-free part is the minimal
    polynomial of the residue value; the full characteristic polynomial keeps
    the closed-place multiplicity needed by Trager-style residue descent.
    """

    local_residue: AlgebraicLocalDifferentialResidue
    residue_symbol: Symbol
    characteristic_polynomial: Poly = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.local_residue, AlgebraicLocalDifferentialResidue):
            raise TypeError("a residue-profile factor needs a local residue")
        if not isinstance(self.residue_symbol, Symbol):
            raise TypeError("the residue indeterminate must be a Symbol")
        object.__setattr__(
            self,
            "characteristic_polynomial",
            _local_residue_characteristic_polynomial(
                self.local_residue,
                self.residue_symbol,
            ),
        )

    @property
    def minimal_polynomial(self) -> Poly:
        """Return the monic minimal polynomial of this residue over ``C``."""
        return self.characteristic_polynomial.sqf_part().monic()


def _factor_product(
    factors: tuple[AlgebraicResidueProfileFactor, ...],
    residue_symbol: Symbol,
    constants: Domain,
) -> Poly:
    """Multiply characteristic factors in their stable divisor-support order."""
    result = Poly(1, residue_symbol, domain=constants)
    for factor in factors:
        result *= factor.characteristic_polynomial
    return result.monic()


@dataclass(frozen=True, slots=True)
class AlgebraicResidueProfile:
    r"""The all-place residue polynomial of an algebraic Hermite remainder.

    For every pole place ``P`` of a collected differential ``omega``, the
    profile records

    ``R_omega(Z) = product_P Norm_{k(P)/C}(Z - res_P)``.

    It is an exact global invariant of the local residues, including branch
    and infinity places and zero residues.  It is not a common constant field
    or a residue basis: compatible geometric embeddings must be constructed
    later before logarithmic divisors can be assembled.
    """

    collection: AlgebraicAllPlaceResidueResult
    residue_symbol: Symbol
    status: AlgebraicResidueProfileStatus
    factors: tuple[AlgebraicResidueProfileFactor, ...]
    residue_polynomial: Poly | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.collection, AlgebraicAllPlaceResidueResult):
            raise TypeError("a residue profile needs an all-place collection")
        if not isinstance(self.residue_symbol, Symbol):
            raise TypeError("the residue indeterminate must be a Symbol")
        if not isinstance(self.status, AlgebraicResidueProfileStatus):
            raise TypeError("the residue-profile status has the wrong type")
        if not isinstance(self.factors, tuple) or not all(
            isinstance(factor, AlgebraicResidueProfileFactor) for factor in self.factors
        ):
            raise TypeError("a residue profile needs immutable factor records")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a residue profile needs a detail")

        constants = _profile_constant_domain(
            self.collection.basis.extension,
            self.residue_symbol,
        )
        match self.collection:
            case AlgebraicHermiteExact():
                if self.status is not AlgebraicResidueProfileStatus.EXACT:
                    raise ValueError(
                        "an exact collection needs an exact residue profile"
                    )
            case AlgebraicAllPlacesUnsupported():
                if self.status is not AlgebraicResidueProfileStatus.UNSUPPORTED:
                    raise ValueError("an unsupported collection cannot have a profile")
            case AlgebraicAllPlacesCollected():
                if self.status not in {
                    AlgebraicResidueProfileStatus.PROFILED,
                    AlgebraicResidueProfileStatus.UNSUPPORTED,
                }:
                    raise ValueError(
                        "a collected result needs a profiled or unsupported result"
                    )

        if self.status is AlgebraicResidueProfileStatus.UNSUPPORTED:
            if self.factors or self.residue_polynomial is not None:
                raise ValueError("an unsupported residue profile cannot be partial")
            return

        if not isinstance(self.residue_polynomial, Poly):
            raise TypeError("a supported residue profile needs a polynomial")
        if self.residue_polynomial.gens != (self.residue_symbol,):
            raise ValueError("the residue polynomial uses another indeterminate")
        if self.residue_polynomial.domain != constants:
            raise ValueError("the residue polynomial uses another constant field")
        if self.residue_polynomial.monic() != self.residue_polynomial:
            raise ValueError("the residue polynomial must be monic")

        if self.status is AlgebraicResidueProfileStatus.EXACT:
            if self.factors or self.residue_polynomial != Poly(
                1,
                self.residue_symbol,
                domain=constants,
            ):
                raise ValueError("an exact residue profile must be the unit polynomial")
            return

        if not isinstance(self.collection, AlgebraicAllPlacesCollected):
            raise ValueError("a profiled residue result needs collected all-place data")
        if len(self.factors) != len(self.collection.residues):
            raise ValueError("the residue-profile factors do not cover every pole")
        for factor, local_residue in zip(
            self.factors,
            self.collection.residues,
            strict=True,
        ):
            if factor.local_residue is not local_residue:
                raise ValueError("a residue-profile factor belongs to another pole")
            if factor.residue_symbol != self.residue_symbol:
                raise ValueError("a residue-profile factor uses another indeterminate")
        expected = _factor_product(self.factors, self.residue_symbol, constants)
        if self.residue_polynomial != expected:
            raise ValueError("the residue polynomial does not match its local factors")

    @property
    def square_free_residue_polynomial(self) -> Poly | None:
        """Return the monic square-free profile, if profile construction worked."""
        if self.residue_polynomial is None:
            return None
        if self.residue_polynomial.degree() <= 0:
            return self.residue_polynomial
        return self.residue_polynomial.sqf_part().monic()

    @property
    def finite_unramified_simple_pole_polynomial(self) -> Poly | None:
        r"""Return the finite unramified simple-pole subprofile.

        This deliberately omits ramified and infinity factors, making it the
        directly comparable slice of the profile for Trager's plane-model
        resultant.  The full :attr:`residue_polynomial` remains the canonical
        all-place invariant.
        """
        if self.residue_polynomial is None:
            return None
        from ..algebraic_geometry.finite_places import FiniteAlgebraicPlace

        constants = _profile_constant_domain(
            self.collection.basis.extension,
            self.residue_symbol,
        )
        finite_factors = tuple(
            factor
            for factor in self.factors
            if isinstance(factor.local_residue.place, FiniteAlgebraicPlace)
            and not factor.local_residue.place.is_ramified
            and factor.local_residue.pole_order == 1
        )
        return _factor_product(finite_factors, self.residue_symbol, constants)


def build_algebraic_residue_profile(
    collection: AlgebraicAllPlaceResidueResult,
    residue_symbol: Symbol,
) -> AlgebraicResidueProfile:
    r"""Build the closed-place profile of an all-place residue collection.

    This computes the product of the characteristic polynomials of
    multiplication by each local residue.  It therefore extends the finite,
    unramified Trager resultant to every collected branch and infinity place
    without choosing incompatible local-field embeddings.  A later constant
    base-change and place-splitting algorithm will use its factors to build a
    genuine logarithmic residue basis.
    """
    if not isinstance(collection, AlgebraicAllPlaceResidueResult):
        raise TypeError("residue profiling requires an all-place collection")
    if not isinstance(residue_symbol, Symbol):
        raise TypeError("the residue indeterminate must be a Symbol")

    constants = _profile_constant_domain(
        collection.basis.extension,
        residue_symbol,
    )
    match collection:
        case AlgebraicHermiteExact():
            return AlgebraicResidueProfile(
                collection,
                residue_symbol,
                AlgebraicResidueProfileStatus.EXACT,
                (),
                Poly(1, residue_symbol, domain=constants),
                "Hermite reduction proved exactness, so the all-place residue "
                "profile is 1",
            )
        case AlgebraicAllPlacesUnsupported(detail=detail):
            return AlgebraicResidueProfile(
                collection,
                residue_symbol,
                AlgebraicResidueProfileStatus.UNSUPPORTED,
                (),
                None,
                "the all-place collection is unsupported: " + detail,
            )
        case AlgebraicAllPlacesCollected():
            pass

    try:
        factors = tuple(
            AlgebraicResidueProfileFactor(local_residue, residue_symbol)
            for local_residue in collection.residues
        )
    except NotImplementedError as exc:
        return AlgebraicResidueProfile(
            collection,
            residue_symbol,
            AlgebraicResidueProfileStatus.UNSUPPORTED,
            (),
            None,
            f"the local residue fields do not support exact profile coordinates: {exc}",
        )
    return AlgebraicResidueProfile(
        collection,
        residue_symbol,
        AlgebraicResidueProfileStatus.PROFILED,
        factors,
        _factor_product(factors, residue_symbol, constants),
        "computed characteristic-polynomial factors for every finite and "
        "infinite pole place",
    )
