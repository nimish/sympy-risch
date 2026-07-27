r"""First logarithmic reconstruction step for algebraic differentials.

The all-place collector gives each residue in its own closed residue field.
Before compatible algebraic constant base change is implemented, there is one
useful locally decidable case: every residue is already a rational constant. In
that case their primitive rational scale turns the residues into an integral
degree-zero divisor.  A verified principal witness for that divisor produces
one exact logarithmic derivative.

This module keeps that narrow result separate from the general elementary
dispatcher.  A residual differential with zero residues is explicitly retained
for the later second-kind exactness decision; it is never called
non-elementary here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd, lcm

from sympy.core.expr import Expr
from sympy.functions.elementary.exponential import log
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed

from .._typing import FieldElement
from ..algebraic_geometry.function_field import AlgebraicDivisor
from ..algebraic_geometry.principal import (
    AlgebraicPrincipalDivisorSearch,
    AlgebraicPrincipalDivisorStatus,
    search_algebraic_principal_divisor,
)
from ..tower import Derivation
from ._field import _base_polynomial_ring, _extension_element
from .all_places import (
    AlgebraicAllPlaceResidueResult,
    AlgebraicAllPlaceResidueStatus,
    AlgebraicLocalDifferentialResidue,
    collect_algebraic_all_place_residues,
)
from .basis import IntegralBasis


class AlgebraicIntegrationStatus(Enum):
    """Current outcome of the algebraic logarithmic integration slice."""

    EXACT = "exact"
    ELEMENTARY = "elementary"
    RESIDUE_FREE_REMAINDER = "residue_free_remainder"
    NEEDS_CONSTANT_BASE_CHANGE = "needs_constant_base_change"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicIntegrationResult:
    r"""One exact algebraic Hermite/logarithmic reduction result.

    Its native-field invariant is

    ``integrand = D(exact_part) + logarithmic_derivative + remainder``.

    ``ELEMENTARY`` and ``EXACT`` are constructive successes. A
    ``RESIDUE_FREE_REMAINDER`` has had every rational logarithmic residue
    removed, but still needs the later exactness decision for first- and
    second-kind differentials. ``INCOMPLETE`` means only that the requested
    finite principal-divisor search bound did not find a witness.
    """

    collection: AlgebraicAllPlaceResidueResult
    status: AlgebraicIntegrationStatus
    logarithmic_coefficient: FieldElement | None
    residue_divisor: AlgebraicDivisor | None
    principal_search: AlgebraicPrincipalDivisorSearch | None
    logarithmic_argument: ExtensionElement | None
    logarithmic_derivative: ExtensionElement
    remainder: ExtensionElement
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.collection, AlgebraicAllPlaceResidueResult):
            raise TypeError("an algebraic integration result needs all-place data")
        if not isinstance(self.status, AlgebraicIntegrationStatus):
            raise TypeError("the algebraic integration status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic integration result needs a detail")

        extension = self.collection.basis.extension
        logarithmic_derivative = _extension_element(
            extension,
            self.logarithmic_derivative,
        )
        remainder = _extension_element(extension, self.remainder)
        object.__setattr__(self, "logarithmic_derivative", logarithmic_derivative)
        object.__setattr__(self, "remainder", remainder)
        reconstructed = (
            self.collection.derivation(self.collection.exact_part)
            + logarithmic_derivative
            + remainder
        )
        if reconstructed - self.collection.integrand != extension.field.zero:
            raise ValueError("the algebraic integration result does not reconstruct")

        argument = self.logarithmic_argument
        if argument is not None:
            argument = _extension_element(extension, argument)
            if argument == extension.field.zero:
                raise ValueError("a logarithmic argument must be nonzero")
            object.__setattr__(self, "logarithmic_argument", argument)

        if self.status is AlgebraicIntegrationStatus.EXACT:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.EXACT
                or remainder != extension.field.zero
            ):
                raise ValueError("an exact result needs an exact zero-remainder input")
            return

        if self.status is AlgebraicIntegrationStatus.ELEMENTARY:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
                or remainder != extension.field.zero
                or any(
                    value is None
                    for value in (
                        self.logarithmic_coefficient,
                        self.residue_divisor,
                        self.principal_search,
                        argument,
                    )
                )
            ):
                raise ValueError("an elementary result needs a logarithmic witness")
            return

        if self.status is AlgebraicIntegrationStatus.INCOMPLETE:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
                or self.residue_divisor is None
                or self.principal_search is None
                or argument is not None
                or logarithmic_derivative != extension.field.zero
            ):
                raise ValueError("an incomplete result needs its failed divisor search")
            return

        if self.status is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE:
            if argument is not None or logarithmic_derivative != extension.field.zero:
                raise ValueError("a base-change boundary cannot carry a logarithm")
            return

        if self.status in {
            AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER,
            AlgebraicIntegrationStatus.UNSUPPORTED,
        }:
            return
        raise ValueError("the algebraic integration result has an unknown status")

    @property
    def is_elementary(self) -> bool:
        """Whether this slice constructed a complete antiderivative."""
        return self.status in {
            AlgebraicIntegrationStatus.EXACT,
            AlgebraicIntegrationStatus.ELEMENTARY,
        }

    @property
    def reconstructs(self) -> bool:
        """Whether the retained native decomposition reconstructs the input."""
        reconstructed = (
            self.collection.derivation(self.collection.exact_part)
            + self.logarithmic_derivative
            + self.remainder
        )
        return (
            reconstructed - self.collection.integrand
            == self.collection.basis.extension.field.zero
        )

    @property
    def antiderivative(self) -> Expr | None:
        r"""Return the displayed antiderivative on constructive success.

        As with other tower results, ordinary SymPy differentiation needs the
        algebraic generator's defining relation and derivation to verify the
        displayed logarithm.
        """
        if not self.is_elementary:
            return None
        field = self.collection.basis.extension.field
        exact_part = field.to_sympy(self.collection.exact_part)
        if self.logarithmic_argument is None:
            return exact_part
        if self.logarithmic_coefficient is None:
            raise ValueError("a logarithmic argument lacks its coefficient")
        coefficient = QQ.to_sympy(QQ.convert(self.logarithmic_coefficient))
        return exact_part + coefficient * log(field.to_sympy(self.logarithmic_argument))


def _rational_residue(
    local_residue: AlgebraicLocalDifferentialResidue,
) -> FieldElement | None:
    """Return a residue in ``QQ``, or ``None`` when it needs base change."""
    coordinates = local_residue.constant_coordinates
    if not coordinates or any(coordinate != QQ.zero for coordinate in coordinates[1:]):
        return None
    try:
        return QQ.convert(coordinates[0])
    except (CoercionFailed, TypeError, ValueError):
        return None


def _primitive_rational_scale(residues: tuple[FieldElement, ...]) -> FieldElement:
    """Return the positive rational content of nonzero rational residues."""
    numerator_gcd = 0
    denominator_lcm = 1
    for residue in residues:
        rational = QQ.convert(residue)
        numerator_gcd = gcd(numerator_gcd, abs(int(QQ.numer(rational))))
        denominator_lcm = lcm(denominator_lcm, int(QQ.denom(rational)))
    if numerator_gcd == 0:
        raise ValueError("rational residue content needs a nonzero residue")
    return QQ.exquo(QQ.convert(numerator_gcd), QQ.convert(denominator_lcm))


def _integer_multiple(residue: FieldElement, scale: FieldElement) -> int:
    """Return the exact integral quotient ``residue / scale``."""
    quotient = QQ.exquo(QQ.convert(residue), QQ.convert(scale))
    if QQ.denom(quotient) != 1:
        raise ValueError("the rational residue scale did not clear denominators")
    return int(QQ.numer(quotient))


def reconstruct_rational_residue_logarithms(
    collection: AlgebraicAllPlaceResidueResult,
    *,
    max_x_degree: int,
) -> AlgebraicIntegrationResult:
    r"""Reconstruct rational-residue logarithms from all-place data.

    This first scope requires displayed constants exactly ``QQ`` and every
    local residue to have constant-coordinate form ``(q, 0, ..., 0)``. It
    then searches for a principal witness of the integral residue divisor.
    Other algebraic residues need compatible constant base change and
    split-place maps, which are intentionally not guessed here.
    """
    if not isinstance(collection, AlgebraicAllPlaceResidueResult):
        raise TypeError("logarithmic reconstruction needs all-place residue data")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the finite lattice degree bound must be nonnegative")

    extension = collection.basis.extension
    zero = extension.field.zero
    if collection.status is AlgebraicAllPlaceResidueStatus.EXACT:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.EXACT,
            None,
            None,
            None,
            None,
            zero,
            zero,
            "Hermite reduction already produced an exact antiderivative",
        )
    if collection.status is AlgebraicAllPlaceResidueStatus.UNSUPPORTED:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            None,
            None,
            None,
            None,
            zero,
            collection.remainder,
            "all-place residue collection is unsupported: " + collection.detail,
        )
    if collection.model is None or collection.constant_field_analysis is None:
        raise ValueError("a collected algebraic differential lacks global data")

    _parent, constants, _variable = _base_polynomial_ring(extension)
    if constants != QQ:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            None,
            None,
            None,
            None,
            zero,
            collection.remainder,
            "rational-residue reconstruction currently requires displayed constants QQ",
        )

    rational_records: list[tuple[AlgebraicLocalDifferentialResidue, FieldElement]] = []
    for record in collection.residues:
        residue = _rational_residue(record)
        if residue is None:
            return AlgebraicIntegrationResult(
                collection,
                AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE,
                None,
                None,
                None,
                None,
                zero,
                collection.remainder,
                "a local residue is not a rational displayed constant; "
                "compatible constant base change is required",
            )
        if residue != QQ.zero:
            rational_records.append((record, residue))

    if not rational_records:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER,
            None,
            None,
            None,
            None,
            zero,
            collection.remainder,
            "all collected residues vanish; first- and second-kind exactness "
            "remains to be decided",
        )

    scale = _primitive_rational_scale(
        tuple(residue for _record, residue in rational_records),
    )
    divisor = AlgebraicDivisor(
        collection.model,
        tuple(
            (record.place, _integer_multiple(residue, scale))
            for record, residue in rational_records
        ),
    )
    if divisor.degree != 0:
        raise ValueError("rational residues violated the global residue theorem")

    search = search_algebraic_principal_divisor(
        collection.model,
        divisor,
        collection.constant_field_analysis,
        max_x_degree=max_x_degree,
    )
    if search.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.INCOMPLETE,
            scale,
            divisor,
            search,
            None,
            zero,
            collection.remainder,
            "the rational residue divisor needs a larger principal-divisor "
            "search bound",
        )
    if search.status is not AlgebraicPrincipalDivisorStatus.PRINCIPAL:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            scale,
            divisor,
            search,
            None,
            zero,
            collection.remainder,
            "the rational residue divisor could not be reconstructed in the "
            "current principal-divisor kernel: " + search.detail,
        )
    if search.witness is None:
        raise ValueError("a principal-divisor search omitted its witness")

    argument = search.witness
    coefficient = extension.field.from_sympy(QQ.to_sympy(scale))
    logarithmic_derivative = coefficient * collection.derivation(argument) / argument
    remainder = collection.remainder - logarithmic_derivative
    status = (
        AlgebraicIntegrationStatus.ELEMENTARY
        if remainder == zero
        else AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    )
    detail = (
        "rational residues were reconstructed by a verified principal divisor"
        if status is AlgebraicIntegrationStatus.ELEMENTARY
        else "rational residues were reconstructed; the remaining residue-free "
        "differential needs first- and second-kind exactness"
    )
    return AlgebraicIntegrationResult(
        collection,
        status,
        scale,
        divisor,
        search,
        argument,
        logarithmic_derivative,
        remainder,
        detail,
    )


def integrate_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    max_x_degree: int = 1,
    lazy: bool = False,
) -> AlgebraicIntegrationResult:
    r"""Run the current algebraic Hermite/rational-residue integration slice.

    The function is constructive for exact differentials and for the
    rational-residue cases whose divisor has a witness within the requested
    finite lattice bound. It returns structured boundaries for algebraic
    residue base change, bounded misses, and the residue-free exactness work
    that remains outside this first slice.
    """
    collection = collect_algebraic_all_place_residues(
        integrand,
        derivation,
        basis,
        lazy=lazy,
    )
    return reconstruct_rational_residue_logarithms(
        collection,
        max_x_degree=max_x_degree,
    )
