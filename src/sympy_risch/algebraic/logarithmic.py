r"""First logarithmic reconstruction step for algebraic differentials.

The all-place collector gives each residue in its own closed residue field.
When every residue is already in the displayed constant field, its exact
coordinates over ``QQ`` turn the residue data into finitely many integral
degree-zero divisors.  A verified principal witness for each divisor produces
the corresponding logarithmic terms.  Residues that live only in a closed
residue extension still need compatible algebraic constant base change.

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
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
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
class AlgebraicLogarithmicTerm:
    r"""One verified term ``c * log(f)`` in an algebraic integral.

    ``divisor`` is the integral divisor carrying the rational coordinate of
    the residues, while ``coefficient`` is its displayed algebraic-constant
    basis factor.  The principal-divisor witness therefore verifies the
    field-level identity ``c * D(f) / f`` without choosing logarithm branches.
    """

    coefficient: FieldElement
    divisor: AlgebraicDivisor
    principal_search: AlgebraicPrincipalDivisorSearch
    argument: ExtensionElement

    def __post_init__(self) -> None:
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a logarithmic term needs an algebraic divisor")
        if not isinstance(self.principal_search, AlgebraicPrincipalDivisorSearch):
            raise TypeError("a logarithmic term needs a principal-divisor search")
        if self.principal_search.model is not self.divisor.model:
            raise ValueError("the principal search belongs to another model")
        if self.principal_search.divisor != self.divisor:
            raise ValueError("the principal search has another residue divisor")
        if (
            self.principal_search.status
            is not AlgebraicPrincipalDivisorStatus.PRINCIPAL
        ):
            raise ValueError("a logarithmic term needs a principal divisor witness")
        if self.principal_search.witness is None:
            raise ValueError("a principal logarithmic term needs its witness")

        extension = self.divisor.model.extension
        _parent, constants, _variable = _base_polynomial_ring(extension)
        try:
            coefficient = constants.convert(self.coefficient)
        except (CoercionFailed, TypeError, ValueError) as exc:
            raise ValueError(
                "a logarithmic coefficient must lie in the displayed constants"
            ) from exc
        argument = _extension_element(extension, self.argument)
        if argument == extension.field.zero:
            raise ValueError("a logarithmic argument must be nonzero")
        if argument - self.principal_search.witness != extension.field.zero:
            raise ValueError("a logarithmic argument differs from its witness")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "argument", argument)


@dataclass(frozen=True, slots=True)
class AlgebraicIntegrationResult:
    r"""One exact algebraic Hermite/logarithmic reduction result.

    Its native-field invariant is

    ``integrand = D(exact_part) + logarithmic_derivative + remainder``.

    ``ELEMENTARY`` and ``EXACT`` are constructive successes. A
    ``RESIDUE_FREE_REMAINDER`` has had every displayed-constant logarithmic
    residue removed, but still needs the later exactness decision for first-
    and second-kind differentials. ``INCOMPLETE`` means only that the
    requested finite principal-divisor search bound did not find every
    witness.
    """

    collection: AlgebraicAllPlaceResidueResult
    status: AlgebraicIntegrationStatus
    logarithmic_terms: tuple[AlgebraicLogarithmicTerm, ...]
    principal_searches: tuple[AlgebraicPrincipalDivisorSearch, ...]
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
        terms = tuple(self.logarithmic_terms)
        if not all(isinstance(term, AlgebraicLogarithmicTerm) for term in terms):
            raise TypeError("an algebraic integration result needs logarithmic terms")
        if any(term.divisor.model is not self.collection.model for term in terms):
            raise ValueError("a logarithmic term belongs to another function field")
        searches = tuple(self.principal_searches)
        if not all(
            isinstance(search, AlgebraicPrincipalDivisorSearch) for search in searches
        ):
            raise TypeError("an algebraic integration result needs principal searches")
        if self.collection.model is not None and any(
            search.model is not self.collection.model for search in searches
        ):
            raise ValueError("a principal search belongs to another function field")
        if any(term.principal_search not in searches for term in terms):
            raise ValueError("a logarithmic term lacks its recorded principal search")
        object.__setattr__(self, "logarithmic_terms", terms)
        object.__setattr__(self, "principal_searches", searches)

        remainder = _extension_element(extension, self.remainder)
        object.__setattr__(self, "remainder", remainder)
        reconstructed = (
            self.collection.derivation(self.collection.exact_part)
            + self.logarithmic_derivative
            + remainder
        )
        if reconstructed - self.collection.integrand != extension.field.zero:
            raise ValueError("the algebraic integration result does not reconstruct")

        if self.status is AlgebraicIntegrationStatus.EXACT:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.EXACT
                or remainder != extension.field.zero
                or terms
                or searches
            ):
                raise ValueError("an exact result needs an exact zero-remainder input")
            return

        if self.status is AlgebraicIntegrationStatus.ELEMENTARY:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
                or remainder != extension.field.zero
                or not terms
                or len(searches) != len(terms)
            ):
                raise ValueError("an elementary result needs a logarithmic witness")
            return

        if self.status is AlgebraicIntegrationStatus.INCOMPLETE:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
                or terms
                or not searches
                or not any(
                    search.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE
                    for search in searches
                )
            ):
                raise ValueError("an incomplete result needs its failed divisor search")
            return

        if self.status is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE:
            if terms or searches:
                raise ValueError("a base-change boundary cannot carry a logarithm")
            return

        if self.status is AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER:
            if remainder == extension.field.zero:
                raise ValueError("a residue-free remainder result needs a remainder")
            if len(searches) != len(terms):
                raise ValueError("a reconstructed logarithmic term needs its search")
            return

        if self.status is AlgebraicIntegrationStatus.UNSUPPORTED:
            if terms:
                raise ValueError("an unsupported result cannot carry logarithmic terms")
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
    def logarithmic_derivative(self) -> ExtensionElement:
        """Return the field derivative represented by every logarithmic term."""
        extension = self.collection.basis.extension
        _parent, constants, _variable = _base_polynomial_ring(extension)
        result = extension.field.zero
        for term in self.logarithmic_terms:
            coefficient = extension.field.from_sympy(
                constants.to_sympy(constants.convert(term.coefficient)),
            )
            result += (
                coefficient * self.collection.derivation(term.argument) / term.argument
            )
        return result

    @property
    def logarithmic_coefficient(self) -> FieldElement | None:
        """Return the coefficient when the result has exactly one log term."""
        if len(self.logarithmic_terms) != 1:
            return None
        return self.logarithmic_terms[0].coefficient

    @property
    def residue_divisor(self) -> AlgebraicDivisor | None:
        """Return the divisor when the result has exactly one log term."""
        if len(self.logarithmic_terms) != 1:
            return None
        return self.logarithmic_terms[0].divisor

    @property
    def principal_search(self) -> AlgebraicPrincipalDivisorSearch | None:
        """Return the search when the result has exactly one divisor search."""
        if len(self.principal_searches) != 1:
            return None
        return self.principal_searches[0]

    @property
    def logarithmic_argument(self) -> ExtensionElement | None:
        """Return the argument when the result has exactly one log term."""
        if len(self.logarithmic_terms) != 1:
            return None
        return self.logarithmic_terms[0].argument

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
        _parent, constants, _variable = _base_polynomial_ring(
            self.collection.basis.extension,
        )
        exact_part = field.to_sympy(self.collection.exact_part)
        return exact_part + sum(
            constants.to_sympy(constants.convert(term.coefficient))
            * log(field.to_sympy(term.argument))
            for term in self.logarithmic_terms
        )


def _displayed_constant_residue(
    local_residue: AlgebraicLocalDifferentialResidue,
    constants: Domain,
) -> FieldElement | None:
    """Return a residue in the displayed constants, if it has one.

    A closed-place coordinate tuple begins with the base constant and then
    uses positive powers of the local residue generators.  Thus precisely
    ``(c, 0, ..., 0)`` represents a value already in the displayed constant
    field; any other coordinate needs a compatible constant-base change.
    """
    coordinates = local_residue.constant_coordinates
    try:
        zero = constants.zero
    except AttributeError:
        return None
    if not coordinates or any(coordinate != zero for coordinate in coordinates[1:]):
        return None
    try:
        return constants.convert(coordinates[0])
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


def _constant_coordinate_basis(
    constants: Domain,
) -> tuple[FieldElement, ...] | None:
    """Return the exact ``QQ`` power basis of supported displayed constants."""
    if constants == QQ:
        return (QQ.one,)
    if not isinstance(constants, AlgebraicField) or constants.dom != QQ:
        return None
    degree = constants.ext.minpoly.degree()
    return tuple(constants.unit**power for power in range(degree))


def _rational_coordinates_of_constant(
    constants: Domain,
    value: FieldElement,
) -> tuple[FieldElement, ...] | None:
    """Express one displayed constant in its exact ``QQ`` power basis."""
    if constants == QQ:
        try:
            return (QQ.convert(value),)
        except (CoercionFailed, TypeError, ValueError):
            return None
    if not isinstance(constants, AlgebraicField) or constants.dom != QQ:
        return None
    try:
        converted = constants.convert(value)
        coefficients = tuple(QQ.convert(coefficient) for coefficient in converted.rep)
    except (AttributeError, CoercionFailed, TypeError, ValueError):
        return None
    degree = constants.ext.minpoly.degree()
    if len(coefficients) > degree:
        raise ValueError("an algebraic constant has an overlong power representation")
    # SymPy stores ANP coefficients from highest to lowest power.  Pad on the
    # high side, then reverse to match (1, alpha, ..., alpha**(d - 1)).
    return tuple(reversed((QQ.zero,) * (degree - len(coefficients)) + coefficients))


def reconstruct_constant_residue_logarithms(
    collection: AlgebraicAllPlaceResidueResult,
    *,
    max_x_degree: int,
) -> AlgebraicIntegrationResult:
    r"""Reconstruct displayed-constant logarithms from all-place data.

    Each local residue must have constant-coordinate form ``(c, 0, ..., 0)``.
    Over ``QQ(alpha)`` the values ``c`` are then expanded in the exact power
    basis ``(1, alpha, ..., alpha**(d - 1))``.  Every rational coordinate
    gives an integral residue divisor and, when that divisor has a verified
    principal witness, one term in the logarithmic part.  This is constructive
    descent over the *displayed* algebraic constants; local residues outside
    that field still require compatible constant base change and place
    splitting.
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
            (),
            (),
            zero,
            "Hermite reduction already produced an exact antiderivative",
        )
    if collection.status is AlgebraicAllPlaceResidueStatus.UNSUPPORTED:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            (),
            (),
            collection.remainder,
            "all-place residue collection is unsupported: " + collection.detail,
        )
    if collection.model is None or collection.constant_field_analysis is None:
        raise ValueError("a collected algebraic differential lacks global data")

    _parent, constants, _variable = _base_polynomial_ring(extension)
    constant_basis = _constant_coordinate_basis(constants)
    if constant_basis is None:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            (),
            (),
            collection.remainder,
            "constant-residue reconstruction currently requires QQ or a finite "
            "algebraic extension of QQ",
        )

    constant_records: list[
        tuple[AlgebraicLocalDifferentialResidue, tuple[FieldElement, ...]]
    ] = []
    for record in collection.residues:
        residue = _displayed_constant_residue(record, constants)
        if residue is None:
            return AlgebraicIntegrationResult(
                collection,
                AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE,
                (),
                (),
                collection.remainder,
                "a local residue is not a displayed constant; "
                "compatible constant base change is required",
            )
        coordinates = _rational_coordinates_of_constant(constants, residue)
        if coordinates is None:
            return AlgebraicIntegrationResult(
                collection,
                AlgebraicIntegrationStatus.UNSUPPORTED,
                (),
                (),
                collection.remainder,
                "the displayed constant could not be expanded over QQ",
            )
        if len(coordinates) != len(constant_basis):
            raise ValueError("a displayed constant has the wrong coordinate dimension")
        if any(coordinate != QQ.zero for coordinate in coordinates):
            constant_records.append((record, coordinates))

    if not constant_records:
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER,
            (),
            (),
            collection.remainder,
            "all collected residues vanish; first- and second-kind exactness "
            "remains to be decided",
        )

    plans: list[
        tuple[FieldElement, AlgebraicDivisor, AlgebraicPrincipalDivisorSearch]
    ] = []
    for coordinate_index, basis_element in enumerate(constant_basis):
        coordinate_records = tuple(
            (record, coordinates[coordinate_index])
            for record, coordinates in constant_records
            if coordinates[coordinate_index] != QQ.zero
        )
        if not coordinate_records:
            continue
        scale = _primitive_rational_scale(
            tuple(residue for _record, residue in coordinate_records),
        )
        divisor = AlgebraicDivisor(
            collection.model,
            tuple(
                (record.place, _integer_multiple(residue, scale))
                for record, residue in coordinate_records
            ),
        )
        if divisor.degree != 0:
            raise ValueError(
                "displayed-constant residue coordinates violated the global "
                "residue theorem",
            )
        search = search_algebraic_principal_divisor(
            collection.model,
            divisor,
            collection.constant_field_analysis,
            max_x_degree=max_x_degree,
        )
        coefficient = constants.convert(basis_element * scale)
        plans.append((coefficient, divisor, search))

    searches = tuple(search for _coefficient, _divisor, search in plans)
    if any(
        search.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE
        for search in searches
    ):
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.INCOMPLETE,
            (),
            searches,
            collection.remainder,
            "one displayed-constant residue divisor needs a larger "
            "principal-divisor search bound",
        )
    if any(
        search.status is not AlgebraicPrincipalDivisorStatus.PRINCIPAL
        for search in searches
    ):
        return AlgebraicIntegrationResult(
            collection,
            AlgebraicIntegrationStatus.UNSUPPORTED,
            (),
            searches,
            collection.remainder,
            "a displayed-constant residue divisor could not be reconstructed "
            "in the current principal-divisor kernel",
        )
    terms = tuple(
        AlgebraicLogarithmicTerm(coefficient, divisor, search, search.witness)
        for coefficient, divisor, search in plans
        if search.witness is not None
    )
    if len(terms) != len(plans):
        raise ValueError("a principal-divisor search omitted its witness")
    logarithmic_derivative = sum(
        (
            extension.field.from_sympy(
                constants.to_sympy(constants.convert(term.coefficient)),
            )
            * collection.derivation(term.argument)
            / term.argument
            for term in terms
        ),
        extension.field.zero,
    )
    remainder = collection.remainder - logarithmic_derivative
    status = (
        AlgebraicIntegrationStatus.ELEMENTARY
        if remainder == zero
        else AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    )
    detail = (
        "displayed-constant residues were reconstructed by verified principal divisors"
        if status is AlgebraicIntegrationStatus.ELEMENTARY
        else "displayed-constant residues were reconstructed; the remaining "
        "residue-free "
        "differential needs first- and second-kind exactness"
    )
    return AlgebraicIntegrationResult(
        collection,
        status,
        terms,
        searches,
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
    r"""Run the current algebraic Hermite/constant-residue integration slice.

    The function is constructive for exact differentials and for the
    displayed-constant residue cases whose divisors have witnesses within the
    requested finite lattice bound. It returns structured boundaries for
    algebraic residue base change, bounded misses, and the residue-free
    exactness work that remains outside this first slice.
    """
    collection = collect_algebraic_all_place_residues(
        integrand,
        derivation,
        basis,
        lazy=lazy,
    )
    return reconstruct_constant_residue_logarithms(
        collection,
        max_x_degree=max_x_degree,
    )
