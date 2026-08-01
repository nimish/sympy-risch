r"""First logarithmic reconstruction step for algebraic differentials.

The all-place collector gives each residue in its own closed residue field.
When every residue is already in the displayed constant field, its exact
coordinates over ``QQ`` turn the residue data into finitely many integral
degree-zero divisors.  A verified principal witness for each divisor produces
the corresponding logarithmic terms.  Residues that live only in a closed
residue extension still need compatible algebraic constant base change.

This module keeps that result separate from the general elementary dispatcher.
A certified nontorsion residue coordinate is a Liouville obstruction.  After
all residues have been removed, a complete primitive-space decision also
certifies non-elementarity when the remaining differential is not exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from sympy.core.expr import Expr
from sympy.functions.elementary.exponential import log
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed

from .._typing import FieldElement
from ..algebraic_geometry.divisor_classes import (
    AlgebraicTorsionStatus,
    search_algebraic_divisor_torsion,
)
from ..algebraic_geometry.finite_places import FiniteAlgebraicPlace
from ..algebraic_geometry.finite_torsion import (
    AlgebraicDivisorNontorsion,
    AlgebraicDivisorTorsion,
    decide_algebraic_divisor_torsion,
    prepare_algebraic_torsion_reductions,
)
from ..algebraic_geometry.function_field import AlgebraicDivisor
from ..algebraic_geometry.principal import (
    AlgebraicPrincipalityDecision,
    AlgebraicPrincipalityStatus,
    decide_algebraic_principality,
    principality_from_torsion_search,
)
from ..constant_adapters import adapt_risch_constant_field_analysis
from ..tower import Derivation
from ._field import (
    _base_polynomial_ring,
    _extension_element,
    _quotient_extension_elements,
)
from .all_places import (
    AlgebraicAllPlaceResidueResult,
    AlgebraicAllPlaceResidueStatus,
    AlgebraicLocalDifferentialResidue,
    collect_algebraic_all_place_residues,
)
from .basis import IntegralBasis
from .constant_analysis import AlgebraicConstantFieldStatus
from .exactness import (
    AlgebraicExactnessCertificate,
    AlgebraicExactnessResult,
    AlgebraicExactnessStatus,
    solve_algebraic_exactness,
)
from .liouville import (
    AlgebraicExactnessLiouvilleObstruction,
    AlgebraicLiouvilleObstruction,
)
from .residue_coordinates import (
    AlgebraicResidueCoordinateCertificate,
    constant_coordinate_basis,
    displayed_constant_residue,
    integer_multiple,
    primitive_rational_scale,
    rational_coordinates_of_constant,
)


class AlgebraicIntegrationStatus(Enum):
    """Derived tag for the concrete algebraic integration result variants."""

    EXACT = "exact"
    ELEMENTARY = "elementary"
    NON_ELEMENTARY = "non_elementary"
    RESIDUE_FREE_REMAINDER = "residue_free_remainder"
    NEEDS_CONSTANT_BASE_CHANGE = "needs_constant_base_change"
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
    principality: AlgebraicPrincipalityDecision
    argument: ExtensionElement

    def __post_init__(self) -> None:
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a logarithmic term needs an algebraic divisor")
        if not isinstance(self.principality, AlgebraicPrincipalityDecision):
            raise TypeError("a logarithmic term needs a principality decision")
        if self.principality.model is not self.divisor.model:
            raise ValueError("the principality decision belongs to another model")
        if self.principality.divisor != self.divisor:
            raise ValueError("the principality decision has another residue divisor")
        if self.principality.status is not AlgebraicPrincipalityStatus.PRINCIPAL:
            raise ValueError("a logarithmic term needs a principal divisor witness")
        if self.principality.witness is None:
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
        if argument - self.principality.witness != extension.field.zero:
            raise ValueError("a logarithmic argument differs from its witness")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "argument", argument)


class _AlgebraicIntegrationResult:
    r"""One exact algebraic Hermite/logarithmic reduction result.

    Its native-field invariant is

    ``integrand = D(exact_part) + logarithmic_derivative + remainder``.

    Here ``exact_part`` is the Hermite exact part plus an optional exact
    correction found after logarithmic residues have been removed.

    ``ELEMENTARY`` and ``EXACT`` are constructive successes.
    ``NON_ELEMENTARY`` carries either a nontorsion residue obstruction or a
    complete residue-free nonexactness obstruction. A
    ``RESIDUE_FREE_REMAINDER`` has had every displayed-constant logarithmic
    residue removed. Its optional ``exactness`` record says whether the
    remaining first-/second-kind differential was decided in the current
    algebraic field. Both logarithmic reconstruction and residue-free
    exactness use complete function-field spaces when their AFF arithmetic is
    supported.
    """

    collection: AlgebraicAllPlaceResidueResult
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus]
    remainder: ExtensionElement
    exact_correction: ExtensionElement

    def _validate_common(self) -> None:
        if not isinstance(self.collection, AlgebraicAllPlaceResidueResult):
            raise TypeError("an algebraic integration result needs all-place data")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic integration result needs a detail")
        obstruction = getattr(self, "liouville_obstruction", None)
        if obstruction is not None and (
            not isinstance(
                obstruction,
                (
                    AlgebraicLiouvilleObstruction,
                    AlgebraicExactnessLiouvilleObstruction,
                ),
            )
            or obstruction.collection is not self.collection
        ):
            raise ValueError("the Liouville obstruction belongs to another result")

        extension = self.collection.basis.extension
        terms = getattr(self, "logarithmic_terms", ())
        if not isinstance(terms, tuple) or not all(
            isinstance(term, AlgebraicLogarithmicTerm) for term in terms
        ):
            raise TypeError("an algebraic integration result needs logarithmic terms")
        if any(term.divisor.model is not self.collection.model for term in terms):
            raise ValueError("a logarithmic term belongs to another function field")
        decisions = getattr(self, "principality_decisions", ())
        if not isinstance(decisions, tuple) or not all(
            isinstance(decision, AlgebraicPrincipalityDecision)
            for decision in decisions
        ):
            raise TypeError(
                "an algebraic integration result needs principality decisions"
            )
        if self.collection.model is not None and any(
            decision.model is not self.collection.model for decision in decisions
        ):
            raise ValueError(
                "a principality decision belongs to another function field"
            )
        if any(term.principality not in decisions for term in terms):
            raise ValueError(
                "a logarithmic term lacks its recorded principality decision"
            )
        remainder = _extension_element(extension, self.remainder)
        if remainder != self.remainder:
            raise ValueError("an algebraic remainder is not in canonical field form")
        correction = _extension_element(extension, self.exact_correction)
        exactness = getattr(self, "exactness", None)
        if exactness is not None:
            if not isinstance(exactness, AlgebraicExactnessResult):
                raise TypeError("an algebraic integration result has invalid exactness")
            if (
                self.collection.model is None
                or exactness.model is not self.collection.model
            ):
                raise ValueError("the exactness result belongs to another model")
            if not exactness.derivation.has_same_position(self.collection.derivation):
                raise ValueError("the exactness result belongs to another derivation")
            if (
                exactness.coefficient
                - (self.collection.derivation(correction) + remainder)
                != extension.field.zero
            ):
                raise ValueError("the exactness result has another remainder")
            if exactness.is_exact:
                if (
                    correction != exactness.primitive
                    or remainder != extension.field.zero
                ):
                    raise ValueError("an exactness witness was not applied completely")
            elif correction != extension.field.zero:
                raise ValueError(
                    "only an exactness witness can add an exact correction"
                )
        elif correction != extension.field.zero:
            raise ValueError("an exact correction needs its exactness result")
        reconstructed = (
            self.collection.derivation(self.collection.exact_part + correction)
            + self.logarithmic_derivative
            + remainder
        )
        if reconstructed - self.collection.integrand != extension.field.zero:
            raise ValueError("the algebraic integration result does not reconstruct")

        if self.status is AlgebraicIntegrationStatus.EXACT:
            if (
                self.collection.status
                not in {
                    AlgebraicAllPlaceResidueStatus.EXACT,
                    AlgebraicAllPlaceResidueStatus.COLLECTED,
                }
                or remainder != extension.field.zero
                or terms
                or decisions
                or obstruction is not None
            ):
                raise ValueError("an exact result needs an exact zero-remainder input")
            if self.collection.status is AlgebraicAllPlaceResidueStatus.COLLECTED and (
                exactness is None or not exactness.is_exact
            ):
                raise ValueError("a collected exact result needs an exactness witness")
            return

        if self.status is AlgebraicIntegrationStatus.ELEMENTARY:
            if (
                self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED
                or remainder != extension.field.zero
                or not terms
                or len(decisions) != len(terms)
                or obstruction is not None
            ):
                raise ValueError("an elementary result needs a logarithmic witness")
            return

        if self.status is AlgebraicIntegrationStatus.NON_ELEMENTARY:
            if self.collection.status is not AlgebraicAllPlaceResidueStatus.COLLECTED:
                raise ValueError(
                    "a non-elementary result needs complete all-place residue data"
                )
            if isinstance(obstruction, AlgebraicLiouvilleObstruction):
                valid = (
                    not terms
                    and obstruction.principality in decisions
                    and exactness is None
                    and correction == extension.field.zero
                    and remainder == self.collection.remainder
                )
            elif isinstance(obstruction, AlgebraicExactnessLiouvilleObstruction):
                valid = (
                    exactness is obstruction.exactness
                    and correction == extension.field.zero
                    and remainder == exactness.coefficient
                    and len(decisions) == len(terms)
                )
            else:
                valid = False
            if not valid:
                raise ValueError(
                    "a non-elementary result needs its certified Liouville obstruction"
                )
            return

        if self.status is AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE:
            if terms or decisions or obstruction is not None:
                raise ValueError("a base-change boundary cannot carry a logarithm")
            return

        if self.status is AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER:
            if remainder == extension.field.zero:
                raise ValueError("a residue-free remainder result needs a remainder")
            if len(decisions) != len(terms):
                raise ValueError("a reconstructed logarithmic term needs its decision")
            if (
                exactness is not None
                and exactness.status is AlgebraicExactnessStatus.EXACT
            ):
                raise ValueError("an exactness witness must promote the integration")
            if obstruction is not None:
                raise ValueError("a residue-free result cannot carry an obstruction")
            return

        if self.status is AlgebraicIntegrationStatus.UNSUPPORTED:
            if terms or obstruction is not None:
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
    def complete_negative(self) -> bool:
        """Whether a Liouville obstruction proves non-elementarity."""
        return self.status is AlgebraicIntegrationStatus.NON_ELEMENTARY

    @property
    def reconstructs(self) -> bool:
        """Whether the retained native decomposition reconstructs the input."""
        reconstructed = (
            self.collection.derivation(
                self.collection.exact_part + self.exact_correction
            )
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
        for term in getattr(self, "logarithmic_terms", ()):
            coefficient = extension.field.from_sympy(
                constants.to_sympy(constants.convert(term.coefficient)),
            )
            result += coefficient * _quotient_extension_elements(
                extension,
                self.collection.derivation(term.argument),
                term.argument,
            )
        return result

    @property
    def exact_part(self) -> ExtensionElement:
        """Return the complete algebraic exact part of the decomposition."""
        return self.collection.exact_part + self.exact_correction

    @property
    def logarithmic_coefficient(self) -> FieldElement | None:
        """Return the coefficient when the result has exactly one log term."""
        terms = getattr(self, "logarithmic_terms", ())
        if len(terms) != 1:
            return None
        return terms[0].coefficient

    @property
    def residue_divisor(self) -> AlgebraicDivisor | None:
        """Return the divisor when the result has exactly one log term."""
        terms = getattr(self, "logarithmic_terms", ())
        if len(terms) != 1:
            return None
        return terms[0].divisor

    @property
    def principality_decision(self) -> AlgebraicPrincipalityDecision | None:
        """Return the decision when the result has exactly one residue divisor."""
        decisions = getattr(self, "principality_decisions", ())
        if len(decisions) != 1:
            return None
        return decisions[0]

    @property
    def logarithmic_argument(self) -> ExtensionElement | None:
        """Return the argument when the result has exactly one log term."""
        terms = getattr(self, "logarithmic_terms", ())
        if len(terms) != 1:
            return None
        return terms[0].argument

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
        exact_part = field.to_sympy(self.exact_part)
        return exact_part + sum(
            constants.to_sympy(constants.convert(term.coefficient))
            * log(field.to_sympy(term.argument))
            for term in getattr(self, "logarithmic_terms", ())
        )


@dataclass(frozen=True, slots=True)
class AlgebraicExactIntegral(_AlgebraicIntegrationResult):
    """A purely algebraic antiderivative with no logarithmic terms."""

    collection: AlgebraicAllPlaceResidueResult
    detail: str
    exactness: AlgebraicExactnessResult | None = None
    status: ClassVar[AlgebraicIntegrationStatus] = AlgebraicIntegrationStatus.EXACT

    @property
    def remainder(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    @property
    def exact_correction(self) -> ExtensionElement:
        if self.exactness is not None and self.exactness.primitive is not None:
            return self.exactness.primitive
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicElementaryIntegral(_AlgebraicIntegrationResult):
    """An algebraic exact part plus one or more verified logarithmic terms."""

    collection: AlgebraicAllPlaceResidueResult
    logarithmic_terms: tuple[AlgebraicLogarithmicTerm, ...] = field()
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = field()
    detail: str
    exactness: AlgebraicExactnessResult | None = None
    status: ClassVar[AlgebraicIntegrationStatus] = AlgebraicIntegrationStatus.ELEMENTARY

    @property
    def remainder(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    @property
    def exact_correction(self) -> ExtensionElement:
        if self.exactness is not None and self.exactness.primitive is not None:
            return self.exactness.primitive
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicResidueNonElementary(_AlgebraicIntegrationResult):
    """Non-elementarity proved by a nontorsion residue coordinate."""

    collection: AlgebraicAllPlaceResidueResult
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = field()
    liouville_obstruction: AlgebraicLiouvilleObstruction = field()
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.NON_ELEMENTARY
    )

    @property
    def remainder(self) -> ExtensionElement:
        return self.collection.remainder

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicExactnessNonElementary(_AlgebraicIntegrationResult):
    """Non-elementarity proved by complete residue-free nonexactness."""

    collection: AlgebraicAllPlaceResidueResult
    logarithmic_terms: tuple[AlgebraicLogarithmicTerm, ...] = field()
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = field()
    remainder: ExtensionElement
    exactness: AlgebraicExactnessResult = field()
    liouville_obstruction: AlgebraicExactnessLiouvilleObstruction = field()
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.NON_ELEMENTARY
    )

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicResidueFreeRemainder(_AlgebraicIntegrationResult):
    """A nonzero residue-free remainder whose exactness has not been run."""

    collection: AlgebraicAllPlaceResidueResult
    logarithmic_terms: tuple[AlgebraicLogarithmicTerm, ...] = field()
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = field()
    remainder: ExtensionElement
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    )

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicExactnessBoundary(_AlgebraicIntegrationResult):
    """A residue-free remainder with an explicit incomplete exactness result."""

    collection: AlgebraicAllPlaceResidueResult
    logarithmic_terms: tuple[AlgebraicLogarithmicTerm, ...] = field()
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = field()
    remainder: ExtensionElement
    exactness: AlgebraicExactnessResult = field()
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.RESIDUE_FREE_REMAINDER
    )

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicNeedsConstantBaseChange(_AlgebraicIntegrationResult):
    """A complete local-residue collection requiring larger constants."""

    collection: AlgebraicAllPlaceResidueResult
    detail: str
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.NEEDS_CONSTANT_BASE_CHANGE
    )

    @property
    def remainder(self) -> ExtensionElement:
        return self.collection.remainder

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


@dataclass(frozen=True, slots=True)
class AlgebraicIntegrationUnsupported(_AlgebraicIntegrationResult):
    """An exact unsupported boundary with no negative mathematical claim."""

    collection: AlgebraicAllPlaceResidueResult
    detail: str
    principality_decisions: tuple[AlgebraicPrincipalityDecision, ...] = ()
    status: ClassVar[AlgebraicIntegrationStatus] = (
        AlgebraicIntegrationStatus.UNSUPPORTED
    )

    @property
    def remainder(self) -> ExtensionElement:
        return self.collection.remainder

    @property
    def exact_correction(self) -> ExtensionElement:
        return self.collection.basis.extension.field.zero

    def __post_init__(self) -> None:
        self._validate_common()


AlgebraicIntegrationResult = (
    AlgebraicExactIntegral
    | AlgebraicElementaryIntegral
    | AlgebraicResidueNonElementary
    | AlgebraicExactnessNonElementary
    | AlgebraicResidueFreeRemainder
    | AlgebraicExactnessBoundary
    | AlgebraicNeedsConstantBaseChange
    | AlgebraicIntegrationUnsupported
)


def _rational_parameter_plans(
    collection: AlgebraicAllPlaceResidueResult,
    constants: Domain,
    records: tuple[
        tuple[
            AlgebraicLocalDifferentialResidue,
            FieldElement,
            tuple[FieldElement, ...],
        ],
        ...,
    ],
) -> (
    tuple[tuple[FieldElement, AlgebraicDivisor, AlgebraicPrincipalityDecision], ...]
    | None
):
    r"""Reconstruct degree-one residues through a rational parameter.

    If ``t`` has one simple pole, then ``div(t - t(P)) = P - P_infinity`` for
    every degree-one finite point ``P``.  This avoids expanding algebraic
    residues into a rational power basis, which can create needlessly large
    integer-coordinate divisors.  Every proposed divisor and witness is still
    verified; failure simply returns to the general coordinate algorithm.
    """
    model = collection.model
    if model is None:
        return None
    extension = model.extension
    _parent, _constants, variable = _base_polynomial_ring(extension)
    parameters = (
        extension.field.generator,
        extension.field.from_sympy(variable),
    )
    for parameter in parameters:
        parameter_divisor = model.principal_divisor(parameter)
        pole_divisor = parameter_divisor.negative_part
        if (
            pole_divisor.degree != 1
            or len(pole_divisor.terms) != 1
            or pole_divisor.terms[0][1] != 1
        ):
            continue
        pole = pole_divisor.terms[0][0]
        plans: list[
            tuple[FieldElement, AlgebraicDivisor, AlgebraicPrincipalityDecision]
        ] = []
        nonpole_sum = constants.zero
        pole_residue = constants.zero
        supported = True
        for record, residue, _coordinates in records:
            converted_residue = constants.convert(residue)
            if record.place is pole:
                pole_residue += converted_residue
                continue
            if (
                not isinstance(record.place, FiniteAlgebraicPlace)
                or record.place.degree != 1
            ):
                supported = False
                break
            value_at_place = record.place.lift_residue(
                record.place.reduce_local(parameter)
            )
            argument = parameter - value_at_place
            divisor = AlgebraicDivisor(
                model,
                ((record.place, 1), (pole, -1)),
            )
            if model.principal_divisor(argument) != divisor:
                supported = False
                break
            decision = decide_algebraic_principality(model, divisor)
            if decision.status is not AlgebraicPrincipalityStatus.PRINCIPAL:
                supported = False
                break
            plans.append((converted_residue, divisor, decision))
            nonpole_sum += converted_residue
        if supported and plans and pole_residue == -nonpole_sum:
            return tuple(plans)
    return None


def reconstruct_constant_residue_logarithms(
    collection: AlgebraicAllPlaceResidueResult,
    *,
    torsion_order_bound: int | None = None,
) -> AlgebraicIntegrationResult:
    r"""Reconstruct displayed-constant logarithms from all-place data.

    Each local residue must have constant-coordinate form ``(c, 0, ..., 0)``.
    Over ``QQ(alpha)`` the values ``c`` are then expanded in the exact power
    basis ``(1, alpha, ..., alpha**(d - 1))``.  Every rational coordinate
    gives an integral residue divisor and, when that divisor has a verified
    principal witness, one term in the logarithmic part.  If the primitive
    coordinate divisor is nonprincipal, finite-prime divisor-class arithmetic
    determines a candidate torsion order and verifies it in characteristic
    zero.  Supplying ``torsion_order_bound`` instead requests the older bounded
    positive search.

    This is constructive descent over the *displayed* algebraic constants;
    local residues outside that field still require compatible constant base
    change and place splitting.
    """
    if not isinstance(collection, AlgebraicAllPlaceResidueResult):
        raise TypeError("logarithmic reconstruction needs all-place residue data")
    extension = collection.basis.extension
    zero = extension.field.zero
    if collection.status is AlgebraicAllPlaceResidueStatus.EXACT:
        return AlgebraicExactIntegral(
            collection,
            "Hermite reduction already produced an exact antiderivative",
        )
    if collection.status is AlgebraicAllPlaceResidueStatus.UNSUPPORTED:
        return AlgebraicIntegrationUnsupported(
            collection,
            "all-place residue collection is unsupported: " + collection.detail,
        )
    if collection.model is None or collection.constant_field_analysis is None:
        raise ValueError("a collected algebraic differential lacks global data")
    if (
        collection.constant_field_analysis.status
        is not AlgebraicConstantFieldStatus.PRESERVED
    ):
        return AlgebraicIntegrationUnsupported(
            collection,
            "displayed-constant logarithmic reconstruction requires a certified "
            "preserved constant field",
        )

    _parent, constants, _variable = _base_polynomial_ring(extension)
    constant_basis = constant_coordinate_basis(constants)
    if constant_basis is None:
        return AlgebraicIntegrationUnsupported(
            collection,
            "constant-residue reconstruction currently requires QQ or a finite "
            "algebraic extension of QQ",
        )

    constant_records: list[
        tuple[
            AlgebraicLocalDifferentialResidue,
            FieldElement,
            tuple[FieldElement, ...],
        ]
    ] = []
    for record in collection.residues:
        residue = displayed_constant_residue(record, constants)
        if residue is None:
            return AlgebraicNeedsConstantBaseChange(
                collection,
                "a local residue is not a displayed constant; "
                "compatible constant base change is required",
            )
        coordinates = rational_coordinates_of_constant(constants, residue)
        if coordinates is None:
            return AlgebraicIntegrationUnsupported(
                collection,
                "the displayed constant could not be expanded over QQ",
            )
        if len(coordinates) != len(constant_basis):
            raise ValueError("a displayed constant has the wrong coordinate dimension")
        if any(coordinate != QQ.zero for coordinate in coordinates):
            constant_records.append((record, residue, coordinates))

    if not constant_records:
        return AlgebraicResidueFreeRemainder(
            collection,
            (),
            (),
            collection.remainder,
            "all collected residues vanish; first- and second-kind exactness "
            "remains to be decided",
        )

    # For small displayed constant bases the coordinate decomposition is both
    # compact and gives the traditional combined logarithmic presentation.
    # The parameter form matters once a larger compositum would inflate those
    # coordinates into high-power principal-divisor searches.
    parameter_plans = (
        _rational_parameter_plans(
            collection,
            constants,
            tuple(constant_records),
        )
        if len(constant_basis) > 2
        else None
    )
    plans: list[
        tuple[
            FieldElement,
            AlgebraicDivisor,
            AlgebraicPrincipalityDecision,
            AlgebraicResidueCoordinateCertificate | None,
        ]
    ] = [] if parameter_plans is None else [(*plan, None) for plan in parameter_plans]
    if parameter_plans is None:
        for coordinate_index, basis_element in enumerate(constant_basis):
            coordinate_records = tuple(
                (record, coordinates[coordinate_index])
                for record, _residue, coordinates in constant_records
                if coordinates[coordinate_index] != QQ.zero
            )
            if not coordinate_records:
                continue
            scale = primitive_rational_scale(
                tuple(residue for _record, residue in coordinate_records),
            )
            divisor = AlgebraicDivisor(
                collection.model,
                tuple(
                    (record.place, integer_multiple(residue, scale))
                    for record, residue in coordinate_records
                ),
            )
            if divisor.degree != 0:
                raise ValueError(
                    "displayed-constant residue coordinates violated the global "
                    "residue theorem",
                )
            decision = decide_algebraic_principality(collection.model, divisor)
            coefficient = constants.convert(basis_element * scale)
            coordinate = AlgebraicResidueCoordinateCertificate(
                collection,
                coordinate_index,
                coefficient,
                divisor,
            )
            plans.append((coefficient, divisor, decision, coordinate))

        pending = tuple(
            (index, divisor)
            for index, (_coefficient, divisor, decision, _coordinate) in enumerate(
                plans
            )
            if decision.status is AlgebraicPrincipalityStatus.NONPRINCIPAL
        )
        torsion_reductions = (
            prepare_algebraic_torsion_reductions(
                collection.model,
                adapt_risch_constant_field_analysis(
                    collection.model,
                    collection.constant_field_analysis,
                ),
                divisors=tuple(divisor for _index, divisor in pending),
            )
            if pending and torsion_order_bound is None
            else None
        )
        liouville_obstruction: AlgebraicLiouvilleObstruction | None = None
        for index, divisor in pending:
            coefficient, _old_divisor, decision, coordinate = plans[index]
            if torsion_order_bound is not None:
                search = search_algebraic_divisor_torsion(
                    collection.model,
                    divisor,
                    max_order=torsion_order_bound,
                )
            else:
                if torsion_reductions is None:
                    raise ValueError("finite torsion preparation unexpectedly failed")
                torsion = decide_algebraic_divisor_torsion(
                    collection.model,
                    divisor,
                    reduction_batch=torsion_reductions,
                )
                if isinstance(torsion, AlgebraicDivisorNontorsion):
                    if coordinate is None:
                        raise ValueError(
                            "a nontorsion coordinate lacks replayable residue evidence"
                        )
                    liouville_obstruction = AlgebraicLiouvilleObstruction(
                        coordinate,
                        decision,
                        torsion,
                    )
                    break
                search = (
                    torsion.characteristic_zero_search
                    if isinstance(torsion, AlgebraicDivisorTorsion)
                    else None
                )
            if search is not None and search.status is AlgebraicTorsionStatus.TORSION:
                if search.order is None:
                    raise ValueError("a torsion relation omitted its order")
                divisor = divisor.scale(search.order)
                decision = principality_from_torsion_search(search, search.order)
                coefficient = constants.quo(
                    coefficient,
                    constants.convert(search.order),
                )
            plans[index] = (coefficient, divisor, decision, coordinate)
        if liouville_obstruction is not None:
            return AlgebraicResidueNonElementary(
                collection,
                tuple(
                    decision for _coefficient, _divisor, decision, _coordinate in plans
                ),
                liouville_obstruction,
                "Liouville's residue condition fails: " + liouville_obstruction.detail,
            )

    decisions = tuple(
        decision for _coefficient, _divisor, decision, _coordinate in plans
    )
    if any(
        decision.status is not AlgebraicPrincipalityStatus.PRINCIPAL
        for decision in decisions
    ):
        return AlgebraicIntegrationUnsupported(
            collection,
            "a displayed-constant residue divisor could not be reconstructed "
            "by the current complete principality service",
            decisions,
        )
    terms = tuple(
        AlgebraicLogarithmicTerm(coefficient, divisor, decision, decision.witness)
        for coefficient, divisor, decision, _coordinate in plans
        if decision.witness is not None
    )
    if len(terms) != len(plans):
        raise ValueError("a principality decision omitted its witness")
    logarithmic_derivative = sum(
        (
            extension.field.from_sympy(
                constants.to_sympy(constants.convert(term.coefficient)),
            )
            * _quotient_extension_elements(
                extension,
                collection.derivation(term.argument),
                term.argument,
            )
            for term in terms
        ),
        extension.field.zero,
    )
    remainder = collection.remainder - logarithmic_derivative
    if remainder == zero:
        return AlgebraicElementaryIntegral(
            collection,
            terms,
            decisions,
            "displayed-constant residues were reconstructed by verified "
            "principal divisors",
        )
    return AlgebraicResidueFreeRemainder(
        collection,
        terms,
        decisions,
        remainder,
        "displayed-constant residues were reconstructed; the remaining "
        "residue-free differential needs first- and second-kind exactness",
    )


def integrate_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool = False,
    torsion_order_bound: int | None = None,
) -> AlgebraicIntegrationResult:
    r"""Run the current algebraic Hermite/constant-residue integration slice.

    The function is constructive for exact differentials and for
    displayed-constant residues whose divisors are decided principal by the
    complete function-field service. A residue-free remainder is then tested
    for an algebraic primitive using a complete certified Riemann--Roch space.
    It returns structured boundaries for algebraic residue base change and
    unsupported exactness spaces.  By default, positive divisor-class torsion
    is decided from two good finite reductions and verified in characteristic
    zero.  ``torsion_order_bound`` remains available as an explicit bounded
    positive search.
    """
    collection = collect_algebraic_all_place_residues(
        integrand,
        derivation,
        basis,
        lazy=lazy,
    )
    result = reconstruct_constant_residue_logarithms(
        collection,
        torsion_order_bound=torsion_order_bound,
    )
    if not isinstance(result, AlgebraicResidueFreeRemainder):
        return result

    model = collection.model
    analysis = collection.constant_field_analysis
    if model is None or analysis is None:
        raise ValueError("a residue-free algebraic result lacks global data")
    exactness = solve_algebraic_exactness(
        result.remainder,
        collection.derivation,
        model,
        analysis,
    )
    if exactness.is_exact:
        if exactness.primitive is None:
            raise ValueError("an exactness witness lacks its primitive")
        if result.logarithmic_terms:
            return AlgebraicElementaryIntegral(
                collection,
                result.logarithmic_terms,
                result.principality_decisions,
                "the residue-free remainder has a verified algebraic primitive",
                exactness,
            )
        return AlgebraicExactIntegral(
            collection,
            "the residue-free remainder has a verified algebraic primitive",
            exactness,
        )
    if (
        exactness.status is AlgebraicExactnessStatus.NO_PRIMITIVE
        and exactness.certificate
        is AlgebraicExactnessCertificate.COMPLETE_PRIMITIVE_SPACE
    ):
        obstruction = AlgebraicExactnessLiouvilleObstruction(
            collection,
            exactness,
        )
        return AlgebraicExactnessNonElementary(
            collection,
            result.logarithmic_terms,
            result.principality_decisions,
            result.remainder,
            exactness,
            obstruction,
            "Liouville's residue-free exactness condition fails: " + obstruction.detail,
        )
    return AlgebraicExactnessBoundary(
        collection,
        result.logarithmic_terms,
        result.principality_decisions,
        result.remainder,
        exactness,
        result.detail + "; " + exactness.detail,
    )
