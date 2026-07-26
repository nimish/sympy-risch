r"""Finite logarithmic-relation systems for supported monomial towers.

The central object here is the rational relation space

.. math::

   \{q\in\mathbb{Q}^r : \sum_i q_i a_i=Dv/v\text{ after an integer scale}\}.

It is the Chapter-9 structure-theorem fallback used once a tower contains a
hyperexponential level.  The compiler works downward through the tower.  At
each primitive level it records normal-factor valuations; at each
hyperexponential level it additionally records the special generator
valuation.  Every equality in an intermediate field is recursively expanded
to rational coordinates, so Gaussian constants never accidentally permit
Gaussian (rather than rational) valuations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import lcm

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly, cancel

from ._typing import FieldElement
from .constant_fields import (
    is_supported_finite_constant_field,
    rational_constant_coordinates,
)
from .tower import (
    AlgebraicExtension,
    ConstantFieldStatus,
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)


class FundamentalSignatureOutcome(Enum):
    """Whether a tower has a certified P/E/T fundamental signature."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class FundamentalSignatureReason(Enum):
    """Stable reason a tower lacks a certified fundamental signature."""

    NO_RATIONAL_BASE = "no_rational_base"
    BASE_FIELD = "base_field"
    BASE_DERIVATIVE = "base_derivative"
    ALGEBRAIC_EXTENSION = "algebraic_extension"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    CONSTANT_FIELD_UNKNOWN = "constant_field_unknown"
    CONSTANT_FIELD_ENLARGED = "constant_field_enlarged"
    CONSTANT_FIELD_PROVENANCE = "constant_field_provenance"
    UNSUPPORTED_MONOMIAL_FAMILY = "unsupported_monomial_family"


@dataclass(frozen=True, slots=True)
class PrimitiveFundamentalDirection:
    r"""One primitive direction ``Dt`` in its native parent field."""

    level: int
    extension: MonomialExtension
    derivative: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.PRIMITIVE:
            raise ValueError("a primitive direction needs a primitive extension")
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(0),
        )
        if self.derivative != expected:
            raise ValueError("a primitive direction must equal Dt")

    @property
    def generator(self) -> Symbol:
        """Return the primitive generator."""
        return self.extension.generator


@dataclass(frozen=True, slots=True)
class HyperexponentialFundamentalDirection:
    r"""One exponential direction ``Dt/t`` in its native parent field."""

    level: int
    extension: MonomialExtension
    logarithmic_derivative: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.HYPEREXPONENTIAL:
            raise ValueError(
                "a hyperexponential direction needs a hyperexponential extension"
            )
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(1),
        )
        if self.logarithmic_derivative != expected:
            raise ValueError("a hyperexponential direction must equal Dt/t")

    @property
    def generator(self) -> Symbol:
        """Return the hyperexponential generator."""
        return self.extension.generator


@dataclass(frozen=True, slots=True)
class HypertangentFundamentalDirection:
    r"""One real tangent direction ``eta`` with ``Dt=eta*(1+t**2)``.

    ``cayley_coefficient`` remains in the real source parent field.  Its
    independently verified Gaussian image is the logarithmic derivative
    ``2*i*eta`` of the Cayley coordinate.
    """

    level: int
    extension: MonomialExtension
    cayley_coefficient: FieldElement

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("a fundamental direction needs a nonnegative level")
        if self.extension.family is not MonomialFamily.HYPERTANGENT:
            raise ValueError("a hypertangent direction needs a hypertangent extension")
        expected = self.extension.parent_field.convert(
            self.extension.generator_derivative.nth(2),
        )
        if self.cayley_coefficient != expected:
            raise ValueError("a hypertangent direction must equal eta")

    @property
    def generator(self) -> Symbol:
        """Return the hypertangent generator."""
        return self.extension.generator


FundamentalDirection = (
    PrimitiveFundamentalDirection
    | HyperexponentialFundamentalDirection
    | HypertangentFundamentalDirection
)


@dataclass(frozen=True, slots=True)
class FundamentalSignature:
    r"""Certified structural directions for one immutable P/E/T tower prefix.

    The signature records the standard rational base as its level-zero
    primitive direction, then each active primitive, hyperexponential, or
    hypertangent extension in order.  It is a structure-theorem input, not a
    search over all Darboux or special polynomials.  In particular, a
    hypertangent source direction is real data; its Cayley image is checked
    separately by :mod:`sympy_risch.gaussian`.
    """

    derivation: Derivation
    outcome: FundamentalSignatureOutcome
    reason: FundamentalSignatureReason | None
    detail: str
    certified_constant_levels: tuple[int, ...]
    primitive_directions: tuple[PrimitiveFundamentalDirection, ...]
    hyperexponential_directions: tuple[HyperexponentialFundamentalDirection, ...]
    hypertangent_directions: tuple[HypertangentFundamentalDirection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.derivation, Derivation):
            raise TypeError("a fundamental signature requires a Derivation")
        if not isinstance(self.outcome, FundamentalSignatureOutcome):
            raise TypeError(
                "a fundamental signature outcome must be a FundamentalSignatureOutcome"
            )
        if self.reason is not None and not isinstance(
            self.reason,
            FundamentalSignatureReason,
        ):
            raise TypeError(
                "a fundamental signature reason must be a FundamentalSignatureReason"
            )
        if not isinstance(self.detail, str):
            raise TypeError("a fundamental signature detail must be a string")
        direction_groups = (
            (self.primitive_directions, PrimitiveFundamentalDirection),
            (
                self.hyperexponential_directions,
                HyperexponentialFundamentalDirection,
            ),
            (self.hypertangent_directions, HypertangentFundamentalDirection),
        )
        for directions, direction_type in direction_groups:
            if not isinstance(directions, tuple) or not all(
                isinstance(direction, direction_type) for direction in directions
            ):
                raise TypeError(
                    "fundamental signature directions must be immutable "
                    "direction tuples"
                )
        if not isinstance(self.certified_constant_levels, tuple) or not all(
            isinstance(level, int) for level in self.certified_constant_levels
        ):
            raise TypeError(
                "fundamental signature certified levels must be an integer tuple"
            )
        if self.outcome is FundamentalSignatureOutcome.SUPPORTED:
            if self.reason is not None:
                raise ValueError("a supported signature has no rejection reason")
            if self.derivation.level < 0:
                raise ValueError("a supported signature requires a rational base level")
            expected_levels = tuple(range(self.derivation.level + 1))
            if self.certified_constant_levels != expected_levels:
                raise ValueError(
                    "a supported signature must certify every active tower level"
                )
            if (
                tuple(direction.level for direction in self.directions)
                != expected_levels
            ):
                raise ValueError(
                    "a supported signature needs one ordered direction per level"
                )
            base = self.derivation.tower.levels[0]
            if not isinstance(
                base, MonomialExtension
            ) or not _is_standard_relation_base(base):
                raise ValueError(
                    "a supported signature requires C(x) with Dx = 1 over an "
                    "effective finite constant field C"
                )
            if base.parent_field not in {QQ, QQ_I} and self.hypertangent_directions:
                raise ValueError(
                    "hypertangent directions currently require QQ or QQ_I constants"
                )
            if base.parent_field not in {QQ, QQ_I} and self.derivation.level > 0:
                raise ValueError(
                    "finite algebraic constant signatures currently certify only "
                    "the standard rational base"
                )
            for direction in self.directions:
                extension = self.derivation.tower.levels[direction.level]
                if direction.extension is not extension:
                    raise ValueError(
                        "a fundamental direction must retain its exact tower extension"
                    )
            if self.derivation.level > 0:
                # A supported signature is itself a structural theorem
                # certificate, so it cannot be authorized by advisory
                # extension metadata. Reuse the prefix replay rather than
                # independently scanning every direction.
                from .constant_proofs import first_nonpreserved_constant_level

                evidence = first_nonpreserved_constant_level(self.derivation)
                if evidence is not None:
                    raise ValueError(
                        "a non-base fundamental direction requires replayed "
                        "constant preservation"
                    )
        else:
            if self.reason is None:
                raise ValueError("an unsupported signature needs a reason")
            if (
                self.certified_constant_levels
                or self.primitive_directions
                or self.hyperexponential_directions
                or self.hypertangent_directions
            ):
                raise ValueError(
                    "an unsupported signature must not expose partial directions"
                )

    @property
    def supported(self) -> bool:
        """Whether this is a certified P/E/T structural signature."""
        return self.outcome is FundamentalSignatureOutcome.SUPPORTED

    @property
    def unsupported(self) -> bool:
        """Whether the tower is outside this signature's theorem scope."""
        return self.outcome is FundamentalSignatureOutcome.UNSUPPORTED

    @property
    def directions(self) -> tuple[FundamentalDirection, ...]:
        """Return every source direction in ascending active-tower order."""
        return tuple(
            sorted(
                (
                    *self.primitive_directions,
                    *self.hyperexponential_directions,
                    *self.hypertangent_directions,
                ),
                key=lambda direction: direction.level,
            )
        )

    def direction_at(self, level: int) -> FundamentalDirection | None:
        """Return the exact source direction at one active tower level."""
        for direction in self.directions:
            if direction.level == level:
                return direction
        return None


def _unsupported_fundamental_signature(
    derivation: Derivation,
    reason: FundamentalSignatureReason,
    detail: str,
) -> FundamentalSignature:
    return FundamentalSignature(
        derivation,
        FundamentalSignatureOutcome.UNSUPPORTED,
        reason,
        detail,
        (),
        (),
        (),
        (),
    )


def _is_standard_relation_base(extension: MonomialExtension) -> bool:
    """Whether ``extension`` is a supported standard ``C(x)`` base."""
    return is_supported_finite_constant_field(
        extension.parent_field,
    ) and extension.generator_derivative == Poly(
        1, extension.generator, domain=extension.parent_field
    )


def build_fundamental_signature(derivation: Derivation) -> FundamentalSignature:
    r"""Describe the certified P/E/T directions of an active tower prefix.

    The construction replays the implemented constant-field criteria but never
    solves an input relation.  It reports ``UNSUPPORTED`` rather than a
    mathematical negative conclusion when a required structure-theorem
    hypothesis is absent.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("fundamental signatures require a Derivation")
    if derivation.level < 0:
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.NO_RATIONAL_BASE,
            "the constant field alone has no standard rational base",
        )

    active_levels = derivation.tower.levels[: derivation.level + 1]
    base = active_levels[0]
    if not isinstance(base, MonomialExtension):
        return _unsupported_fundamental_signature(
            derivation,
            (
                FundamentalSignatureReason.ALGEBRAIC_EXTENSION
                if isinstance(base, AlgebraicExtension)
                else FundamentalSignatureReason.UNSUPPORTED_EXTENSION
            ),
            "level 0 is not a monomial standard rational base",
        )
    if not is_supported_finite_constant_field(base.parent_field):
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.BASE_FIELD,
            "the supported relation base must be over an effective finite "
            "constant field over QQ",
        )
    if not _is_standard_relation_base(base):
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.BASE_DERIVATIVE,
            "the supported relation base must have Dx = 1",
        )

    primitive_directions: list[PrimitiveFundamentalDirection] = [
        PrimitiveFundamentalDirection(
            0,
            base,
            base.parent_field.convert(base.generator_derivative.nth(0)),
        ),
    ]
    hyperexponential_directions: list[HyperexponentialFundamentalDirection] = []
    hypertangent_directions: list[HypertangentFundamentalDirection] = []
    constant_failure = None
    if base.parent_field in {QQ, QQ_I}:
        # Replay the full prefix once. The result cache is bottom-up and
        # identity-safe, so later factories and signatures reuse this check
        # instead of each rescanning every lower level.
        from .constant_proofs import first_nonpreserved_constant_level

        constant_failure = first_nonpreserved_constant_level(derivation)

    for level, extension in enumerate(active_levels[1:], start=1):
        if not isinstance(extension, MonomialExtension):
            return _unsupported_fundamental_signature(
                derivation,
                (
                    FundamentalSignatureReason.ALGEBRAIC_EXTENSION
                    if isinstance(extension, AlgebraicExtension)
                    else FundamentalSignatureReason.UNSUPPORTED_EXTENSION
                ),
                f"level {level} is not a monomial extension",
            )
        if extension.family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
            MonomialFamily.HYPERTANGENT,
        }:
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
                f"level {level} has unsupported monomial family {extension.family.value}",
            )
        if base.parent_field not in {QQ, QQ_I}:
            if extension.family is MonomialFamily.HYPERTANGENT:
                return _unsupported_fundamental_signature(
                    derivation,
                    FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
                    "hypertangent fundamental directions over finite algebraic "
                    "constant fields are not yet supported",
                )
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.CONSTANT_FIELD_PROVENANCE,
                "finite algebraic constant towers need proof-carrying "
                "no-new-constants evidence before recursive P/E signatures "
                "can be certified",
            )
        if (
            constant_failure is not None
            and constant_failure.derivation.level == level
            and constant_failure.status is ConstantFieldStatus.UNKNOWN
        ):
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.CONSTANT_FIELD_UNKNOWN,
                f"level {level} has no replayed preserved-constants proof",
            )
        if (
            constant_failure is not None
            and constant_failure.derivation.level == level
            and constant_failure.status is ConstantFieldStatus.ENLARGED
        ):
            return _unsupported_fundamental_signature(
                derivation,
                FundamentalSignatureReason.CONSTANT_FIELD_ENLARGED,
                f"level {level} enlarges the constant field under replay",
            )
        if extension.family is MonomialFamily.PRIMITIVE:
            primitive_directions.append(
                PrimitiveFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(0),
                    ),
                ),
            )
            continue
        if extension.family is MonomialFamily.HYPEREXPONENTIAL:
            hyperexponential_directions.append(
                HyperexponentialFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(1),
                    ),
                ),
            )
            continue
        if extension.family is MonomialFamily.HYPERTANGENT:
            hypertangent_directions.append(
                HypertangentFundamentalDirection(
                    level,
                    extension,
                    extension.parent_field.convert(
                        extension.generator_derivative.nth(2),
                    ),
                ),
            )
            continue
        return _unsupported_fundamental_signature(
            derivation,
            FundamentalSignatureReason.UNSUPPORTED_MONOMIAL_FAMILY,
            f"level {level} has unsupported monomial family {extension.family.value}",
        )

    return FundamentalSignature(
        derivation,
        FundamentalSignatureOutcome.SUPPORTED,
        None,
        "the active tower has certified primitive, hyperexponential, and "
        "hypertangent fundamental directions",
        tuple(range(derivation.level + 1)),
        tuple(primitive_directions),
        tuple(hyperexponential_directions),
        tuple(hypertangent_directions),
    )


@dataclass(frozen=True, slots=True)
class LogarithmicRelationWitness:
    r"""One integral relation ``sum(c_i*a_i)=D(v)/v``.

    ``source_coefficients`` contains the integral multiple of the rational
    relation vector.  ``variable_values`` additionally exposes the recorded
    normal and special valuations used to construct ``argument``.
    """

    source_coefficients: tuple[int, ...]
    variable_values: tuple[int, ...]
    argument: FieldElement


@dataclass(frozen=True, slots=True)
class LogarithmicRelationSystemResult:
    r"""Complete rational kernel for finitely many logarithmic directions."""

    derivation: Derivation
    source_terms: tuple[FieldElement, ...]
    equations: tuple[tuple[FieldElement, ...], ...]
    witness_factors: tuple[tuple[int, Expr], ...]
    relations: tuple[LogarithmicRelationWitness, ...]
    fundamental_signature: FundamentalSignature

    @property
    def source_dimension(self) -> int:
        """Return the number of input directions."""
        return len(self.source_terms)

    @property
    def variable_dimension(self) -> int:
        """Return the total number of rational system variables."""
        if self.equations:
            return len(self.equations[0])
        if self.relations:
            return len(self.relations[0].variable_values)
        return self.source_dimension

    @property
    def reconstructs(self) -> bool:
        """Whether every stored kernel vector reconstructs exactly."""
        field = self.derivation.field
        for relation in self.relations:
            left = field.zero
            for coefficient, term in zip(
                relation.source_coefficients,
                self.source_terms,
                strict=True,
            ):
                left += coefficient * term
            argument = _coerce(field, relation.argument)
            if argument == field.zero:
                return False
            difference = left - self.derivation(argument) / argument
            if difference != field.zero and cancel(field.to_sympy(difference)) != 0:
                return False
        return True


@dataclass(frozen=True, slots=True)
class _LinearFieldTerm:
    """A field element multiplied by a sparse rational linear form."""

    element: FieldElement
    coefficients: tuple[tuple[int, FieldElement], ...]


def _is_zero(value: FieldElement, field: Domain) -> bool:
    """Test zero while tolerating independently normalized fractions."""
    return value == field.zero or cancel(field.to_sympy(value)) == 0


def _fraction_polynomials(
    element: FieldElement,
    extension: MonomialExtension,
) -> tuple[Poly, Poly]:
    """Return coprime numerator and denominator polynomials in the top symbol."""
    numerator_expression, denominator_expression = cancel(
        extension.field.to_sympy(element),
    ).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=extension.parent_field,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    return numerator.mul_ground(coefficient), denominator


def _top_coordinates(
    elements: tuple[FieldElement, ...],
    extension: MonomialExtension,
) -> tuple[tuple[tuple[FieldElement, ...], ...], tuple[FieldElement, ...]]:
    r"""Return nonconstant and constant coordinates over ``extension.parent``.

    For a linear combination of ``elements``, all returned nonconstant rows
    must vanish exactly when that combination belongs to the parent field.
    The final tuple gives the resulting parent-field constant coordinate.
    """
    fractions = tuple(_fraction_polynomials(element, extension) for element in elements)
    polynomial_parts: list[Poly] = []
    proper_numerators: list[Poly] = []
    denominators: list[Poly] = []
    common_denominator = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    maximum_degree = 0
    for numerator, denominator in fractions:
        polynomial_part, proper_numerator = numerator.div(denominator)
        polynomial_parts.append(polynomial_part)
        proper_numerators.append(proper_numerator)
        denominators.append(denominator)
        common_denominator = common_denominator.lcm(denominator)
        if not polynomial_part.is_zero:
            maximum_degree = max(maximum_degree, polynomial_part.degree())

    parent = extension.parent_field
    rows: list[tuple[FieldElement, ...]] = []
    for degree in range(1, maximum_degree + 1):
        rows.append(
            tuple(
                parent.convert(polynomial.nth(degree))
                for polynomial in polynomial_parts
            )
        )

    if common_denominator.degree() > 0:
        aligned_numerators = tuple(
            proper * common_denominator.exquo(denominator)
            for proper, denominator in zip(
                proper_numerators,
                denominators,
                strict=True,
            )
        )
        for degree in range(common_denominator.degree()):
            rows.append(
                tuple(
                    parent.convert(numerator.nth(degree))
                    for numerator in aligned_numerators
                )
            )

    constants = tuple(
        parent.convert(polynomial.nth(0)) for polynomial in polynomial_parts
    )
    return tuple(rows), constants


def _field_zero_constraints(
    columns: tuple[FieldElement, ...],
    derivation: Derivation,
) -> tuple[tuple[FieldElement, ...], ...]:
    r"""Compile ``sum(q_i*columns[i])=0`` into rational-coordinate rows."""
    field = derivation.field
    if derivation.level == -1:
        coordinate_columns = tuple(
            rational_constant_coordinates(field, column) for column in columns
        )
        if not coordinate_columns:
            return ()
        coordinate_count = len(coordinate_columns[0])
        if any(
            len(coordinates) != coordinate_count for coordinates in coordinate_columns
        ):
            raise ValueError("constant coordinates have inconsistent dimensions")
        rows: list[tuple[FieldElement, ...]] = []
        for coordinate_index in range(coordinate_count):
            row = tuple(
                coordinates[coordinate_index] for coordinates in coordinate_columns
            )
            if any(value != QQ.zero for value in row):
                rows.append(row)
        return tuple(rows)

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "logarithmic relation systems do not support algebraic levels"
        )
    fractions = tuple(
        _fraction_polynomials(_coerce(field, column), extension) for column in columns
    )
    common_denominator = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    for _, denominator in fractions:
        common_denominator = common_denominator.lcm(denominator)
    cleared_numerators = tuple(
        numerator * common_denominator.exquo(denominator)
        for numerator, denominator in fractions
    )
    maximum_degree = max(
        (
            numerator.degree()
            for numerator in cleared_numerators
            if not numerator.is_zero
        ),
        default=0,
    )
    rows: list[tuple[FieldElement, ...]] = []
    for degree in range(maximum_degree + 1):
        coefficient_columns = tuple(
            extension.parent_field.convert(numerator.nth(degree))
            for numerator in cleared_numerators
        )
        rows.extend(_field_zero_constraints(coefficient_columns, derivation.lower()))
    return tuple(rows)


def _rational_nullspace(
    rows: tuple[tuple[FieldElement, ...], ...],
    column_count: int,
) -> tuple[tuple[FieldElement, ...], ...]:
    """Return a row basis for a homogeneous kernel over ``QQ``."""
    if column_count == 0:
        return ()
    matrix = DomainMatrix(
        [[QQ.convert(value) for value in row] for row in rows],
        (len(rows), column_count),
        QQ,
    )
    return tuple(
        tuple(QQ.convert(value) for value in row)
        for row in matrix.nullspace().to_list()
    )


def _integer_vector(
    vector: tuple[FieldElement, ...],
) -> tuple[int, ...]:
    """Clear all rational denominators in one kernel vector."""
    scale = 1
    rational_vector = tuple(QQ.convert(value) for value in vector)
    for value in rational_vector:
        scale = lcm(scale, int(QQ.denom(value)))
    return tuple(int(QQ.numer(QQ.convert(scale) * value)) for value in rational_vector)


class _RelationSystemCompiler:
    """Compile a logarithmic relation system from the top of a tower."""

    def __init__(
        self,
        source_terms: tuple[FieldElement, ...],
        derivation: Derivation,
        fundamental_signature: FundamentalSignature,
    ) -> None:
        self.source_terms = source_terms
        self.derivation = derivation
        self.fundamental_signature = fundamental_signature
        self.variable_count = len(source_terms)
        self.rows: list[list[FieldElement]] = []
        self.witness_factors: list[tuple[int, Expr]] = []

    def new_variable(self, factor: Expr) -> int:
        """Allocate a rational valuation variable with its witness factor."""
        index = self.variable_count
        self.variable_count += 1
        for row in self.rows:
            row.append(QQ.zero)
        self.witness_factors.append((index, factor))
        return index

    def add_row(self, row: tuple[FieldElement, ...]) -> None:
        """Append one rational homogeneous equation."""
        if len(row) != self.variable_count:
            raise ValueError("relation-system row has the wrong width")
        converted = [QQ.convert(value) for value in row]
        if any(value != QQ.zero for value in converted):
            self.rows.append(converted)

    def aggregate(
        self,
        terms: tuple[_LinearFieldTerm, ...],
        coordinates: tuple[FieldElement, ...],
        field: Domain,
    ) -> tuple[FieldElement, ...]:
        """Collect a field-valued coordinate for every system variable."""
        if len(terms) != len(coordinates):
            raise ValueError("relation-system coordinates are misaligned")
        columns = [field.zero for _ in range(self.variable_count)]
        for term, coordinate in zip(terms, coordinates, strict=True):
            coordinate = field.convert(coordinate)
            if _is_zero(coordinate, field):
                continue
            for index, coefficient in term.coefficients:
                columns[index] += QQ.convert(coefficient) * coordinate
        return tuple(columns)

    def add_field_zero_constraints(
        self,
        terms: tuple[_LinearFieldTerm, ...],
        coordinates: tuple[FieldElement, ...],
        derivation: Derivation,
    ) -> None:
        """Enforce that the indicated field-valued linear coordinate vanishes."""
        columns = self.aggregate(terms, coordinates, derivation.field)
        for row in _field_zero_constraints(columns, derivation):
            self.add_row(row)

    def compile(
        self, terms: tuple[_LinearFieldTerm, ...], derivation: Derivation
    ) -> None:
        """Recursively compile logarithmic-derivative structure constraints."""
        if derivation.level == -1:
            columns = self.aggregate(
                terms,
                tuple(term.element for term in terms),
                derivation.field,
            )
            for row in _field_zero_constraints(columns, derivation):
                self.add_row(row)
            return

        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "logarithmic relation systems do not support algebraic levels"
            )
        if derivation.level == 0:
            if (
                extension.family is not MonomialFamily.PRIMITIVE
                or not _is_standard_relation_base(extension)
            ):
                raise NotImplementedError(
                    "logarithmic relation systems require C(x) with Dx = 1 "
                    "over an effective finite constant field C"
                )
        else:
            from .constant_proofs import require_preserved_constant_field

            require_preserved_constant_field(
                derivation,
                "logarithmic relation systems",
            )
        direction = (
            self.fundamental_signature.direction_at(derivation.level)
            if self.fundamental_signature.supported
            else None
        )
        family = (
            direction.extension.family if direction is not None else extension.family
        )
        if family not in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
        }:
            raise NotImplementedError(
                "logarithmic relation systems support only primitive and "
                "hyperexponential monomial levels"
            )

        top_terms = list(terms)
        seen_factors: set[Expr] = set()
        for term in terms:
            _, denominator = _fraction_polynomials(term.element, extension)
            for raw_factor, _ in denominator.factor_list()[1]:
                factor = raw_factor.monic()
                factor_expression = factor.as_expr()
                if factor_expression in seen_factors:
                    continue
                seen_factors.add(factor_expression)
                derivative = derivation.polynomial(factor)
                is_special = factor.gcd(derivative).degree() > 0
                generator_factor = Poly(
                    extension.generator,
                    extension.generator,
                    domain=extension.parent_field,
                )
                if is_special:
                    if (
                        family is MonomialFamily.HYPEREXPONENTIAL
                        and factor == generator_factor
                    ):
                        continue
                    raise NotImplementedError(
                        "logarithmic relation systems encountered a "
                        "non-generator special polynomial"
                    )
                valuation = self.new_variable(factor_expression)
                logarithmic_derivative = _coerce(
                    extension.field,
                    derivative.as_expr() / factor_expression,
                )
                top_terms.append(
                    _LinearFieldTerm(
                        -logarithmic_derivative,
                        ((valuation, QQ.one),),
                    )
                )

        if family is MonomialFamily.HYPEREXPONENTIAL:
            if not isinstance(direction, HyperexponentialFundamentalDirection):
                raise ValueError(
                    "a supported relation signature lacks the active "
                    "hyperexponential direction"
                )
            special_valuation = self.new_variable(extension.generator)
            top_terms.append(
                _LinearFieldTerm(
                    _coerce(
                        extension.field,
                        -direction.logarithmic_derivative,
                    ),
                    ((special_valuation, QQ.one),),
                )
            )

        elements = tuple(term.element for term in top_terms)
        nonconstant_rows, constants = _top_coordinates(elements, extension)
        for coordinates in nonconstant_rows:
            self.add_field_zero_constraints(
                tuple(top_terms),
                coordinates,
                derivation.lower(),
            )

        lower_terms = tuple(
            _LinearFieldTerm(
                extension.parent_field.convert(constant),
                term.coefficients,
            )
            for term, constant in zip(top_terms, constants, strict=True)
            if not _is_zero(
                extension.parent_field.convert(constant),
                extension.parent_field,
            )
        )
        self.compile(lower_terms, derivation.lower())

    def finish(self) -> LogarithmicRelationSystemResult:
        """Solve the accumulated QQ system and construct every basis witness."""
        kernel = _rational_nullspace(
            tuple(tuple(row) for row in self.rows),
            self.variable_count,
        )
        field = self.derivation.field
        relations = []
        for vector in kernel:
            values = _integer_vector(vector)
            argument = field.one
            for index, factor in self.witness_factors:
                exponent = values[index]
                if exponent:
                    argument *= field.from_sympy(factor) ** exponent
            relations.append(
                LogarithmicRelationWitness(
                    values[: len(self.source_terms)],
                    values,
                    argument,
                )
            )
        result = LogarithmicRelationSystemResult(
            self.derivation,
            self.source_terms,
            tuple(tuple(row) for row in self.rows),
            tuple(self.witness_factors),
            tuple(relations),
            self.fundamental_signature,
        )
        if not result.reconstructs:
            raise ValueError("logarithmic relation-system witness did not reconstruct")
        return result


def solve_logarithmic_relation_system(
    terms: tuple[object, ...],
    derivation: Derivation,
) -> LogarithmicRelationSystemResult:
    r"""Compute the complete rational logarithmic-relation kernel.

    Each returned witness represents an integral kernel row.  Their rational
    span contains exactly the input combinations that become logarithmic
    derivatives after a positive integer scale in the supported tower.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("logarithmic relation systems require a Derivation")
    if not terms:
        raise ValueError("logarithmic relation systems require at least one term")
    field = derivation.field
    source_terms = tuple(_coerce(field, term) for term in terms)
    fundamental_signature = build_fundamental_signature(derivation)
    if fundamental_signature.unsupported and derivation.tower.constants not in {
        QQ,
        QQ_I,
    }:
        raise NotImplementedError(fundamental_signature.detail)
    compiler = _RelationSystemCompiler(
        source_terms,
        derivation,
        fundamental_signature,
    )
    initial_terms = tuple(
        _LinearFieldTerm(
            term,
            ((index, QQ.one),),
        )
        for index, term in enumerate(source_terms)
    )
    compiler.compile(initial_terms, derivation)
    return compiler.finish()
