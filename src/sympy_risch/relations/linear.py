r"""Finite logarithmic-relation systems for supported monomial towers.

The central object here is the rational relation space

.. math::

   \{q\in\mathbb{Q}^r : \sum_i q_i a_i=Dv/v\text{ after an integer scale}\}.

It is the Chapter-9 structure-theorem fallback for mixed P/E/T towers. The
compiler works downward through the tower. At each primitive level it records
normal-factor valuations; at each hyperexponential level it additionally
records the special generator valuation; and at each hypertangent level it
records the real special divisor ``1 + t**2``. Every equality in an
intermediate field is recursively expanded to rational coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm

from sympy.core.expr import Expr
from sympy.polys.domains import QQ, QQ_I, ZZ
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..constant_fields import rational_constant_coordinates
from ..tower import Derivation, MonomialExtension, MonomialFamily, _coerce
from .signature import (
    FundamentalSignature,
    HyperexponentialFundamentalDirection,
    _is_standard_relation_base,
    build_fundamental_signature,
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


def _saturated_integer_kernel(
    rows: tuple[tuple[FieldElement, ...], ...],
    column_count: int,
) -> tuple[tuple[int, ...], ...]:
    r"""Return a lattice basis for the integral kernel of ``rows``.

    Clearing each row's denominators preserves its kernel. Smith
    decomposition then supplies the unimodular change of variables whose
    trailing columns form the saturated integer kernel. Unlike independently
    clearing denominators in a rational nullspace basis, this does not miss a
    smaller integral source coefficient formed by combining basis vectors.
    """
    integer_rows: list[list[object]] = []
    for row in rows:
        denominators = [int(QQ.denom(QQ.convert(value))) for value in row]
        scale = 1
        for denominator in denominators:
            scale = lcm(scale, denominator)
        integer_rows.append(
            [ZZ.convert(QQ.convert(scale) * QQ.convert(value)) for value in row]
        )

    matrix = DomainMatrix(
        integer_rows,
        (len(integer_rows), column_count),
        ZZ,
    )
    diagonal, _, right_transform = smith_normal_decomp(matrix)
    rank = sum(
        diagonal[index, index].element != ZZ.zero
        for index in range(min(diagonal.shape))
    )
    transform = right_transform.to_list()
    return tuple(
        tuple(int(transform[row][column]) for row in range(column_count))
        for column in range(rank, column_count)
    )


def minimal_source_logarithmic_relation(
    system: LogarithmicRelationSystemResult,
    source_index: int = 0,
) -> LogarithmicRelationWitness | None:
    r"""Return the relation with minimal positive coefficient at one source.

    The projection of the integral kernel onto one source coordinate is an
    ideal ``n*ZZ``. A saturated Smith kernel basis generates that ideal, and
    Bézout combination of its projected coordinates constructs the witness
    with its positive generator ``n``.
    """
    if not 0 <= source_index < system.source_dimension:
        raise IndexError("source index is outside the relation system")
    basis = system.relations

    gcd_value = ZZ.zero
    combination = [ZZ.zero for _ in basis]
    for index, relation in enumerate(basis):
        coefficient = ZZ.convert(relation.source_coefficients[source_index])
        left, right, new_gcd = ZZ.gcdex(gcd_value, coefficient)
        combination = [left * value for value in combination]
        combination[index] += right
        gcd_value = new_gcd

    if gcd_value == ZZ.zero:
        return None
    values = tuple(
        int(
            sum(
                multiplier * ZZ.convert(relation.variable_values[column])
                for multiplier, relation in zip(combination, basis, strict=True)
            )
        )
        for column in range(system.variable_dimension)
    )
    if values[source_index] <= 0:
        raise ValueError("Smith kernel projection did not produce a positive generator")

    field = system.derivation.field
    argument = field.one
    for multiplier, basis_relation in zip(combination, basis, strict=True):
        if multiplier:
            argument *= basis_relation.argument ** int(multiplier)
    relation = LogarithmicRelationWitness(
        values[: system.source_dimension],
        values,
        argument,
    )
    left = field.zero
    for coefficient, term in zip(
        relation.source_coefficients,
        system.source_terms,
        strict=True,
    ):
        left += coefficient * term
    if left != system.derivation(argument) / argument:
        raise ValueError("minimal logarithmic relation failed to reconstruct")
    return relation


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
            from ..constant_proofs import require_preserved_constant_field

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
            MonomialFamily.HYPERTANGENT,
        }:
            raise NotImplementedError(
                "logarithmic relation systems support only primitive, "
                "hyperexponential, and hypertangent monomial levels"
            )

        top_terms = list(terms)
        seen_factors: set[Expr] = set()
        generator_factor = Poly(
            extension.generator,
            extension.generator,
            domain=extension.parent_field,
        )
        tangent_special = Poly(
            1 + extension.generator**2,
            extension.generator,
            domain=extension.parent_field,
        )
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
                if is_special:
                    if (
                        family is MonomialFamily.HYPEREXPONENTIAL
                        and factor == generator_factor
                    ):
                        continue
                    if not (
                        family is MonomialFamily.HYPERTANGENT
                        and tangent_special.rem(factor).is_zero
                    ):
                        raise NotImplementedError(
                            "logarithmic relation systems encountered an "
                            "unsupported special polynomial"
                        )
                    continue
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
        elif family is MonomialFamily.HYPERTANGENT:
            for raw_factor, _ in tangent_special.factor_list()[1]:
                factor = raw_factor.monic()
                factor_expression = factor.as_expr()
                special_valuation = self.new_variable(factor_expression)
                logarithmic_derivative = _coerce(
                    extension.field,
                    derivation.polynomial(factor).as_expr() / factor_expression,
                )
                top_terms.append(
                    _LinearFieldTerm(
                        -logarithmic_derivative,
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
        kernel = _saturated_integer_kernel(
            tuple(tuple(row) for row in self.rows),
            self.variable_count,
        )
        field = self.derivation.field
        relations = []
        for values in kernel:
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
