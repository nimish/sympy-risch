"""Independent matrix oracle over one hyperexponential extension."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly

from sympy_risch.rde._tower_polynomials import _is_standard_rational_derivation
from sympy_risch.rde.hyperexponential import _hyperexponential_laurent_rde_extension
from sympy_risch.tower import Derivation, MonomialExtension, _coerce

from ._common import as_qq_bivariate_fraction
from .models import BoundedHyperexponentialLaurentRDEMatrixResult


def solve_bounded_hyperexponential_laurent_rde_matrix(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    minimum_power: int,
    maximum_power: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedHyperexponentialLaurentRDEMatrixResult:
    r"""Solve a bounded rational RDE ansatz by exact global linear algebra.

    This independent oracle is limited to one hyperexponential level over
    ``QQ(x)``.  It searches

    ``sum(c_ij*x**i*t**j/(lower_denominator*top_denominator))``

    for ``minimum_power <= j <= maximum_power``.  It deliberately does not
    perform weak normalization, denominator reduction, special-denominator
    reduction, SPDE, or lower-field RDE solving.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded hyperexponential Laurent matrix oracle requires "
            "QQ(x) below the hyperexponential level"
        )
    if (
        not isinstance(lower_numerator_degree_bound, int)
        or isinstance(lower_numerator_degree_bound, bool)
        or lower_numerator_degree_bound < 0
    ):
        raise ValueError("the lower numerator degree bound must be a nonnegative int")
    for name, power in (
        ("minimum", minimum_power),
        ("maximum", maximum_power),
    ):
        if not isinstance(power, int) or isinstance(power, bool):
            raise ValueError(f"the {name} power must be an int")
    if minimum_power > maximum_power:
        raise ValueError("the minimum power must not exceed the maximum power")

    lower_extension = derivation.tower.levels[lower_derivation.level]
    if not isinstance(lower_extension, MonomialExtension):
        raise TypeError("the rational lower level is not monomial")
    x = lower_extension.generator
    t = extension.generator
    denominator = Poly(lower_denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the lower ansatz denominator must be nonzero")
    denominator = denominator.monic()
    top_denominator_polynomial = Poly(
        top_denominator,
        t,
        domain=extension.parent_field,
    )
    if top_denominator_polynomial.is_zero:
        raise ValueError("the top ansatz denominator must be nonzero")
    top_denominator_polynomial = top_denominator_polynomial.monic()

    field = extension.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    basis = [
        _coerce(
            field,
            x**lower_degree
            * t**top_power
            / (denominator.as_expr() * top_denominator_polynomial.as_expr()),
        )
        for top_power in range(minimum_power, maximum_power + 1)
        for lower_degree in range(lower_numerator_degree_bound + 1)
    ]
    columns = [
        derivation(candidate) + coefficient_element * candidate for candidate in basis
    ]
    fractions = [
        as_qq_bivariate_fraction(field.to_sympy(element), x, t)
        for element in [*columns, rhs_element]
    ]
    common_denominator = Poly(1, x, t, domain=QQ)
    for _, element_denominator in fractions:
        common_denominator = common_denominator.lcm(element_denominator)
    cleared = [
        numerator * common_denominator.exquo(element_denominator)
        for numerator, element_denominator in fractions
    ]
    column_polynomials = cleared[:-1]
    rhs_polynomial = cleared[-1]
    monomials = sorted(
        {monomial for polynomial in cleared for monomial, _ in polynomial.terms()},
        reverse=True,
    )
    if not monomials:
        monomials = [(0, 0)]

    unknown_count = len(basis)
    coefficient_rows = [
        [QQ.convert(polynomial.nth(*monomial)) for polynomial in column_polynomials]
        for monomial in monomials
    ]
    rhs_rows = [[QQ.convert(rhs_polynomial.nth(*monomial))] for monomial in monomials]
    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        rhs_rows,
        (len(rhs_rows), 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedHyperexponentialLaurentRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            minimum_power,
            maximum_power,
            None,
            coefficient_rank,
            augmented_rank,
            top_denominator_polynomial,
        )

    reduced_matrix, pivots = augmented.rref()
    rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = rows[row][unknown_count]
    solution = field.zero
    for value, candidate in zip(coefficients, basis, strict=True):
        solution += field.convert(value) * candidate
    if derivation(solution) + coefficient_element * solution != rhs_element:
        raise ValueError(
            "bounded hyperexponential Laurent matrix RDE reconstruction failed"
        )
    return BoundedHyperexponentialLaurentRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        minimum_power,
        maximum_power,
        solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )
