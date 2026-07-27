"""Independent matrix oracles over one primitive extension."""

from __future__ import annotations

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polytools import Poly

from sympy_risch.rde._tower_polynomials import _is_standard_rational_derivation
from sympy_risch.rde.primitive import _primitive_rde_extension
from sympy_risch.tower import Derivation, MonomialExtension, _coerce

from ._common import as_qq_bivariate_fraction
from .models import (
    BoundedPrimitiveCoupledRDEMatrixResult,
    BoundedPrimitiveRDEMatrixResult,
)


def solve_bounded_primitive_rde_matrix(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    top_degree_bound: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedPrimitiveRDEMatrixResult:
    r"""Solve a bounded primitive-field RDE by global exact linear algebra.

    This independent oracle is limited to one primitive level over ``QQ(x)``.
    It searches

    ``sum(c_ij*x**i*t**j/(lower_denominator*top_denominator))``

    over the caller-supplied rectangular degree bounds. It shares no
    denominator theorem, SPDE, or leading recurrence with the production
    primitive solver.
    """
    extension = _primitive_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded primitive matrix oracle requires QQ(x) below the "
            "primitive level"
        )
    for name, bound in (
        ("lower numerator", lower_numerator_degree_bound),
        ("top", top_degree_bound),
    ):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
            raise ValueError(f"the {name} degree bound must be nonnegative")

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
            * t**top_degree
            / (denominator.as_expr() * top_denominator_polynomial.as_expr()),
        )
        for top_degree in range(top_degree_bound + 1)
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
        return BoundedPrimitiveRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            top_degree_bound,
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
        raise ValueError("bounded primitive matrix RDE reconstruction failed")
    return BoundedPrimitiveRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        top_degree_bound,
        solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )


def solve_bounded_primitive_coupled_rde_matrix(
    lambda_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    top_degree_bound: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedPrimitiveCoupledRDEMatrixResult:
    r"""Solve a bounded coupled ansatz by direct real block linear algebra.

    This independent oracle is limited to one primitive level over ``QQ(x)``.
    It constructs both real equations directly and never rebases the tower to
    ``QQ_I`` or invokes a scalar RDE solver.
    """
    extension = _primitive_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded primitive coupled oracle requires QQ(x) below "
            "the primitive level"
        )
    for name, bound in (
        ("lower numerator", lower_numerator_degree_bound),
        ("top", top_degree_bound),
    ):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
            raise ValueError(f"the {name} degree bound must be nonnegative")

    lower_extension = derivation.tower.levels[0]
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
    lambda_element = _coerce(field, lambda_coefficient)
    if lambda_element == field.zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs = _coerce(field, real_right_hand_side)
    imaginary_rhs = _coerce(field, imaginary_right_hand_side)
    basis = [
        _coerce(
            field,
            x**lower_degree
            * t**top_degree
            / (denominator.as_expr() * top_denominator_polynomial.as_expr()),
        )
        for top_degree in range(top_degree_bound + 1)
        for lower_degree in range(lower_numerator_degree_bound + 1)
    ]
    basis_derivatives = [derivation(candidate) for candidate in basis]
    real_columns = [
        *basis_derivatives,
        *(-lambda_element * candidate for candidate in basis),
    ]
    imaginary_columns = [
        *(lambda_element * candidate for candidate in basis),
        *basis_derivatives,
    ]
    all_elements = [
        *real_columns,
        real_rhs,
        *imaginary_columns,
        imaginary_rhs,
    ]
    fractions = [
        as_qq_bivariate_fraction(field.to_sympy(element), x, t)
        for element in all_elements
    ]
    common_denominator = Poly(1, x, t, domain=QQ)
    for _, element_denominator in fractions:
        common_denominator = common_denominator.lcm(element_denominator)
    cleared = [
        numerator * common_denominator.exquo(element_denominator)
        for numerator, element_denominator in fractions
    ]
    unknown_count = 2 * len(basis)
    real_columns_polynomial = cleared[:unknown_count]
    real_rhs_polynomial = cleared[unknown_count]
    imaginary_start = unknown_count + 1
    imaginary_columns_polynomial = cleared[
        imaginary_start : imaginary_start + unknown_count
    ]
    imaginary_rhs_polynomial = cleared[-1]
    all_polynomials = [
        *real_columns_polynomial,
        real_rhs_polynomial,
        *imaginary_columns_polynomial,
        imaginary_rhs_polynomial,
    ]
    monomials = sorted(
        {
            monomial
            for polynomial in all_polynomials
            for monomial, _ in polynomial.terms()
        },
        reverse=True,
    )
    if not monomials:
        monomials = [(0, 0)]

    coefficient_rows = [
        [QQ.convert(polynomial.nth(*monomial)) for polynomial in columns]
        for columns in (
            real_columns_polynomial,
            imaginary_columns_polynomial,
        )
        for monomial in monomials
    ]
    rhs_rows = [
        [QQ.convert(polynomial.nth(*monomial))]
        for polynomial in (
            real_rhs_polynomial,
            imaginary_rhs_polynomial,
        )
        for monomial in monomials
    ]
    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(rhs_rows, (len(rhs_rows), 1), QQ)
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedPrimitiveCoupledRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            top_degree_bound,
            None,
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
    real_solution = field.zero
    imaginary_solution = field.zero
    basis_count = len(basis)
    for value, candidate in zip(
        coefficients[:basis_count],
        basis,
        strict=True,
    ):
        real_solution += field.convert(value) * candidate
    for value, candidate in zip(
        coefficients[basis_count:],
        basis,
        strict=True,
    ):
        imaginary_solution += field.convert(value) * candidate
    if (
        derivation(real_solution) - lambda_element * imaginary_solution != real_rhs
        or derivation(imaginary_solution) + lambda_element * real_solution
        != imaginary_rhs
    ):
        raise ValueError("bounded primitive coupled matrix reconstruction failed")
    return BoundedPrimitiveCoupledRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        top_degree_bound,
        real_solution,
        imaginary_solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )
