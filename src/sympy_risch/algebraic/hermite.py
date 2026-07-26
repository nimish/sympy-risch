r"""Algebraic Hermite reduction over caller-certified integral bases."""

from __future__ import annotations

from typing import cast

from sympy.core.symbol import Symbol
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.matrices.normalforms import smith_normal_decomp
from sympy.polys.polytools import Poly, invert

from ..tower import Derivation
from ._field import (
    _base_polynomial_ring,
    _common_denominator,
    _embed_parent,
    _extension_element,
    _parent_element_polynomial,
    _polynomial_parent_element,
)
from .basis import IntegralBasis


def _coordinate_matrix(
    basis: IntegralBasis,
    elements: tuple[ExtensionElement, ...],
) -> DomainMatrix:
    parent = basis.extension.parent_field
    rank = len(basis.elements)
    columns = [basis.coordinates(element) for element in elements]
    return DomainMatrix(
        [[columns[column][row] for column in range(rank)] for row in range(rank)],
        (rank, rank),
        parent,
    )


def _primitive_polynomial_vector(
    elements: tuple[FracElement, ...],
    parent: FractionField,
    constants: Domain,
    variable: Symbol,
) -> tuple[Poly, ...]:
    """Clear denominators and remove the polynomial gcd from a vector."""
    denominator = _common_denominator(elements, constants, variable)
    denominator_element = _polynomial_parent_element(denominator, parent)
    numerators = tuple(
        _parent_element_polynomial(
            element * denominator_element,
            parent,
            constants,
            variable,
        )
        for element in elements
    )
    common_factor = Poly(0, variable, domain=constants)
    for numerator in numerators:
        common_factor = common_factor.gcd(numerator)
    if common_factor.is_zero:
        raise ValueError("expected a nonzero polynomial relation")
    common_factor = common_factor.monic()
    return tuple(numerator.exquo(common_factor) for numerator in numerators)


def _denominator_excess(denominator: Poly) -> int:
    return sum(
        (multiplicity - 1) * factor.degree()
        for factor, multiplicity in denominator.sqf_list()[1]
    )


def _adjoin_integral_element(
    basis: IntegralBasis,
    numerators: tuple[Poly, ...],
    denominator: Poly,
) -> IntegralBasis:
    r"""Adjoin ``w0 = (C_1*w_1 + ... + C_n*w_n)/F`` to the basis module.

    Relative to ``(w_1/F, ..., w_n/F)``, the old basis and ``w0`` are the
    rows of ``[F*I; C]``.  Bronstein reduces this matrix by Hermitian row
    reduction after Theorem 3 in section 2.1 of his ISSAC'98 tutorial.  A
    Smith decomposition performs equivalent unimodular row operations over
    the PID ``K[x]``; the nonzero rows of ``left*[F*I; C]`` are therefore a
    basis of the enlarged module.
    """
    extension = basis.extension
    parent, constants, variable = _base_polynomial_ring(extension)
    polynomial_ring = constants.poly_ring(variable)
    rank = len(basis.elements)
    denominator_coefficient = polynomial_ring.from_sympy(denominator.as_expr())
    zero = polynomial_ring.zero
    rows = [
        [denominator_coefficient if row == column else zero for column in range(rank)]
        for row in range(rank)
    ]
    rows.append(
        [polynomial_ring.from_sympy(numerator.as_expr()) for numerator in numerators]
    )
    generators = DomainMatrix(rows, (rank + 1, rank), polynomial_ring)
    _, left, _ = smith_normal_decomp(generators)
    reduced_rows = [
        row
        for row in (left * generators).to_list()
        if any(coefficient != zero for coefficient in row)
    ]
    if len(reduced_rows) != rank:
        raise ValueError("integral-element adjunction changed the field rank")

    denominator_element = _polynomial_parent_element(denominator, parent)
    elements = tuple(
        basis.element(
            tuple(
                parent.from_sympy(polynomial_ring.to_sympy(coefficient))
                / denominator_element
                for coefficient in row
            )
        )
        for row in reduced_rows
    )
    return IntegralBasis(extension, elements)


def _hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
    *,
    lazy: bool,
) -> tuple[ExtensionElement, ExtensionElement, IntegralBasis]:
    r"""Implement equations (7)--(9) of Bronstein's ISSAC'98 tutorial.

    For denominator ``D = U*V**m``, the algorithm seeks ``B`` such that
    subtracting ``derivation(B/V**(m-1))`` removes the poles of order ``m``.
    The lazy branch implements Bronstein's three missing-integral-element
    certificates: dependent ``S_i``, a solution denominator intersecting
    ``V``, and a nonsquarefree basis-derivative denominator.
    """
    extension = basis.extension
    if derivation.level < 0 or derivation.tower.levels[derivation.level] != extension:
        raise ValueError("the derivation must be bound to the algebraic extension")

    parent, constants, variable = _base_polynomial_ring(extension)
    lower_derivation = derivation.lower()
    if (
        lower_derivation.field != parent
        or lower_derivation(parent.from_sympy(variable)) != parent.one
    ):
        raise NotImplementedError(
            "algebraic Hermite reduction currently requires the standard "
            "rational derivation Dx = 1",
        )
    remainder = _extension_element(extension, integrand)
    exact_part = extension.field.zero
    differentiated_basis: IntegralBasis | None = None
    basis_derivatives: tuple[ExtensionElement, ...] | None = None
    derivative_matrix: DomainMatrix | None = None

    while True:
        representation = basis.fraction(remainder)
        repeated_factors = [
            (factor, multiplicity)
            for factor, multiplicity in representation.denominator.sqf_list()[1]
            if multiplicity > 1
        ]
        if not repeated_factors:
            return exact_part, remainder, basis

        repeated_factor, multiplicity = max(
            repeated_factors,
            key=lambda item: item[1],
        )
        denominator = representation.denominator
        cofactor = denominator.exquo(repeated_factor**multiplicity)
        scale = cofactor * repeated_factor**multiplicity
        candidate_denominator = repeated_factor ** (multiplicity - 1)

        if basis is not differentiated_basis:
            # D(w_i) depends only on the basis and derivation, not on the pole
            # currently being reduced.  Cache its coordinate matrix until a
            # lazy refinement replaces the basis.
            basis_derivatives = cast(
                tuple[ExtensionElement, ...],
                tuple(derivation(element) for element in basis.elements),
            )
            derivative_matrix = _coordinate_matrix(
                basis,
                basis_derivatives,
            )
            differentiated_basis = basis
        assert basis_derivatives is not None
        assert derivative_matrix is not None
        derivative_rows = derivative_matrix.to_list()

        if lazy:
            refined_from_derivative = False
            for column, basis_derivative in enumerate(basis_derivatives):
                derivative_denominator = _common_denominator(
                    tuple(
                        derivative_rows[row][column]
                        for row in range(len(basis.elements))
                    ),
                    constants,
                    variable,
                )
                if (
                    derivative_denominator.gcd(
                        derivative_denominator.diff(),
                    ).degree()
                    == 0
                ):
                    continue

                # Bronstein, Theorem 3: if F = product(F_j**j) is the
                # nonsquarefree denominator of D(w_i), then
                # product(F_j)*D(w_i) is integral but missing from the current
                # K[x]-module.
                squarefree_part = Poly(1, variable, domain=constants)
                for factor, _ in derivative_denominator.sqf_list()[1]:
                    squarefree_part *= factor
                integral_element = (
                    _embed_parent(extension, squarefree_part) * basis_derivative
                )
                integral_fraction = basis.fraction(integral_element)
                if integral_fraction.denominator.degree() == 0:
                    raise ValueError(
                        "nonsquarefree derivative did not enlarge the basis module",
                    )
                basis = _adjoin_integral_element(
                    basis,
                    integral_fraction.numerators,
                    integral_fraction.denominator,
                )
                refined_from_derivative = True
                break
            if refined_from_derivative:
                continue

        scale_parent = _polynomial_parent_element(scale, parent)
        candidate_denominator_parent = _polynomial_parent_element(
            candidate_denominator,
            parent,
        )
        candidate_denominator_derivative = derivation.lower()(
            candidate_denominator_parent,
        )
        candidate_denominator_element = _embed_parent(
            extension,
            candidate_denominator,
        )
        # Bronstein (8)--(9): S_i = U*V**m * D(w_i/V**(m-1)).
        # Solving sum(f_i*S_i) = numerator determines the correction modulo V.
        derivative_scale = scale_parent / candidate_denominator_parent
        diagonal_scale = (
            scale_parent
            * candidate_denominator_derivative
            / candidate_denominator_parent**2
        )
        rank = len(basis.elements)
        system = DomainMatrix(
            [
                [
                    derivative_scale * derivative_rows[row][column]
                    - (diagonal_scale if row == column else parent.zero)
                    for column in range(rank)
                ]
                for row in range(rank)
            ],
            (rank, rank),
            parent,
        )
        numerator_element = remainder * _embed_parent(extension, denominator)
        numerator_coordinates = basis.coordinates(numerator_element)
        right_hand_side = DomainMatrix(
            [[coordinate] for coordinate in numerator_coordinates],
            (rank, 1),
            parent,
        )

        try:
            # Fraction-free solving retains one common denominator.  Besides
            # avoiding expression swell, that denominator is exactly what the
            # lazy-refinement test below must compare with V.
            solution_numerators_matrix, solution_denominator_element = system.solve_den(
                right_hand_side
            )
        except DMNonInvertibleMatrixError as exc:
            if lazy:
                # Bronstein, Theorem 1: a primitive polynomial relation
                # sum(T_i*S_i) = 0 exposes the missing integral element
                # U/V * sum(T_i*w_i).
                relations = system.nullspace().to_list()
                if not relations:
                    raise ValueError(
                        "the singular Hermite system has no nullspace",
                    ) from exc
                relation = _primitive_polynomial_vector(
                    tuple(relations[0]),
                    parent,
                    constants,
                    variable,
                )
                basis = _adjoin_integral_element(
                    basis,
                    tuple(cofactor * coefficient for coefficient in relation),
                    repeated_factor,
                )
                continue
            raise ValueError(
                "the supplied integral basis does not support Hermite reduction",
            ) from exc
        solution = tuple(
            row[0] / solution_denominator_element
            for row in solution_numerators_matrix.to_list()
        )

        solution_denominator = _common_denominator(
            solution,
            constants,
            variable,
        )
        solution_denominator_element = _polynomial_parent_element(
            solution_denominator,
            parent,
        )
        solution_numerators = tuple(
            _parent_element_polynomial(
                coefficient * solution_denominator_element,
                parent,
                constants,
                variable,
            )
            for coefficient in solution
        )
        shared_factor = solution_denominator.gcd(repeated_factor)
        if shared_factor.degree() != 0:
            if lazy:
                # Bronstein, Theorem 2: for f_i = T_i/Q and G = gcd(V, Q),
                #
                #     w0 = U*(V/G)/G * sum(T_i*w_i)
                #
                # is integral but absent from the current K[x]-module.  Its
                # adjunction strictly enlarges that module before we retry V.
                multiplier = cofactor * repeated_factor.exquo(shared_factor)
                basis = _adjoin_integral_element(
                    basis,
                    tuple(multiplier * numerator for numerator in solution_numerators),
                    shared_factor,
                )
                continue
            raise ValueError(
                "the supplied basis is not an integral basis at the repeated factor",
            )
        # When gcd(Q, V) = 1, Q has an inverse modulo V.  Taking
        # B_i = Q**(-1)*T_i mod V gives B/V**(m-1), the exact part in (7).
        inverse = invert(solution_denominator, repeated_factor)
        correction_coordinates = tuple(
            _polynomial_parent_element(
                (inverse * numerator).rem(repeated_factor),
                parent,
            )
            for numerator in solution_numerators
        )
        correction = (
            basis.element(correction_coordinates) / candidate_denominator_element
        )

        previous_excess = _denominator_excess(denominator)
        new_remainder = remainder - derivation(correction)
        new_denominator = basis.fraction(new_remainder).denominator
        if _denominator_excess(new_denominator) >= previous_excess:
            raise ValueError(
                "algebraic Hermite reduction did not lower the repeated pole; "
                "the supplied basis may not be integral",
            )

        exact_part += correction
        remainder = new_remainder


def hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[ExtensionElement, ExtensionElement]:
    r"""Write ``integrand = D(g) + h`` using a supplied integral basis."""
    exact, remainder, _ = _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=False,
    )
    return exact, remainder


def lazy_hermite_reduce_algebraic(
    integrand: object,
    derivation: Derivation,
    basis: IntegralBasis,
) -> tuple[ExtensionElement, ExtensionElement, IntegralBasis]:
    r"""Hermite-reduce while refining denominator-obstructed integral bases.

    This implements the three lazy-refinement theorems in Manuel Bronstein,
    *Symbolic Integration Tutorial*, ISSAC 1998, section 2.1, which summarizes
    the fuller treatment in *The Lazy Hermite Reduction*, INRIA RR-3562.
    """
    return _hermite_reduce_algebraic(
        integrand,
        derivation,
        basis,
        lazy=True,
    )
