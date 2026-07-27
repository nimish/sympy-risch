r"""Polynomial reduction and obstruction coordinates for primitive monomials."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel, invert

from .._typing import FieldElement
from ..constant_proofs import require_preserved_constant_field
from ..rational import (
    _solve_rational_exact_derivative_over_domain,
)
from ..reduction.fractions import (
    _reduced_polynomial_fraction,
)
from ..tower.core import (
    Derivation,
    MonomialExtension,
    _coerce,
)
from .rational_base import _standard_rational_base_variable


@dataclass(frozen=True, slots=True)
class PrimitivePolynomialStep:
    r"""One coefficient reduction modulo ``D(K) + C*Dt``.

    For the current coefficient ``a``, the fields record

    ``a = D(exact_coefficient) + eta_multiple*Dt + obstruction``.

    All field-valued entries are native elements of the parent SymPy domain.
    """

    degree: int
    coefficient: FieldElement
    exact_coefficient: FieldElement
    eta_multiple: FieldElement
    obstruction: FieldElement


@dataclass(frozen=True, slots=True)
class PrimitivePolynomialReduction:
    r"""Certified polynomial reduction in a primitive extension."""

    exact_part: Poly
    remainder: Poly
    steps: tuple[PrimitivePolynomialStep, ...]

    @property
    def has_obstruction(self) -> bool:
        """Whether the polynomial is not an exact derivative in the extension."""
        return not self.remainder.is_zero


@dataclass(frozen=True, slots=True)
class _RationalObstructionPivot:
    """One nonzero rational residue coordinate used to quotient by ``C*eta``."""

    factor: Poly
    degree: int
    value: FieldElement


@dataclass(frozen=True, slots=True)
class _PrimitiveObstructionPivot:
    """An iterated Laurent coordinate on a primitive exact remainder."""

    exponents: tuple[int, ...]
    value: FieldElement


def _reduced_fraction_over_constants(
    expression: object,
    variable: Symbol,
    constants: Domain,
) -> tuple[Poly, Poly]:
    """Return a fraction over ``constants`` with a monic denominator."""
    numerator_expression, denominator_expression = cancel(
        expression,
    ).as_numer_denom()
    numerator = Poly(numerator_expression, variable, domain=constants)
    denominator = Poly(denominator_expression, variable, domain=constants)
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(coefficient)
    leading_inverse = constants.one / denominator.LC()
    return numerator.mul_ground(leading_inverse), denominator.monic()


def _partial_numerator(
    obstruction: FieldElement,
    factor: Poly,
    field: Domain,
    variable: Symbol,
    constants: Domain,
) -> Poly:
    r"""Return the numerator of the ``1/factor`` partial-fraction component."""
    numerator, denominator = _reduced_fraction_over_constants(
        field.to_sympy(obstruction),
        variable,
        constants,
    )
    if not denominator.rem(factor).is_zero:
        return factor.zero
    cofactor = denominator.exquo(factor)
    return (numerator * invert(cofactor, factor)).rem(factor)


def _rational_obstruction_pivot(
    eta_obstruction: FieldElement,
    field: Domain,
    variable: Symbol,
    constants: Domain,
) -> _RationalObstructionPivot:
    """Choose a deterministic nonzero coordinate of the class of ``eta``."""
    _, denominator = _reduced_fraction_over_constants(
        field.to_sympy(eta_obstruction),
        variable,
        constants,
    )
    factors = denominator.factor_list()[1]
    if not factors:
        raise ValueError("a nonzero rational obstruction needs a pole")
    factor = factors[0][0]
    partial_numerator = _partial_numerator(
        eta_obstruction,
        factor,
        field,
        variable,
        constants,
    )
    (degree,), value = partial_numerator.terms()[0]
    return _RationalObstructionPivot(
        factor,
        degree,
        constants.convert(value),
    )


def _rational_coefficient_reduction(
    coefficient: FieldElement,
    eta_exact_part: FieldElement,
    eta_obstruction: FieldElement,
    pivot: _RationalObstructionPivot,
    parent: Domain,
    variable: Symbol,
    constants: Domain,
) -> tuple[FieldElement, FieldElement, FieldElement]:
    r"""Return ``b, c, r`` such that ``a = Db + c*eta + r``."""
    result = _solve_rational_exact_derivative_over_domain(
        parent.to_sympy(coefficient),
        variable,
        constants,
    )
    coefficient_exact_part = _coerce(parent, result.exact_part)
    coefficient_obstruction = _coerce(parent, result.obstruction)

    if coefficient_obstruction == parent.zero:
        eta_multiple = constants.zero
    else:
        coordinate = _partial_numerator(
            coefficient_obstruction,
            pivot.factor,
            parent,
            variable,
            constants,
        ).nth(pivot.degree)
        eta_multiple = constants.exquo(
            constants.convert(coordinate),
            pivot.value,
        )

    embedded_multiple = _coerce(parent, constants.to_sympy(eta_multiple))
    exact_coefficient = coefficient_exact_part - embedded_multiple * eta_exact_part
    obstruction = coefficient_obstruction - embedded_multiple * eta_obstruction
    return exact_coefficient, eta_multiple, obstruction


def _laurent_coefficient_at_infinity(
    element: object,
    extension: MonomialExtension,
    exponent: int,
) -> FieldElement:
    r"""Return the coefficient of ``t**exponent`` at the place at infinity.

    For ``f = N(t)/Q(t)``, substitute ``t = 1/z`` and divide the reversed
    numerator series by the reversed denominator series.  The denominator's
    constant term is its nonzero leading coefficient, so the recurrence is
    exact over the parent field and defines this coefficient for every
    rational function in ``K(t)``.
    """
    parent = extension.parent_field
    numerator, denominator = _reduced_polynomial_fraction(
        _coerce(extension.field, element),
        extension,
    )
    if numerator.is_zero:
        return parent.zero

    numerator_degree = numerator.degree()
    denominator_degree = denominator.degree()
    leading_exponent = numerator_degree - denominator_degree
    if exponent > leading_exponent:
        return parent.zero

    target_order = leading_exponent - exponent
    quotient_coefficients: list[FieldElement] = []
    denominator_leading = parent.convert(denominator.LC())
    for order in range(target_order + 1):
        numerator_index = numerator_degree - order
        right_hand_side = (
            parent.convert(numerator.nth(numerator_index))
            if numerator_index >= 0
            else parent.zero
        )
        for denominator_order in range(
            1,
            min(order, denominator_degree) + 1,
        ):
            right_hand_side -= (
                parent.convert(denominator.nth(denominator_degree - denominator_order))
                * quotient_coefficients[order - denominator_order]
            )
        quotient_coefficients.append(parent.exquo(right_hand_side, denominator_leading))
    return quotient_coefficients[target_order]


def _primitive_obstruction_pivot(
    eta_obstruction: FieldElement,
    derivation: Derivation,
) -> _PrimitiveObstructionPivot:
    r"""Choose a deterministic nonzero constant coordinate of ``[eta]``.

    At each lower monomial level, choose the leading exponent in the Laurent
    expansion at infinity and descend through its nonzero coefficient.  The
    resulting iterated coefficient is in the constant field.  Holding those
    exponents fixed defines a constant-linear functional on every element of
    the lower rational-function tower.

    The complement to ``C*[eta]`` is not unique: any constant-linear
    functional nonzero on ``[eta]`` defines a valid projection.  Iterated
    Laurent coefficients are used here because, unlike point evaluation,
    they are defined on every rational function and introduce neither
    factorization nor choices of pole-avoiding values.
    """
    exponents: list[int] = []
    coefficient = eta_obstruction
    current = derivation
    while current.level >= 0:
        extension = current.tower.levels[current.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "recursive primitive coordinates require monomial lower levels"
            )
        numerator, denominator = _reduced_polynomial_fraction(
            _coerce(extension.field, coefficient),
            extension,
        )
        if numerator.is_zero:
            raise ValueError("a primitive obstruction pivot must be nonzero")
        exponent = numerator.degree() - denominator.degree()
        exponents.append(exponent)
        coefficient = _laurent_coefficient_at_infinity(
            coefficient,
            extension,
            exponent,
        )
        current = current.lower()

    value = _coerce(derivation.tower.constants, coefficient)
    if value == derivation.tower.constants.zero:
        raise ValueError("primitive obstruction pivot unexpectedly vanished")
    return _PrimitiveObstructionPivot(tuple(exponents), value)


def _primitive_obstruction_coordinate(
    obstruction: FieldElement,
    derivation: Derivation,
    pivot: _PrimitiveObstructionPivot,
) -> FieldElement:
    """Apply a fixed iterated Laurent coordinate to an obstruction."""
    coefficient = obstruction
    current = derivation
    for exponent in pivot.exponents:
        if current.level < 0:
            raise ValueError("primitive obstruction pivot exceeds the tower")
        extension = current.tower.levels[current.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "recursive primitive coordinates require monomial lower levels"
            )
        coefficient = _laurent_coefficient_at_infinity(
            coefficient,
            extension,
            exponent,
        )
        current = current.lower()
    if current.level != -1:
        raise ValueError("primitive obstruction pivot does not reach constants")
    return _coerce(derivation.tower.constants, coefficient)


def _recursive_coefficient_reduction(
    coefficient: FieldElement,
    eta: FieldElement,
    eta_exact_part: FieldElement,
    eta_obstruction: FieldElement,
    pivot: _PrimitiveObstructionPivot,
    parent: Domain,
    lower_derivation: Derivation,
) -> tuple[FieldElement, FieldElement, FieldElement]:
    r"""Return ``b, c, r`` with ``a = Db + c*eta + r`` over a lower tower.

    The recursive exact-derivative solver supplies representatives of the
    classes of ``a`` and ``eta`` modulo ``D(K)``.  As in the optimized
    rational base case, ``pivot`` is a fixed nonzero constant-valued
    coordinate of the class of ``eta``.  Subtracting the matching multiple
    makes the returned obstruction's pivot coordinate zero and therefore
    gives an idempotent representative modulo ``D(K) + C*eta``.
    """
    # Local import avoids an import cycle: exact-derivative dispatch uses the
    # primitive solver defined in this module, while recursion strictly lowers
    # the tower level at runtime.
    from ..exact import solve_exact_derivative

    result = solve_exact_derivative(coefficient, lower_derivation)
    coefficient_exact_part = _coerce(parent, result.exact_part)
    coefficient_obstruction = _coerce(parent, result.obstruction)

    constants = lower_derivation.tower.constants
    coordinate = _primitive_obstruction_coordinate(
        coefficient_obstruction,
        lower_derivation,
        pivot,
    )
    eta_multiple = constants.exquo(coordinate, pivot.value)
    embedded_multiple = _coerce(parent, constants.to_sympy(eta_multiple))
    exact_coefficient = coefficient_exact_part - embedded_multiple * eta_exact_part
    obstruction = coefficient_obstruction - embedded_multiple * eta_obstruction
    if (
        _primitive_obstruction_coordinate(
            obstruction,
            lower_derivation,
            pivot,
        )
        != constants.zero
    ):
        raise ValueError("primitive obstruction projection did not remove its pivot")

    if (
        lower_derivation(exact_coefficient) + embedded_multiple * eta + obstruction
        != coefficient
    ):
        raise ValueError(
            "recursive primitive coefficient reduction failed to reconstruct"
        )
    return exact_coefficient, eta_multiple, obstruction


def polynomial_reduce_primitive(
    polynomial: Poly,
    derivation: Derivation,
) -> PrimitivePolynomialReduction:
    r"""Reduce a polynomial over a primitive extension ``K(t)``.

    Let ``Dt = eta``.  At each degree this solves

    .. math::

        a = Db + c\eta + r,\qquad b\in K,\ c\in C,

    using exact-derivative remainders as representatives modulo ``D(K)``.  It
    then subtracts

    .. math::

        D\left(bt^n + \frac{c}{n+1}t^{n+1}\right).

    The returned fields satisfy ``polynomial = D(exact_part) + remainder``.
    Every nonzero coefficient of ``remainder`` is reduced modulo
    ``D(K) + C*eta``.

    Over the standard rational base, Bronstein's residue-pivot normalization
    supplies a constant coordinate of the obstruction space.  Over a
    supported primitive lower tower, a fixed iterated Laurent coefficient at
    infinity supplies the analogous coordinate.  Both paths remove the
    ``eta`` direction even when additional obstruction coordinates remain.
    """
    if not isinstance(polynomial, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if derivation.level < 0:
        raise TypeError("primitive polynomial reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("primitive polynomial reduction requires a monomial level")

    derivation.polynomial(polynomial)
    if extension.generator_derivative.degree() != 0:
        raise ValueError("primitive polynomial reduction requires Dt in K")

    lower_derivation = derivation.lower()
    parent = extension.parent_field
    constants = derivation.tower.constants
    eta = parent.convert(extension.generator_derivative.nth(0))

    variable = None
    if lower_derivation.level == 0:
        variable = _standard_rational_base_variable(lower_derivation)
    require_preserved_constant_field(derivation, "primitive polynomial reduction")
    if variable is not None:
        eta_result = _solve_rational_exact_derivative_over_domain(
            parent.to_sympy(eta),
            variable,
            constants,
        )
    else:
        # See the local-import note in _recursive_coefficient_reduction.
        from ..exact import solve_exact_derivative

        eta_result = solve_exact_derivative(eta, lower_derivation)

    if eta_result.has_solution:
        raise ValueError(
            "constant-field replay is preserved, but Dt is exact in the lower field"
        )
    eta_exact_part = _coerce(parent, eta_result.exact_part)
    eta_obstruction = _coerce(parent, eta_result.obstruction)
    pivot = (
        _rational_obstruction_pivot(
            eta_obstruction,
            parent,
            variable,
            constants,
        )
        if variable is not None
        else _primitive_obstruction_pivot(
            eta_obstruction,
            lower_derivation,
        )
    )

    exact_part = polynomial.zero
    remainder = polynomial.zero
    working = polynomial
    steps: list[PrimitivePolynomialStep] = []

    while not working.is_zero:
        degree = working.degree()
        coefficient = parent.convert(working.nth(degree))
        if variable is not None:
            if not isinstance(pivot, _RationalObstructionPivot):
                raise TypeError("rational primitive reduction needs a rational pivot")
            exact_coefficient, eta_multiple, obstruction = (
                _rational_coefficient_reduction(
                    coefficient,
                    eta_exact_part,
                    eta_obstruction,
                    pivot,
                    parent,
                    variable,
                    constants,
                )
            )
        else:
            if not isinstance(pivot, _PrimitiveObstructionPivot):
                raise TypeError("recursive primitive reduction needs a Laurent pivot")
            exact_coefficient, eta_multiple, obstruction = (
                _recursive_coefficient_reduction(
                    coefficient,
                    eta,
                    eta_exact_part,
                    eta_obstruction,
                    pivot,
                    parent,
                    lower_derivation,
                )
            )
        correction = Poly.from_dict(
            {
                (degree,): exact_coefficient,
                (degree + 1,): _coerce(
                    parent,
                    constants.to_sympy(eta_multiple / constants.convert(degree + 1)),
                ),
            },
            extension.generator,
            domain=parent,
        )
        obstruction_term = Poly.from_dict(
            {(degree,): obstruction},
            extension.generator,
            domain=parent,
        )
        new_working = working - derivation.polynomial(correction) - obstruction_term
        if not new_working.is_zero and new_working.degree() >= degree:
            raise ValueError("primitive polynomial reduction did not lower the degree")

        exact_part += correction
        remainder += obstruction_term
        steps.append(
            PrimitivePolynomialStep(
                degree,
                coefficient,
                exact_coefficient,
                eta_multiple,
                obstruction,
            )
        )
        working = new_working

    return PrimitivePolynomialReduction(
        exact_part,
        remainder,
        tuple(steps),
    )
