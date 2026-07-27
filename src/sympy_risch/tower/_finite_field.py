r"""Finite-field backend operations used by algebraic tower levels."""

from __future__ import annotations

from typing import cast

from flint import fmpz_mod_mpoly_ctx
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly


def finite_field_function_polynomial_is_irreducible(modulus: Poly) -> bool:
    r"""Factor an integral ``GF(p)[x,y]`` model with python-flint.

    SymPy does not currently factor multivariate finite-field polynomials or
    reliably factor a univariate polynomial over ``GF(p)(x)``. Gauss's lemma
    identifies irreducibility over ``GF(p)(x)[y]`` with irreducibility of the
    primitive polynomial in ``GF(p)[x,y]``. Positive-characteristic function
    field models are therefore admitted only when their coefficients are
    polynomial in the rational parameter.
    """
    parent = modulus.domain
    if not isinstance(parent, FractionField):
        raise NotImplementedError(
            "positive-characteristic algebraic extensions require GF(p)(x)"
        )
    constants = parent.domain
    symbols = parent.symbols
    if (
        constants is None
        or not constants.is_FiniteField
        or symbols is None
        or len(symbols) != 1
    ):
        raise NotImplementedError(
            "positive-characteristic algebraic extensions require GF(p)(x)"
        )
    variable = symbols[0]
    coefficient_polynomials: list[Poly] = []
    for power in range(modulus.degree() + 1):
        coefficient = cast(FracElement, parent.convert(modulus.nth(power)))
        denominator_terms = coefficient.denom.to_dict()
        if any(monomial != (0,) for monomial in denominator_terms):
            raise NotImplementedError(
                "positive-characteristic models currently require polynomial "
                "coefficients"
            )
        denominator = constants.convert(denominator_terms.get((0,), constants.one))
        coefficient_polynomials.append(
            Poly.from_dict(
                {
                    monomial: constants.exquo(
                        constants.convert(value),
                        denominator,
                    )
                    for monomial, value in coefficient.numer.to_dict().items()
                },
                variable,
                domain=constants,
            )
        )

    content = coefficient_polynomials[0]
    for coefficient in coefficient_polynomials[1:]:
        content = content.gcd(coefficient)
    if content.is_zero:
        return False
    primitive = tuple(
        coefficient.exquo(content) for coefficient in coefficient_polynomials
    )
    context = fmpz_mod_mpoly_ctx.get(
        (str(modulus.gens[0]), str(variable)),
        modulus=int(constants.characteristic()),
    )
    terms: dict[tuple[int, ...], int] = {}
    for generator_power, coefficient in enumerate(primitive):
        for (variable_power,), value in coefficient.terms():
            terms[(generator_power, variable_power)] = int(value)
    finite_polynomial = context.from_dict(terms)
    _unit, factors = finite_polynomial.factor()
    return (
        len(factors) == 1
        and int(factors[0][1]) == 1
        and factors[0][0].degrees()[0] == modulus.degree()
    )


__all__ = ["finite_field_function_polynomial_is_irreducible"]
