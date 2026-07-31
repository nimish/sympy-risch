"""Exact residue maps for algebraic number-field constants."""

from dataclasses import replace

import pytest
from sympy import sqrt
from sympy.polys.domains import QQ

from sympy_risch import (
    BadConstantFieldReduction,
    constant_field_prime_reductions,
)


def test_inert_prime_builds_the_full_finite_residue_field() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    reduction = constant_field_prime_reductions(constants, 3)[0]
    generator = constants.from_sympy(sqrt(2))
    value = constants.from_sympy(1 + sqrt(2))

    assert reduction.residue_degree == 2
    assert reduction.field_size == 9
    assert reduction.reduce(generator) ** 2 == reduction.target.convert(2)
    assert reduction.reduce(value * value) == reduction.reduce(value) ** 2


def test_split_prime_returns_each_prime_above_the_characteristic() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    reductions = constant_field_prime_reductions(constants, 7)
    generator = constants.from_sympy(sqrt(2))

    assert len(reductions) == 2
    assert {reduction.reduce(generator) for reduction in reductions} == {
        reductions[0].target.convert(3),
        reductions[0].target.convert(-3),
    }
    assert all(reduction.residue_degree == 1 for reduction in reductions)


def test_ramified_prime_and_vanishing_denominator_are_rejected() -> None:
    constants = QQ.algebraic_field(sqrt(2))

    with pytest.raises(BadConstantFieldReduction, match="ramified"):
        constant_field_prime_reductions(constants, 2)

    reduction = constant_field_prime_reductions(constants, 7)[1]
    pole = constants.from_sympy(1 / (sqrt(2) - 3))
    with pytest.raises(BadConstantFieldReduction, match="denominator"):
        reduction.reduce(pole)

    with pytest.raises(TypeError):
        replace(reduction, rational_prime=5)
