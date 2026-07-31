"""Independent genus checks from the different and Hess Algorithm 13."""

from unittest.mock import patch

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    HessGenusConsistency,
    MonomialExtension,
    certify_function_field_constants,
    hess_genus_consistency,
)

x, y = symbols("x y")


def algebraic_model(modulus) -> AlgebraicFunctionField:
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension)


@pytest.mark.parametrize(
    ("modulus", "expected_genus"),
    [
        (y**2 - x, 0),
        (y**2 - x**3 + x, 1),
        (y**2 - x**5 + x, 2),
        (y**3 - x**4 - x - 1, 3),
    ],
)
def test_different_and_direct_canonical_space_give_the_same_genus(
    modulus,
    expected_genus: int,
) -> None:
    model = algebraic_model(modulus)

    constant_field = certify_function_field_constants(model)

    result = hess_genus_consistency(model, constant_field)

    assert isinstance(result, HessGenusConsistency)
    assert result.canonical_degree == 2 * expected_genus - 2
    assert result.riemann_hurwitz_genus == expected_genus
    assert result.holomorphic_dimension == expected_genus
    assert result.consistent


def test_direct_genus_check_does_not_consume_the_model_genus_property() -> None:
    model = algebraic_model(y**2 - x**3 + x)
    constant_field = certify_function_field_constants(model)

    def fail_if_used(_model) -> int:
        raise AssertionError("the direct Algorithm 13 check consumed model.genus")

    with patch.object(AlgebraicFunctionField, "genus", property(fail_if_used)):
        result = hess_genus_consistency(model, constant_field)

    assert result.riemann_hurwitz_genus == 1
    assert result.holomorphic_dimension == 1
    assert result.consistent


def test_split_infinity_contributes_every_reciprocal_place() -> None:
    model = algebraic_model(y**2 - x**2 + 1)
    constant_field = certify_function_field_constants(model)

    result = hess_genus_consistency(model, constant_field)

    assert len(model.infinite_places()) == 2
    assert result.canonical_degree == -2
    assert result.riemann_hurwitz_genus == 0
    assert result.holomorphic_dimension == 0
    assert result.consistent


def test_enlarged_displayed_constants_are_not_admitted_as_full() -> None:
    constant_extension = algebraic_model(y**2 + 1)
    certified_model = algebraic_model(y**2 - x)
    foreign_certificate = certify_function_field_constants(certified_model)

    with pytest.raises(NotImplementedError, match="no degree-one place"):
        certify_function_field_constants(constant_extension)
    with pytest.raises(ValueError, match="belongs to another model"):
        hess_genus_consistency(constant_extension, foreign_certificate)
