"""Hess finite/infinite ideal representations and Riemann--Roch bases."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
    compute_riemann_roch_space,
    hess_divisor_ideals,
)
from sympy_risch.algebraic_geometry.riemann_roch import RiemannRochCertificate

x, y = symbols("x y")


def elliptic_model() -> tuple[AlgebraicFunctionField, AlgebraicExtension]:
    """Return the smooth elliptic field ``QQ(x, y)``, ``y**2 = x**3 - x``."""
    rational = DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ)))
    extension = AlgebraicExtension(
        Poly(y**2 - x**3 + x, y, domain=rational.field()),
    )
    return AlgebraicFunctionField(extension), extension


def test_hess_representation_separates_finite_and_infinite_conditions() -> None:
    model, extension = elliptic_model()
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinite_place, 1)))
    representation = hess_divisor_ideals(model, divisor)

    assert representation.finite_ideal.order is model.order
    assert representation.infinite_ideal.order is model.reciprocal_chart.closure.order
    for element in (extension.field.one, extension.field.from_sympy(x)):
        assert (
            representation.contains(element)
            == (model.principal_divisor(element) + divisor).is_effective
        )


def test_hess_computes_a_mixed_support_space_without_local_expansions() -> None:
    model, _extension = elliptic_model()
    (finite_place,) = model.finite_places(Poly(x, x, domain=QQ))
    (infinite_place,) = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((finite_place, 1), (infinite_place, 2)))

    space = compute_riemann_roch_space(model, divisor)

    assert space.certificate is RiemannRochCertificate.HESS_IDEAL_INTERSECTION
    assert space.dimension == divisor.degree
    assert all(space.contains(element) for element in space.basis)
