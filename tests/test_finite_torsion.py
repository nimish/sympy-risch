"""Good reduction and finite-prime divisor-class torsion decisions."""

from sympy import Poly, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicDivisorTorsionStatus,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
    decide_algebraic_divisor_torsion,
    finite_divisor_class_order,
    reduce_algebraic_function_field,
)

x, y = symbols("x y")


def algebraic_model(modulus):
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    extension = AlgebraicExtension(Poly(modulus, y, domain=rational.field()))
    return AlgebraicFunctionField(extension)


def test_good_reduction_preserves_the_genus_and_finds_order_29() -> None:
    polynomial = x**6 + 4 * x**5 + 6 * x**4 - 12 * x**3 + 33 * x**2 - 16 * x
    model = algebraic_model(y**2 - polynomial)
    left, right = model.infinite_places()
    divisor = AlgebraicDivisor(model, ((left, 1), (right, -1)))

    reduction = reduce_algebraic_function_field(model, 3)
    finite_order = finite_divisor_class_order(model, divisor, 3)

    assert reduction.target.genus == model.genus == 2
    assert reduction.reduce_divisor(divisor).degree == 0
    assert finite_order.order == 29
    assert finite_order.search.witness is not None
    assert len(finite_order.search.reductions) < finite_order.weil_bound
    assert "baby-step/giant-step" in finite_order.search.detail


def test_incompatible_finite_orders_prove_nontorsion() -> None:
    model = algebraic_model(y**2 - x**3 + x - 1)
    point = model.finite_places(Poly(x, x, domain=QQ))[0]
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((point, 1), (infinity, -1)))

    decision = decide_algebraic_divisor_torsion(
        model,
        divisor,
        primes=(5, 7),
    )

    assert decision.status is AlgebraicDivisorTorsionStatus.NON_TORSION
    assert [(result.prime, result.order) for result in decision.finite_orders] == [
        (5, 8),
        (7, 4),
    ]
    assert decision.order is None
    assert decision.witness is None
