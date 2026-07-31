"""Good reduction and finite-prime divisor-class torsion decisions."""

import pytest
from sympy import Poly, Rational, sqrt, symbols
from sympy.polys.domains import QQ

from sympy_risch import (
    AlgebraicDivisor,
    AlgebraicDivisorTorsionStatus,
    AlgebraicExtension,
    AlgebraicFunctionField,
    DifferentialTower,
    MonomialExtension,
    constant_field_prime_reductions,
    decide_algebraic_divisor_torsion,
    finite_divisor_class_order,
    prepare_algebraic_torsion_reductions,
    reduce_algebraic_function_field,
)

x, y = symbols("x y")


def algebraic_model(modulus, *, constants=QQ):
    """Return the direct algebraic function field ``QQ(x)(y)``."""
    rational = DifferentialTower(constants).extend(
        MonomialExtension(Poly(1, x, domain=constants)),
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


def test_algebraic_constants_use_split_prime_ideals_for_curve_reduction() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    model = algebraic_model(
        y**2 - x * (x - sqrt(2)) * (x - 1),
        constants=constants,
    )

    constant_primes = constant_field_prime_reductions(constants, 7)
    reductions = tuple(
        reduce_algebraic_function_field(model, constant_prime)
        for constant_prime in constant_primes
    )

    assert len(reductions) == 2
    assert all(reduction.prime == 7 for reduction in reductions)
    assert all(reduction.field_size == 7 for reduction in reductions)
    assert all(reduction.target.genus == model.genus == 1 for reduction in reductions)
    source_constant = model.extension.field.from_sympy(sqrt(2))
    assert {
        reduction.target.extension.field.to_sympy(
            reduction.reduce_element(source_constant)
        )
        for reduction in reductions
    } == {-3, 3}


def test_curve_reduction_keeps_nonprime_residue_fields_explicitly_unsupported() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    model = algebraic_model(
        y**2 - x * (x - sqrt(2)) * (x - 1),
        constants=constants,
    )

    with pytest.raises(NotImplementedError, match=r"GF\(p\^f\)"):
        reduce_algebraic_function_field(model, 3)


def test_prepared_reductions_are_reused_for_multiple_divisors() -> None:
    constants = QQ.algebraic_field(sqrt(2))
    model = algebraic_model(
        y**2 - x * (x - sqrt(2)) * (x - 1),
        constants=constants,
    )
    point = model.finite_places(Poly(x - sqrt(2), x, domain=constants))[0]
    infinity = model.infinite_places()[0]
    divisor = AlgebraicDivisor(model, ((point, 1), (infinity, -1)))
    batch = prepare_algebraic_torsion_reductions(
        model,
        primes=(7, 17),
        divisors=(divisor,),
    )

    first = decide_algebraic_divisor_torsion(
        model,
        divisor,
        reduction_batch=batch,
    )
    second = decide_algebraic_divisor_torsion(
        model,
        divisor,
        reduction_batch=batch,
    )

    assert first.status is second.status is AlgebraicDivisorTorsionStatus.TORSION
    assert first.order == second.order == 2
    assert tuple(order.reduction for order in first.finite_orders) == batch.reductions
    assert tuple(order.reduction for order in second.finite_orders) == batch.reductions


def test_preparation_skips_a_curve_good_prime_that_is_bad_for_the_divisor() -> None:
    model = algebraic_model(y**2 - x)
    place = model.finite_places(
        Poly(x - Rational(1, 3), x, domain=QQ),
    )[0]
    divisor = AlgebraicDivisor(model, ((place, 1),))

    batch = prepare_algebraic_torsion_reductions(
        model,
        primes=(3, 5, 7),
        divisors=(divisor,),
    )

    assert [reduction.prime for reduction in batch.reductions] == [5, 7]
    assert batch.reduced_divisor(batch.reductions[0], divisor).model is (
        batch.reductions[0].target
    )
    assert batch.failures and batch.failures[0].startswith("3:")
