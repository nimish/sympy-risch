r"""Finite-prime decisions for characteristic-zero divisor-class torsion.

For good reduction at a rational prime ``p``, reduction on the prime-to-``p``
torsion of the Jacobian is injective.  If a characteristic-zero divisor class
has order ``n`` and its reduction has order ``m_p``, then

``n / m_p`` is a power of ``p``.

Two distinct good primes therefore determine the only possible order:
``n = lcm(m_p, m_q)``.  Incompatible reduced orders prove nontorsion; a
compatible candidate is verified with characteristic-zero reduced divisor
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isqrt, lcm

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.fractionfield import FractionField

from .divisor_classes import (
    AlgebraicTorsionSearch,
    AlgebraicTorsionStatus,
    _search_algebraic_divisor_torsion_bsgs,
    _validate_anchor,
)
from .finite_reduction import (
    AlgebraicFunctionFieldReduction,
    BadAlgebraicReduction,
    reduce_algebraic_function_field,
)
from .function_field import AlgebraicDivisor, AlgebraicFunctionField


def _weil_jacobian_upper_bound(field_size: int, genus: int) -> int:
    """Return a safe integral form of ``(sqrt(q) + 1) ** (2*g)``."""
    root = isqrt(field_size)
    if root * root < field_size:
        root += 1
    return (root + 1) ** (2 * genus)


@dataclass(frozen=True, slots=True)
class FiniteDivisorClassOrder:
    """The exact order of a reduced class in one finite Jacobian."""

    reduction: AlgebraicFunctionFieldReduction
    source_divisor: AlgebraicDivisor
    target_divisor: AlgebraicDivisor
    weil_bound: int
    search: AlgebraicTorsionSearch

    def __post_init__(self) -> None:
        if self.source_divisor.model is not self.reduction.source:
            raise ValueError("the source divisor belongs to another reduction")
        if self.target_divisor.model is not self.reduction.target:
            raise ValueError("the target divisor belongs to another reduction")
        if self.search.model is not self.reduction.target:
            raise ValueError("the finite search belongs to another reduction")
        if self.search.divisor != self.target_divisor:
            raise ValueError("the finite search used another divisor")
        if (
            self.search.status is not AlgebraicTorsionStatus.TORSION
            or self.search.order is None
            or self.search.order > self.weil_bound
        ):
            raise ValueError("the finite Jacobian search did not determine an order")

    @property
    def prime(self) -> int:
        """The residue characteristic."""
        return self.reduction.prime

    @property
    def order(self) -> int:
        """The exact reduced divisor-class order."""
        if self.search.order is None:
            raise ValueError("the finite order certificate is incomplete")
        return self.search.order


def finite_divisor_class_order(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    prime: int,
) -> FiniteDivisorClassOrder:
    """Compute the order of ``divisor`` after one supported good reduction."""
    reduction = reduce_algebraic_function_field(model, prime)
    target_divisor = reduction.reduce_divisor(divisor)
    parent = reduction.target.extension.parent_field
    if not isinstance(parent, FractionField):
        raise TypeError("a reduced function field needs a rational parent")
    constants = parent.domain
    if constants is None:
        raise TypeError("a reduced function field needs a constant domain")
    field_size = int(constants.characteristic())
    weil_bound = _weil_jacobian_upper_bound(field_size, reduction.target.genus)
    search = _search_algebraic_divisor_torsion_bsgs(
        reduction.target,
        target_divisor,
        max_order=max(1, weil_bound),
        anchor=_validate_anchor(reduction.target, None),
    )
    if search.status is not AlgebraicTorsionStatus.TORSION:
        raise NotImplementedError(
            "the finite class order was not found within the Weil bound: "
            + search.detail
        )
    return FiniteDivisorClassOrder(
        reduction,
        divisor,
        target_divisor,
        weil_bound,
        search,
    )


class AlgebraicDivisorTorsionStatus(Enum):
    """Outcome of a finite-prime characteristic-zero torsion decision."""

    TORSION = "torsion"
    NON_TORSION = "non_torsion"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicDivisorTorsionDecision:
    """A verified torsion relation, nontorsion proof, or exact boundary."""

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    status: AlgebraicDivisorTorsionStatus
    order: int | None
    witness: ExtensionElement | None
    finite_orders: tuple[FiniteDivisorClassOrder, ...]
    characteristic_zero_search: AlgebraicTorsionSearch | None
    detail: str

    def __post_init__(self) -> None:
        if self.divisor.model is not self.model or self.divisor.degree != 0:
            raise ValueError(
                "a torsion decision needs a model-owned degree-zero divisor"
            )
        if len(self.finite_orders) > 2:
            raise ValueError("a torsion decision uses at most two finite reductions")
        if any(
            order.reduction.source is not self.model
            or order.source_divisor != self.divisor
            for order in self.finite_orders
        ):
            raise ValueError("a finite order belongs to another torsion problem")
        search = self.characteristic_zero_search
        if search is not None and (
            search.model is not self.model or search.divisor != self.divisor
        ):
            raise ValueError("the characteristic-zero search used another divisor")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a torsion decision needs a detail")

        if self.status is AlgebraicDivisorTorsionStatus.TORSION:
            if (
                self.order is None
                or self.witness is None
                or search is None
                or search.status is not AlgebraicTorsionStatus.TORSION
                or search.order != self.order
                or search.witness != self.witness
            ):
                raise ValueError("a torsion decision lacks its verified relation")
            return
        if self.order is not None or self.witness is not None:
            raise ValueError("only a torsion decision carries an order and witness")
        if self.status is AlgebraicDivisorTorsionStatus.NON_TORSION:
            if len(self.finite_orders) != 2:
                raise ValueError("a nontorsion decision needs two good reductions")
            return
        if self.status is AlgebraicDivisorTorsionStatus.UNSUPPORTED:
            return
        raise ValueError("the divisor torsion status is unknown")

    @property
    def torsion(self) -> bool:
        """Whether this decision reconstructed a finite-order relation."""
        return self.status is AlgebraicDivisorTorsionStatus.TORSION

    @property
    def complete_negative(self) -> bool:
        """Whether the finite-prime data proves the class nontorsion."""
        return self.status is AlgebraicDivisorTorsionStatus.NON_TORSION


def _is_power_of(value: int, prime: int) -> bool:
    if value <= 0:
        return False
    while value % prime == 0:
        value //= prime
    return value == 1


def decide_algebraic_divisor_torsion(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    *,
    primes: tuple[int, ...] = (3, 5, 7, 11, 13, 17, 19),
) -> AlgebraicDivisorTorsionDecision:
    r"""Decide torsion from two supported good reductions and exact replay."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("torsion decision needs an algebraic function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("torsion decision needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if divisor.degree != 0:
        raise ValueError("only degree-zero divisor classes can be torsion")

    finite_orders: list[FiniteDivisorClassOrder] = []
    failures: list[str] = []
    for prime in primes:
        if any(existing.prime == prime for existing in finite_orders):
            continue
        try:
            finite_orders.append(finite_divisor_class_order(model, divisor, prime))
        except (
            BadAlgebraicReduction,
            NotImplementedError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            failures.append(f"{prime}: {exc}")
            continue
        if len(finite_orders) == 2:
            break

    if len(finite_orders) < 2:
        detail = "fewer than two supported good reductions were available"
        if failures:
            detail += ": " + "; ".join(failures)
        return AlgebraicDivisorTorsionDecision(
            model,
            divisor,
            AlgebraicDivisorTorsionStatus.UNSUPPORTED,
            None,
            None,
            tuple(finite_orders),
            None,
            detail,
        )

    left, right = finite_orders
    candidate = lcm(left.order, right.order)
    compatible = _is_power_of(candidate // left.order, left.prime) and _is_power_of(
        candidate // right.order,
        right.prime,
    )
    if not compatible:
        return AlgebraicDivisorTorsionDecision(
            model,
            divisor,
            AlgebraicDivisorTorsionStatus.NON_TORSION,
            None,
            None,
            tuple(finite_orders),
            None,
            "the two reduced orders cannot be reductions of one "
            "characteristic-zero torsion order",
        )

    search = _search_algebraic_divisor_torsion_bsgs(
        model,
        divisor,
        max_order=max(1, candidate),
        anchor=_validate_anchor(model, None),
    )
    if search.status is AlgebraicTorsionStatus.TORSION:
        if search.order != candidate or search.witness is None:
            return AlgebraicDivisorTorsionDecision(
                model,
                divisor,
                AlgebraicDivisorTorsionStatus.UNSUPPORTED,
                None,
                None,
                tuple(finite_orders),
                search,
                "finite reduction and characteristic-zero orders disagree",
            )
        return AlgebraicDivisorTorsionDecision(
            model,
            divisor,
            AlgebraicDivisorTorsionStatus.TORSION,
            candidate,
            search.witness,
            tuple(finite_orders),
            search,
            "two good reductions proposed and characteristic-zero replay "
            f"verified order {candidate}",
        )
    if search.status is AlgebraicTorsionStatus.INCONCLUSIVE:
        return AlgebraicDivisorTorsionDecision(
            model,
            divisor,
            AlgebraicDivisorTorsionStatus.NON_TORSION,
            None,
            None,
            tuple(finite_orders),
            search,
            "the unique finite-prime candidate failed characteristic-zero replay",
        )
    return AlgebraicDivisorTorsionDecision(
        model,
        divisor,
        AlgebraicDivisorTorsionStatus.UNSUPPORTED,
        None,
        None,
        tuple(finite_orders),
        search,
        "characteristic-zero verification is unsupported: " + search.detail,
    )


__all__ = [
    "AlgebraicDivisorTorsionDecision",
    "AlgebraicDivisorTorsionStatus",
    "FiniteDivisorClassOrder",
    "decide_algebraic_divisor_torsion",
    "finite_divisor_class_order",
]
