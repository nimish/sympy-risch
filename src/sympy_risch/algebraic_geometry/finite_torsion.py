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

from ..algebraic._field import _base_polynomial_ring
from .constant_reduction import constant_field_prime_reductions
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

_DEFAULT_REDUCTION_PRIMES = (3, 5, 7, 11, 13, 17, 19)


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
    prime: int | AlgebraicFunctionFieldReduction,
) -> FiniteDivisorClassOrder:
    """Compute the order of ``divisor`` after one supported good reduction."""
    if isinstance(prime, AlgebraicFunctionFieldReduction):
        reduction = prime
        if reduction.source is not model:
            raise ValueError("the finite reduction belongs to another model")
    else:
        reduction = reduce_algebraic_function_field(model, prime)
    target_divisor = reduction.reduce_divisor(divisor)
    return _finite_divisor_class_order_from_reduction(
        reduction,
        divisor,
        target_divisor,
    )


def _finite_divisor_class_order_from_reduction(
    reduction: AlgebraicFunctionFieldReduction,
    source_divisor: AlgebraicDivisor,
    target_divisor: AlgebraicDivisor,
) -> FiniteDivisorClassOrder:
    """Compute a finite order from an already reduced divisor."""
    field_size = reduction.field_size
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
        source_divisor,
        target_divisor,
        weil_bound,
        search,
    )


class AlgebraicDivisorTorsionStatus(Enum):
    """Outcome of a finite-prime characteristic-zero torsion decision."""

    TORSION = "torsion"
    NON_TORSION = "non_torsion"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicFiniteReductionBatch:
    """Factory-owned good reductions reusable across many source divisors."""

    model: AlgebraicFunctionField
    source_divisors: tuple[AlgebraicDivisor, ...]
    reductions: tuple[AlgebraicFunctionFieldReduction, ...]
    target_divisors: tuple[tuple[AlgebraicDivisor, ...], ...]
    failures: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("finite-reduction batches must be created by the factory")

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        source_divisors: tuple[AlgebraicDivisor, ...],
        reductions: tuple[AlgebraicFunctionFieldReduction, ...],
        target_divisors: tuple[tuple[AlgebraicDivisor, ...], ...],
        failures: tuple[str, ...],
    ) -> AlgebraicFiniteReductionBatch:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "source_divisors", source_divisors)
        object.__setattr__(result, "reductions", reductions)
        object.__setattr__(result, "target_divisors", target_divisors)
        object.__setattr__(result, "failures", failures)
        if any(divisor.model is not model for divisor in source_divisors):
            raise ValueError("a finite-reduction batch contains a foreign divisor")
        if any(reduction.source is not model for reduction in reductions):
            raise ValueError("a finite-reduction batch contains a foreign model")
        if len(target_divisors) != len(reductions) or any(
            len(row) != len(source_divisors) for row in target_divisors
        ):
            raise ValueError("a finite-reduction batch has inconsistent divisor rows")
        if any(
            target.model is not reduction.target
            for reduction, row in zip(reductions, target_divisors, strict=True)
            for target in row
        ):
            raise ValueError("a reduced divisor belongs to another target model")
        characteristics = tuple(reduction.prime for reduction in reductions)
        if len(characteristics) != len(set(characteristics)):
            raise ValueError("a finite-reduction batch repeats a characteristic")
        return result

    def reduced_divisor(
        self,
        reduction: AlgebraicFunctionFieldReduction,
        divisor: AlgebraicDivisor,
    ) -> AlgebraicDivisor:
        """Return the cached target divisor for one prepared pair."""
        try:
            reduction_index = next(
                index
                for index, candidate in enumerate(self.reductions)
                if candidate is reduction
            )
            divisor_index = self.source_divisors.index(divisor)
        except (StopIteration, ValueError) as exc:
            raise ValueError(
                "the reduction batch was not prepared for this divisor"
            ) from exc
        return self.target_divisors[reduction_index][divisor_index]


def prepare_algebraic_torsion_reductions(
    model: AlgebraicFunctionField,
    *,
    primes: tuple[int, ...] = _DEFAULT_REDUCTION_PRIMES,
    count: int = 2,
    divisors: tuple[AlgebraicDivisor, ...] = (),
) -> AlgebraicFiniteReductionBatch:
    """Build reductions valid for every supplied divisor and cache their images."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("finite-reduction preparation needs a function field")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("the requested reduction count must be positive")
    if any(
        not isinstance(divisor, AlgebraicDivisor) or divisor.model is not model
        for divisor in divisors
    ):
        raise ValueError("prepared divisors must belong to the source model")
    unique_divisors: list[AlgebraicDivisor] = []
    for divisor in divisors:
        if divisor not in unique_divisors:
            unique_divisors.append(divisor)
    source_divisors = tuple(unique_divisors)
    _parent, constants, _variable = _base_polynomial_ring(model.extension)

    reductions: list[AlgebraicFunctionFieldReduction] = []
    target_divisors: list[tuple[AlgebraicDivisor, ...]] = []
    failures: list[str] = []
    for prime in dict.fromkeys(primes):
        if any(reduction.prime == prime for reduction in reductions):
            continue
        try:
            constant_reductions = constant_field_prime_reductions(constants, prime)
        except (NotImplementedError, TypeError, ValueError) as exc:
            failures.append(f"{prime}: {exc}")
            continue

        selected: AlgebraicFunctionFieldReduction | None = None
        selected_divisors: tuple[AlgebraicDivisor, ...] | None = None
        candidate_failures: list[str] = []
        for constant_reduction in constant_reductions:
            try:
                selected = reduce_algebraic_function_field(
                    model,
                    constant_reduction,
                )
                selected_divisors = tuple(
                    selected.reduce_divisor(divisor) for divisor in source_divisors
                )
            except (
                BadAlgebraicReduction,
                NotImplementedError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                candidate_failures.append(str(exc))
                selected = None
                selected_divisors = None
                continue
            break
        if selected is None:
            detail = "; ".join(candidate_failures) or "no residue prime was available"
            failures.append(f"{prime}: {detail}")
            continue
        reductions.append(selected)
        if selected_divisors is None:
            raise ValueError("a selected reduction omitted its divisor images")
        target_divisors.append(selected_divisors)
        if len(reductions) == count:
            break

    return AlgebraicFiniteReductionBatch._create(
        model,
        source_divisors,
        tuple(reductions),
        tuple(target_divisors),
        tuple(failures),
    )


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
    primes: tuple[int, ...] | None = None,
    reduction_batch: AlgebraicFiniteReductionBatch | None = None,
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

    if reduction_batch is not None:
        if primes is not None:
            raise ValueError("pass either primes or a prepared reduction batch")
        if reduction_batch.model is not model:
            raise ValueError("the prepared reductions belong to another model")
        if divisor not in reduction_batch.source_divisors:
            raise ValueError("the prepared reductions do not contain this divisor")
        batch = reduction_batch
    else:
        batch = prepare_algebraic_torsion_reductions(
            model,
            primes=_DEFAULT_REDUCTION_PRIMES if primes is None else primes,
            divisors=(divisor,),
        )

    finite_orders: list[FiniteDivisorClassOrder] = []
    failures = list(batch.failures)
    for reduction in batch.reductions:
        try:
            finite_orders.append(
                _finite_divisor_class_order_from_reduction(
                    reduction,
                    divisor,
                    batch.reduced_divisor(reduction, divisor),
                )
            )
        except (
            BadAlgebraicReduction,
            NotImplementedError,
            TypeError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            failures.append(f"{reduction.prime}: {exc}")
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
    "AlgebraicFiniteReductionBatch",
    "FiniteDivisorClassOrder",
    "decide_algebraic_divisor_torsion",
    "finite_divisor_class_order",
    "prepare_algebraic_torsion_reductions",
]
