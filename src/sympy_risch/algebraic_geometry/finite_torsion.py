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

from dataclasses import dataclass, field
from enum import Enum
from math import isqrt, lcm

from sympy.polys.agca.extensions import ExtensionElement

from ..algebraic._field import _base_polynomial_ring
from .constant_fields import (
    FunctionFieldConstantCertificate,
    certify_function_field_constants,
)
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


def _validate_constant_field(
    model: AlgebraicFunctionField,
    certificate: FunctionFieldConstantCertificate,
) -> None:
    if not isinstance(certificate, FunctionFieldConstantCertificate):
        raise TypeError("finite torsion requires full-constant-field evidence")
    if certificate.model is not model:
        raise ValueError("the constant-field evidence belongs to another model")


def _weil_jacobian_upper_bound(field_size: int, genus: int) -> int:
    """Return a safe integral form of ``(sqrt(q) + 1) ** (2*g)``."""
    root = isqrt(field_size)
    if root * root < field_size:
        root += 1
    return (root + 1) ** (2 * genus)


def _source_divisor_key(divisor: AlgebraicDivisor) -> tuple[tuple[int, int], ...]:
    """Hash exact model-owned place identity without expensive ideal equality."""
    return tuple(
        sorted((id(place), coefficient) for place, coefficient in divisor.terms)
    )


@dataclass(frozen=True, slots=True, init=False)
class FiniteDivisorClassOrder:
    """The exact order of a reduced class in one finite Jacobian."""

    reduction: AlgebraicFunctionFieldReduction
    source_constant_field: FunctionFieldConstantCertificate
    target_constant_field: FunctionFieldConstantCertificate
    source_divisor: AlgebraicDivisor
    target_divisor: AlgebraicDivisor
    weil_bound: int
    search: AlgebraicTorsionSearch

    def __init__(self) -> None:
        raise TypeError("finite-order certificates must be created by the solver")

    @classmethod
    def _create(
        cls,
        reduction: AlgebraicFunctionFieldReduction,
        source_constant_field: FunctionFieldConstantCertificate,
        target_constant_field: FunctionFieldConstantCertificate,
        source_divisor: AlgebraicDivisor,
        target_divisor: AlgebraicDivisor,
        weil_bound: int,
        search: AlgebraicTorsionSearch,
    ) -> FiniteDivisorClassOrder:
        result = object.__new__(cls)
        object.__setattr__(result, "reduction", reduction)
        object.__setattr__(result, "source_constant_field", source_constant_field)
        object.__setattr__(result, "target_constant_field", target_constant_field)
        object.__setattr__(result, "source_divisor", source_divisor)
        object.__setattr__(result, "target_divisor", target_divisor)
        object.__setattr__(result, "weil_bound", weil_bound)
        object.__setattr__(result, "search", search)
        result._validate()
        return result

    def _validate(self) -> None:
        if self.source_constant_field.model is not self.reduction.source:
            raise ValueError("the source constant proof belongs to another model")
        if self.target_constant_field.model is not self.reduction.target:
            raise ValueError("the target constant proof belongs to another model")
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
            or self.weil_bound <= 0
            or self.search.max_order != self.weil_bound
            or self.search.anchor.degree != 1
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
    *,
    constant_field: FunctionFieldConstantCertificate,
) -> FiniteDivisorClassOrder:
    """Compute an exact finite order from a full-constant-field source."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("finite order computation needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("finite order computation needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if divisor.degree != 0:
        raise ValueError("finite Jacobian order requires a degree-zero divisor")
    _validate_constant_field(model, constant_field)
    if isinstance(prime, AlgebraicFunctionFieldReduction):
        reduction = prime
        if reduction.source is not model:
            raise ValueError("the finite reduction belongs to another model")
    else:
        reduction = reduce_algebraic_function_field(model, prime)
    target_divisor = reduction.reduce_divisor(divisor)
    context = _prepare_finite_order_context(reduction)
    return _finite_divisor_class_order_from_reduction(
        reduction,
        constant_field,
        divisor,
        target_divisor,
        context,
    )


@dataclass(frozen=True, slots=True)
class _FiniteOrderContext:
    """Reduction-wide theorem data reused across all target divisors."""

    anchor: AlgebraicDivisor
    target_constant_field: FunctionFieldConstantCertificate
    weil_bound: int


def _prepare_finite_order_context(
    reduction: AlgebraicFunctionFieldReduction,
) -> _FiniteOrderContext:
    anchor = _validate_anchor(reduction.target, None)
    if anchor.degree != 1:
        raise NotImplementedError(
            "automatic finite-Jacobian order computation requires a degree-one "
            "reduction anchor; use the explicitly bounded divisor-class search"
        )
    target_constant_field = certify_function_field_constants(
        reduction.target,
        anchor.support[0],
    )
    return _FiniteOrderContext(
        anchor,
        target_constant_field,
        _weil_jacobian_upper_bound(reduction.field_size, reduction.target.genus),
    )


def _finite_divisor_class_order_from_reduction(
    reduction: AlgebraicFunctionFieldReduction,
    source_constant_field: FunctionFieldConstantCertificate,
    source_divisor: AlgebraicDivisor,
    target_divisor: AlgebraicDivisor,
    context: _FiniteOrderContext,
) -> FiniteDivisorClassOrder:
    """Compute a finite order from an already reduced divisor."""
    _validate_constant_field(reduction.source, source_constant_field)
    search = _search_algebraic_divisor_torsion_bsgs(
        reduction.target,
        target_divisor,
        max_order=context.weil_bound,
        anchor=context.anchor,
    )
    if search.status is not AlgebraicTorsionStatus.TORSION:
        raise NotImplementedError(
            "the finite class order was not found within the Weil bound: "
            + search.detail
        )
    return FiniteDivisorClassOrder._create(
        reduction,
        source_constant_field,
        context.target_constant_field,
        source_divisor,
        target_divisor,
        context.weil_bound,
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
    constant_field: FunctionFieldConstantCertificate
    source_divisors: tuple[AlgebraicDivisor, ...]
    reductions: tuple[AlgebraicFunctionFieldReduction, ...]
    finite_orders: tuple[tuple[FiniteDivisorClassOrder | None, ...], ...]
    failures: tuple[str, ...]
    _divisor_indices: dict[tuple[tuple[int, int], ...], int] = field(
        repr=False,
        compare=False,
    )

    def __init__(self) -> None:
        raise TypeError("finite-reduction batches must be created by the factory")

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        constant_field: FunctionFieldConstantCertificate,
        source_divisors: tuple[AlgebraicDivisor, ...],
        reductions: tuple[AlgebraicFunctionFieldReduction, ...],
        finite_orders: tuple[tuple[FiniteDivisorClassOrder | None, ...], ...],
        failures: tuple[str, ...],
    ) -> AlgebraicFiniteReductionBatch:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "constant_field", constant_field)
        object.__setattr__(result, "source_divisors", source_divisors)
        object.__setattr__(result, "reductions", reductions)
        object.__setattr__(result, "finite_orders", finite_orders)
        object.__setattr__(result, "failures", failures)
        divisor_indices = {
            _source_divisor_key(divisor): index
            for index, divisor in enumerate(source_divisors)
        }
        object.__setattr__(result, "_divisor_indices", divisor_indices)
        _validate_constant_field(model, constant_field)
        if any(divisor.model is not model for divisor in source_divisors):
            raise ValueError("a finite-reduction batch contains a foreign divisor")
        if len(divisor_indices) != len(source_divisors):
            raise ValueError("a finite-reduction batch repeats a divisor")
        if any(reduction.source is not model for reduction in reductions):
            raise ValueError("a finite-reduction batch contains a foreign model")
        if len(finite_orders) != len(reductions) or any(
            len(row) != len(source_divisors) for row in finite_orders
        ):
            raise ValueError("a finite-reduction batch has inconsistent divisor rows")
        if any(
            order is not None
            and (order.reduction is not reduction or order.source_divisor != divisor)
            for reduction, row in zip(reductions, finite_orders, strict=True)
            for divisor, order in zip(source_divisors, row, strict=True)
        ):
            raise ValueError("a finite order belongs to another prepared pair")
        return result

    def orders_for(
        self, divisor: AlgebraicDivisor
    ) -> tuple[FiniteDivisorClassOrder, ...]:
        """Return cached orders at distinct residue characteristics."""
        try:
            divisor_index = self._divisor_indices[_source_divisor_key(divisor)]
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                "the reduction batch was not prepared for this divisor"
            ) from exc
        result: list[FiniteDivisorClassOrder] = []
        characteristics: set[int] = set()
        for row in self.finite_orders:
            order = row[divisor_index]
            if order is not None and order.prime not in characteristics:
                result.append(order)
                characteristics.add(order.prime)
        return tuple(result)


def prepare_algebraic_torsion_reductions(
    model: AlgebraicFunctionField,
    constant_field: FunctionFieldConstantCertificate,
    *,
    primes: tuple[int, ...] = _DEFAULT_REDUCTION_PRIMES,
    divisors: tuple[AlgebraicDivisor, ...] = (),
) -> AlgebraicFiniteReductionBatch:
    """Cache two exact finite orders per divisor, allowing different primes."""
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("finite-reduction preparation needs a function field")
    _validate_constant_field(model, constant_field)
    unique_divisors: dict[tuple[tuple[int, int], ...], AlgebraicDivisor] = {}
    for divisor in divisors:
        if not isinstance(divisor, AlgebraicDivisor):
            raise TypeError("finite-order preparation needs algebraic divisors")
        if divisor.model is not model:
            raise ValueError("prepared divisors must belong to the source model")
        if divisor.degree != 0:
            raise ValueError("prepared divisors must have degree zero")
        unique_divisors.setdefault(_source_divisor_key(divisor), divisor)
    source_divisors = tuple(unique_divisors.values())
    if not source_divisors:
        raise ValueError("finite-order preparation needs at least one divisor")
    _parent, constants, _variable = _base_polynomial_ring(model.extension)

    reductions: list[AlgebraicFunctionFieldReduction] = []
    finite_orders: list[tuple[FiniteDivisorClassOrder | None, ...]] = []
    successful_characteristics = [set() for _divisor in source_divisors]
    failures: list[str] = []
    for prime in dict.fromkeys(primes):
        try:
            constant_reductions = constant_field_prime_reductions(constants, prime)
        except (NotImplementedError, TypeError, ValueError) as exc:
            failures.append(f"{prime}: {exc}")
            continue

        candidate_failures: list[str] = []
        for constant_reduction in constant_reductions:
            try:
                reduction = reduce_algebraic_function_field(
                    model,
                    constant_reduction,
                )
            except (
                BadAlgebraicReduction,
                NotImplementedError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ) as exc:
                candidate_failures.append(str(exc))
                continue
            try:
                context = _prepare_finite_order_context(reduction)
            except (NotImplementedError, TypeError, ValueError) as exc:
                candidate_failures.append(str(exc))
                continue
            row: list[FiniteDivisorClassOrder | None] = []
            for index, divisor in enumerate(source_divisors):
                if prime in successful_characteristics[index]:
                    row.append(None)
                    continue
                try:
                    target_divisor = reduction.reduce_divisor(divisor)
                    order = _finite_divisor_class_order_from_reduction(
                        reduction,
                        constant_field,
                        divisor,
                        target_divisor,
                        context,
                    )
                except (
                    BadAlgebraicReduction,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                ) as exc:
                    candidate_failures.append(f"divisor {index}: {exc}")
                    row.append(None)
                    continue
                row.append(order)
                successful_characteristics[index].add(prime)
            if any(order is not None for order in row):
                reductions.append(reduction)
                finite_orders.append(tuple(row))
        if not any(reduction.prime == prime for reduction in reductions):
            detail = "; ".join(candidate_failures) or "no residue prime was available"
            failures.append(f"{prime}: {detail}")
        if all(
            len(characteristics) >= 2 for characteristics in successful_characteristics
        ):
            break

    return AlgebraicFiniteReductionBatch._create(
        model,
        constant_field,
        source_divisors,
        tuple(reductions),
        tuple(finite_orders),
        tuple(failures),
    )


@dataclass(frozen=True, slots=True, init=False)
class AlgebraicDivisorTorsionDecision:
    """A verified torsion relation, nontorsion proof, or exact boundary."""

    model: AlgebraicFunctionField
    constant_field: FunctionFieldConstantCertificate
    divisor: AlgebraicDivisor
    status: AlgebraicDivisorTorsionStatus
    order: int | None
    witness: ExtensionElement | None
    finite_orders: tuple[FiniteDivisorClassOrder, ...]
    characteristic_zero_search: AlgebraicTorsionSearch | None
    detail: str

    def __init__(self) -> None:
        raise TypeError("torsion decisions must be created by the solver")

    @classmethod
    def _create(
        cls,
        model: AlgebraicFunctionField,
        constant_field: FunctionFieldConstantCertificate,
        divisor: AlgebraicDivisor,
        status: AlgebraicDivisorTorsionStatus,
        order: int | None,
        witness: ExtensionElement | None,
        finite_orders: tuple[FiniteDivisorClassOrder, ...],
        characteristic_zero_search: AlgebraicTorsionSearch | None,
        detail: str,
    ) -> AlgebraicDivisorTorsionDecision:
        result = object.__new__(cls)
        object.__setattr__(result, "model", model)
        object.__setattr__(result, "constant_field", constant_field)
        object.__setattr__(result, "divisor", divisor)
        object.__setattr__(result, "status", status)
        object.__setattr__(result, "order", order)
        object.__setattr__(result, "witness", witness)
        object.__setattr__(result, "finite_orders", finite_orders)
        object.__setattr__(
            result,
            "characteristic_zero_search",
            characteristic_zero_search,
        )
        object.__setattr__(result, "detail", detail)
        result._validate()
        return result

    def _validate(self) -> None:
        _validate_constant_field(self.model, self.constant_field)
        if self.divisor.model is not self.model or self.divisor.degree != 0:
            raise ValueError(
                "a torsion decision needs a model-owned degree-zero divisor"
            )
        if len(self.finite_orders) > 2:
            raise ValueError("a torsion decision uses at most two finite reductions")
        if any(
            order.reduction.source is not self.model
            or order.source_constant_field is not self.constant_field
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
        if not isinstance(self.status, AlgebraicDivisorTorsionStatus):
            raise TypeError("a torsion decision has an unknown status")

        if len(self.finite_orders) == 2:
            left, right = self.finite_orders
            if left.prime == right.prime:
                raise ValueError("a two-prime decision repeated a characteristic")
            candidate = lcm(left.order, right.order)
            compatible = _is_power_of(
                candidate // left.order,
                left.prime,
            ) and _is_power_of(candidate // right.order, right.prime)
        else:
            candidate = None
            compatible = False

        if self.status is AlgebraicDivisorTorsionStatus.TORSION:
            if (
                len(self.finite_orders) != 2
                or not compatible
                or self.order is None
                or self.order != candidate
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
            if search is None:
                if compatible:
                    raise ValueError(
                        "compatible finite orders alone do not prove nontorsion"
                    )
            elif (
                not compatible
                or search.status is not AlgebraicTorsionStatus.INCONCLUSIVE
                or search.max_order != candidate
            ):
                raise ValueError("the nontorsion replay does not exhaust its candidate")
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
    constant_field: FunctionFieldConstantCertificate | None = None,
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
        if (
            constant_field is not None
            and constant_field is not reduction_batch.constant_field
        ):
            raise ValueError("the explicit and prepared constant proofs differ")
        batch = reduction_batch
    else:
        if constant_field is None:
            raise TypeError("torsion decision requires full-constant-field evidence")
        batch = prepare_algebraic_torsion_reductions(
            model,
            constant_field,
            primes=_DEFAULT_REDUCTION_PRIMES if primes is None else primes,
            divisors=(divisor,),
        )

    finite_orders = list(batch.orders_for(divisor)[:2])
    failures = list(batch.failures)

    if len(finite_orders) < 2:
        detail = "fewer than two supported good reductions were available"
        if failures:
            detail += ": " + "; ".join(failures)
        return AlgebraicDivisorTorsionDecision._create(
            model,
            batch.constant_field,
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
        return AlgebraicDivisorTorsionDecision._create(
            model,
            batch.constant_field,
            divisor,
            AlgebraicDivisorTorsionStatus.NON_TORSION,
            None,
            None,
            tuple(finite_orders),
            None,
            "the two reduced orders cannot be reductions of one "
            "characteristic-zero torsion order",
        )

    anchor = _validate_anchor(model, None)
    if anchor.degree != 1:
        return AlgebraicDivisorTorsionDecision._create(
            model,
            batch.constant_field,
            divisor,
            AlgebraicDivisorTorsionStatus.UNSUPPORTED,
            None,
            None,
            tuple(finite_orders),
            None,
            "automatic characteristic-zero verification requires a degree-one "
            "anchor; use the explicitly bounded divisor-class search",
        )
    search = _search_algebraic_divisor_torsion_bsgs(
        model,
        divisor,
        max_order=max(1, candidate),
        anchor=anchor,
    )
    if search.status is AlgebraicTorsionStatus.TORSION:
        if search.order != candidate or search.witness is None:
            return AlgebraicDivisorTorsionDecision._create(
                model,
                batch.constant_field,
                divisor,
                AlgebraicDivisorTorsionStatus.UNSUPPORTED,
                None,
                None,
                tuple(finite_orders),
                search,
                "finite reduction and characteristic-zero orders disagree",
            )
        return AlgebraicDivisorTorsionDecision._create(
            model,
            batch.constant_field,
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
        return AlgebraicDivisorTorsionDecision._create(
            model,
            batch.constant_field,
            divisor,
            AlgebraicDivisorTorsionStatus.NON_TORSION,
            None,
            None,
            tuple(finite_orders),
            search,
            "the unique finite-prime candidate failed characteristic-zero replay",
        )
    return AlgebraicDivisorTorsionDecision._create(
        model,
        batch.constant_field,
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
