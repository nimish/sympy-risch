r"""Bounded exact searches for principal divisors on algebraic curves.

The global algebraic Risch logarithmic step needs a function whose principal
divisor realizes an integral residue divisor. A scalar shift of the integral
order followed by exact valuation constraints gives a sound bounded
semialgorithm: search a finite slice of ``L(-D)``. When ``deg(D) = 0``, any
nonzero element of that space has divisor exactly ``D``. Exhausting a supplied
finite slice is not a non-principality proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.polyerrors import HeuristicGCDFailed
from sympy.polys.polytools import Poly

from ..algebraic._field import (
    _base_polynomial_ring,
    _embed_parent,
    _extension_element,
    _quotient_extension_elements,
)
from ..algebraic.constant_analysis import (
    AlgebraicConstantFieldAnalysis,
    AlgebraicConstantFieldStatus,
)
from .constraints import (
    AlgebraicValuationKernelStatus,
    solve_algebraic_valuation_constraints,
)
from .finite_places import FiniteAlgebraicPlace
from .function_field import AlgebraicDivisor, AlgebraicFunctionField


class AlgebraicPrincipalDivisorStatus(Enum):
    """Outcome of a bounded principal-divisor search."""

    PRINCIPAL = "principal"
    NONPRINCIPAL = "nonprincipal"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicPrincipalDivisorSearch:
    r"""A verified principal witness or an honest bounded-search outcome.

    For a degree-zero divisor ``D``, the search enumerates a finite subspace
    of ``L(-D)``. A nonzero witness ``f`` there satisfies
    ``div(f) - D >= 0``. Both sides have degree zero, so exact divisor
    verification proves ``div(f) = D``. An empty finite slice is only
    :attr:`INCOMPLETE`, since a larger slice may contain a witness.
    """

    model: AlgebraicFunctionField
    divisor: AlgebraicDivisor
    constant_field_analysis: AlgebraicConstantFieldAnalysis
    max_x_degree: int
    status: AlgebraicPrincipalDivisorStatus
    witness: ExtensionElement | None
    searched_dimension: int | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlgebraicFunctionField):
            raise TypeError("a principal-divisor search needs a function field")
        if not isinstance(self.divisor, AlgebraicDivisor):
            raise TypeError("a principal-divisor search needs an algebraic divisor")
        if self.divisor.model is not self.model:
            raise ValueError("the divisor belongs to another function field")
        if not isinstance(
            self.constant_field_analysis,
            AlgebraicConstantFieldAnalysis,
        ):
            raise TypeError("a principal-divisor search needs constant analysis")
        if self.constant_field_analysis.extension is not self.model.extension:
            raise ValueError("the constant analysis belongs to another field")
        if not isinstance(self.max_x_degree, int) or self.max_x_degree < 0:
            raise ValueError("the finite lattice degree bound must be nonnegative")
        if not isinstance(self.status, AlgebraicPrincipalDivisorStatus):
            raise TypeError("the principal-divisor status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("a principal-divisor search needs a detail")

        witness = self.witness
        if witness is not None:
            witness = _extension_element(self.model.extension, witness)
            object.__setattr__(self, "witness", witness)

        if self.status is AlgebraicPrincipalDivisorStatus.UNSUPPORTED:
            if witness is not None or self.searched_dimension is not None:
                raise ValueError("an unsupported search cannot carry partial data")
            return

        if self.status is AlgebraicPrincipalDivisorStatus.NONPRINCIPAL:
            if self.divisor.degree == 0:
                raise ValueError(
                    "a bounded search cannot disprove a degree-zero divisor"
                )
            if witness is not None or self.searched_dimension is not None:
                raise ValueError("a nonprincipal result cannot carry search data")
            return

        if (
            self.constant_field_analysis.status
            is not AlgebraicConstantFieldStatus.PRESERVED
        ):
            raise ValueError("a bounded search needs a preserved constant field")
        if self.divisor.degree != 0:
            raise ValueError("a bounded search requires a degree-zero divisor")

        if self.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE:
            if witness is not None or self.searched_dimension != 0:
                raise ValueError("an incomplete search needs an empty finite kernel")
            return

        if self.status is not AlgebraicPrincipalDivisorStatus.PRINCIPAL:
            raise ValueError("a principal-divisor search has an unknown status")
        if witness is None or witness == self.model.extension.field.zero:
            raise ValueError("a principal result needs a nonzero witness")
        if not isinstance(self.searched_dimension, int) or self.searched_dimension < 1:
            raise ValueError("a principal result needs a nonempty finite kernel")
        if self.model.principal_divisor(witness) != self.divisor:
            raise ValueError("the principal witness has another divisor")

    @property
    def principal(self) -> bool:
        """Whether the search found and verified a principal witness."""
        return self.status is AlgebraicPrincipalDivisorStatus.PRINCIPAL

    @property
    def complete_negative(self) -> bool:
        """Whether non-principality follows from the degree obstruction."""
        return self.status is AlgebraicPrincipalDivisorStatus.NONPRINCIPAL

    @property
    def incomplete(self) -> bool:
        """Whether a larger finite lattice slice could still find a witness."""
        return self.status is AlgebraicPrincipalDivisorStatus.INCOMPLETE


def _principal_candidate_lattice_slice(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    max_x_degree: int,
) -> tuple[ExtensionElement, ...]:
    r"""Build a bounded rational-denominator candidate space for ``div(f)>=D``.

    For every finite base prime ``p``, set

    ``s_p = min_{P | p} floor(D(P) / e(P | p))``.

    The rational function ``S = product(p**s_p)`` has finite divisor bounded
    by ``D``. Therefore every ``f`` with ``div(f) >= D`` has ``f / S`` in the
    maximal finite order, so the exact finite lattice is generated by
    ``x**k * b * S`` for an integral order basis ``b``. The valuation kernel
    below still imposes every finite and infinite inequality exactly, and a
    returned witness is independently checked by its complete principal
    divisor. ``max_x_degree`` bounds the remaining polynomial degree.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("a principal candidate slice needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("a principal candidate slice needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the finite lattice degree bound must be nonnegative")

    extension = model.extension
    _parent, constants, variable = _base_polynomial_ring(extension)
    base_primes = tuple(
        dict.fromkeys(
            place.base_prime
            for place, _coefficient in divisor.terms
            if isinstance(place, FiniteAlgebraicPlace)
        )
    )
    numerator = Poly(1, variable, domain=constants)
    denominator = Poly(1, variable, domain=constants)
    for base_prime in base_primes:
        exponent = min(
            divisor.coefficient(place) // place.ramification_index
            for place in model.finite_places(base_prime)
        )
        if exponent > 0:
            numerator *= base_prime**exponent
        elif exponent < 0:
            denominator *= base_prime ** (-exponent)
    scale = _quotient_extension_elements(
        extension,
        _embed_parent(extension, numerator),
        _embed_parent(extension, denominator),
    )
    base_variable = _embed_parent(
        extension,
        Poly(variable, variable, domain=constants),
    )
    return tuple(
        base_variable**power * basis_element * scale
        for power in range(max_x_degree + 1)
        for basis_element in model.order.basis.elements
    )


def search_algebraic_principal_divisor(
    model: AlgebraicFunctionField,
    divisor: AlgebraicDivisor,
    constant_field_analysis: AlgebraicConstantFieldAnalysis,
    *,
    max_x_degree: int,
) -> AlgebraicPrincipalDivisorSearch:
    r"""Search a finite exact candidate slice for ``f`` with ``div(f) = divisor``.

    A nonzero-degree divisor is immediately nonprincipal. For degree zero,
    this is a semialgorithm: it searches a finite rational-denominator
    candidate space, imposes all valuations exactly, and verifies any witness
    using the complete principal-divisor computation. A failed bound is
    therefore returned as :attr:`AlgebraicPrincipalDivisorStatus.INCOMPLETE`,
    never as a degree-zero non-principality claim.
    """
    if not isinstance(model, AlgebraicFunctionField):
        raise TypeError("principal-divisor search needs a function field")
    if not isinstance(divisor, AlgebraicDivisor):
        raise TypeError("principal-divisor search needs an algebraic divisor")
    if divisor.model is not model:
        raise ValueError("the divisor belongs to another function field")
    if not isinstance(constant_field_analysis, AlgebraicConstantFieldAnalysis):
        raise TypeError("principal-divisor search needs constant analysis")
    if constant_field_analysis.extension is not model.extension:
        raise ValueError("the constant analysis belongs to another field")
    if not isinstance(max_x_degree, int) or max_x_degree < 0:
        raise ValueError("the finite lattice degree bound must be nonnegative")

    if divisor.degree != 0:
        return AlgebraicPrincipalDivisorSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicPrincipalDivisorStatus.NONPRINCIPAL,
            None,
            None,
            "a principal divisor must have degree zero",
        )
    if constant_field_analysis.status is not AlgebraicConstantFieldStatus.PRESERVED:
        return AlgebraicPrincipalDivisorSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicPrincipalDivisorStatus.UNSUPPORTED,
            None,
            None,
            "principal-divisor search requires a certified preserved constant "
            f"field; the current presentation is "
            f"{constant_field_analysis.status.value}",
        )

    try:
        ambient_basis = _principal_candidate_lattice_slice(
            model,
            divisor,
            max_x_degree,
        )
        kernel = solve_algebraic_valuation_constraints(
            model,
            -divisor,
            constant_field_analysis,
            ambient_basis,
        )
    except (HeuristicGCDFailed, NotImplementedError) as exc:
        return AlgebraicPrincipalDivisorSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicPrincipalDivisorStatus.UNSUPPORTED,
            None,
            None,
            f"principal-divisor search requires unavailable exact local data: {exc}",
        )
    if kernel.status is not AlgebraicValuationKernelStatus.CERTIFIED:
        return AlgebraicPrincipalDivisorSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicPrincipalDivisorStatus.UNSUPPORTED,
            None,
            None,
            "principal-divisor search could not construct its exact valuation "
            f"kernel: {kernel.detail}",
        )
    if kernel.dimension == 0:
        return AlgebraicPrincipalDivisorSearch(
            model,
            divisor,
            constant_field_analysis,
            max_x_degree,
            AlgebraicPrincipalDivisorStatus.INCOMPLETE,
            None,
            0,
            "the supplied finite lattice slice contains no principal witness; "
            "a larger slice may still succeed",
        )

    return AlgebraicPrincipalDivisorSearch(
        model,
        divisor,
        constant_field_analysis,
        max_x_degree,
        AlgebraicPrincipalDivisorStatus.PRINCIPAL,
        kernel.basis[0],
        kernel.dimension,
        "a nonzero degree-zero valuation-kernel element was verified to have "
        "the requested principal divisor",
    )
