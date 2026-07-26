"""Result models for independent RDE matrix oracles."""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.polys.polytools import Poly

from sympy_risch._typing import FieldElement


def _validate_matrix_ranks(
    *,
    has_solution: bool,
    coefficient_rank: int,
    augmented_rank: int,
    label: str,
) -> None:
    """Validate the exact rank certificate for a complete bounded search."""
    for value in (coefficient_rank, augmented_rank):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} ranks must be nonnegative integers")
    if augmented_rank < coefficient_rank:
        raise ValueError(f"{label} augmented rank cannot be smaller")
    if has_solution and augmented_rank != coefficient_rank:
        raise ValueError(f"{label} solved system must have equal ranks")
    if not has_solution and augmented_rank <= coefficient_rank:
        raise ValueError(
            f"{label} unsolved system needs augmented rank greater than coefficient rank"
        )


@dataclass(frozen=True, slots=True)
class BoundedRationalRDEMatrixResult:
    """Exact matrix result for a caller-supplied rational ansatz."""

    denominator: Poly
    numerator_degree_bound: int
    solution: Expr | None
    coefficient_rank: int
    augmented_rank: int

    def __post_init__(self) -> None:
        """Validate the bounded exact-linear-algebra outcome."""
        _validate_matrix_ranks(
            has_solution=self.solution is not None,
            coefficient_rank=self.coefficient_rank,
            augmented_rank=self.augmented_rank,
            label="bounded rational RDE",
        )

    @property
    def has_solution(self) -> bool:
        """Whether the bounded rational ansatz contains a solution."""
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class BoundedPrimitiveRDEMatrixResult:
    """Exact matrix result for a caller-supplied primitive-field ansatz."""

    lower_denominator: Poly
    lower_numerator_degree_bound: int
    top_degree_bound: int
    solution: FieldElement | None
    coefficient_rank: int
    augmented_rank: int
    top_denominator: Poly | None = None

    def __post_init__(self) -> None:
        """Validate the bounded primitive matrix outcome."""
        _validate_matrix_ranks(
            has_solution=self.solution is not None,
            coefficient_rank=self.coefficient_rank,
            augmented_rank=self.augmented_rank,
            label="bounded primitive RDE",
        )

    @property
    def has_solution(self) -> bool:
        """Whether the bounded two-variable ansatz contains a solution."""
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class BoundedPrimitiveCoupledRDEMatrixResult:
    """Direct real block-matrix oracle over one primitive level."""

    lower_denominator: Poly
    lower_numerator_degree_bound: int
    top_degree_bound: int
    real_solution: FieldElement | None
    imaginary_solution: FieldElement | None
    coefficient_rank: int
    augmented_rank: int
    top_denominator: Poly

    def __post_init__(self) -> None:
        """Validate the bounded primitive coupled matrix outcome."""
        if (self.real_solution is None) != (self.imaginary_solution is None):
            raise ValueError(
                "a bounded primitive coupled RDE needs both solution coordinates"
            )
        _validate_matrix_ranks(
            has_solution=self.real_solution is not None,
            coefficient_rank=self.coefficient_rank,
            augmented_rank=self.augmented_rank,
            label="bounded primitive coupled RDE",
        )

    @property
    def has_solution(self) -> bool:
        """Whether the bounded real-pair ansatz contains a solution."""
        return self.real_solution is not None and self.imaginary_solution is not None


@dataclass(frozen=True, slots=True)
class BoundedHyperexponentialLaurentRDEMatrixResult:
    """Exact matrix oracle for a bounded hyperexponential Laurent ansatz."""

    lower_denominator: Poly
    lower_numerator_degree_bound: int
    minimum_power: int
    maximum_power: int
    solution: FieldElement | None
    coefficient_rank: int
    augmented_rank: int
    top_denominator: Poly | None = None

    def __post_init__(self) -> None:
        """Validate the bounded Laurent matrix outcome."""
        _validate_matrix_ranks(
            has_solution=self.solution is not None,
            coefficient_rank=self.coefficient_rank,
            augmented_rank=self.augmented_rank,
            label="bounded Laurent RDE",
        )

    @property
    def has_solution(self) -> bool:
        """Whether the bounded rectangular Laurent ansatz has a solution."""
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class BoundedCoupledRDEMatrixResult:
    """Exact block-matrix result for a caller-supplied coupled ansatz."""

    denominator: Poly
    numerator_degree_bound: int
    real_solution: Expr | None
    imaginary_solution: Expr | None
    coefficient_rank: int
    augmented_rank: int

    def __post_init__(self) -> None:
        """Validate the bounded real block-matrix outcome."""
        if (self.real_solution is None) != (self.imaginary_solution is None):
            raise ValueError(
                "a bounded coupled RDE needs both solution coordinates or neither"
            )
        _validate_matrix_ranks(
            has_solution=self.real_solution is not None,
            coefficient_rank=self.coefficient_rank,
            augmented_rank=self.augmented_rank,
            label="bounded coupled RDE",
        )

    @property
    def has_solution(self) -> bool:
        """Whether the bounded real-pair ansatz contains a solution."""
        return self.real_solution is not None and self.imaginary_solution is not None
