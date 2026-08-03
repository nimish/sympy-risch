r"""Immutable certificates returned by Risch differential equation solvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..tower import Derivation
from ._common import (
    _field_element_is_zero,
    _field_elements_equal,
    _validate_decided_outcome,
    _validate_failed_division,
    _validate_paired_outcome,
)

if TYPE_CHECKING:
    from ..dependency import ParametricLogarithmicDerivativeResult
    from ..exact import ExactDerivativeResult
    from ..gaussian import HypertangentGaussianNormalForm
    from ..monomial import RadicalLogarithmicDerivativeResult


class PolynomialRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution over an exact constant field."""

    DEGREE = "degree"
    NONZERO_REMAINDER = "nonzero_remainder"
    INCONSISTENT_SYSTEM = "inconsistent_system"


class RationalRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in the rational function field."""

    DENOMINATOR = "denominator"
    SPDE_DEGREE = "spde_degree"
    SPDE_DIVISIBILITY = "spde_divisibility"
    POLYNOMIAL = "polynomial"


class PrimitivePolynomialRDEObstruction(Enum):
    """Why a polynomial RDE has no solution in a primitive field."""

    DEGREE = "degree"
    NONZERO_REMAINDER = "nonzero_remainder"
    LOWER_FIELD = "lower_field"
    EXACT_DERIVATIVE = "exact_derivative"


class PrimitiveRDEObstruction(Enum):
    """Why a supported RDE has no solution in a primitive field."""

    DENOMINATOR = "denominator"
    SPDE_DEGREE = "spde_degree"
    SPDE_DIVISIBILITY = "spde_divisibility"
    POLYNOMIAL = "polynomial"
    EXACT_DERIVATIVE = "exact_derivative"


class HyperexponentialLaurentRDEObstruction(Enum):
    """Why a Laurent RDE fails at a hyperexponential level."""

    LOWER_FIELD = "lower_field"
    NONZERO_REMAINDER = "nonzero_remainder"


class HyperexponentialRDEObstruction(Enum):
    """Why a rational RDE fails at a hyperexponential level."""

    DENOMINATOR = "denominator"
    SPDE_DEGREE = "spde_degree"
    SPDE_DIVISIBILITY = "spde_divisibility"
    LAURENT = "laurent"


class HyperexponentialLaurentRDEDirection(Enum):
    """Direction used by a hyperexponential Laurent recurrence."""

    DIAGONAL = "diagonal"
    DESCENDING = "descending"
    ASCENDING = "ascending"


class CoupledRDEObstruction(Enum):
    r"""Why the real system ``Dp-lambda*q=u``, ``Dq+lambda*p=v`` fails."""

    DENOMINATOR = "denominator"
    DEGREE = "degree"
    LOWER_FIELD = "lower_field"
    NONZERO_REMAINDER = "nonzero_remainder"


class CoupledHypertangentPolynomialRDEMode(Enum):
    """Terminal polynomial branch used at a hypertangent level."""

    NON_CANCELLATION = "non_cancellation"
    CANCELLATION = "cancellation"


@dataclass(frozen=True, slots=True)
class PolynomialRischDifferentialEquationResult:
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``C[x]``.

    ``C`` is any supported exact rational constant field.

    A rational solution cannot have a finite pole: at a pole of ``y``,
    ``Dy`` has strictly higher order than ``a*y`` and cannot be cancelled by
    polynomial ``b``.  It is therefore enough to search for polynomial
    solutions.

    When ``a`` and ``b`` are nonzero, leading degrees force

    ``degree(y) = degree(b) - degree(a)``.

    ``degree_bound`` stores this forced degree.  The value ``-1`` is the
    degree certificate that no nonzero polynomial can solve a nonzero right
    hand side.

    The recurrence solver certifies a later failure with ``partial_solution``
    and ``residual`` satisfying

    ``b = D(partial_solution) + a*partial_solution + residual``,

    where the nonzero residual has degree smaller than ``a`` and therefore
    cannot be cancelled by another polynomial term.  The independent matrix
    solver instead records coefficient and augmented ranks.
    """

    a: Poly
    b: Poly
    degree_bound: int
    solution: Poly | None
    obstruction: PolynomialRDEObstruction | None
    coefficient_rank: int | None = None
    augmented_rank: int | None = None
    partial_solution: Poly | None = None
    residual: Poly | None = None

    def __post_init__(self) -> None:
        """Validate the complete polynomial decision certificate."""
        if not isinstance(self.a, Poly) or not isinstance(self.b, Poly):
            raise TypeError("polynomial RDE certificates require Poly inputs")
        if (
            len(self.a.gens) != 1
            or self.a.gens != self.b.gens
            or self.a.domain != self.b.domain
        ):
            raise ValueError(
                "polynomial RDE certificate inputs must share one polynomial domain"
            )
        domain = self.a.domain
        if not domain.is_Field or not domain.is_Exact or domain.characteristic() != 0:
            raise ValueError(
                "polynomial RDE certificates require an exact "
                "characteristic-zero constant field"
            )
        if (
            not isinstance(self.degree_bound, int)
            or isinstance(self.degree_bound, bool)
            or self.degree_bound < -1
        ):
            raise ValueError("a polynomial RDE degree bound must be an integer >= -1")
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            PolynomialRDEObstruction,
        ):
            raise TypeError("invalid polynomial RDE obstruction")

        if self.solution is not None:
            self._validate_polynomial(self.solution, role="solution")
            if self.obstruction is not None:
                raise ValueError(
                    "a polynomial RDE certificate cannot contain both a solution "
                    "and an obstruction"
                )
            if self.solution.diff() + self.a * self.solution != self.b:
                raise ValueError(
                    "the stored polynomial RDE solution does not reconstruct"
                )
        elif self.obstruction is None:
            raise ValueError(
                "a polynomial RDE certificate needs a solution or an obstruction"
            )

        if (self.partial_solution is None) != (self.residual is None):
            raise ValueError(
                "a polynomial RDE reduction needs both partial solution and residual"
            )
        if self.partial_solution is not None and self.residual is not None:
            self._validate_polynomial(
                self.partial_solution,
                role="partial solution",
            )
            self._validate_polynomial(self.residual, role="residual")
            if (
                self.partial_solution.diff()
                + self.a * self.partial_solution
                + self.residual
                != self.b
            ):
                raise ValueError(
                    "the polynomial RDE partial reduction does not reconstruct"
                )

        if (self.coefficient_rank is None) != (self.augmented_rank is None):
            raise ValueError(
                "a matrix certificate needs both coefficient and augmented ranks"
            )
        if self.coefficient_rank is not None and self.augmented_rank is not None:
            if (
                not isinstance(self.coefficient_rank, int)
                or isinstance(self.coefficient_rank, bool)
                or not isinstance(self.augmented_rank, int)
                or isinstance(self.augmented_rank, bool)
                or self.coefficient_rank < 0
                or self.augmented_rank < 0
            ):
                raise ValueError("matrix ranks must be nonnegative integers")
            if (
                self.obstruction is PolynomialRDEObstruction.INCONSISTENT_SYSTEM
                and self.augmented_rank <= self.coefficient_rank
            ):
                raise ValueError(
                    "an inconsistent-system obstruction needs augmented rank "
                    "greater than coefficient rank"
                )
            if self.solution is not None and (
                self.coefficient_rank != self.augmented_rank
            ):
                raise ValueError("a solved matrix system must have equal ranks")

        if (
            self.obstruction is PolynomialRDEObstruction.DEGREE
            and self.degree_bound != -1
        ):
            raise ValueError("a degree obstruction requires degree bound -1")
        if self.obstruction is PolynomialRDEObstruction.NONZERO_REMAINDER and (
            self.residual is None or self.residual.is_zero
        ):
            raise ValueError("a remainder obstruction requires a nonzero residual")
        if (
            self.obstruction is PolynomialRDEObstruction.INCONSISTENT_SYSTEM
            and self.coefficient_rank is None
        ):
            raise ValueError("an inconsistent-system obstruction needs matrix ranks")

    def _validate_polynomial(self, polynomial: Poly, *, role: str) -> None:
        if not isinstance(polynomial, Poly):
            raise TypeError(f"polynomial RDE {role} must be a Poly")
        if polynomial.gens != self.a.gens or polynomial.domain != self.a.domain:
            raise ValueError(
                f"polynomial RDE {role} must belong to the coefficient domain"
            )

    @property
    def has_solution(self) -> bool:
        """Whether the differential equation has a rational solution."""
        return self.solution is not None

    @property
    def solution_expression(self) -> Expr | None:
        """Return the canonical polynomial solution as a SymPy expression."""
        if self.solution is None:
            return None
        return self.solution.as_expr()

    @property
    def has_degree_obstruction(self) -> bool:
        """Whether leading degrees alone prove nonexistence."""
        return self.obstruction is PolynomialRDEObstruction.DEGREE

    @property
    def has_inconsistent_system_obstruction(self) -> bool:
        """Whether exact rank comparison proves the coefficient system fails."""
        return self.obstruction is PolynomialRDEObstruction.INCONSISTENT_SYSTEM

    @property
    def has_nonzero_remainder_obstruction(self) -> bool:
        """Whether leading-term reduction leaves an uncancellable remainder."""
        return self.obstruction is PolynomialRDEObstruction.NONZERO_REMAINDER


@dataclass(frozen=True, slots=True)
class RationalRischDifferentialEquationResult:
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``C(x)``.

    ``C`` is any supported exact rational constant field.

    ``normalization_factor`` records the weak normalization

    ``a_normal = a - D(normalization_factor)/normalization_factor``.

    When denominator reduction succeeds, every solution is represented as

    ``y = q/(normalization_factor*denominator_bound)``

    for a polynomial ``q`` of degree at most ``degree_bound``.  A successful
    SPDE reduction finishes with ``polynomial_result``.  On a denominator or
    SPDE divisibility failure, ``obstruction_polynomial`` and
    ``obstruction_remainder`` retain the failed exact division.
    """

    coefficient: Expr
    right_hand_side: Expr
    solution: Expr | None
    obstruction: RationalRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: Expr
    normalized_right_hand_side: Expr
    denominator_bound: Poly | None = None
    degree_bound: int | None = None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    def __post_init__(self) -> None:
        """Validate the scalar rational decision certificate."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            RationalRDEObstruction,
        ):
            raise TypeError("invalid rational RDE obstruction")
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="rational RDE certificate",
        )
        variable = cast(Symbol, self.normalization_factor.gens[0])
        solution_expression = (
            cancel(self.solution) if self.solution is not None else None
        )
        if solution_expression is not None and (
            cancel(
                solution_expression.diff(variable)
                + self.coefficient * solution_expression
                - self.right_hand_side
            )
            != 0
        ):
            raise ValueError("the stored rational RDE solution does not reconstruct")

        division_obstruction = self.obstruction in {
            RationalRDEObstruction.DENOMINATOR,
            RationalRDEObstruction.SPDE_DIVISIBILITY,
        }
        _validate_failed_division(
            self.obstruction_polynomial,
            self.obstruction_remainder,
            required=division_obstruction,
            label="rational RDE",
        )
        if self.obstruction is RationalRDEObstruction.POLYNOMIAL and (
            self.polynomial_result is None or self.polynomial_result.has_solution
        ):
            raise ValueError(
                "a polynomial obstruction needs an obstructed polynomial result"
            )
        if (
            self.solution is not None
            and self.polynomial_result is not None
            and not self.polynomial_result.has_solution
        ):
            raise ValueError(
                "a solved rational RDE cannot retain an obstructed polynomial result"
            )

    @property
    def has_solution(self) -> bool:
        """Whether the Risch differential equation has a rational solution."""
        return self.solution is not None

    @property
    def has_denominator_obstruction(self) -> bool:
        """Whether the normal-denominator theorem proves nonexistence."""
        return self.obstruction is RationalRDEObstruction.DENOMINATOR

    @property
    def has_spde_obstruction(self) -> bool:
        """Whether SPDE degree or divisibility proves nonexistence."""
        return self.obstruction in {
            RationalRDEObstruction.SPDE_DEGREE,
            RationalRDEObstruction.SPDE_DIVISIBILITY,
        }

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether the final polynomial RDE has no solution."""
        return self.obstruction is RationalRDEObstruction.POLYNOMIAL


@dataclass(frozen=True, slots=True)
class PrimitivePolynomialRischDifferentialEquationResult:
    r"""Certified polynomial RDE result at one primitive tower level.

    For a positive-degree coefficient, ``a*y`` strictly dominates ``D(y)``
    and every correction is forced.  For a lower-field coefficient, the
    Bronstein 6.6 cancellation recurrence solves one lower-field RDE for each
    leading coefficient.  If that coefficient is itself a logarithmic
    derivative, ``exact_derivative_result`` records the equivalent exact
    differentiation problem.
    """

    coefficient: Poly
    right_hand_side: Poly
    degree_bound: int
    solution: Poly | None
    obstruction: PrimitivePolynomialRDEObstruction | None
    partial_solution: Poly
    residual: Poly
    lower_field_results: tuple[RischDifferentialEquationResult, ...] = ()
    logarithmic_derivative_result: RadicalLogarithmicDerivativeResult | None = None
    exact_derivative_result: ExactDerivativeResult | None = None

    def __post_init__(self) -> None:
        """Validate the primitive polynomial decision state."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            PrimitivePolynomialRDEObstruction,
        ):
            raise TypeError("invalid primitive polynomial RDE obstruction")
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="primitive polynomial RDE certificate",
        )
        if self.solution is not None and (
            self.partial_solution != self.solution or not self.residual.is_zero
        ):
            raise ValueError(
                "a solved primitive polynomial RDE needs its solution as the "
                "complete reduction"
            )
        if (
            self.obstruction is PrimitivePolynomialRDEObstruction.DEGREE
            and self.degree_bound != -1
        ):
            raise ValueError("a primitive polynomial degree obstruction needs bound -1")
        if (
            self.obstruction is PrimitivePolynomialRDEObstruction.NONZERO_REMAINDER
            and self.residual.is_zero
        ):
            raise ValueError(
                "a primitive polynomial remainder obstruction must be nonzero"
            )
        if self.obstruction is PrimitivePolynomialRDEObstruction.LOWER_FIELD and (
            not self.lower_field_results
            or all(result.has_solution for result in self.lower_field_results)
        ):
            raise ValueError(
                "a lower-field obstruction needs an obstructed lower-field result"
            )
        if self.obstruction is PrimitivePolynomialRDEObstruction.EXACT_DERIVATIVE and (
            self.exact_derivative_result is None
            or self.exact_derivative_result.has_solution
        ):
            raise ValueError(
                "an exact-derivative obstruction needs an obstructed exact result"
            )
        if (
            self.solution is not None
            and self.exact_derivative_result is not None
            and not self.exact_derivative_result.has_solution
        ):
            raise ValueError(
                "a solved primitive polynomial RDE cannot retain an obstructed "
                "exact-derivative result"
            )

    @property
    def has_solution(self) -> bool:
        """Whether the equation has a solution in the primitive field."""
        return self.solution is not None

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether a recursively solved leading coefficient failed."""
        return self.obstruction is PrimitivePolynomialRDEObstruction.LOWER_FIELD

    @property
    def has_exact_derivative_obstruction(self) -> bool:
        """Whether the transformed exact-derivative problem failed."""
        return self.obstruction is PrimitivePolynomialRDEObstruction.EXACT_DERIVATIVE


@dataclass(frozen=True, slots=True)
class PrimitiveRischDifferentialEquationResult:
    r"""Certified RDE result over primitive ``K(t)`` with ``Dt`` in ``K``.

    ``normalization_factor`` records Bronstein's weak normalization

    ``a_normal = a - D(normalization_factor)/normalization_factor``.

    Under the substitution ``z = normalization_factor*y``, the original
    equation becomes

    ``Dz + a_normal*z = normalized_right_hand_side``.

    The normal-denominator theorem next reduces a possible normalized solution
    to ``q/denominator_bound``. Rothstein SPDE then reduces the numerator
    equation to a polynomial RDE, including the primitive lower-field
    cancellation cases.
    """

    coefficient: FieldElement
    right_hand_side: FieldElement
    solution: FieldElement | None
    obstruction: PrimitiveRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: FieldElement
    normalized_right_hand_side: FieldElement
    denominator_bound: Poly
    degree_bound: int | None = None
    polynomial_result: PrimitivePolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    logarithmic_derivative_result: RadicalLogarithmicDerivativeResult | None = None
    exact_derivative_result: ExactDerivativeResult | None = None

    def __post_init__(self) -> None:
        """Validate the primitive-field decision state and subcertificates."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            PrimitiveRDEObstruction,
        ):
            raise TypeError("invalid primitive RDE obstruction")
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="primitive RDE certificate",
        )
        division_obstruction = self.obstruction in {
            PrimitiveRDEObstruction.DENOMINATOR,
            PrimitiveRDEObstruction.SPDE_DIVISIBILITY,
        }
        _validate_failed_division(
            self.obstruction_polynomial,
            self.obstruction_remainder,
            required=division_obstruction,
            label="primitive RDE",
        )
        if self.obstruction is PrimitiveRDEObstruction.POLYNOMIAL and (
            self.polynomial_result is None or self.polynomial_result.has_solution
        ):
            raise ValueError(
                "a primitive polynomial obstruction needs an obstructed "
                "polynomial result"
            )
        if self.obstruction is PrimitiveRDEObstruction.EXACT_DERIVATIVE and (
            self.exact_derivative_result is None
            or self.exact_derivative_result.has_solution
        ):
            raise ValueError(
                "a primitive exact-derivative obstruction needs an obstructed "
                "exact result"
            )
        if self.solution is not None:
            if (
                self.polynomial_result is not None
                and not self.polynomial_result.has_solution
            ):
                raise ValueError(
                    "a solved primitive RDE cannot retain an obstructed polynomial "
                    "result"
                )
            if (
                self.exact_derivative_result is not None
                and not self.exact_derivative_result.has_solution
            ):
                raise ValueError(
                    "a solved primitive RDE cannot retain an obstructed exact result"
                )

    @property
    def has_solution(self) -> bool:
        """Whether the equation has a solution in the primitive field."""
        return self.solution is not None

    @property
    def has_denominator_obstruction(self) -> bool:
        """Whether normal pole orders prove nonexistence."""
        return self.obstruction is PrimitiveRDEObstruction.DENOMINATOR

    @property
    def has_spde_obstruction(self) -> bool:
        """Whether SPDE degree or divisibility proves nonexistence."""
        return self.obstruction in {
            PrimitiveRDEObstruction.SPDE_DEGREE,
            PrimitiveRDEObstruction.SPDE_DIVISIBILITY,
        }

    @property
    def has_polynomial_obstruction(self) -> bool:
        """Whether the terminal polynomial recurrence proves failure."""
        return self.obstruction is PrimitiveRDEObstruction.POLYNOMIAL

    @property
    def has_exact_derivative_obstruction(self) -> bool:
        """Whether a transformed exact-derivative problem proves failure."""
        return self.obstruction is PrimitiveRDEObstruction.EXACT_DERIVATIVE


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentRDEStep:
    r"""One diagonal Laurent-mode equation over the lower field.

    If ``Dt = eta*t``, the coefficient of ``t**j`` in

    ``D(y) + a*y = b``

    is decided by the lower-field equation

    ``D(y_j) + (a + j*eta)*y_j = b_j``.
    """

    power: int
    right_hand_side: FieldElement
    shifted_coefficient: FieldElement
    solution: FieldElement | None
    lower_field_result: RischDifferentialEquationResult

    def __post_init__(self) -> None:
        """Validate the retained lower-field mode decision."""
        if not isinstance(self.power, int) or isinstance(self.power, bool):
            raise ValueError("a Laurent RDE mode power must be an integer")
        lower_solution = self.lower_field_result.solution
        if (self.solution is None) != (lower_solution is None):
            raise ValueError(
                "a Laurent RDE mode solution must match its lower-field result"
            )
        if (
            self.solution is not None
            and lower_solution is not None
            and not _field_elements_equal(
                self.solution,
                lower_solution,
                self.lower_field_result.derivation.field,
            )
        ):
            raise ValueError(
                "a Laurent RDE mode solution differs from its lower-field result"
            )

    @property
    def has_solution(self) -> bool:
        """Whether this Laurent mode has a lower-field solution."""
        return self.solution is not None


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentRDECorrection:
    r"""One forced leading or trailing Laurent correction.

    ``correction`` is ``coefficient*t**power``. The stored residuals satisfy

    ``residual_after = residual_before - D(correction) - a*correction``.
    """

    power: int
    coefficient: FieldElement
    correction: FieldElement
    residual_before: FieldElement
    residual_after: FieldElement


@dataclass(frozen=True, slots=True)
class HyperexponentialLaurentRischDifferentialEquationResult:
    r"""Certified RDE result for the Laurent branch of ``K(t)``.

    The coefficient and right-hand side belong to ``K[t, t**-1]``. With
    ``Dt = eta*t``, ``minimum_power_bound`` and ``maximum_power_bound`` are
    Bronstein's valuation bounds at zero and infinity. A nonconstant
    coefficient then forces the leading or trailing corrections retained in
    ``corrections``. A coefficient in ``K`` preserves every Laurent mode, so
    ``steps`` instead retains the resulting lower-field equations.

    These restrictions also make the decision complete for rational
    solutions in ``K(t)``: a normal finite pole of a solution would acquire
    one additional order under differentiation, while multiplication by the
    Laurent coefficient would not introduce that normal factor. It therefore
    could not cancel against a Laurent right-hand side. The only possible
    top-level poles are powers of the special generator ``t``.
    """

    coefficient: FieldElement
    right_hand_side: FieldElement
    logarithmic_derivative: FieldElement
    solution: FieldElement | None
    obstruction: HyperexponentialLaurentRDEObstruction | None
    partial_solution: FieldElement
    residual: FieldElement
    steps: tuple[HyperexponentialLaurentRDEStep, ...]
    minimum_power_bound: int | None = None
    maximum_power_bound: int | None = None
    direction: HyperexponentialLaurentRDEDirection = (
        HyperexponentialLaurentRDEDirection.DIAGONAL
    )
    corrections: tuple[HyperexponentialLaurentRDECorrection, ...] = ()
    boundary_logarithmic_result: ParametricLogarithmicDerivativeResult | None = None
    exceptional_power: int | None = None

    def __post_init__(self) -> None:
        """Validate the complete Laurent-branch decision state."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            HyperexponentialLaurentRDEObstruction,
        ):
            raise TypeError("invalid hyperexponential Laurent RDE obstruction")
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="hyperexponential Laurent RDE certificate",
        )
        if self.obstruction is HyperexponentialLaurentRDEObstruction.LOWER_FIELD and (
            not self.steps or all(step.has_solution for step in self.steps)
        ):
            raise ValueError(
                "a Laurent lower-field obstruction needs an obstructed mode"
            )
        if self.solution is not None and any(
            not step.has_solution for step in self.steps
        ):
            raise ValueError(
                "a solved Laurent RDE cannot retain an obstructed lower-field mode"
            )

    @property
    def has_solution(self) -> bool:
        """Whether the RDE has a rational solution in this branch."""
        return self.solution is not None

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether at least one diagonal lower-field equation failed."""
        return self.obstruction is HyperexponentialLaurentRDEObstruction.LOWER_FIELD

    @property
    def has_nonzero_remainder_obstruction(self) -> bool:
        """Whether forced recurrence leaves an uncancellable residual."""
        return (
            self.obstruction is HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
        )

    @property
    def obstructed_steps(self) -> tuple[HyperexponentialLaurentRDEStep, ...]:
        """Return the Laurent modes with unsolved lower-field equations."""
        return tuple(step for step in self.steps if not step.has_solution)


@dataclass(frozen=True, slots=True)
class HyperexponentialRischDifferentialEquationResult:
    r"""Certified rational RDE result over ``K(t)`` with ``Dt = eta*t``.

    ``normalization_factor`` records Bronstein's weak normalization

    ``a_normal = a - D(normalization_factor)/normalization_factor``.

    Normal-denominator reduction proves that every normalized solution has
    denominator dividing ``normal_denominator_bound`` away from ``t = 0``.
    The special-denominator step then bounds its remaining power of ``t``:

    ``y_normal = polynomial_numerator
                  / (normal_denominator_bound*t**special_denominator_power)``.

    SPDE reduces the resulting polynomial equation to a Laurent RDE. The
    terminal Laurent certificate is retained in ``laurent_result``; failed
    exact divisions retain their polynomial and remainder.
    """

    coefficient: FieldElement
    right_hand_side: FieldElement
    solution: FieldElement | None
    obstruction: HyperexponentialRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: FieldElement
    normalized_right_hand_side: FieldElement
    normal_denominator_bound: Poly
    special_denominator_power: int | None = None
    special_clearing_power: int | None = None
    degree_bound: int | None = None
    laurent_result: HyperexponentialLaurentRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    special_boundary_result: ParametricLogarithmicDerivativeResult | None = None
    special_exceptional_power: int | None = None
    degree_boundary_result: ParametricLogarithmicDerivativeResult | None = None
    degree_exceptional_power: int | None = None

    def __post_init__(self) -> None:
        """Validate the hyperexponential decision state and terminal result."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            HyperexponentialRDEObstruction,
        ):
            raise TypeError("invalid hyperexponential RDE obstruction")
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="hyperexponential RDE certificate",
        )
        division_obstruction = self.obstruction in {
            HyperexponentialRDEObstruction.DENOMINATOR,
            HyperexponentialRDEObstruction.SPDE_DIVISIBILITY,
        }
        _validate_failed_division(
            self.obstruction_polynomial,
            self.obstruction_remainder,
            required=division_obstruction,
            label="hyperexponential RDE",
        )
        if self.obstruction is HyperexponentialRDEObstruction.LAURENT and (
            self.laurent_result is None or self.laurent_result.has_solution
        ):
            raise ValueError(
                "a Laurent obstruction needs an obstructed terminal result"
            )
        if (
            self.solution is not None
            and self.laurent_result is not None
            and not self.laurent_result.has_solution
        ):
            raise ValueError(
                "a solved hyperexponential RDE cannot retain an obstructed "
                "Laurent result"
            )

    @property
    def has_solution(self) -> bool:
        """Whether the RDE has a rational solution in this field."""
        return self.solution is not None

    @property
    def has_denominator_obstruction(self) -> bool:
        """Whether the normal-denominator theorem proves nonexistence."""
        return self.obstruction is HyperexponentialRDEObstruction.DENOMINATOR

    @property
    def has_spde_obstruction(self) -> bool:
        """Whether SPDE degree or divisibility proves nonexistence."""
        return self.obstruction in {
            HyperexponentialRDEObstruction.SPDE_DEGREE,
            HyperexponentialRDEObstruction.SPDE_DIVISIBILITY,
        }

    @property
    def has_laurent_obstruction(self) -> bool:
        """Whether the terminal complete Laurent decision proves failure."""
        return self.obstruction is HyperexponentialRDEObstruction.LAURENT


@dataclass(frozen=True, slots=True)
class RischDifferentialEquationResult:
    """Level-independent RDE result over an explicitly supplied tower."""

    derivation: Derivation
    coefficient: FieldElement
    right_hand_side: FieldElement
    solution: FieldElement | None
    obstruction: (
        RationalRDEObstruction
        | PrimitiveRDEObstruction
        | HyperexponentialRDEObstruction
        | HyperexponentialLaurentRDEObstruction
        | None
    )
    certificate: (
        RationalRischDifferentialEquationResult
        | PrimitiveRischDifferentialEquationResult
        | HyperexponentialRischDifferentialEquationResult
        | HyperexponentialLaurentRischDifferentialEquationResult
        | HypertangentGaussianRischDifferentialEquationResult
    )

    def __post_init__(self) -> None:
        """Validate dispatch consistency and exact reconstruction."""
        _validate_decided_outcome(
            self.solution,
            self.obstruction,
            label="recursive RDE result",
        )
        if self.certificate.has_solution != (self.solution is not None):
            raise ValueError(
                "the recursive RDE outcome disagrees with its level certificate"
            )
        if self.certificate.obstruction is not self.obstruction:
            raise ValueError(
                "the recursive RDE obstruction disagrees with its level certificate"
            )
        if self.solution is not None and not self.reconstructs:
            raise ValueError("the recursive RDE solution does not reconstruct")

    @property
    def has_solution(self) -> bool:
        """Whether the supported RDE has a solution in the current field."""
        return self.solution is not None

    @property
    def reconstructs(self) -> bool:
        """Whether a returned solution satisfies the differential equation."""
        return self.solution is not None and _field_element_is_zero(
            self.derivation(self.solution)
            + self.coefficient * self.solution
            - self.right_hand_side,
            self.derivation.field,
        )


@dataclass(frozen=True, slots=True)
class HypertangentGaussianRischDifferentialEquationResult:
    r"""Scalar real RDE certificate obtained through the Cayley isomorphism."""

    normal_form: HypertangentGaussianNormalForm
    target_result: RischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether the Gaussian target equation has a solution."""
        return self.target_result.has_solution

    @property
    def obstruction(
        self,
    ) -> (
        RationalRDEObstruction
        | PrimitiveRDEObstruction
        | HyperexponentialRDEObstruction
        | HyperexponentialLaurentRDEObstruction
        | None
    ):
        """Return the obstruction transported from the target equation."""
        return self.target_result.obstruction


@dataclass(frozen=True, slots=True)
class CoupledHyperexponentialLaurentRDEStep:
    """One independent Laurent mode in a coupled hyperexponential RDE."""

    power: int
    real_right_hand_side: FieldElement
    imaginary_right_hand_side: FieldElement
    shifted_diagonal_coefficient: FieldElement
    lower_result: CoupledRischDifferentialEquationResult

    @property
    def has_solution(self) -> bool:
        """Whether the lower coupled equation for this mode is solvable."""
        return self.lower_result.has_solution


@dataclass(frozen=True, slots=True)
class CoupledHyperexponentialLaurentRDEResult:
    r"""Direct Laurent-mode result for a hyperexponential top generator.

    For ``Dt = eta*t`` and lower-field ``a`` and ``lambda``, the coefficient
    of ``t**m`` in

    ``D(p) + a*p - lambda*q`` and
    ``D(q) + lambda*p + a*q``

    is a coupled equation over the lower field with diagonal coefficient
    ``a + m*eta``.  Different Laurent powers never mix.
    """

    derivation: Derivation
    diagonal_coefficient: FieldElement
    rotation_coefficient: FieldElement
    real_right_hand_side: FieldElement
    imaginary_right_hand_side: FieldElement
    real_solution: FieldElement | None
    imaginary_solution: FieldElement | None
    partial_real_solution: FieldElement
    partial_imaginary_solution: FieldElement
    real_residual: FieldElement
    imaginary_residual: FieldElement
    steps: tuple[CoupledHyperexponentialLaurentRDEStep, ...]
    obstruction: CoupledRDEObstruction | None

    def __post_init__(self) -> None:
        """Validate the direct Laurent decomposition and outcome."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            CoupledRDEObstruction,
        ):
            raise TypeError("invalid coupled Laurent RDE obstruction")
        _validate_paired_outcome(
            self.real_solution,
            self.imaginary_solution,
            self.obstruction,
            label="coupled Laurent RDE certificate",
        )
        if not self.reconstructs:
            raise ValueError(
                "the coupled Laurent partial reduction does not reconstruct"
            )
        if self.obstruction is CoupledRDEObstruction.LOWER_FIELD and (
            not self.steps or all(step.has_solution for step in self.steps)
        ):
            raise ValueError(
                "a coupled Laurent lower-field obstruction needs an obstructed mode"
            )
        if self.real_solution is not None and self.imaginary_solution is not None:
            field = self.derivation.field
            if not _field_elements_equal(
                self.derivation(self.real_solution)
                + self.diagonal_coefficient * self.real_solution
                - self.rotation_coefficient * self.imaginary_solution,
                self.real_right_hand_side,
                field,
            ) or not _field_elements_equal(
                self.derivation(self.imaginary_solution)
                + self.rotation_coefficient * self.real_solution
                + self.diagonal_coefficient * self.imaginary_solution,
                self.imaginary_right_hand_side,
                field,
            ):
                raise ValueError(
                    "the stored coupled Laurent solution does not reconstruct"
                )

    @property
    def has_solution(self) -> bool:
        """Whether every Laurent mode has a lower-field solution."""
        return self.real_solution is not None and self.imaginary_solution is not None

    @property
    def reconstructs(self) -> bool:
        """Whether the partial solutions and residuals reconstruct the input."""
        return (
            self.derivation(self.partial_real_solution)
            + self.diagonal_coefficient * self.partial_real_solution
            - self.rotation_coefficient * self.partial_imaginary_solution
            + self.real_residual
            == self.real_right_hand_side
            and self.derivation(self.partial_imaginary_solution)
            + self.rotation_coefficient * self.partial_real_solution
            + self.diagonal_coefficient * self.partial_imaginary_solution
            + self.imaginary_residual
            == self.imaginary_right_hand_side
        )


@dataclass(frozen=True, slots=True)
class CoupledHypertangentPolynomialRDECorrection:
    """One forced leading correction in the nonlinear no-cancellation case."""

    degree: int
    real_coefficient: FieldElement
    imaginary_coefficient: FieldElement
    real_residual_before: Poly
    imaginary_residual_before: Poly
    real_residual_after: Poly
    imaginary_residual_after: Poly


@dataclass(frozen=True, slots=True)
class CoupledHypertangentPolynomialRDEResult:
    r"""Direct polynomial coupled-RDE result at a hypertangent level.

    ``NON_CANCELLATION`` repeatedly removes the forced leading term created
    by ``Dt = eta*(1+t**2)``.  ``CANCELLATION`` is Bronstein's
    ``CoupledDECancelTan`` recurrence from Chapter 8, Section 8.4.
    """

    derivation: Derivation
    mode: CoupledHypertangentPolynomialRDEMode
    diagonal_coefficient: Poly
    rotation_coefficient: FieldElement
    real_right_hand_side: Poly
    imaginary_right_hand_side: Poly
    degree_bound: int
    real_solution: Poly | None
    imaginary_solution: Poly | None
    partial_real_solution: Poly
    partial_imaginary_solution: Poly
    real_residual: Poly
    imaginary_residual: Poly
    obstruction: CoupledRDEObstruction | None
    corrections: tuple[CoupledHypertangentPolynomialRDECorrection, ...] = ()
    boundary_result: CoupledRischDifferentialEquationResult | None = None
    recursive_result: CoupledHypertangentPolynomialRDEResult | None = None

    def __post_init__(self) -> None:
        """Validate the direct hypertangent polynomial certificate."""
        _validate_paired_outcome(
            self.real_solution,
            self.imaginary_solution,
            self.obstruction,
            label="coupled hypertangent polynomial RDE certificate",
        )
        if not self.reconstructs:
            raise ValueError(
                "the coupled hypertangent polynomial reduction does not reconstruct"
            )
        subordinate_results = tuple(
            result
            for result in (self.boundary_result, self.recursive_result)
            if result is not None
        )
        if self.obstruction is CoupledRDEObstruction.LOWER_FIELD and (
            not subordinate_results
            or all(result.has_solution for result in subordinate_results)
        ):
            raise ValueError(
                "a coupled hypertangent lower-field obstruction needs an "
                "obstructed subordinate result"
            )
        if self.real_solution is not None and self.imaginary_solution is not None:
            if any(not result.has_solution for result in subordinate_results):
                raise ValueError(
                    "a solved coupled hypertangent RDE cannot retain an obstructed "
                    "subordinate result"
                )
            rotation = self.diagonal_coefficient.domain.convert(
                self.rotation_coefficient
            )
            if (
                self.derivation.polynomial(self.real_solution)
                + self.diagonal_coefficient * self.real_solution
                - self.imaginary_solution.mul_ground(rotation)
                != self.real_right_hand_side
                or self.derivation.polynomial(self.imaginary_solution)
                + self.real_solution.mul_ground(rotation)
                + self.diagonal_coefficient * self.imaginary_solution
                != self.imaginary_right_hand_side
            ):
                raise ValueError(
                    "the stored coupled hypertangent solution does not reconstruct"
                )

    @property
    def has_solution(self) -> bool:
        """Whether the polynomial coupled system has a solution."""
        return self.real_solution is not None and self.imaginary_solution is not None

    @property
    def reconstructs(self) -> bool:
        """Whether the partial pair and residual reconstruct both equations."""
        rotation = self.diagonal_coefficient.domain.convert(self.rotation_coefficient)
        return (
            self.derivation.polynomial(self.partial_real_solution)
            + self.diagonal_coefficient * self.partial_real_solution
            - self.partial_imaginary_solution.mul_ground(rotation)
            + self.real_residual
            == self.real_right_hand_side
            and self.derivation.polynomial(self.partial_imaginary_solution)
            + self.partial_real_solution.mul_ground(rotation)
            + self.diagonal_coefficient * self.partial_imaginary_solution
            + self.imaginary_residual
            == self.imaginary_right_hand_side
        )


@dataclass(frozen=True, slots=True)
class CoupledRationalRDEResult:
    r"""Certified real solution of a conjugate pair of rational RDEs.

    The equations are

    ``Dp - lambda*q = u`` and ``Dq + lambda*p = v``,

    where all quantities belong to ``QQ(x)`` and ``lambda`` is nonzero.  This
    is the real restriction of

    ``D(p + i*q) + i*lambda*(p + i*q) = u + i*v``.

    For polynomial ``lambda``, the real leading recurrence directly supplies
    ``partial_*`` and ``*_residual`` certificates.  For rational ``lambda``,
    Bronstein weak normalization is performed over ``QQ_I(x)``; the
    ``gaussian_*`` fields retain that exact scalar certificate while the
    returned solution and candidate denominator remain real.
    """

    lambda_coefficient: Expr | Poly
    real_right_hand_side: Expr
    imaginary_right_hand_side: Expr
    real_solution: Expr | None
    imaginary_solution: Expr | None
    obstruction: CoupledRDEObstruction | None
    denominator_bound: Poly | None = None
    degree_bound: int | None = None
    partial_real_numerator: Poly | None = None
    partial_imaginary_numerator: Poly | None = None
    real_residual: Poly | None = None
    imaginary_residual: Poly | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    gaussian_normalization_factor: Poly | None = None
    gaussian_normalized_coefficient: Expr | None = None
    gaussian_denominator_bound: Poly | None = None
    gaussian_obstruction: RationalRDEObstruction | None = None
    gaussian_obstruction_polynomial: Poly | None = None
    gaussian_obstruction_remainder: Poly | None = None

    def __post_init__(self) -> None:
        """Validate the real rational pair and Gaussian subcertificate."""
        _validate_paired_outcome(
            self.real_solution,
            self.imaginary_solution,
            self.obstruction,
            label="coupled rational RDE certificate",
        )
        _validate_failed_division(
            self.obstruction_polynomial,
            self.obstruction_remainder,
            required=(
                self.obstruction is CoupledRDEObstruction.DENOMINATOR
                and self.gaussian_obstruction is None
            ),
            label="coupled rational RDE",
        )

        reduction_fields = (
            self.partial_real_numerator,
            self.partial_imaginary_numerator,
            self.real_residual,
            self.imaginary_residual,
        )
        if any(value is None for value in reduction_fields) and not all(
            value is None for value in reduction_fields
        ):
            raise ValueError(
                "a coupled rational reduction needs both partial numerators "
                "and both residuals"
            )

        _validate_failed_division(
            self.gaussian_obstruction_polynomial,
            self.gaussian_obstruction_remainder,
            required=self.gaussian_obstruction
            in {
                RationalRDEObstruction.DENOMINATOR,
                RationalRDEObstruction.SPDE_DIVISIBILITY,
            },
            label="Gaussian scalar RDE",
        )
        if self.gaussian_obstruction is not None:
            expected_obstruction = {
                RationalRDEObstruction.DENOMINATOR: (CoupledRDEObstruction.DENOMINATOR),
                RationalRDEObstruction.SPDE_DEGREE: CoupledRDEObstruction.DEGREE,
                RationalRDEObstruction.SPDE_DIVISIBILITY: (
                    CoupledRDEObstruction.NONZERO_REMAINDER
                ),
                RationalRDEObstruction.POLYNOMIAL: (
                    CoupledRDEObstruction.NONZERO_REMAINDER
                ),
            }[self.gaussian_obstruction]
            if self.obstruction is not expected_obstruction:
                raise ValueError(
                    "the coupled obstruction disagrees with its Gaussian certificate"
                )
        if self.obstruction is CoupledRDEObstruction.NONZERO_REMAINDER and (
            all(value is None for value in reduction_fields)
            and self.gaussian_obstruction is None
        ):
            raise ValueError(
                "a coupled remainder obstruction needs a recurrence or Gaussian "
                "subcertificate"
            )
        if self.real_solution is not None and self.imaginary_solution is not None:
            if self.denominator_bound is None:
                raise ValueError("a solved coupled rational RDE needs a denominator")
            variable = cast(Symbol, self.denominator_bound.gens[0])
            real_solution = cancel(self.real_solution)
            imaginary_solution = cancel(self.imaginary_solution)
            lambda_expression = (
                self.lambda_coefficient.as_expr()
                if isinstance(self.lambda_coefficient, Poly)
                else self.lambda_coefficient
            )
            if (
                cancel(
                    real_solution.diff(variable)
                    - lambda_expression * imaginary_solution
                    - self.real_right_hand_side
                )
                != 0
                or cancel(
                    imaginary_solution.diff(variable)
                    + lambda_expression * real_solution
                    - self.imaginary_right_hand_side
                )
                != 0
            ):
                raise ValueError(
                    "the stored coupled rational solution does not reconstruct"
                )

    @property
    def has_solution(self) -> bool:
        """Whether both coupled equations have a rational solution."""
        return self.real_solution is not None and self.imaginary_solution is not None

    @property
    def has_denominator_obstruction(self) -> bool:
        """Whether finite-pole orders prove nonexistence."""
        return self.obstruction is CoupledRDEObstruction.DENOMINATOR

    @property
    def has_degree_obstruction(self) -> bool:
        """Whether a degree bound or SPDE degree proves nonexistence."""
        return self.obstruction is CoupledRDEObstruction.DEGREE

    @property
    def has_nonzero_remainder_obstruction(self) -> bool:
        """Whether recurrence or SPDE leaves an uncancellable remainder."""
        return self.obstruction is CoupledRDEObstruction.NONZERO_REMAINDER


@dataclass(frozen=True, slots=True)
class CoupledRischDifferentialEquationResult:
    r"""Certified coupled RDE result over a supported differential tower.

    The real system

    ``Dp + a*p - lambda*q = u`` and
    ``Dq + lambda*p + a*q = v``

    is solved as the scalar equation

    ``D(p + i*q) + (a + i*lambda)*(p + i*q) = u + i*v``

    whenever adjoining ``i`` preserves the monomial families in the active
    tower.  If a hypertangent level is present, ``gaussian_normal_form``
    records the Cayley isomorphism used before solving that scalar equation.
    Direct Chapter 8 terminal recurrences are retained in ``direct_result``.
    """

    derivation: Derivation
    diagonal_coefficient: FieldElement
    lambda_coefficient: FieldElement
    real_right_hand_side: FieldElement
    imaginary_right_hand_side: FieldElement
    real_solution: FieldElement | None
    imaginary_solution: FieldElement | None
    obstruction: CoupledRDEObstruction | None
    scalar_result: RischDifferentialEquationResult | None = None
    direct_result: (
        CoupledHyperexponentialLaurentRDEResult
        | CoupledHypertangentPolynomialRDEResult
        | None
    ) = None
    gaussian_normal_form: HypertangentGaussianNormalForm | None = None

    def __post_init__(self) -> None:
        """Validate the recursive coupled outcome and exact solution."""
        if self.obstruction is not None and not isinstance(
            self.obstruction,
            CoupledRDEObstruction,
        ):
            raise TypeError("invalid coupled RDE obstruction")
        _validate_paired_outcome(
            self.real_solution,
            self.imaginary_solution,
            self.obstruction,
            label="recursive coupled RDE certificate",
        )
        if (self.scalar_result is None) == (self.direct_result is None):
            raise ValueError(
                "a recursive coupled RDE needs exactly one scalar or direct result"
            )
        subordinate = self.scalar_result or self.direct_result
        if subordinate is None:
            raise ValueError("a recursive coupled RDE lacks its subordinate result")
        if subordinate.has_solution != self.has_solution:
            raise ValueError(
                "the recursive coupled outcome disagrees with its subordinate result"
            )
        if self.direct_result is not None and (
            self.direct_result.obstruction is not self.obstruction
        ):
            raise ValueError(
                "the recursive coupled obstruction disagrees with its direct result"
            )
        if self.gaussian_normal_form is not None and self.scalar_result is None:
            raise ValueError("a Gaussian normal form requires a scalar RDE result")
        if self.has_solution and not self.reconstructs:
            raise ValueError("the recursive coupled RDE solution does not reconstruct")

    @property
    def has_solution(self) -> bool:
        """Whether both coupled equations have a solution."""
        return self.real_solution is not None and self.imaginary_solution is not None

    @property
    def reconstructs(self) -> bool:
        """Whether the returned pair satisfies both coupled equations."""
        if self.real_solution is None or self.imaginary_solution is None:
            return False
        return _field_element_is_zero(
            self.derivation(self.real_solution)
            + self.diagonal_coefficient * self.real_solution
            - self.lambda_coefficient * self.imaginary_solution
            - self.real_right_hand_side,
            self.derivation.field,
        ) and _field_element_is_zero(
            self.derivation(self.imaginary_solution)
            + self.lambda_coefficient * self.real_solution
            + self.diagonal_coefficient * self.imaginary_solution
            - self.imaginary_right_hand_side,
            self.derivation.field,
        )
