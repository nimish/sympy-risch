r"""Risch differential equations over exact polynomial domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.polytools import Poly, cancel, invert, resultant

from ._typing import FieldElement
from .polynomial import integrate_polynomial
from .rational import (
    hermite_reduce_rational,
    solve_rational_logarithmic_derivative,
)
from .tower import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
    _rebase_monomial_derivation,
)

if TYPE_CHECKING:
    from .dependency import ParametricLogarithmicDerivativeResult
    from .exact import ExactDerivativeResult
    from .gaussian import HypertangentGaussianNormalForm
    from .monomial import RadicalLogarithmicDerivativeResult


class PolynomialRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in ``QQ(x)``."""

    DEGREE = "degree"
    NONZERO_REMAINDER = "nonzero_remainder"
    INCONSISTENT_SYSTEM = "inconsistent_system"


class RationalRDEObstruction(Enum):
    """Why ``Dy + a*y = b`` has no solution in ``QQ(x)``."""

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


def _field_element_is_zero(value: FieldElement, field: Domain) -> bool:
    """Test exact zero despite noncanonical Gaussian fraction-field units."""
    return value == field.zero or cancel(field.to_sympy(value)) == 0


@dataclass(frozen=True, slots=True)
class PolynomialRischDifferentialEquationResult:
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``QQ[x]``.

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
    r"""Certified result for ``Dy + a*y = b`` with ``a,b`` in ``QQ(x)``.

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
    logarithmic_derivative_result: (
        RadicalLogarithmicDerivativeResult | None
    ) = None
    exact_derivative_result: ExactDerivativeResult | None = None

    @property
    def has_solution(self) -> bool:
        """Whether the equation has a solution in the primitive field."""
        return self.solution is not None

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether a recursively solved leading coefficient failed."""
        return (
            self.obstruction
            is PrimitivePolynomialRDEObstruction.LOWER_FIELD
        )

    @property
    def has_exact_derivative_obstruction(self) -> bool:
        """Whether the transformed exact-derivative problem failed."""
        return (
            self.obstruction
            is PrimitivePolynomialRDEObstruction.EXACT_DERIVATIVE
        )


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
    polynomial_result: (
        PrimitivePolynomialRischDifferentialEquationResult | None
    ) = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    logarithmic_derivative_result: (
        RadicalLogarithmicDerivativeResult | None
    ) = None
    exact_derivative_result: ExactDerivativeResult | None = None

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
    boundary_logarithmic_result: (
        ParametricLogarithmicDerivativeResult | None
    ) = None
    exceptional_power: int | None = None

    @property
    def has_solution(self) -> bool:
        """Whether the RDE has a rational solution in this branch."""
        return self.solution is not None

    @property
    def has_lower_field_obstruction(self) -> bool:
        """Whether at least one diagonal lower-field equation failed."""
        return (
            self.obstruction
            is HyperexponentialLaurentRDEObstruction.LOWER_FIELD
        )

    @property
    def has_nonzero_remainder_obstruction(self) -> bool:
        """Whether forced recurrence leaves an uncancellable residual."""
        return (
            self.obstruction
            is HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
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
    laurent_result: (
        HyperexponentialLaurentRischDifferentialEquationResult | None
    ) = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None
    special_boundary_result: (
        ParametricLogarithmicDerivativeResult | None
    ) = None
    special_exceptional_power: int | None = None
    degree_boundary_result: (
        ParametricLogarithmicDerivativeResult | None
    ) = None
    degree_exceptional_power: int | None = None

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
    )

    @property
    def has_solution(self) -> bool:
        """Whether the supported RDE has a solution in the current field."""
        return self.solution is not None

    @property
    def reconstructs(self) -> bool:
        """Whether a returned solution satisfies the differential equation."""
        return self.solution is not None and _field_element_is_zero(
            self.derivation(self.solution)
            + self.coefficient*self.solution
            - self.right_hand_side,
            self.derivation.field,
        )


@dataclass(frozen=True, slots=True)
class BoundedRationalRDEMatrixResult:
    """Exact matrix result for a caller-supplied rational ansatz."""

    denominator: Poly
    numerator_degree_bound: int
    solution: Expr | None
    coefficient_rank: int
    augmented_rank: int

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

    @property
    def has_solution(self) -> bool:
        """Whether the bounded real-pair ansatz contains a solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )


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

    @property
    def has_solution(self) -> bool:
        """Whether the bounded rectangular Laurent ansatz has a solution."""
        return self.solution is not None


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

    @property
    def has_solution(self) -> bool:
        """Whether every Laurent mode has a lower-field solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the partial solutions and residuals reconstruct the input."""
        return (
            self.derivation(self.partial_real_solution)
            + self.diagonal_coefficient*self.partial_real_solution
            - self.rotation_coefficient*self.partial_imaginary_solution
            + self.real_residual
            == self.real_right_hand_side
            and self.derivation(self.partial_imaginary_solution)
            + self.rotation_coefficient*self.partial_real_solution
            + self.diagonal_coefficient*self.partial_imaginary_solution
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

    @property
    def has_solution(self) -> bool:
        """Whether the polynomial coupled system has a solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the partial pair and residual reconstruct both equations."""
        rotation = self.diagonal_coefficient.domain.convert(
            self.rotation_coefficient
        )
        return (
            self.derivation.polynomial(self.partial_real_solution)
            + self.diagonal_coefficient*self.partial_real_solution
            - self.partial_imaginary_solution.mul_ground(rotation)
            + self.real_residual
            == self.real_right_hand_side
            and self.derivation.polynomial(self.partial_imaginary_solution)
            + self.partial_real_solution.mul_ground(rotation)
            + self.diagonal_coefficient*self.partial_imaginary_solution
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

    @property
    def has_solution(self) -> bool:
        """Whether both coupled equations have a rational solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )

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

    @property
    def has_solution(self) -> bool:
        """Whether both coupled equations have a solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )

    @property
    def reconstructs(self) -> bool:
        """Whether the returned pair satisfies both coupled equations."""
        if (
            self.real_solution is None
            or self.imaginary_solution is None
        ):
            return False
        return _field_element_is_zero(
            self.derivation(self.real_solution)
            + self.diagonal_coefficient*self.real_solution
            - self.lambda_coefficient*self.imaginary_solution
            - self.real_right_hand_side,
            self.derivation.field,
        ) and _field_element_is_zero(
            self.derivation(self.imaginary_solution)
            + self.lambda_coefficient*self.real_solution
            + self.diagonal_coefficient*self.imaginary_solution
            - self.imaginary_right_hand_side,
            self.derivation.field,
        )


@dataclass(frozen=True, slots=True)
class BoundedCoupledRDEMatrixResult:
    """Exact block-matrix result for a caller-supplied coupled ansatz."""

    denominator: Poly
    numerator_degree_bound: int
    real_solution: Expr | None
    imaginary_solution: Expr | None
    coefficient_rank: int
    augmented_rank: int

    @property
    def has_solution(self) -> bool:
        """Whether the bounded real-pair ansatz contains a solution."""
        return (
            self.real_solution is not None
            and self.imaginary_solution is not None
        )


def _polynomial_rde_system(
    a: Poly,
    b: Poly,
    degree_bound: int,
) -> tuple[DomainMatrix, DomainMatrix]:
    """Build the exact coefficient system for ``Dy + a*y = b``."""
    variable = a.gens[0]
    columns = []
    for degree in range(degree_bound + 1):
        monomial = Poly(variable**degree, variable, domain=QQ)
        columns.append(monomial.diff() + a*monomial)

    equation_count = max(
        b.degree(),
        max(column.degree() for column in columns),
    ) + 1
    system = DomainMatrix(
        [
            [
                QQ.convert(column.nth(degree))
                for column in columns
            ]
            for degree in range(equation_count)
        ],
        (equation_count, degree_bound + 1),
        QQ,
    )
    right_hand_side = DomainMatrix(
        [
            [QQ.convert(b.nth(degree))]
            for degree in range(equation_count)
        ],
        (equation_count, 1),
        QQ,
    )
    return system, right_hand_side


def _coerce_polynomial_rde_inputs(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> tuple[Poly, Poly]:
    """Coerce a polynomial Risch equation to its exact ``QQ[x]`` domain."""
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    return Poly(a, x, domain=QQ), Poly(b, x, domain=QQ)


def _solve_trivial_polynomial_rde(
    a: Poly,
    b: Poly,
) -> PolynomialRischDifferentialEquationResult | None:
    """Handle the zero cases shared by both independent solvers."""
    x = cast(Symbol, a.gens[0])

    if b.is_zero:
        zero = Poly(0, x, domain=QQ)
        return PolynomialRischDifferentialEquationResult(
            a,
            b,
            0,
            zero,
            None,
        )

    if a.is_zero:
        solution = Poly(
            integrate_polynomial(b, x),
            x,
            domain=QQ,
        )
        if solution.diff() != b:
            raise ValueError("polynomial Risch equation integration failed")
        return PolynomialRischDifferentialEquationResult(
            a,
            b,
            b.degree() + 1,
            solution,
            None,
        )

    return None


def solve_polynomial_risch_differential_equation(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> PolynomialRischDifferentialEquationResult:
    r"""Solve ``Dy + a*y = b`` for rational ``y`` when ``a,b`` are in ``QQ[x]``.

    This is the ``D = d/dt`` specialization of Bronstein,
    *Symbolic Integration I*, Section 6.5, particularly Lemma 6.5.1(i) and
    ``PolyRischDENoCancel``.  In this specialization the derivative always
    lowers polynomial degree, so the leading term of ``a*y`` cannot cancel.
    The solver therefore cancels the forced leading term repeatedly instead
    of constructing a coefficient matrix.

    The output is a decision certificate, not merely an ansatz search:

    * finite-pole analysis proves every rational solution is polynomial;
    * leading degrees either determine a finite degree or prove failure;
    * each recurrence step is forced and strictly lowers the residual degree.

    When ``a = 0``, the equation is ordinary polynomial integration and the
    integration constant is normalized to zero.
    """
    a_polynomial, b_polynomial = _coerce_polynomial_rde_inputs(a, b, x)
    trivial_result = _solve_trivial_polynomial_rde(
        a_polynomial,
        b_polynomial,
    )
    if trivial_result is not None:
        return trivial_result

    degree_bound = b_polynomial.degree() - a_polynomial.degree()
    if degree_bound < 0:
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            -1,
            None,
            PolynomialRDEObstruction.DEGREE,
            partial_solution=Poly(0, x, domain=QQ),
            residual=b_polynomial,
        )

    solution = Poly(0, x, domain=QQ)
    residual = b_polynomial
    while not residual.is_zero:
        candidate_degree = residual.degree() - a_polynomial.degree()
        if candidate_degree < 0:
            if (
                solution.diff()
                + a_polynomial*solution
                + residual
                != b_polynomial
            ):
                raise ValueError(
                    "polynomial Risch equation reduction invariant failed"
                )
            return PolynomialRischDifferentialEquationResult(
                a_polynomial,
                b_polynomial,
                degree_bound,
                None,
                PolynomialRDEObstruction.NONZERO_REMAINDER,
                partial_solution=solution,
                residual=residual,
            )

        term = Poly.from_dict(
            {
                (candidate_degree,): (
                    residual.LC()/a_polynomial.LC()
                ),
            },
            x,
            domain=QQ,
        )
        new_residual = residual - term.diff() - a_polynomial*term
        if (
            not new_residual.is_zero
            and new_residual.degree() >= residual.degree()
        ):
            raise ValueError(
                "polynomial Risch equation residual degree did not decrease"
            )
        solution += term
        residual = new_residual

    if solution.diff() + a_polynomial*solution != b_polynomial:
        raise ValueError(
            "polynomial Risch equation reconstruction failed"
        )
    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        degree_bound,
        solution,
        None,
    )


def solve_polynomial_risch_differential_equation_matrix(
    a: Expr | Poly,
    b: Expr | Poly,
    x: Symbol,
) -> PolynomialRischDifferentialEquationResult:
    r"""Solve ``Dy + a*y = b`` with an independent exact matrix method.

    This deliberately slower reference implementation searches every
    polynomial degree through ``degree(b)`` and uses exact rank comparison.
    It is retained to cross-check the leading-term recurrence without sharing
    that algorithm's forced-degree calculation.
    """
    a_polynomial, b_polynomial = _coerce_polynomial_rde_inputs(a, b, x)
    trivial_result = _solve_trivial_polynomial_rde(
        a_polynomial,
        b_polynomial,
    )
    if trivial_result is not None:
        return trivial_result

    forced_degree = b_polynomial.degree() - a_polynomial.degree()
    search_degree = b_polynomial.degree()
    system, right_hand_side = _polynomial_rde_system(
        a_polynomial,
        b_polynomial,
        search_degree,
    )
    try:
        solution_numerators, solution_denominator = system.solve_den(
            right_hand_side,
            method="rref",
        )
    except DMNonInvertibleMatrixError as exc:
        coefficient_rank = system.rank()
        augmented_rank = system.hstack(right_hand_side).rank()
        if augmented_rank <= coefficient_rank:
            raise ValueError(
                "polynomial Risch equation unexpectedly lacks a unique solution"
            ) from exc
        return PolynomialRischDifferentialEquationResult(
            a_polynomial,
            b_polynomial,
            max(-1, forced_degree),
            None,
            PolynomialRDEObstruction.INCONSISTENT_SYSTEM,
            coefficient_rank,
            augmented_rank,
        )

    coefficients = [
        row[0]/solution_denominator
        for row in solution_numerators.to_list()
    ]
    solution = Poly.from_dict(
        {
            (degree,): coefficient
            for degree, coefficient in enumerate(coefficients)
        },
        x,
        domain=QQ,
    )
    if solution.diff() + a_polynomial*solution != b_polynomial:
        raise ValueError(
            "matrix polynomial Risch equation reconstruction failed"
        )

    coefficient_rank = system.rank()
    augmented_rank = system.hstack(right_hand_side).rank()
    if (
        coefficient_rank != search_degree + 1
        or augmented_rank != coefficient_rank
    ):
        raise ValueError(
            "matrix polynomial Risch equation returned invalid ranks"
        )

    # The matrix searches beyond the forced degree on purpose.  The extra
    # solution coefficients are zero, but the reported ranks describe the
    # complete independently searched system.
    return PolynomialRischDifferentialEquationResult(
        a_polynomial,
        b_polynomial,
        forced_degree,
        solution,
        None,
        coefficient_rank,
        augmented_rank,
    )


def _as_qq_rational(
    value: Expr | Poly,
    x: Symbol,
) -> tuple[Expr, Poly, Poly]:
    """Return an expression and coprime numerator/denominator in ``QQ[x]``."""
    if isinstance(value, Poly):
        value = value.as_expr()
    numerator, denominator = cancel(value).as_numer_denom()
    numerator_polynomial = Poly(numerator, x, domain=QQ)
    denominator_polynomial = Poly(denominator, x, domain=QQ)
    coefficient, numerator_polynomial, denominator_polynomial = (
        numerator_polynomial.cancel(denominator_polynomial)
    )
    numerator_polynomial = numerator_polynomial.mul_ground(coefficient)
    expression = cancel(
        numerator_polynomial.as_expr()/denominator_polynomial.as_expr()
    )
    return expression, numerator_polynomial, denominator_polynomial


def _weakly_normalize_rational_rde_coefficient(
    coefficient: Expr,
    x: Symbol,
) -> tuple[Poly, Expr]:
    r"""Return ``q`` and ``coefficient - Dq/q`` with no positive residues."""
    _, residue_part = hermite_reduce_rational(coefficient, x)
    logarithmic_result = solve_rational_logarithmic_derivative(
        residue_part,
        x,
    )
    normalizer = Poly(1, x, domain=QQ)
    for factor in logarithmic_result.factors:
        if factor.multiplicity is None:
            continue
        multiplicity = QQ.convert(factor.multiplicity)
        if (
            QQ.denom(multiplicity) == 1
            and QQ.numer(multiplicity) > 0
        ):
            normalizer *= factor.factor**int(QQ.numer(multiplicity))

    normalized = cancel(
        coefficient - normalizer.diff().as_expr()/normalizer.as_expr()
    )
    if cancel(
        normalized
        + normalizer.diff().as_expr()/normalizer.as_expr()
        - coefficient
    ) != 0:
        raise ValueError("weak normalization reconstruction failed")
    return normalizer, normalized


@dataclass(frozen=True, slots=True)
class _NormalDenominatorReduction:
    denominator_bound: Poly
    derivative_coefficient: Poly | None
    solution_coefficient: Poly | None
    right_hand_side: Poly | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None

    @property
    def equation(self) -> tuple[Poly, Poly, Poly] | None:
        """Return the transformed polynomial equation when reduction succeeds."""
        if (
            self.derivative_coefficient is None
            or self.solution_coefficient is None
            or self.right_hand_side is None
        ):
            return None
        return (
            self.derivative_coefficient,
            self.solution_coefficient,
            self.right_hand_side,
        )


def _reduce_normal_denominator(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
) -> _NormalDenominatorReduction:
    r"""Reduce a weakly normalized equation to ``A*Dq + B*q = C``.

    This is the ``D = d/dx`` specialization of Bronstein's normal
    denominator theorem.  Every nonconstant polynomial in ``QQ[x]`` is
    normal, so no separate special-denominator step remains.
    """
    _, coefficient_numerator, coefficient_denominator = _as_qq_rational(
        coefficient,
        x,
    )
    _, rhs_numerator, rhs_denominator = _as_qq_rational(
        right_hand_side,
        x,
    )

    shared_denominator = coefficient_denominator.gcd(rhs_denominator)
    denominator_bound = (
        rhs_denominator.gcd(rhs_denominator.diff())
        .exquo(
            shared_denominator.gcd(shared_denominator.diff())
        )
        .monic()
    )

    derivative_coefficient = coefficient_denominator*denominator_bound
    solution_coefficient = (
        denominator_bound*coefficient_numerator
        - coefficient_denominator*denominator_bound.diff()
    )
    multiplier = (
        coefficient_denominator
        * denominator_bound
        * denominator_bound
    )
    quotient, remainder = multiplier.div(rhs_denominator)
    if not remainder.is_zero:
        return _NormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_denominator,
            remainder,
        )

    transformed_rhs = quotient*rhs_numerator
    return _NormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )


def _base_polynomial_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
) -> int:
    r"""Bound polynomial solutions of ``A*Dq + B*q = C`` over ``QQ[x]``."""
    if right_hand_side.is_zero:
        return 0

    derivative_degree = derivative_coefficient.degree()
    rhs_degree = right_hand_side.degree()
    if solution_coefficient.is_zero:
        dominant_degree = derivative_degree - 1
    else:
        dominant_degree = max(
            solution_coefficient.degree(),
            derivative_degree - 1,
        )
    bound = max(0, rhs_degree - dominant_degree)

    if (
        not solution_coefficient.is_zero
        and solution_coefficient.degree() == derivative_degree - 1
    ):
        cancellation_degree = QQ.convert(
            -solution_coefficient.LC()/derivative_coefficient.LC()
        )
        if QQ.denom(cancellation_degree) == 1:
            bound = max(
                bound,
                int(QQ.numer(cancellation_degree)),
                rhs_degree - solution_coefficient.degree(),
            )
    return bound


@dataclass(frozen=True, slots=True)
class _SPDEResult:
    solution: Poly | None
    obstruction: RationalRDEObstruction | None
    polynomial_result: PolynomialRischDifferentialEquationResult | None = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_special_polynomial_differential_equation(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
) -> _SPDEResult:
    r"""Apply Rothstein's SPDE reduction, then the polynomial RDE solver."""
    x = cast(Symbol, derivative_coefficient.gens[0])
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = Poly(1, x, domain=QQ)
    beta = Poly(0, x, domain=QQ)
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient*beta.diff()
                + original_solution_coefficient*beta
                != original_rhs
            ):
                raise ValueError("SPDE zero-tail reconstruction failed")
            return _SPDEResult(beta, None)

        if current_bound < 0:
            return _SPDEResult(
                None,
                RationalRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _SPDEResult(
                None,
                RationalRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse_leading_coefficient = QQ.one/derivative_coefficient.LC()
            normalized_coefficient = solution_coefficient.mul_ground(
                inverse_leading_coefficient
            )
            normalized_rhs = right_hand_side.mul_ground(
                inverse_leading_coefficient
            )
            polynomial_result = solve_polynomial_risch_differential_equation(
                normalized_coefficient,
                normalized_rhs,
                x,
            )
            if not polynomial_result.has_solution:
                return _SPDEResult(
                    None,
                    RationalRDEObstruction.POLYNOMIAL,
                    polynomial_result,
                )
            tail = polynomial_result.solution
            if tail is None:
                raise ValueError("a solved polynomial RDE lacks its solution")
            if (
                not tail.is_zero
                and tail.degree() > current_bound
            ):
                return _SPDEResult(
                    None,
                    RationalRDEObstruction.SPDE_DEGREE,
                    polynomial_result,
                )
            solution = alpha*tail + beta
            if (
                original_derivative_coefficient*solution.diff()
                + original_solution_coefficient*solution
                != original_rhs
            ):
                raise ValueError("SPDE reconstruction failed")
            return _SPDEResult(
                solution,
                None,
                polynomial_result,
            )

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side*inverse).rem(
            derivative_coefficient
        )
        quotient = (
            right_hand_side - solution_coefficient*residue
        ).exquo(derivative_coefficient)

        right_hand_side = quotient - residue.diff()
        solution_coefficient += derivative_coefficient.diff()
        current_bound -= derivative_coefficient.degree()
        beta += alpha*residue
        alpha *= derivative_coefficient


def solve_rational_risch_differential_equation(
    coefficient: Expr | Poly,
    right_hand_side: Expr | Poly,
    x: Symbol,
) -> RationalRischDifferentialEquationResult:
    r"""Decide ``Dy + coefficient*y = right_hand_side`` in ``QQ(x)``.

    The algorithm is the base-field specialization of Bronstein's Chapter 6
    Risch differential equation:

    1. weakly normalize the coefficient by removing positive integer
       logarithmic residues;
    2. compute the forced normal denominator of a rational solution;
    3. bound the remaining polynomial numerator;
    4. apply Rothstein's SPDE reduction;
    5. finish with :func:`solve_polynomial_risch_differential_equation`.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    coefficient, _, _ = _as_qq_rational(coefficient, x)
    right_hand_side, _, _ = _as_qq_rational(right_hand_side, x)

    normalization_factor, normalized_coefficient = (
        _weakly_normalize_rational_rde_coefficient(coefficient, x)
    )
    normalized_rhs = cancel(
        normalization_factor.as_expr()*right_hand_side
    )

    if normalized_rhs == 0:
        zero = S.Zero
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            zero,
            None,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            Poly(1, x, domain=QQ),
            0,
        )

    denominator_reduction = _reduce_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        x,
    )
    equation = denominator_reduction.equation
    if equation is None:
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            None,
            RationalRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            obstruction_polynomial=(
                denominator_reduction.obstruction_polynomial
            ),
            obstruction_remainder=(
                denominator_reduction.obstruction_remainder
            ),
        )

    denominator_bound = denominator_reduction.denominator_bound
    (
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    ) = equation
    degree_bound = _base_polynomial_rde_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )
    spde_result = _solve_special_polynomial_differential_equation(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
    )
    if spde_result.solution is None:
        return RationalRischDifferentialEquationResult(
            coefficient,
            right_hand_side,
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_bound,
            degree_bound,
            spde_result.polynomial_result,
            spde_result.obstruction_polynomial,
            spde_result.obstruction_remainder,
        )

    solution = cancel(
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_bound.as_expr()
        )
    )
    if cancel(
        solution.diff(x)
        + coefficient*solution
        - right_hand_side
    ) != 0:
        raise ValueError("rational Risch equation reconstruction failed")
    return RationalRischDifferentialEquationResult(
        coefficient,
        right_hand_side,
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        normalized_rhs,
        denominator_bound,
        degree_bound,
        spde_result.polynomial_result,
    )


def _primitive_rde_extension(
    derivation: Derivation,
) -> MonomialExtension:
    """Validate a primitive top level with the no-new-constants property."""
    if derivation.level < 0:
        raise TypeError("a primitive RDE requires a differential extension")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "primitive RDE solving does not support algebraic tower levels"
        )
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError("primitive RDE solving requires Dt in the lower field")
    extension.require_preserved_constants("primitive RDE solving")
    return extension


def _as_primitive_fraction(
    value: object,
    extension: MonomialExtension,
) -> tuple[FieldElement, Poly, Poly]:
    """Return an element and coprime polynomials in the top generator."""
    field = extension.field
    parent = extension.parent_field
    element = _coerce(
        field,
        value.as_expr() if isinstance(value, Poly) else value,
    )
    numerator_expression, denominator_expression = cancel(
        field.to_sympy(element)
    ).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        extension.generator,
        domain=parent,
    )
    denominator = Poly(
        denominator_expression,
        extension.generator,
        domain=parent,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(parent.convert(coefficient))
    leading_inverse = parent.one/parent.convert(denominator.LC())
    numerator = numerator.mul_ground(leading_inverse)
    denominator = denominator.monic()
    if _coerce(field, numerator.as_expr()/denominator.as_expr()) != element:
        raise ValueError("primitive fraction normalization failed")
    return element, numerator, denominator


def _as_primitive_polynomial(
    value: object,
    extension: MonomialExtension,
) -> Poly:
    """Coerce a top-field element known to be polynomial in the generator."""
    _, numerator, denominator = _as_primitive_fraction(
        value,
        extension,
    )
    if denominator.degree() > 0:
        raise NotImplementedError(
            "primitive RDE coefficients with top-generator poles are not "
            "implemented"
        )
    scale = extension.parent_field.one/extension.parent_field.convert(
        denominator.LC()
    )
    return numerator.mul_ground(scale)


def _positive_integer_rational(value: object) -> int | None:
    """Return a positive rational integer, or ``None`` otherwise."""
    rational = QQ.convert(value)
    if QQ.denom(rational) != 1 or QQ.numer(rational) <= 0:
        return None
    return int(QQ.numer(rational))


def _positive_integer_in_domain(value: object, domain: Domain) -> int | None:
    """Return a positive integer represented in ``domain``, if any."""
    expression = domain.to_sympy(domain.convert(value))
    if expression.is_Integer is not True or expression.is_positive is not True:
        return None
    return int(expression)


def _weakly_normalize_hyperexponential_rde_coefficient(
    coefficient: object,
    derivation: Derivation,
) -> tuple[Poly, FieldElement]:
    r"""Apply Bronstein's general weak-normalizer resultant construction.

    Let ``d1`` be the product of the normal irreducible denominator factors
    that occur exactly once. If ``a`` is the partial numerator of ``f`` at
    ``d1``, the positive integer roots of

    ``resultant_t(a - z*D(d1), d1)``

    are precisely the logarithmic residues that weak normalization removes.
    For every such root ``n``, the factor
    ``gcd(a - n*D(d1), d1)**n`` is included in the normalizer.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    element, numerator, denominator = _as_primitive_fraction(
        coefficient,
        extension,
    )
    from .monomial import split_normal_special

    normal_denominator, _ = split_normal_special(denominator, derivation)
    repeated_part = normal_denominator.gcd(normal_denominator.diff())
    squarefree_part = normal_denominator.exquo(repeated_part)
    simple_normal_part = squarefree_part.exquo(
        squarefree_part.gcd(repeated_part)
    ).monic()
    one = Poly(1, extension.generator, domain=parent)
    if simple_normal_part.degree() == 0:
        return one, element

    cofactor = denominator.exquo(simple_normal_part)
    partial_numerator = (
        numerator*invert(cofactor, simple_normal_part)
    ).rem(simple_normal_part)
    simple_derivative = derivation.polynomial(simple_normal_part)
    residue_symbol = Dummy("z")
    residue_polynomial = Poly(
        resultant(
            (
                partial_numerator.as_expr()
                - residue_symbol*simple_derivative.as_expr()
            ),
            simple_normal_part.as_expr(),
            extension.generator,
        ),
        residue_symbol,
        domain=parent,
    )

    normalizer = one
    for root in residue_polynomial.ground_roots():
        multiplicity = _positive_integer_in_domain(root, parent)
        if multiplicity is None:
            continue
        factor = (
            partial_numerator
            - simple_derivative.mul_ground(parent.convert(multiplicity))
        ).gcd(simple_normal_part)
        normalizer *= factor**multiplicity
    normalizer = normalizer.monic()
    if normalizer.degree() == 0:
        return normalizer, element

    normalizer_element = _coerce(field, normalizer.as_expr())
    normalized = element - derivation(normalizer_element)/normalizer_element
    if (
        normalized
        + derivation(normalizer_element)/normalizer_element
        != element
    ):
        raise ValueError(
            "hyperexponential weak normalization failed to reconstruct"
        )
    return normalizer, normalized


@dataclass(frozen=True, slots=True)
class _HyperexponentialNormalDenominatorReduction:
    """Internal result of Bronstein's normal-denominator theorem."""

    denominator_bound: Poly
    derivative_coefficient: FieldElement | None
    solution_coefficient: FieldElement | None
    right_hand_side: FieldElement | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None


def _reduce_hyperexponential_normal_denominator(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _HyperexponentialNormalDenominatorReduction:
    r"""Bound all normal factors in a hyperexponential RDE solution.

    This is ``RdeNormalDenominator`` from Bronstein, Section 6.1. The returned
    equation ``A*Dq + B*q = C`` has only special powers of ``t`` in ``B`` and
    ``C``; every normalized solution is ``q/h`` where ``h`` is the returned
    denominator bound.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    _, _, coefficient_denominator = _as_primitive_fraction(
        coefficient,
        extension,
    )
    _, _, rhs_denominator = _as_primitive_fraction(
        right_hand_side,
        extension,
    )
    from .monomial import split_normal_special

    coefficient_normal, _ = split_normal_special(
        coefficient_denominator,
        derivation,
    )
    rhs_normal, _ = split_normal_special(rhs_denominator, derivation)
    shared_normal = coefficient_normal.gcd(rhs_normal)
    denominator_bound = (
        rhs_normal.gcd(rhs_normal.diff())
        .exquo(shared_normal.gcd(shared_normal.diff()))
        .monic()
    )
    divisibility_target = coefficient_normal*denominator_bound**2
    _, remainder = divisibility_target.div(rhs_normal)
    if not remainder.is_zero:
        return _HyperexponentialNormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_normal,
            remainder,
        )

    coefficient_element = _coerce(field, coefficient)
    rhs_element = _coerce(field, right_hand_side)
    normal_element = _coerce(field, coefficient_normal.as_expr())
    bound_element = _coerce(field, denominator_bound.as_expr())
    derivative_coefficient = normal_element*bound_element
    solution_coefficient = (
        derivative_coefficient*coefficient_element
        - normal_element*derivation(bound_element)
    )
    transformed_rhs = (
        normal_element*bound_element*bound_element*rhs_element
    )
    return _HyperexponentialNormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )


def _weakly_normalize_primitive_rde_coefficient(
    coefficient: object,
    derivation: Derivation,
) -> tuple[Poly, FieldElement]:
    r"""Return ``q`` and ``coefficient - Dq/q`` weakly normalized in ``t``.

    At a normal irreducible factor ``p``, weak normalization removes the
    simple logarithmic residue only when it is a positive integer ``n``.
    Multiplying ``q`` by ``p**n`` subtracts ``n*Dp/p`` from the coefficient.

    A primitive monomial with preserved constants has no nonconstant special
    factors, so the complete principal-part signature supplies every factor
    that must be inspected.
    """
    extension = _primitive_rde_extension(derivation)
    field = extension.field
    element = _coerce(field, coefficient)
    _, _, denominator = _as_primitive_fraction(element, extension)
    if denominator.degree() == 0:
        return (
            Poly(
                1,
                extension.generator,
                domain=extension.parent_field,
            ),
            element,
        )

    # Local import avoids the module cycle: primitive monomial reduction uses
    # the RDE dispatcher defined in this module.
    from .monomial import primitive_logarithmic_derivative_signature

    signature = primitive_logarithmic_derivative_signature(
        element,
        derivation,
    )
    normalizer = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    for factor in signature.factors:
        if factor.logarithmic_multiplicity is None:
            continue
        multiplicity = _positive_integer_rational(
            factor.logarithmic_multiplicity,
        )
        if multiplicity is not None:
            normalizer *= factor.factor**multiplicity
    normalizer = normalizer.monic()
    if normalizer.degree() == 0:
        return normalizer, element

    normalizer_element = _coerce(field, normalizer.as_expr())
    normalized = element - derivation(normalizer_element)/normalizer_element
    if (
        normalized
        + derivation(normalizer_element)/normalizer_element
        != element
    ):
        raise ValueError("primitive weak normalization failed to reconstruct")

    normalized_signature = primitive_logarithmic_derivative_signature(
        normalized,
        derivation,
    )
    for factor in normalized_signature.factors:
        if (
            factor.logarithmic_multiplicity is not None
            and _positive_integer_rational(
                factor.logarithmic_multiplicity,
            )
            is not None
        ):
            raise ValueError(
                "primitive weak normalization left a positive integer residue"
            )
    return normalizer, normalized


def _reduce_primitive_normal_denominator(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _NormalDenominatorReduction:
    r"""Reduce a weakly normalized primitive RDE to ``A*Dq+B*q=C``.

    Every nonconstant polynomial is normal in a primitive extension with
    preserved constants. Thus Bronstein's normal-denominator formula uses the
    complete top-generator denominators and no special-factor component.
    Formal differentiation detects their repeated factors, while ``D(h)`` in
    the transformed equation is the total tower derivative.
    """
    extension = _primitive_rde_extension(derivation)
    _, coefficient_numerator, coefficient_denominator = (
        _as_primitive_fraction(coefficient, extension)
    )
    _, rhs_numerator, rhs_denominator = _as_primitive_fraction(
        right_hand_side,
        extension,
    )

    shared_denominator = coefficient_denominator.gcd(rhs_denominator)
    denominator_bound = (
        rhs_denominator.gcd(rhs_denominator.diff())
        .exquo(
            shared_denominator.gcd(shared_denominator.diff())
        )
        .monic()
    )
    derivative_coefficient = coefficient_denominator*denominator_bound
    solution_coefficient = (
        denominator_bound*coefficient_numerator
        - coefficient_denominator
        * derivation.polynomial(denominator_bound)
    )
    multiplier = (
        coefficient_denominator
        * denominator_bound
        * denominator_bound
    )
    quotient, remainder = multiplier.div(rhs_denominator)
    if not remainder.is_zero:
        return _NormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_denominator,
            remainder,
        )
    return _NormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        quotient*rhs_numerator,
    )


def _primitive_limited_integration_relation(
    value: object,
    derivation: Derivation,
) -> tuple[int, FieldElement] | None:
    r"""Find ``m`` and ``z`` with ``value = m*Dt + Dz``, if supported.

    Reducing a constant polynomial at the current primitive level computes
    exactly the quotient by ``D(K) + Const(K)*Dt``. The degree-bound theorem
    additionally requires the coefficient of ``Dt`` to be an integer.
    """
    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    element = parent.convert(value)
    if element == parent.zero:
        return 0, parent.zero

    # See the cycle note in primitive weak normalization above.
    from .monomial import polynomial_reduce_primitive

    reduction = polynomial_reduce_primitive(
        Poly(
            element,
            extension.generator,
            domain=parent,
        ),
        derivation,
    )
    if reduction.has_obstruction or len(reduction.steps) != 1:
        return None
    step = reduction.steps[0]
    integer_multiple = _integer_constant(
        step.eta_multiple,
        derivation.tower.constants,
    )
    if integer_multiple is None:
        return None
    exact_part = parent.convert(step.exact_coefficient)
    eta = parent.convert(extension.generator_derivative.nth(0))
    if (
        derivation.lower()(exact_part)
        + parent.convert(integer_multiple)*eta
        != element
    ):
        raise ValueError("primitive limited-integration relation is invalid")
    return integer_multiple, exact_part


def _primitive_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    derivation: Derivation,
) -> int:
    r"""Implement Bronstein's primitive polynomial RDE degree bound.

    This is ``RdeBoundDegreePrim`` from Section 6.3. Besides the generic
    degree estimate, the two possible leading cancellations require one
    limited-integration decision and, at equal degrees, one logarithmic
    derivative decision in the lower field.
    """
    if derivative_coefficient.is_zero:
        raise ValueError("the derivative coefficient must be nonzero")
    if right_hand_side.is_zero:
        return 0

    degree_a = derivative_coefficient.degree()
    degree_b = (
        solution_coefficient.degree()
        if not solution_coefficient.is_zero
        else -1
    )
    degree_c = right_hand_side.degree()
    if degree_b > degree_a:
        bound = max(0, degree_c - degree_b)
    else:
        bound = max(0, degree_c - degree_a + 1)

    parent = derivative_coefficient.domain
    if (
        not solution_coefficient.is_zero
        and degree_b == degree_a - 1
    ):
        alpha = parent.convert(
            -solution_coefficient.LC()/derivative_coefficient.LC()
        )
        relation = _primitive_limited_integration_relation(
            alpha,
            derivation,
        )
        if relation is not None:
            bound = max(bound, relation[0])

    if degree_b == degree_a:
        alpha = parent.convert(
            -solution_coefficient.LC()/derivative_coefficient.LC()
        )
        from .monomial import solve_radical_logarithmic_derivative

        logarithmic_result = solve_radical_logarithmic_derivative(
            alpha,
            derivation.lower(),
        )
        if (
            logarithmic_result.has_solution
            and logarithmic_result.minimal_power == 1
        ):
            z = parent.convert(logarithmic_result.argument)
            dz = derivation.lower()(z)
            transformed = (
                derivative_coefficient.mul_ground(dz)
                + solution_coefficient.mul_ground(z)
            )
            beta = parent.convert(
                -transformed.LC()
                / (z*parent.convert(derivative_coefficient.LC()))
            )
            relation = _primitive_limited_integration_relation(
                beta,
                derivation,
            )
            if relation is not None:
                bound = max(bound, relation[0])
    return bound


def _solve_primitive_polynomial_exact_case(
    coefficient_polynomial: Poly,
    rhs_polynomial: Poly,
    derivation: Derivation,
    multiplier: object,
    logarithmic_result: RadicalLogarithmicDerivativeResult | None = None,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Solve a polynomial cancellation case by exact differentiation.

    If the coefficient is zero, ``multiplier`` is one.  Otherwise the caller
    has certified ``coefficient = D(multiplier)/multiplier``, so

    ``D(y) + coefficient*y = c``

    is equivalent to ``D(multiplier*y) = multiplier*c``.
    """
    from .exact import solve_exact_derivative

    extension = _primitive_rde_extension(derivation)
    field = extension.field
    zero = rhs_polynomial.zero
    multiplier_element = _coerce(field, multiplier)
    transformed_rhs = (
        multiplier_element
        * _coerce(field, rhs_polynomial.as_expr())
    )
    exact_result = solve_exact_derivative(
        transformed_rhs,
        derivation,
    )
    degree_bound = (
        0 if rhs_polynomial.is_zero else rhs_polynomial.degree() + 1
    )
    if not exact_result.has_solution:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            degree_bound,
            None,
            PrimitivePolynomialRDEObstruction.EXACT_DERIVATIVE,
            zero,
            rhs_polynomial,
            logarithmic_derivative_result=logarithmic_result,
            exact_derivative_result=exact_result,
        )

    solution_element = exact_result.antiderivative/multiplier_element
    solution = _as_primitive_polynomial(
        solution_element,
        extension,
    )
    if not solution.is_zero and solution.degree() > degree_bound:
        raise ValueError(
            "primitive exact cancellation exceeded its degree bound"
        )
    if (
        derivation.polynomial(solution)
        + coefficient_polynomial*solution
        != rhs_polynomial
    ):
        raise ValueError(
            "primitive exact cancellation reconstruction failed"
        )
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        zero,
        logarithmic_derivative_result=logarithmic_result,
        exact_derivative_result=exact_result,
    )


def _solve_primitive_polynomial_lower_coefficient(
    coefficient_polynomial: Poly,
    rhs_polynomial: Poly,
    derivation: Derivation,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Apply Bronstein's ``PolyRischDECancelPrim`` recurrence."""
    from .monomial import solve_radical_logarithmic_derivative

    extension = _primitive_rde_extension(derivation)
    parent = extension.parent_field
    lower_derivation = derivation.lower()
    coefficient = parent.convert(coefficient_polynomial.nth(0))
    zero = rhs_polynomial.zero

    if coefficient == parent.zero:
        return _solve_primitive_polynomial_exact_case(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
            parent.one,
        )

    logarithmic_result = solve_radical_logarithmic_derivative(
        coefficient,
        lower_derivation,
    )
    if (
        logarithmic_result.has_solution
        and logarithmic_result.minimal_power == 1
    ):
        return _solve_primitive_polynomial_exact_case(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
            logarithmic_result.argument,
            logarithmic_result,
        )

    if rhs_polynomial.is_zero:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            0,
            zero,
            None,
            zero,
            zero,
            logarithmic_derivative_result=logarithmic_result,
        )

    degree_bound = rhs_polynomial.degree()
    solution = zero
    residual = rhs_polynomial
    lower_results = []
    while not residual.is_zero:
        degree = residual.degree()
        lower_rhs = parent.convert(residual.LC())
        lower_result = solve_risch_differential_equation(
            coefficient,
            lower_rhs,
            lower_derivation,
        )
        lower_results.append(lower_result)
        if not lower_result.has_solution:
            if (
                derivation.polynomial(solution)
                + coefficient_polynomial*solution
                + residual
                != rhs_polynomial
            ):
                raise ValueError(
                    "primitive cancellation obstruction invariant failed"
                )
            return PrimitivePolynomialRischDifferentialEquationResult(
                coefficient_polynomial,
                rhs_polynomial,
                degree_bound,
                None,
                PrimitivePolynomialRDEObstruction.LOWER_FIELD,
                solution,
                residual,
                lower_field_results=tuple(lower_results),
                logarithmic_derivative_result=logarithmic_result,
            )

        correction = Poly.from_dict(
            {
                (degree,): _coerce(parent, lower_result.solution),
            },
            extension.generator,
            domain=parent,
        )
        new_residual = (
            residual
            - derivation.polynomial(correction)
            - coefficient_polynomial*correction
        )
        if (
            not new_residual.is_zero
            and new_residual.degree() >= residual.degree()
        ):
            raise ValueError(
                "primitive cancellation residual degree did not decrease"
            )
        solution += correction
        residual = new_residual

    if (
        derivation.polynomial(solution)
        + coefficient_polynomial*solution
        != rhs_polynomial
    ):
        raise ValueError("primitive cancellation reconstruction failed")
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        residual,
        lower_field_results=tuple(lower_results),
        logarithmic_derivative_result=logarithmic_result,
    )


def solve_primitive_polynomial_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> PrimitivePolynomialRischDifferentialEquationResult:
    r"""Decide a polynomial RDE at a primitive tower level.

    For ``Dt`` in ``K``, solve

    ``D(y) + a*y = b``, with ``a,b`` in ``K[t]``.

    Positive-degree coefficients use the no-cancellation leading recurrence.
    Lower-field coefficients use Bronstein's cancellation recurrence,
    recursively solving coefficient equations in ``K``. Preservation of
    constants excludes special finite poles, so the polynomial decision also
    covers all rational solutions when both inputs are polynomial.
    """
    extension = _primitive_rde_extension(derivation)
    coefficient_polynomial = _as_primitive_polynomial(
        coefficient,
        extension,
    )
    rhs_polynomial = _as_primitive_polynomial(
        right_hand_side,
        extension,
    )
    if (
        coefficient_polynomial.is_zero
        or coefficient_polynomial.degree() <= 0
    ):
        return _solve_primitive_polynomial_lower_coefficient(
            coefficient_polynomial,
            rhs_polynomial,
            derivation,
        )

    zero = Poly(
        0,
        extension.generator,
        domain=extension.parent_field,
    )
    if rhs_polynomial.is_zero:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            0,
            zero,
            None,
            zero,
            zero,
        )

    degree_bound = (
        rhs_polynomial.degree() - coefficient_polynomial.degree()
    )
    if degree_bound < 0:
        return PrimitivePolynomialRischDifferentialEquationResult(
            coefficient_polynomial,
            rhs_polynomial,
            -1,
            None,
            PrimitivePolynomialRDEObstruction.DEGREE,
            zero,
            rhs_polynomial,
        )

    solution = zero
    residual = rhs_polynomial
    while not residual.is_zero:
        correction_degree = (
            residual.degree() - coefficient_polynomial.degree()
        )
        if correction_degree < 0:
            if (
                derivation.polynomial(solution)
                + coefficient_polynomial*solution
                + residual
                != rhs_polynomial
            ):
                raise ValueError(
                    "primitive polynomial RDE obstruction invariant failed"
                )
            return PrimitivePolynomialRischDifferentialEquationResult(
                coefficient_polynomial,
                rhs_polynomial,
                degree_bound,
                None,
                PrimitivePolynomialRDEObstruction.NONZERO_REMAINDER,
                solution,
                residual,
            )

        correction = Poly.from_dict(
            {
                (correction_degree,): (
                    residual.LC()/coefficient_polynomial.LC()
                )
            },
            extension.generator,
            domain=extension.parent_field,
        )
        new_residual = (
            residual
            - derivation.polynomial(correction)
            - coefficient_polynomial*correction
        )
        if (
            not new_residual.is_zero
            and new_residual.degree() >= residual.degree()
        ):
            raise ValueError(
                "primitive polynomial RDE residual degree did not decrease"
            )
        solution += correction
        residual = new_residual

    if (
        derivation.polynomial(solution)
        + coefficient_polynomial*solution
        != rhs_polynomial
    ):
        raise ValueError("primitive polynomial RDE reconstruction failed")
    return PrimitivePolynomialRischDifferentialEquationResult(
        coefficient_polynomial,
        rhs_polynomial,
        degree_bound,
        solution,
        None,
        solution,
        residual,
    )


@dataclass(frozen=True, slots=True)
class _PrimitiveSPDEResult:
    """Internal numerator result after primitive-field SPDE reduction."""

    solution: Poly | None
    obstruction: PrimitiveRDEObstruction | None
    polynomial_result: (
        PrimitivePolynomialRischDifferentialEquationResult | None
    ) = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_primitive_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
    derivation: Derivation,
) -> _PrimitiveSPDEResult:
    """Apply Rothstein SPDE over a primitive coefficient field."""
    extension = _primitive_rde_extension(derivation)
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = derivative_coefficient.one
    beta = derivative_coefficient.zero
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient
                * derivation.polynomial(beta)
                + original_solution_coefficient*beta
                != original_rhs
            ):
                raise ValueError("primitive SPDE zero-tail reconstruction failed")
            return _PrimitiveSPDEResult(beta, None)
        if current_bound < 0:
            return _PrimitiveSPDEResult(
                None,
                PrimitiveRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _PrimitiveSPDEResult(
                None,
                PrimitiveRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse = (
                extension.parent_field.one
                / extension.parent_field.convert(derivative_coefficient.LC())
            )
            normalized_coefficient = solution_coefficient.mul_ground(inverse)
            normalized_rhs = right_hand_side.mul_ground(inverse)
            polynomial_result = (
                solve_primitive_polynomial_risch_differential_equation(
                    normalized_coefficient,
                    normalized_rhs,
                    derivation,
                )
            )
            if not polynomial_result.has_solution:
                return _PrimitiveSPDEResult(
                    None,
                    PrimitiveRDEObstruction.POLYNOMIAL,
                    polynomial_result,
                )
            tail = polynomial_result.solution
            if tail is None:
                raise ValueError(
                    "a solved primitive polynomial RDE lacks its solution"
                )
            if not tail.is_zero and tail.degree() > current_bound:
                return _PrimitiveSPDEResult(
                    None,
                    PrimitiveRDEObstruction.SPDE_DEGREE,
                    polynomial_result,
                )
            solution = alpha*tail + beta
            if (
                original_derivative_coefficient
                * derivation.polynomial(solution)
                + original_solution_coefficient*solution
                != original_rhs
            ):
                raise ValueError("primitive SPDE reconstruction failed")
            return _PrimitiveSPDEResult(
                solution,
                None,
                polynomial_result,
            )

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side*inverse).rem(derivative_coefficient)
        quotient = (
            right_hand_side - solution_coefficient*residue
        ).exquo(derivative_coefficient)
        right_hand_side = quotient - derivation.polynomial(residue)
        solution_coefficient += derivation.polynomial(
            derivative_coefficient
        )
        current_bound -= derivative_coefficient.degree()
        beta += alpha*residue
        alpha *= derivative_coefficient


def _solve_primitive_rde_exact_case(
    coefficient: FieldElement,
    right_hand_side: FieldElement,
    normalized_coefficient: FieldElement,
    normalized_right_hand_side: FieldElement,
    derivation: Derivation,
    normalization_factor: Poly,
    multiplier: object,
    logarithmic_result: RadicalLogarithmicDerivativeResult | None = None,
) -> PrimitiveRischDifferentialEquationResult:
    """Solve a weakly normalized primitive RDE by exact differentiation."""
    from .exact import solve_exact_derivative

    extension = _primitive_rde_extension(derivation)
    field = extension.field
    one = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    multiplier_element = _coerce(field, multiplier)
    exact_result = solve_exact_derivative(
        multiplier_element*normalized_right_hand_side,
        derivation,
    )
    if not exact_result.has_solution:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient,
            right_hand_side=right_hand_side,
            solution=None,
            obstruction=PrimitiveRDEObstruction.EXACT_DERIVATIVE,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_right_hand_side,
            denominator_bound=one,
            logarithmic_derivative_result=logarithmic_result,
            exact_derivative_result=exact_result,
        )

    normalized_solution = (
        exact_result.antiderivative/multiplier_element
    )
    solution = normalized_solution/normalization_element
    if (
        derivation(solution) + coefficient*solution
        != right_hand_side
    ):
        raise ValueError("primitive exact RDE reconstruction failed")
    return PrimitiveRischDifferentialEquationResult(
        coefficient=coefficient,
        right_hand_side=right_hand_side,
        solution=solution,
        obstruction=None,
        normalization_factor=normalization_factor,
        normalized_coefficient=normalized_coefficient,
        normalized_right_hand_side=normalized_right_hand_side,
        denominator_bound=one,
        logarithmic_derivative_result=logarithmic_result,
        exact_derivative_result=exact_result,
    )


def solve_primitive_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> PrimitiveRischDifferentialEquationResult:
    r"""Decide an RDE with rational coefficient over primitive ``K(t)``.

    This implements the Chapter 6 sequence in Bronstein, *Symbolic
    Integration I*: weak normalization, normal denominator, primitive degree
    bound, Rothstein SPDE, and the polynomial RDE. Polynomial coefficients
    retain the existing no-cancellation and recursive lower-field
    cancellation paths.
    """
    extension = _primitive_rde_extension(derivation)
    field = extension.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_primitive_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    normalized_rhs = normalization_element*rhs_element
    one = Poly(
        1,
        extension.generator,
        domain=extension.parent_field,
    )
    if normalized_rhs == field.zero:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=field.zero,
            obstruction=None,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=one,
            degree_bound=0,
        )

    _, normalized_numerator, normalized_denominator = (
        _as_primitive_fraction(normalized_coefficient, extension)
    )
    normalized_polynomial = None
    if normalized_denominator.degree() == 0:
        inverse = (
            extension.parent_field.one
            / extension.parent_field.convert(normalized_denominator.LC())
        )
        normalized_polynomial = normalized_numerator.mul_ground(inverse)

    coefficient_logarithmic_result = None
    coefficient_is_lower = (
        normalized_polynomial is not None
        and (
            normalized_polynomial.is_zero
            or normalized_polynomial.degree() <= 0
        )
    )
    if coefficient_is_lower:
        coefficient_lower = extension.parent_field.convert(
            normalized_polynomial.nth(0)
        )
        if coefficient_lower == extension.parent_field.zero:
            return _solve_primitive_rde_exact_case(
                coefficient_element,
                rhs_element,
                normalized_coefficient,
                normalized_rhs,
                derivation,
                normalization_factor,
                extension.parent_field.one,
            )

        from .monomial import solve_radical_logarithmic_derivative

        coefficient_logarithmic_result = (
            solve_radical_logarithmic_derivative(
                coefficient_lower,
                derivation.lower(),
            )
        )
        if (
            coefficient_logarithmic_result.has_solution
            and coefficient_logarithmic_result.minimal_power == 1
        ):
            return _solve_primitive_rde_exact_case(
                coefficient_element,
                rhs_element,
                normalized_coefficient,
                normalized_rhs,
                derivation,
                normalization_factor,
                coefficient_logarithmic_result.argument,
                coefficient_logarithmic_result,
            )

    denominator_reduction = _reduce_primitive_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        derivation,
    )
    if not denominator_reduction.succeeded:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=None,
            obstruction=PrimitiveRDEObstruction.DENOMINATOR,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=denominator_reduction.denominator_bound,
            obstruction_polynomial=(
                denominator_reduction.obstruction_polynomial
            ),
            obstruction_remainder=(
                denominator_reduction.obstruction_remainder
            ),
            logarithmic_derivative_result=(
                coefficient_logarithmic_result
            ),
        )

    denominator_bound = denominator_reduction.denominator_bound
    derivative_coefficient = denominator_reduction.derivative_coefficient
    solution_coefficient = denominator_reduction.solution_coefficient
    transformed_rhs = denominator_reduction.right_hand_side
    if (
        derivative_coefficient is None
        or solution_coefficient is None
        or transformed_rhs is None
    ):
        raise ValueError(
            "successful primitive denominator reduction lacks an equation"
        )
    degree_bound = _primitive_rde_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        derivation,
    )
    spde_result = _solve_primitive_spde(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
        derivation,
    )
    if spde_result.solution is None:
        return PrimitiveRischDifferentialEquationResult(
            coefficient=coefficient_element,
            right_hand_side=rhs_element,
            solution=None,
            obstruction=spde_result.obstruction,
            normalization_factor=normalization_factor,
            normalized_coefficient=normalized_coefficient,
            normalized_right_hand_side=normalized_rhs,
            denominator_bound=denominator_bound,
            degree_bound=degree_bound,
            polynomial_result=spde_result.polynomial_result,
            obstruction_polynomial=spde_result.obstruction_polynomial,
            obstruction_remainder=spde_result.obstruction_remainder,
            logarithmic_derivative_result=(
                coefficient_logarithmic_result
            ),
        )

    solution = _coerce(
        field,
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_bound.as_expr()
        ),
    )
    if (
        derivation(solution) + coefficient_element*solution
        != rhs_element
    ):
        raise ValueError("primitive RDE reconstruction failed")
    return PrimitiveRischDifferentialEquationResult(
        coefficient=coefficient_element,
        right_hand_side=rhs_element,
        solution=solution,
        obstruction=None,
        normalization_factor=normalization_factor,
        normalized_coefficient=normalized_coefficient,
        normalized_right_hand_side=normalized_rhs,
        denominator_bound=denominator_bound,
        degree_bound=degree_bound,
        polynomial_result=spde_result.polynomial_result,
        logarithmic_derivative_result=coefficient_logarithmic_result,
    )


def _is_standard_rational_derivation_over(
    derivation: Derivation,
    constants: Domain,
) -> bool:
    """Whether ``derivation`` is ``d/dx`` on ``constants(x)``."""
    if derivation.level != 0:
        return False
    extension = derivation.tower.levels[0]
    return (
        isinstance(extension, MonomialExtension)
        and extension.parent_field == constants
        and extension.generator_derivative
        == Poly(1, extension.generator, domain=constants)
    )


def _is_standard_rational_derivation(derivation: Derivation) -> bool:
    """Whether the derivation is ``d/dx`` on the standard ``QQ(x)`` base."""
    return _is_standard_rational_derivation_over(derivation, QQ)


def _hyperexponential_laurent_rde_extension(
    derivation: Derivation,
) -> MonomialExtension:
    """Validate a hyperexponential top level with unchanged constants."""
    if derivation.level < 1:
        raise TypeError(
            "hyperexponential Laurent RDE solving requires a nonconstant "
            "lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "hyperexponential Laurent RDE solving does not support "
            "algebraic tower levels"
        )
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError(
            "hyperexponential Laurent RDE solving requires Dt = eta*t"
        )
    extension.require_preserved_constants(
        "hyperexponential Laurent RDE solving"
    )
    return extension


def _hyperexponential_laurent_coefficients(
    value: object,
    extension: MonomialExtension,
    role: str,
) -> tuple[FieldElement, dict[int, FieldElement]]:
    """Return the finite Laurent expansion of a top-field element."""
    element, numerator, denominator = _as_primitive_fraction(
        value,
        extension,
    )
    denominator_terms = denominator.terms()
    if len(denominator_terms) != 1:
        raise NotImplementedError(
            f"the hyperexponential Laurent RDE {role} must be a Laurent "
            "polynomial in the top generator"
        )
    (denominator_power,), denominator_coefficient = denominator_terms[0]
    parent = extension.parent_field
    scale = parent.one/parent.convert(denominator_coefficient)
    coefficients = {
        numerator_power - denominator_power: (
            parent.convert(coefficient)*scale
        )
        for (numerator_power,), coefficient in numerator.terms()
        if parent.convert(coefficient) != parent.zero
    }
    return element, coefficients


def _hyperexponential_laurent_term(
    coefficient: FieldElement,
    power: int,
    extension: MonomialExtension,
) -> FieldElement:
    """Embed one lower-field Laurent term in the top field."""
    return _coerce(
        extension.field,
        extension.parent_field.to_sympy(coefficient)
        * extension.generator**power,
    )


def _hyperexponential_boundary_power(
    constant_coefficient: FieldElement,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> tuple[ParametricLogarithmicDerivativeResult, int | None]:
    r"""Decide ``-a_0 = m*eta + D(v)/v`` for an ordinary integer ``m``."""
    from .dependency import solve_parametric_logarithmic_derivative

    result = solve_parametric_logarithmic_derivative(
        -constant_coefficient,
        extension,
        lower_derivation,
    )
    if result.is_unsupported or result.is_inconclusive:
        raise NotImplementedError(
            "hyperexponential Laurent support bounds require a complete "
            "parametric logarithmic-derivative decision in the lower field"
        )
    if result.has_solution and result.n == 1:
        if result.m is None:
            raise ValueError(
                "a solved boundary relation lacks its integer power"
            )
        return result, result.m
    return result, None


@dataclass(frozen=True, slots=True)
class _HyperexponentialSpecialDenominatorReduction:
    """Internal output of ``RdeSpecialDenomExp``."""

    derivative_coefficient: Poly
    solution_coefficient: Poly
    right_hand_side: Poly
    solution_denominator_power: int
    clearing_power: int
    boundary_result: ParametricLogarithmicDerivativeResult | None = None
    exceptional_power: int | None = None


def _hyperexponential_special_denominator(
    derivative_coefficient: object,
    solution_coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> _HyperexponentialSpecialDenominatorReduction:
    r"""Apply Bronstein's hyperexponential special-denominator reduction.

    For ``A*Dq + B*q = C``, all remaining denominators are powers of the
    special polynomial ``t``. The valuation bound ``n <= 0`` implies
    ``r = q*t**(-n)`` is polynomial. A final power ``N`` clears the transformed
    equation

    ``A*Dr + (B + n*A*Dt/t)*r = C*t**(-n)``.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    derivative_polynomial = _as_primitive_polynomial(
        derivative_coefficient,
        extension,
    )
    if derivative_polynomial.eval(parent.zero) == parent.zero:
        raise ValueError(
            "the special-denominator derivative coefficient must be "
            "coprime to the hyperexponential generator"
        )
    solution_element, solution_terms = (
        _hyperexponential_laurent_coefficients(
            solution_coefficient,
            extension,
            "special-denominator solution coefficient",
        )
    )
    rhs_element, rhs_terms = _hyperexponential_laurent_coefficients(
        right_hand_side,
        extension,
        "special-denominator right-hand side",
    )
    if not rhs_terms:
        raise ValueError(
            "special-denominator reduction requires a nonzero right-hand side"
        )

    solution_valuation = min(solution_terms) if solution_terms else None
    rhs_valuation = min(rhs_terms)
    bounded_solution_valuation = min(0, solution_valuation or 0)
    valuation_bound = min(
        0,
        rhs_valuation - bounded_solution_valuation,
    )
    boundary_result = None
    exceptional_power = None
    if solution_valuation == 0:
        alpha = (
            -solution_terms[0]
            / parent.convert(derivative_polynomial.nth(0))
        )
        boundary_result, exceptional_power = (
            _hyperexponential_boundary_power(
                -alpha,
                extension,
                derivation.lower(),
            )
        )
        if exceptional_power is not None:
            valuation_bound = min(valuation_bound, exceptional_power)

    clearing_power = max(
        0,
        -solution_valuation if solution_valuation is not None else 0,
        valuation_bound - rhs_valuation,
    )
    field = extension.field
    t_power = _coerce(field, t**clearing_power)
    eta = parent.convert(extension.generator_derivative.nth(1))
    eta_element = _coerce(field, parent.to_sympy(eta))
    derivative_element = _coerce(
        field,
        derivative_polynomial.as_expr(),
    )
    transformed_derivative = derivative_element*t_power
    transformed_solution = (
        solution_element
        + field.convert(valuation_bound)*derivative_element*eta_element
    )*t_power
    transformed_rhs = (
        rhs_element
        * _coerce(field, t**(clearing_power - valuation_bound))
    )
    derivative_output = _as_primitive_polynomial(
        transformed_derivative,
        extension,
    )
    solution_output = _as_primitive_polynomial(
        transformed_solution,
        extension,
    )
    rhs_output = _as_primitive_polynomial(transformed_rhs, extension)
    return _HyperexponentialSpecialDenominatorReduction(
        derivative_output,
        solution_output,
        rhs_output,
        -valuation_bound,
        clearing_power,
        boundary_result,
        exceptional_power,
    )


def _hyperexponential_rde_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    derivation: Derivation,
) -> tuple[
    int,
    ParametricLogarithmicDerivativeResult | None,
    int | None,
]:
    r"""Implement Bronstein's ``RdeBoundDegreeExp``."""
    if derivative_coefficient.is_zero:
        raise ValueError("the derivative coefficient must be nonzero")
    if right_hand_side.is_zero:
        return 0, None, None

    degree_a = derivative_coefficient.degree()
    degree_b = (
        solution_coefficient.degree()
        if not solution_coefficient.is_zero
        else -1
    )
    degree_c = right_hand_side.degree()
    bound = max(0, degree_c - max(degree_b, degree_a))
    boundary_result = None
    exceptional_power = None
    if degree_a == degree_b:
        parent = derivative_coefficient.domain
        alpha = parent.convert(
            -solution_coefficient.LC()/derivative_coefficient.LC()
        )
        extension = _hyperexponential_laurent_rde_extension(derivation)
        boundary_result, exceptional_power = (
            _hyperexponential_boundary_power(
                -alpha,
                extension,
                derivation.lower(),
            )
        )
        if exceptional_power is not None:
            bound = max(bound, exceptional_power)
    return bound, boundary_result, exceptional_power


@dataclass(frozen=True, slots=True)
class _HyperexponentialSPDEResult:
    """Internal numerator result after hyperexponential SPDE."""

    solution: Poly | None
    obstruction: HyperexponentialRDEObstruction | None
    laurent_result: (
        HyperexponentialLaurentRischDifferentialEquationResult | None
    ) = None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_hyperexponential_spde(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
    derivation: Derivation,
) -> _HyperexponentialSPDEResult:
    """Apply Rothstein SPDE and finish with the complete Laurent solver."""
    extension = _hyperexponential_laurent_rde_extension(derivation)
    parent = extension.parent_field
    original_derivative = derivative_coefficient
    original_solution = solution_coefficient
    original_rhs = right_hand_side
    alpha = derivative_coefficient.one
    beta = derivative_coefficient.zero
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative*derivation.polynomial(beta)
                + original_solution*beta
                != original_rhs
            ):
                raise ValueError(
                    "hyperexponential SPDE zero-tail reconstruction failed"
                )
            return _HyperexponentialSPDEResult(beta, None)
        if current_bound < 0:
            return _HyperexponentialSPDEResult(
                None,
                HyperexponentialRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _HyperexponentialSPDEResult(
                None,
                HyperexponentialRDEObstruction.SPDE_DIVISIBILITY,
                obstruction_polynomial=common_factor,
                obstruction_remainder=remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse = parent.one/parent.convert(derivative_coefficient.LC())
            normalized_coefficient = solution_coefficient.mul_ground(inverse)
            normalized_rhs = right_hand_side.mul_ground(inverse)
            laurent_result = (
                solve_hyperexponential_laurent_risch_differential_equation(
                    normalized_coefficient,
                    normalized_rhs,
                    derivation,
                )
            )
            if not laurent_result.has_solution:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.LAURENT,
                    laurent_result,
                )
            tail_element = laurent_result.solution
            if tail_element is None:
                raise ValueError(
                    "a solved hyperexponential Laurent RDE lacks its solution"
                )
            _, tail_numerator, tail_denominator = _as_primitive_fraction(
                tail_element,
                extension,
            )
            if tail_denominator.degree() > 0:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.SPDE_DEGREE,
                    laurent_result,
                )
            tail = tail_numerator.mul_ground(
                parent.one/parent.convert(tail_denominator.LC())
            )
            if not tail.is_zero and tail.degree() > current_bound:
                return _HyperexponentialSPDEResult(
                    None,
                    HyperexponentialRDEObstruction.SPDE_DEGREE,
                    laurent_result,
                )
            numerator_solution = alpha*tail + beta
            if (
                original_derivative*derivation.polynomial(numerator_solution)
                + original_solution*numerator_solution
                != original_rhs
            ):
                raise ValueError("hyperexponential SPDE reconstruction failed")
            return _HyperexponentialSPDEResult(
                numerator_solution,
                None,
                laurent_result,
            )

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side*inverse).rem(derivative_coefficient)
        quotient = (
            right_hand_side - solution_coefficient*residue
        ).exquo(derivative_coefficient)
        right_hand_side = quotient - derivation.polynomial(residue)
        solution_coefficient += derivation.polynomial(
            derivative_coefficient
        )
        current_bound -= derivative_coefficient.degree()
        beta += alpha*residue
        alpha *= derivative_coefficient


def _hyperexponential_laurent_support_bounds(
    coefficient: dict[int, FieldElement],
    right_hand_side: dict[int, FieldElement],
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> tuple[
    int,
    int,
    ParametricLogarithmicDerivativeResult | None,
    int | None,
]:
    r"""Apply Bronstein's valuations at ``t=0`` and ``t=infinity``.

    This is the standardized ``D(y) + a*y = b`` specialization of Sections
    6.2 and 6.3. The returned interval contains the Laurent support of every
    solution. Equality is forced at an edge where multiplication by a
    nonzero power of ``t`` dominates differentiation.
    """
    if not coefficient or not right_hand_side:
        raise ValueError("Laurent support bounds require nonzero inputs")

    minimum_coefficient_power = min(coefficient)
    maximum_coefficient_power = max(coefficient)
    minimum_rhs_power = min(right_hand_side)
    maximum_rhs_power = max(right_hand_side)
    boundary_result = None
    exceptional_power = None

    if minimum_coefficient_power == 0 or maximum_coefficient_power == 0:
        boundary_result, exceptional_power = (
            _hyperexponential_boundary_power(
                coefficient[0],
                extension,
                lower_derivation,
            )
        )

    if minimum_coefficient_power < 0:
        minimum_solution_power = min(
            0,
            minimum_rhs_power - minimum_coefficient_power,
        )
    elif minimum_coefficient_power > 0:
        minimum_solution_power = min(0, minimum_rhs_power)
    else:
        candidates = [0, minimum_rhs_power]
        if exceptional_power is not None:
            candidates.append(exceptional_power)
        minimum_solution_power = min(candidates)

    if maximum_coefficient_power > 0:
        maximum_solution_power = max(
            0,
            maximum_rhs_power - maximum_coefficient_power,
        )
    elif maximum_coefficient_power < 0:
        maximum_solution_power = max(0, maximum_rhs_power)
    else:
        candidates = [0, maximum_rhs_power]
        if exceptional_power is not None:
            candidates.append(exceptional_power)
        maximum_solution_power = max(candidates)

    return (
        minimum_solution_power,
        maximum_solution_power,
        boundary_result,
        exceptional_power,
    )


def _solve_hyperexponential_diagonal_laurent_rde(
    coefficient_element: FieldElement,
    lower_coefficient: FieldElement,
    rhs_element: FieldElement,
    coefficients: dict[int, FieldElement],
    derivation: Derivation,
    extension: MonomialExtension,
) -> HyperexponentialLaurentRischDifferentialEquationResult:
    """Solve the diagonal ``a in K`` base case mode by mode."""
    field = extension.field
    parent = extension.parent_field
    eta = parent.convert(extension.generator_derivative.nth(1))
    lower_derivation = derivation.lower()
    partial_solution = field.zero
    residual = field.zero
    steps: list[HyperexponentialLaurentRDEStep] = []

    for power in sorted(coefficients):
        rhs_coefficient = coefficients[power]
        shifted_coefficient = (
            lower_coefficient + parent.convert(power)*eta
        )
        lower_result = solve_risch_differential_equation(
            shifted_coefficient,
            rhs_coefficient,
            lower_derivation,
        )
        solution_coefficient = lower_result.solution
        term = _hyperexponential_laurent_term(
            rhs_coefficient,
            power,
            extension,
        )
        if solution_coefficient is None:
            residual += term
        else:
            partial_solution += _hyperexponential_laurent_term(
                solution_coefficient,
                power,
                extension,
            )
        steps.append(
            HyperexponentialLaurentRDEStep(
                power,
                rhs_coefficient,
                shifted_coefficient,
                solution_coefficient,
                lower_result,
            )
        )

    if (
        derivation(partial_solution)
        + coefficient_element*partial_solution
        + residual
        != rhs_element
    ):
        raise ValueError(
            "hyperexponential diagonal RDE decomposition failed to reconstruct"
        )
    has_obstruction = residual != field.zero
    powers = tuple(coefficients)
    return HyperexponentialLaurentRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        eta,
        None if has_obstruction else partial_solution,
        (
            HyperexponentialLaurentRDEObstruction.LOWER_FIELD
            if has_obstruction
            else None
        ),
        partial_solution,
        residual,
        tuple(steps),
        min(powers) if powers else 0,
        max(powers) if powers else 0,
    )


def solve_hyperexponential_laurent_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> HyperexponentialLaurentRischDifferentialEquationResult:
    r"""Decide ``D(y) + a*y = b`` in the Laurent branch of ``K(t)``.

    Both ``a`` and ``b`` may be Laurent polynomials. Bronstein's valuations
    at zero and infinity first bound the support of every solution. If ``a``
    has a positive highest power, its leading product term forces corrections
    downward. Otherwise a negative lowest power forces corrections upward.
    The terminal ``a in K`` case is diagonal and recursively solves

    ``D(y_j) + (a + j*eta)*y_j = b_j``.

    Every exit retains ``partial_solution`` and ``residual`` satisfying

    ``b = D(partial_solution) + a*partial_solution + residual``.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    coefficient_element, coefficient_terms = (
        _hyperexponential_laurent_coefficients(
            coefficient,
            extension,
            "coefficient",
        )
    )
    rhs_element, coefficients = _hyperexponential_laurent_coefficients(
        right_hand_side,
        extension,
        "right-hand side",
    )
    eta = parent.convert(extension.generator_derivative.nth(1))
    if not coefficients:
        return HyperexponentialLaurentRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            eta,
            field.zero,
            None,
            field.zero,
            field.zero,
            (),
            0,
            0,
        )

    if not coefficient_terms or set(coefficient_terms) == {0}:
        return _solve_hyperexponential_diagonal_laurent_rde(
            coefficient_element,
            coefficient_terms.get(0, parent.zero),
            rhs_element,
            coefficients,
            derivation,
            extension,
        )

    (
        minimum_power_bound,
        maximum_power_bound,
        boundary_result,
        exceptional_power,
    ) = _hyperexponential_laurent_support_bounds(
        coefficient_terms,
        coefficients,
        extension,
        derivation.lower(),
    )
    partial_solution = field.zero
    residual = rhs_element
    corrections: list[HyperexponentialLaurentRDECorrection] = []
    if minimum_power_bound > maximum_power_bound:
        raise ValueError(
            "hyperexponential Laurent support bounds are inconsistent"
        )

    maximum_coefficient_power = max(coefficient_terms)
    minimum_coefficient_power = min(coefficient_terms)
    if maximum_coefficient_power > 0:
        direction = HyperexponentialLaurentRDEDirection.DESCENDING
        active_bound = maximum_power_bound
        edge_power = maximum_coefficient_power
        edge_coefficient = coefficient_terms[edge_power]
    else:
        direction = HyperexponentialLaurentRDEDirection.ASCENDING
        active_bound = minimum_power_bound
        edge_power = minimum_coefficient_power
        edge_coefficient = coefficient_terms[edge_power]

    while residual != field.zero:
        _, residual_terms = _hyperexponential_laurent_coefficients(
            residual,
            extension,
            "recurrence residual",
        )
        if not residual_terms:
            raise ValueError("a nonzero Laurent residual has no terms")
        if direction is HyperexponentialLaurentRDEDirection.DESCENDING:
            residual_edge_power = max(residual_terms)
            correction_power = residual_edge_power - edge_power
            inside_support = (
                minimum_power_bound
                <= correction_power
                <= active_bound
            )
        else:
            residual_edge_power = min(residual_terms)
            correction_power = residual_edge_power - edge_power
            inside_support = (
                active_bound
                <= correction_power
                <= maximum_power_bound
            )
        if not inside_support:
            break

        correction_coefficient = (
            residual_terms[residual_edge_power]/edge_coefficient
        )
        correction = _hyperexponential_laurent_term(
            correction_coefficient,
            correction_power,
            extension,
        )
        new_residual = (
            residual
            - derivation(correction)
            - coefficient_element*correction
        )
        if new_residual != field.zero:
            _, new_terms = _hyperexponential_laurent_coefficients(
                new_residual,
                extension,
                "recurrence residual",
            )
            if (
                direction
                is HyperexponentialLaurentRDEDirection.DESCENDING
                and max(new_terms) >= residual_edge_power
            ):
                raise ValueError(
                    "descending Laurent RDE recurrence did not lower degree"
                )
            if (
                direction
                is HyperexponentialLaurentRDEDirection.ASCENDING
                and min(new_terms) <= residual_edge_power
            ):
                raise ValueError(
                    "ascending Laurent RDE recurrence did not raise valuation"
                )
        corrections.append(
            HyperexponentialLaurentRDECorrection(
                correction_power,
                correction_coefficient,
                correction,
                residual,
                new_residual,
            )
        )
        partial_solution += correction
        residual = new_residual
        if direction is HyperexponentialLaurentRDEDirection.DESCENDING:
            active_bound = correction_power - 1
        else:
            active_bound = correction_power + 1

    if (
        derivation(partial_solution)
        + coefficient_element*partial_solution
        + residual
        != rhs_element
    ):
        raise ValueError(
            "hyperexponential Laurent RDE decomposition failed to reconstruct"
        )
    has_obstruction = residual != field.zero
    return HyperexponentialLaurentRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        eta,
        None if has_obstruction else partial_solution,
        (
            HyperexponentialLaurentRDEObstruction.NONZERO_REMAINDER
            if has_obstruction
            else None
        ),
        partial_solution,
        residual,
        (),
        minimum_power_bound,
        maximum_power_bound,
        direction,
        tuple(corrections),
        boundary_result,
        exceptional_power,
    )


def solve_hyperexponential_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> HyperexponentialRischDifferentialEquationResult:
    r"""Decide ``D(y) + a*y = b`` over hyperexponential ``K(t)``.

    This implements the complete Chapter 6 rational frontend for the
    hyperexponential case: general weak normalization, normal-denominator
    reduction, the special ``t``-denominator valuation bound, the
    hyperexponential polynomial degree bound, and Rothstein SPDE. The terminal
    equation is delegated to the complete Laurent-coefficient solver.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_hyperexponential_rde_coefficient(
            coefficient_element,
            derivation,
        )
    )
    normalization_element = _coerce(
        field,
        normalization_factor.as_expr(),
    )
    normalized_rhs = normalization_element*rhs_element
    one = Poly(1, extension.generator, domain=parent)
    if normalized_rhs == field.zero:
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            field.zero,
            None,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            one,
            special_denominator_power=0,
            special_clearing_power=0,
            degree_bound=0,
        )

    denominator_reduction = _reduce_hyperexponential_normal_denominator(
        normalized_coefficient,
        normalized_rhs,
        derivation,
    )
    if not denominator_reduction.succeeded:
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            None,
            HyperexponentialRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            obstruction_polynomial=(
                denominator_reduction.obstruction_polynomial
            ),
            obstruction_remainder=(
                denominator_reduction.obstruction_remainder
            ),
        )
    derivative_element = denominator_reduction.derivative_coefficient
    solution_element = denominator_reduction.solution_coefficient
    transformed_rhs_element = denominator_reduction.right_hand_side
    if (
        derivative_element is None
        or solution_element is None
        or transformed_rhs_element is None
    ):
        raise ValueError(
            "successful hyperexponential denominator reduction lacks "
            "an equation"
        )

    special_reduction = _hyperexponential_special_denominator(
        derivative_element,
        solution_element,
        transformed_rhs_element,
        derivation,
    )
    (
        degree_bound,
        degree_boundary_result,
        degree_exceptional_power,
    ) = _hyperexponential_rde_degree_bound(
        special_reduction.derivative_coefficient,
        special_reduction.solution_coefficient,
        special_reduction.right_hand_side,
        derivation,
    )
    spde_result = _solve_hyperexponential_spde(
        special_reduction.derivative_coefficient,
        special_reduction.solution_coefficient,
        special_reduction.right_hand_side,
        degree_bound,
        derivation,
    )
    if spde_result.solution is None:
        return HyperexponentialRischDifferentialEquationResult(
            coefficient_element,
            rhs_element,
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            normalized_rhs,
            denominator_reduction.denominator_bound,
            special_reduction.solution_denominator_power,
            special_reduction.clearing_power,
            degree_bound,
            spde_result.laurent_result,
            spde_result.obstruction_polynomial,
            spde_result.obstruction_remainder,
            special_reduction.boundary_result,
            special_reduction.exceptional_power,
            degree_boundary_result,
            degree_exceptional_power,
        )

    solution = _coerce(
        field,
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_reduction.denominator_bound.as_expr()
            * extension.generator
            ** special_reduction.solution_denominator_power
        ),
    )
    if (
        derivation(solution) + coefficient_element*solution
        != rhs_element
    ):
        raise ValueError("hyperexponential RDE reconstruction failed")
    return HyperexponentialRischDifferentialEquationResult(
        coefficient_element,
        rhs_element,
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        normalized_rhs,
        denominator_reduction.denominator_bound,
        special_reduction.solution_denominator_power,
        special_reduction.clearing_power,
        degree_bound,
        spde_result.laurent_result,
        special_boundary_result=special_reduction.boundary_result,
        special_exceptional_power=special_reduction.exceptional_power,
        degree_boundary_result=degree_boundary_result,
        degree_exceptional_power=degree_exceptional_power,
    )


def solve_risch_differential_equation(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
) -> RischDifferentialEquationResult:
    r"""Dispatch ``Dy + a*y = b`` over an implemented differential level."""
    field = derivation.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )

    if _is_standard_rational_derivation(derivation):
        extension = derivation.tower.levels[0]
        certificate = solve_rational_risch_differential_equation(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            extension.generator,
        )
        solution = (
            _coerce(field, certificate.solution)
            if certificate.has_solution
            else None
        )
    elif _is_standard_rational_derivation_over(derivation, QQ_I):
        extension = derivation.tower.levels[0]
        domain_result = _solve_rde_over_constant_field(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            extension.generator,
            QQ_I,
        )
        normalized_rhs = cancel(
            domain_result.normalization_factor.as_expr()
            * field.to_sympy(rhs_element)
        )
        certificate = RationalRischDifferentialEquationResult(
            field.to_sympy(coefficient_element),
            field.to_sympy(rhs_element),
            domain_result.solution,
            domain_result.obstruction,
            domain_result.normalization_factor,
            domain_result.normalized_coefficient,
            normalized_rhs,
            domain_result.denominator_bound,
            domain_result.degree_bound,
            obstruction_polynomial=(
                domain_result.obstruction_polynomial
            ),
            obstruction_remainder=(
                domain_result.obstruction_remainder
            ),
        )
        solution = (
            _coerce(field, domain_result.solution)
            if domain_result.solution is not None
            else None
        )
    else:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "RDE dispatch does not yet support algebraic tower levels"
            )
        if extension.family is MonomialFamily.PRIMITIVE:
            certificate = solve_primitive_risch_differential_equation(
                coefficient_element,
                rhs_element,
                derivation,
            )
        elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
            try:
                _hyperexponential_laurent_coefficients(
                    coefficient_element,
                    extension,
                    "coefficient",
                )
                _hyperexponential_laurent_coefficients(
                    rhs_element,
                    extension,
                    "right-hand side",
                )
            except NotImplementedError:
                certificate = (
                    solve_hyperexponential_risch_differential_equation(
                        coefficient_element,
                        rhs_element,
                        derivation,
                    )
                )
            else:
                certificate = (
                    solve_hyperexponential_laurent_risch_differential_equation(
                        coefficient_element,
                        rhs_element,
                        derivation,
                    )
                )
        else:
            raise NotImplementedError(
                f"RDE dispatch does not yet support "
                f"{extension.family.value} monomial levels"
            )
        solution = certificate.solution

    result = RischDifferentialEquationResult(
        derivation,
        coefficient_element,
        rhs_element,
        solution,
        certificate.obstruction,
        certificate,
    )
    if result.has_solution and not result.reconstructs:
        raise ValueError("recursive RDE dispatch reconstruction failed")
    return result


def solve_bounded_rational_rde_matrix(
    coefficient: Expr | Poly,
    right_hand_side: Expr | Poly,
    x: Symbol,
    denominator: Expr | Poly,
    numerator_degree_bound: int,
) -> BoundedRationalRDEMatrixResult:
    r"""Solve a bounded rational RDE ansatz by independent exact linear algebra.

    The candidate is ``y = p/denominator`` with
    ``degree(p) <= numerator_degree_bound``.  This implementation deliberately
    performs no weak normalization, denominator reduction, SPDE, or
    leading-term recurrence and is intended as a differential-test oracle.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    if (
        not isinstance(numerator_degree_bound, int)
        or isinstance(numerator_degree_bound, bool)
        or numerator_degree_bound < 0
    ):
        raise ValueError("the numerator degree bound must be a nonnegative int")

    coefficient, _, _ = _as_qq_rational(coefficient, x)
    right_hand_side, _, _ = _as_qq_rational(right_hand_side, x)
    denominator = Poly(denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the ansatz denominator must be nonzero")
    denominator = denominator.monic()

    column_expressions = []
    for degree in range(numerator_degree_bound + 1):
        candidate = x**degree/denominator.as_expr()
        column_expressions.append(
            cancel(candidate.diff(x) + coefficient*candidate)
        )

    rational_expressions = [*column_expressions, right_hand_side]
    reduced_expressions = [
        _as_qq_rational(expression, x)
        for expression in rational_expressions
    ]
    common_denominator = Poly(1, x, domain=QQ)
    for _, _, expression_denominator in reduced_expressions:
        common_denominator = common_denominator.lcm(expression_denominator)

    cleared_polynomials = [
        Poly(
            cancel(expression*common_denominator.as_expr()),
            x,
            domain=QQ,
        )
        for expression in rational_expressions
    ]
    column_polynomials = cleared_polynomials[:-1]
    rhs_polynomial = cleared_polynomials[-1]
    equation_degree = max(
        (
            polynomial.degree()
            for polynomial in cleared_polynomials
            if not polynomial.is_zero
        ),
        default=0,
    )
    equation_count = equation_degree + 1
    unknown_count = numerator_degree_bound + 1
    system = DomainMatrix(
        [
            [
                QQ.convert(polynomial.nth(degree))
                for polynomial in column_polynomials
            ]
            for degree in range(equation_count)
        ],
        (equation_count, unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        [
            [QQ.convert(rhs_polynomial.nth(degree))]
            for degree in range(equation_count)
        ],
        (equation_count, 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedRationalRDEMatrixResult(
            denominator,
            numerator_degree_bound,
            None,
            coefficient_rank,
            augmented_rank,
        )

    reduced_matrix, pivots = augmented.rref()
    reduced_rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = reduced_rows[row][unknown_count]
    numerator = Poly.from_dict(
        {
            (degree,): value
            for degree, value in enumerate(coefficients)
        },
        x,
        domain=QQ,
    )
    solution = cancel(numerator.as_expr()/denominator.as_expr())
    if cancel(
        solution.diff(x)
        + coefficient*solution
        - right_hand_side
    ) != 0:
        raise ValueError("bounded matrix RDE reconstruction failed")
    return BoundedRationalRDEMatrixResult(
        denominator,
        numerator_degree_bound,
        solution,
        coefficient_rank,
        augmented_rank,
    )


def _as_qq_bivariate_fraction(
    expression: Expr,
    lower_generator: Symbol,
    top_generator: Symbol,
) -> tuple[Poly, Poly]:
    """Return a reduced fraction in ``QQ[lower_generator, top_generator]``."""
    numerator_expression, denominator_expression = cancel(
        expression
    ).as_numer_denom()
    numerator = Poly(
        numerator_expression,
        lower_generator,
        top_generator,
        domain=QQ,
    )
    denominator = Poly(
        denominator_expression,
        lower_generator,
        top_generator,
        domain=QQ,
    )
    coefficient, numerator, denominator = numerator.cancel(denominator)
    numerator = numerator.mul_ground(QQ.convert(coefficient))
    leading_inverse = QQ.one/denominator.LC()
    return (
        numerator.mul_ground(leading_inverse),
        denominator.monic(),
    )


def solve_bounded_primitive_rde_matrix(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    top_degree_bound: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedPrimitiveRDEMatrixResult:
    r"""Solve a bounded primitive-field RDE by global exact linear algebra.

    This independent oracle is limited to one primitive level over ``QQ(x)``.
    It searches

    ``sum(c_ij*x**i*t**j/(lower_denominator*top_denominator))``

    over the caller-supplied rectangular degree bounds. It shares no
    denominator theorem, SPDE, or leading recurrence with the production
    primitive solver.
    """
    extension = _primitive_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded primitive matrix oracle requires QQ(x) below the "
            "primitive level"
        )
    for name, bound in (
        ("lower numerator", lower_numerator_degree_bound),
        ("top", top_degree_bound),
    ):
        if (
            not isinstance(bound, int)
            or isinstance(bound, bool)
            or bound < 0
        ):
            raise ValueError(f"the {name} degree bound must be nonnegative")

    lower_extension = derivation.tower.levels[lower_derivation.level]
    if not isinstance(lower_extension, MonomialExtension):
        raise TypeError("the rational lower level is not monomial")
    x = lower_extension.generator
    t = extension.generator
    denominator = Poly(lower_denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the lower ansatz denominator must be nonzero")
    denominator = denominator.monic()
    top_denominator_polynomial = Poly(
        top_denominator,
        t,
        domain=extension.parent_field,
    )
    if top_denominator_polynomial.is_zero:
        raise ValueError("the top ansatz denominator must be nonzero")
    top_denominator_polynomial = top_denominator_polynomial.monic()

    field = extension.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    basis = [
        _coerce(
            field,
            x**lower_degree*t**top_degree
            / (
                denominator.as_expr()
                * top_denominator_polynomial.as_expr()
            ),
        )
        for top_degree in range(top_degree_bound + 1)
        for lower_degree in range(lower_numerator_degree_bound + 1)
    ]
    columns = [
        derivation(candidate) + coefficient_element*candidate
        for candidate in basis
    ]
    fractions = [
        _as_qq_bivariate_fraction(field.to_sympy(element), x, t)
        for element in [*columns, rhs_element]
    ]
    common_denominator = Poly(1, x, t, domain=QQ)
    for _, element_denominator in fractions:
        common_denominator = common_denominator.lcm(element_denominator)
    cleared = [
        numerator*common_denominator.exquo(element_denominator)
        for numerator, element_denominator in fractions
    ]
    column_polynomials = cleared[:-1]
    rhs_polynomial = cleared[-1]
    monomials = sorted(
        {
            monomial
            for polynomial in cleared
            for monomial, _ in polynomial.terms()
        },
        reverse=True,
    )
    if not monomials:
        monomials = [(0, 0)]

    unknown_count = len(basis)
    coefficient_rows = [
        [
            QQ.convert(polynomial.nth(*monomial))
            for polynomial in column_polynomials
        ]
        for monomial in monomials
    ]
    rhs_rows = [
        [QQ.convert(rhs_polynomial.nth(*monomial))]
        for monomial in monomials
    ]
    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        rhs_rows,
        (len(rhs_rows), 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedPrimitiveRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            top_degree_bound,
            None,
            coefficient_rank,
            augmented_rank,
            top_denominator_polynomial,
        )

    reduced_matrix, pivots = augmented.rref()
    rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = rows[row][unknown_count]
    solution = field.zero
    for value, candidate in zip(coefficients, basis, strict=True):
        solution += field.convert(value)*candidate
    if derivation(solution) + coefficient_element*solution != rhs_element:
        raise ValueError("bounded primitive matrix RDE reconstruction failed")
    return BoundedPrimitiveRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        top_degree_bound,
        solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )


def solve_bounded_primitive_coupled_rde_matrix(
    lambda_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    top_degree_bound: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedPrimitiveCoupledRDEMatrixResult:
    r"""Solve a bounded coupled ansatz by direct real block linear algebra.

    This independent oracle is limited to one primitive level over ``QQ(x)``.
    It constructs both real equations directly and never rebases the tower to
    ``QQ_I`` or invokes a scalar RDE solver.
    """
    extension = _primitive_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded primitive coupled oracle requires QQ(x) below "
            "the primitive level"
        )
    for name, bound in (
        ("lower numerator", lower_numerator_degree_bound),
        ("top", top_degree_bound),
    ):
        if (
            not isinstance(bound, int)
            or isinstance(bound, bool)
            or bound < 0
        ):
            raise ValueError(f"the {name} degree bound must be nonnegative")

    lower_extension = derivation.tower.levels[0]
    if not isinstance(lower_extension, MonomialExtension):
        raise TypeError("the rational lower level is not monomial")
    x = lower_extension.generator
    t = extension.generator
    denominator = Poly(lower_denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the lower ansatz denominator must be nonzero")
    denominator = denominator.monic()
    top_denominator_polynomial = Poly(
        top_denominator,
        t,
        domain=extension.parent_field,
    )
    if top_denominator_polynomial.is_zero:
        raise ValueError("the top ansatz denominator must be nonzero")
    top_denominator_polynomial = top_denominator_polynomial.monic()

    field = extension.field
    lambda_element = _coerce(field, lambda_coefficient)
    if lambda_element == field.zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs = _coerce(field, real_right_hand_side)
    imaginary_rhs = _coerce(field, imaginary_right_hand_side)
    basis = [
        _coerce(
            field,
            x**lower_degree*t**top_degree
            / (
                denominator.as_expr()
                * top_denominator_polynomial.as_expr()
            ),
        )
        for top_degree in range(top_degree_bound + 1)
        for lower_degree in range(lower_numerator_degree_bound + 1)
    ]
    basis_derivatives = [derivation(candidate) for candidate in basis]
    real_columns = [
        *basis_derivatives,
        *(-lambda_element*candidate for candidate in basis),
    ]
    imaginary_columns = [
        *(lambda_element*candidate for candidate in basis),
        *basis_derivatives,
    ]
    all_elements = [
        *real_columns,
        real_rhs,
        *imaginary_columns,
        imaginary_rhs,
    ]
    fractions = [
        _as_qq_bivariate_fraction(field.to_sympy(element), x, t)
        for element in all_elements
    ]
    common_denominator = Poly(1, x, t, domain=QQ)
    for _, element_denominator in fractions:
        common_denominator = common_denominator.lcm(element_denominator)
    cleared = [
        numerator*common_denominator.exquo(element_denominator)
        for numerator, element_denominator in fractions
    ]
    unknown_count = 2*len(basis)
    real_columns_polynomial = cleared[:unknown_count]
    real_rhs_polynomial = cleared[unknown_count]
    imaginary_start = unknown_count + 1
    imaginary_columns_polynomial = cleared[
        imaginary_start:imaginary_start + unknown_count
    ]
    imaginary_rhs_polynomial = cleared[-1]
    all_polynomials = [
        *real_columns_polynomial,
        real_rhs_polynomial,
        *imaginary_columns_polynomial,
        imaginary_rhs_polynomial,
    ]
    monomials = sorted(
        {
            monomial
            for polynomial in all_polynomials
            for monomial, _ in polynomial.terms()
        },
        reverse=True,
    )
    if not monomials:
        monomials = [(0, 0)]

    coefficient_rows = [
        [
            QQ.convert(polynomial.nth(*monomial))
            for polynomial in columns
        ]
        for columns in (
            real_columns_polynomial,
            imaginary_columns_polynomial,
        )
        for monomial in monomials
    ]
    rhs_rows = [
        [QQ.convert(polynomial.nth(*monomial))]
        for polynomial in (
            real_rhs_polynomial,
            imaginary_rhs_polynomial,
        )
        for monomial in monomials
    ]
    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(rhs_rows, (len(rhs_rows), 1), QQ)
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedPrimitiveCoupledRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            top_degree_bound,
            None,
            None,
            coefficient_rank,
            augmented_rank,
            top_denominator_polynomial,
        )

    reduced_matrix, pivots = augmented.rref()
    rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = rows[row][unknown_count]
    real_solution = field.zero
    imaginary_solution = field.zero
    basis_count = len(basis)
    for value, candidate in zip(
        coefficients[:basis_count],
        basis,
        strict=True,
    ):
        real_solution += field.convert(value)*candidate
    for value, candidate in zip(
        coefficients[basis_count:],
        basis,
        strict=True,
    ):
        imaginary_solution += field.convert(value)*candidate
    if (
        derivation(real_solution)
        - lambda_element*imaginary_solution
        != real_rhs
        or derivation(imaginary_solution)
        + lambda_element*real_solution
        != imaginary_rhs
    ):
        raise ValueError(
            "bounded primitive coupled matrix reconstruction failed"
        )
    return BoundedPrimitiveCoupledRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        top_degree_bound,
        real_solution,
        imaginary_solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )


def solve_bounded_hyperexponential_laurent_rde_matrix(
    coefficient: object,
    right_hand_side: object,
    derivation: Derivation,
    lower_denominator: Expr | Poly,
    lower_numerator_degree_bound: int,
    minimum_power: int,
    maximum_power: int,
    top_denominator: Expr | Poly = S.One,
) -> BoundedHyperexponentialLaurentRDEMatrixResult:
    r"""Solve a bounded rational RDE ansatz by exact global linear algebra.

    This independent oracle is limited to one hyperexponential level over
    ``QQ(x)``.  It searches

    ``sum(c_ij*x**i*t**j/(lower_denominator*top_denominator))``

    for ``minimum_power <= j <= maximum_power``.  It deliberately does not
    perform weak normalization, denominator reduction, special-denominator
    reduction, SPDE, or lower-field RDE solving.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    lower_derivation = derivation.lower()
    if not _is_standard_rational_derivation(lower_derivation):
        raise NotImplementedError(
            "the bounded hyperexponential Laurent matrix oracle requires "
            "QQ(x) below the hyperexponential level"
        )
    if (
        not isinstance(lower_numerator_degree_bound, int)
        or isinstance(lower_numerator_degree_bound, bool)
        or lower_numerator_degree_bound < 0
    ):
        raise ValueError(
            "the lower numerator degree bound must be a nonnegative int"
        )
    for name, power in (
        ("minimum", minimum_power),
        ("maximum", maximum_power),
    ):
        if not isinstance(power, int) or isinstance(power, bool):
            raise ValueError(f"the {name} power must be an int")
    if minimum_power > maximum_power:
        raise ValueError("the minimum power must not exceed the maximum power")

    lower_extension = derivation.tower.levels[lower_derivation.level]
    if not isinstance(lower_extension, MonomialExtension):
        raise TypeError("the rational lower level is not monomial")
    x = lower_extension.generator
    t = extension.generator
    denominator = Poly(lower_denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the lower ansatz denominator must be nonzero")
    denominator = denominator.monic()
    top_denominator_polynomial = Poly(
        top_denominator,
        t,
        domain=extension.parent_field,
    )
    if top_denominator_polynomial.is_zero:
        raise ValueError("the top ansatz denominator must be nonzero")
    top_denominator_polynomial = top_denominator_polynomial.monic()

    field = extension.field
    coefficient_element = _coerce(
        field,
        coefficient.as_expr() if isinstance(coefficient, Poly) else coefficient,
    )
    rhs_element = _coerce(
        field,
        (
            right_hand_side.as_expr()
            if isinstance(right_hand_side, Poly)
            else right_hand_side
        ),
    )
    basis = [
        _coerce(
            field,
            x**lower_degree*t**top_power
            / (
                denominator.as_expr()
                * top_denominator_polynomial.as_expr()
            ),
        )
        for top_power in range(minimum_power, maximum_power + 1)
        for lower_degree in range(lower_numerator_degree_bound + 1)
    ]
    columns = [
        derivation(candidate) + coefficient_element*candidate
        for candidate in basis
    ]
    fractions = [
        _as_qq_bivariate_fraction(field.to_sympy(element), x, t)
        for element in [*columns, rhs_element]
    ]
    common_denominator = Poly(1, x, t, domain=QQ)
    for _, element_denominator in fractions:
        common_denominator = common_denominator.lcm(element_denominator)
    cleared = [
        numerator*common_denominator.exquo(element_denominator)
        for numerator, element_denominator in fractions
    ]
    column_polynomials = cleared[:-1]
    rhs_polynomial = cleared[-1]
    monomials = sorted(
        {
            monomial
            for polynomial in cleared
            for monomial, _ in polynomial.terms()
        },
        reverse=True,
    )
    if not monomials:
        monomials = [(0, 0)]

    unknown_count = len(basis)
    coefficient_rows = [
        [
            QQ.convert(polynomial.nth(*monomial))
            for polynomial in column_polynomials
        ]
        for monomial in monomials
    ]
    rhs_rows = [
        [QQ.convert(rhs_polynomial.nth(*monomial))]
        for monomial in monomials
    ]
    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        rhs_rows,
        (len(rhs_rows), 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedHyperexponentialLaurentRDEMatrixResult(
            denominator,
            lower_numerator_degree_bound,
            minimum_power,
            maximum_power,
            None,
            coefficient_rank,
            augmented_rank,
            top_denominator_polynomial,
        )

    reduced_matrix, pivots = augmented.rref()
    rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = rows[row][unknown_count]
    solution = field.zero
    for value, candidate in zip(coefficients, basis, strict=True):
        solution += field.convert(value)*candidate
    if derivation(solution) + coefficient_element*solution != rhs_element:
        raise ValueError(
            "bounded hyperexponential Laurent matrix RDE reconstruction failed"
        )
    return BoundedHyperexponentialLaurentRDEMatrixResult(
        denominator,
        lower_numerator_degree_bound,
        minimum_power,
        maximum_power,
        solution,
        coefficient_rank,
        augmented_rank,
        top_denominator_polynomial,
    )


def _maximum_polynomial_degree(*polynomials: Poly) -> int | None:
    """Return the largest degree among nonzero polynomials, if one exists."""
    degrees = [
        polynomial.degree()
        for polynomial in polynomials
        if not polynomial.is_zero
    ]
    return max(degrees) if degrees else None


def _as_domain_rational(
    value: Expr | Poly,
    x: Symbol,
    domain: Domain,
) -> tuple[Expr, Poly, Poly]:
    """Return a reduced rational function over an exact constant field."""
    if isinstance(value, Poly):
        value = value.as_expr()
    field = domain.frac_field(x)
    element = field.from_sympy(value)
    numerator_polynomial = Poly.from_dict(
        element.numer.to_dict(),
        x,
        domain=domain,
    )
    denominator_polynomial = Poly.from_dict(
        element.denom.to_dict(),
        x,
        domain=domain,
    )
    expression = field.to_sympy(element)
    return expression, numerator_polynomial, denominator_polynomial


def _integer_constant(value: object, domain: Domain) -> int | None:
    """Return an exact integer constant, or ``None`` otherwise."""
    expression = domain.to_sympy(domain.convert(value))
    if expression.is_Integer is not True:
        return None
    return int(expression)


def _weakly_normalize_over_constant_field(
    coefficient: Expr,
    x: Symbol,
    domain: Domain,
) -> tuple[Poly, Expr]:
    r"""Apply Bronstein weak normalization over ``domain(x)``.

    For ``D=d/dx``, an irreducible factor ``p`` occurring simply in the
    denominator has logarithmic residue

    ``numerator / ((denominator/p)*Dp) mod p``.

    Every positive integer residue ``n`` is removed by subtracting
    ``D(p**n)/p**n``.  Higher-order poles need no adjustment.
    """
    coefficient, numerator, denominator = _as_domain_rational(
        coefficient,
        x,
        domain,
    )
    normalizer = Poly(1, x, domain=domain)
    for factor, multiplicity in denominator.factor_list()[1]:
        if multiplicity != 1:
            continue
        cofactor = denominator.exquo(factor)
        local_denominator = (cofactor*factor.diff()).rem(factor)
        residue = (
            numerator*invert(local_denominator, factor)
        ).rem(factor)
        if residue.degree() > 0:
            continue
        integer_residue = _integer_constant(residue.nth(0), domain)
        if integer_residue is not None and integer_residue > 0:
            normalizer *= factor**integer_residue

    normalized = cancel(
        coefficient - normalizer.diff().as_expr()/normalizer.as_expr()
    )
    if cancel(
        normalized
        + normalizer.diff().as_expr()/normalizer.as_expr()
        - coefficient
    ) != 0:
        raise ValueError("constant-field weak normalization failed")
    return normalizer.monic(), normalized


@dataclass(frozen=True, slots=True)
class _DomainNormalDenominatorReduction:
    denominator_bound: Poly
    derivative_coefficient: Poly | None
    solution_coefficient: Poly | None
    right_hand_side: Poly | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None

    @property
    def succeeded(self) -> bool:
        return self.derivative_coefficient is not None

    @property
    def equation(self) -> tuple[Poly, Poly, Poly] | None:
        """Return the transformed polynomial equation when reduction succeeds."""
        if (
            self.derivative_coefficient is None
            or self.solution_coefficient is None
            or self.right_hand_side is None
        ):
            return None
        return (
            self.derivative_coefficient,
            self.solution_coefficient,
            self.right_hand_side,
        )


def _reduce_normal_denominator_over_constant_field(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
    domain: Domain,
) -> _DomainNormalDenominatorReduction:
    """Apply Bronstein's normal-denominator theorem over ``domain(x)``."""
    _, coefficient_numerator, coefficient_denominator = _as_domain_rational(
        coefficient,
        x,
        domain,
    )
    _, rhs_numerator, rhs_denominator = _as_domain_rational(
        right_hand_side,
        x,
        domain,
    )
    shared_denominator = coefficient_denominator.gcd(rhs_denominator)
    denominator_bound = (
        rhs_denominator.gcd(rhs_denominator.diff())
        .exquo(
            shared_denominator.gcd(shared_denominator.diff())
        )
        .monic()
    )
    derivative_coefficient = coefficient_denominator*denominator_bound
    solution_coefficient = (
        denominator_bound*coefficient_numerator
        - coefficient_denominator*denominator_bound.diff()
    )
    multiplier = (
        coefficient_denominator
        * denominator_bound
        * denominator_bound
    )
    quotient, remainder = multiplier.div(rhs_denominator)
    if not remainder.is_zero:
        return _DomainNormalDenominatorReduction(
            denominator_bound,
            None,
            None,
            None,
            rhs_denominator,
            remainder,
        )
    return _DomainNormalDenominatorReduction(
        denominator_bound,
        derivative_coefficient,
        solution_coefficient,
        quotient*rhs_numerator,
    )


def _domain_polynomial_degree_bound(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
) -> int:
    """Bound solutions of ``A*Dq+B*q=C`` over an exact constant field."""
    if right_hand_side.is_zero:
        return 0
    derivative_degree = derivative_coefficient.degree()
    rhs_degree = right_hand_side.degree()
    if solution_coefficient.is_zero:
        dominant_degree = derivative_degree - 1
    else:
        dominant_degree = max(
            solution_coefficient.degree(),
            derivative_degree - 1,
        )
    bound = max(0, rhs_degree - dominant_degree)
    if (
        not solution_coefficient.is_zero
        and solution_coefficient.degree() == derivative_degree - 1
    ):
        domain = derivative_coefficient.domain
        cancellation_degree = _integer_constant(
            domain.exquo(
                -domain.convert(solution_coefficient.LC()),
                domain.convert(derivative_coefficient.LC()),
            ),
            domain,
        )
        if cancellation_degree is not None:
            bound = max(
                bound,
                cancellation_degree,
                rhs_degree - solution_coefficient.degree(),
            )
    return bound


@dataclass(frozen=True, slots=True)
class _DomainPolynomialRDEResult:
    solution: Poly | None
    degree_bound: int
    residual: Poly | None = None


def _solve_polynomial_rde_over_constant_field(
    coefficient: Poly,
    right_hand_side: Poly,
) -> _DomainPolynomialRDEResult:
    """Solve ``Dq+a*q=b`` by the forced leading recurrence."""
    x = coefficient.gens[0]
    domain = coefficient.domain
    zero = Poly(0, x, domain=domain)
    if right_hand_side.is_zero:
        return _DomainPolynomialRDEResult(zero, 0)
    if coefficient.is_zero:
        solution = Poly.from_dict(
            {
                (degree + 1,): domain.exquo(
                    domain.convert(value),
                    domain.convert(degree + 1),
                )
                for (degree,), value in right_hand_side.terms()
            },
            x,
            domain=domain,
        )
        return _DomainPolynomialRDEResult(
            solution,
            right_hand_side.degree() + 1,
        )

    degree_bound = right_hand_side.degree() - coefficient.degree()
    if degree_bound < 0:
        return _DomainPolynomialRDEResult(
            None,
            -1,
            right_hand_side,
        )
    solution = zero
    residual = right_hand_side
    while not residual.is_zero:
        candidate_degree = residual.degree() - coefficient.degree()
        if candidate_degree < 0:
            return _DomainPolynomialRDEResult(
                None,
                degree_bound,
                residual,
            )
        term = Poly.from_dict(
            {
                (candidate_degree,): domain.exquo(
                    domain.convert(residual.LC()),
                    domain.convert(coefficient.LC()),
                ),
            },
            x,
            domain=domain,
        )
        new_residual = residual - term.diff() - coefficient*term
        if (
            not new_residual.is_zero
            and new_residual.degree() >= residual.degree()
        ):
            raise ValueError(
                "constant-field polynomial RDE degree did not decrease"
            )
        solution += term
        residual = new_residual
    if solution.diff() + coefficient*solution != right_hand_side:
        raise ValueError(
            "constant-field polynomial RDE reconstruction failed"
        )
    return _DomainPolynomialRDEResult(solution, degree_bound)


@dataclass(frozen=True, slots=True)
class _DomainSPDEResult:
    solution: Poly | None
    obstruction: RationalRDEObstruction | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_spde_over_constant_field(
    derivative_coefficient: Poly,
    solution_coefficient: Poly,
    right_hand_side: Poly,
    degree_bound: int,
) -> _DomainSPDEResult:
    """Apply Rothstein SPDE reduction over an exact constant field."""
    x = derivative_coefficient.gens[0]
    domain = derivative_coefficient.domain
    original_derivative_coefficient = derivative_coefficient
    original_solution_coefficient = solution_coefficient
    original_rhs = right_hand_side
    alpha = Poly(1, x, domain=domain)
    beta = Poly(0, x, domain=domain)
    current_bound = degree_bound

    while True:
        if right_hand_side.is_zero:
            if (
                original_derivative_coefficient*beta.diff()
                + original_solution_coefficient*beta
                != original_rhs
            ):
                raise ValueError(
                    "constant-field SPDE zero-tail reconstruction failed"
                )
            return _DomainSPDEResult(beta, None)
        if current_bound < 0:
            return _DomainSPDEResult(
                None,
                RationalRDEObstruction.SPDE_DEGREE,
            )

        common_factor = derivative_coefficient.gcd(solution_coefficient)
        quotient, remainder = right_hand_side.div(common_factor)
        if not remainder.is_zero:
            return _DomainSPDEResult(
                None,
                RationalRDEObstruction.SPDE_DIVISIBILITY,
                common_factor,
                remainder,
            )
        derivative_coefficient = derivative_coefficient.exquo(common_factor)
        solution_coefficient = solution_coefficient.exquo(common_factor)
        right_hand_side = quotient

        if derivative_coefficient.degree() == 0:
            inverse_leading_coefficient = domain.exquo(
                domain.one,
                domain.convert(derivative_coefficient.LC()),
            )
            normalized_coefficient = solution_coefficient.mul_ground(
                inverse_leading_coefficient
            )
            normalized_rhs = right_hand_side.mul_ground(
                inverse_leading_coefficient
            )
            polynomial_result = _solve_polynomial_rde_over_constant_field(
                normalized_coefficient,
                normalized_rhs,
            )
            if polynomial_result.solution is None:
                return _DomainSPDEResult(
                    None,
                    RationalRDEObstruction.POLYNOMIAL,
                    normalized_coefficient,
                    polynomial_result.residual,
                )
            tail = polynomial_result.solution
            if not tail.is_zero and tail.degree() > current_bound:
                return _DomainSPDEResult(
                    None,
                    RationalRDEObstruction.SPDE_DEGREE,
                )
            solution = alpha*tail + beta
            if (
                original_derivative_coefficient*solution.diff()
                + original_solution_coefficient*solution
                != original_rhs
            ):
                raise ValueError(
                    "constant-field SPDE reconstruction failed"
                )
            return _DomainSPDEResult(solution, None)

        inverse = invert(solution_coefficient, derivative_coefficient)
        residue = (right_hand_side*inverse).rem(
            derivative_coefficient
        )
        quotient = (
            right_hand_side - solution_coefficient*residue
        ).exquo(derivative_coefficient)
        right_hand_side = quotient - residue.diff()
        solution_coefficient += derivative_coefficient.diff()
        current_bound -= derivative_coefficient.degree()
        beta += alpha*residue
        alpha *= derivative_coefficient


@dataclass(frozen=True, slots=True)
class _DomainRDEResult:
    solution: Expr | None
    obstruction: RationalRDEObstruction | None
    normalization_factor: Poly
    normalized_coefficient: Expr
    denominator_bound: Poly
    degree_bound: int | None
    obstruction_polynomial: Poly | None = None
    obstruction_remainder: Poly | None = None


def _solve_rde_over_constant_field(
    coefficient: Expr,
    right_hand_side: Expr,
    x: Symbol,
    domain: Domain,
) -> _DomainRDEResult:
    """Solve a scalar rational RDE over ``domain(x)``."""
    coefficient, _, _ = _as_domain_rational(coefficient, x, domain)
    right_hand_side, _, _ = _as_domain_rational(
        right_hand_side,
        x,
        domain,
    )
    normalization_factor, normalized_coefficient = (
        _weakly_normalize_over_constant_field(
            coefficient,
            x,
            domain,
        )
    )
    normalized_rhs = cancel(
        normalization_factor.as_expr()*right_hand_side
    )
    if normalized_rhs == 0:
        return _DomainRDEResult(
            S.Zero,
            None,
            normalization_factor,
            normalized_coefficient,
            Poly(1, x, domain=domain),
            0,
        )

    denominator_reduction = (
        _reduce_normal_denominator_over_constant_field(
            normalized_coefficient,
            normalized_rhs,
            x,
            domain,
        )
    )
    equation = denominator_reduction.equation
    if equation is None:
        return _DomainRDEResult(
            None,
            RationalRDEObstruction.DENOMINATOR,
            normalization_factor,
            normalized_coefficient,
            denominator_reduction.denominator_bound,
            None,
            denominator_reduction.obstruction_polynomial,
            denominator_reduction.obstruction_remainder,
        )
    (
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    ) = equation
    degree_bound = _domain_polynomial_degree_bound(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
    )
    spde_result = _solve_spde_over_constant_field(
        derivative_coefficient,
        solution_coefficient,
        transformed_rhs,
        degree_bound,
    )
    if spde_result.solution is None:
        return _DomainRDEResult(
            None,
            spde_result.obstruction,
            normalization_factor,
            normalized_coefficient,
            denominator_reduction.denominator_bound,
            degree_bound,
            spde_result.obstruction_polynomial,
            spde_result.obstruction_remainder,
        )
    solution = cancel(
        spde_result.solution.as_expr()
        / (
            normalization_factor.as_expr()
            * denominator_reduction.denominator_bound.as_expr()
        )
    )
    if cancel(
        solution.diff(x)
        + coefficient*solution
        - right_hand_side
    ) != 0:
        raise ValueError(
            "constant-field rational RDE reconstruction failed"
        )
    return _DomainRDEResult(
        solution,
        None,
        normalization_factor,
        normalized_coefficient,
        denominator_reduction.denominator_bound,
        degree_bound,
    )


def _conjugate_gaussian_polynomial(polynomial: Poly) -> Poly:
    """Conjugate coefficients of a polynomial over ``QQ_I``."""
    x = polynomial.gens[0]
    return Poly(
        polynomial.as_expr().xreplace({I: -I}),
        x,
        domain=QQ_I,
    )


def _real_candidate_denominator(
    normalization_factor: Poly,
    denominator_bound: Poly,
) -> tuple[Poly, int]:
    """Return the least real common denominator and numerator-degree shift."""
    x = normalization_factor.gens[0]
    gaussian_denominator = normalization_factor*denominator_bound
    conjugate_denominator = _conjugate_gaussian_polynomial(
        gaussian_denominator
    )
    real_common = gaussian_denominator.lcm(
        conjugate_denominator
    ).monic()
    real_coefficients = {}
    for monomial, coefficient in real_common.terms():
        coefficient = QQ_I.convert(coefficient)
        if coefficient.y != 0:
            raise ValueError(
                "Gaussian common denominator unexpectedly has "
                "nonreal coefficients"
            )
        real_coefficients[monomial] = QQ.convert(coefficient.x)
    real_denominator = Poly.from_dict(
        real_coefficients,
        x,
        domain=QQ,
    ).monic()
    numerator_degree_shift = (
        real_common.degree() - gaussian_denominator.degree()
    )
    return real_denominator, numerator_degree_shift


def _split_gaussian_rational(
    value: Expr,
    x: Symbol,
) -> tuple[Expr, Expr]:
    """Split a member of ``QQ_I(x)`` into canonical real rational parts."""
    conjugate = cancel(value.xreplace({I: -I}))
    real_part = cancel((value + conjugate)/2)
    imaginary_part = cancel((value - conjugate)/(2*I))
    real_part, _, _ = _as_qq_rational(real_part, x)
    imaginary_part, _, _ = _as_qq_rational(imaginary_part, x)
    return real_part, imaginary_part


def _solve_coupled_rational_rotation(
    lambda_coefficient: Expr,
    real_rhs: Expr,
    imaginary_rhs: Expr,
    x: Symbol,
) -> CoupledRationalRDEResult:
    """Solve a rational rotation through one Gaussian scalar RDE."""
    scalar_result = _solve_rde_over_constant_field(
        I*lambda_coefficient,
        real_rhs + I*imaginary_rhs,
        x,
        QQ_I,
    )
    if real_rhs == 0 and imaginary_rhs == 0:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            S.Zero,
            S.Zero,
            None,
            Poly(1, x, domain=QQ),
            0,
            gaussian_normalization_factor=(
                scalar_result.normalization_factor
            ),
            gaussian_normalized_coefficient=(
                scalar_result.normalized_coefficient
            ),
            gaussian_denominator_bound=(
                scalar_result.denominator_bound
            ),
        )
    real_denominator, numerator_degree_shift = _real_candidate_denominator(
        scalar_result.normalization_factor,
        scalar_result.denominator_bound,
    )
    real_degree_bound = (
        None
        if scalar_result.degree_bound is None
        else scalar_result.degree_bound + numerator_degree_shift
    )
    obstruction_map = {
        RationalRDEObstruction.DENOMINATOR: (
            CoupledRDEObstruction.DENOMINATOR
        ),
        RationalRDEObstruction.SPDE_DEGREE: CoupledRDEObstruction.DEGREE,
        RationalRDEObstruction.SPDE_DIVISIBILITY: (
            CoupledRDEObstruction.NONZERO_REMAINDER
        ),
        RationalRDEObstruction.POLYNOMIAL: (
            CoupledRDEObstruction.NONZERO_REMAINDER
        ),
    }
    if scalar_result.solution is None:
        if scalar_result.obstruction is None:
            raise ValueError("a failed scalar RDE lacks its obstruction")
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            obstruction_map[scalar_result.obstruction],
            real_denominator,
            real_degree_bound,
            gaussian_normalization_factor=(
                scalar_result.normalization_factor
            ),
            gaussian_normalized_coefficient=(
                scalar_result.normalized_coefficient
            ),
            gaussian_denominator_bound=(
                scalar_result.denominator_bound
            ),
            gaussian_obstruction=scalar_result.obstruction,
            gaussian_obstruction_polynomial=(
                scalar_result.obstruction_polynomial
            ),
            gaussian_obstruction_remainder=(
                scalar_result.obstruction_remainder
            ),
        )

    real_solution, imaginary_solution = _split_gaussian_rational(
        scalar_result.solution,
        x,
    )
    if (
        cancel(
            real_solution.diff(x)
            - lambda_coefficient*imaginary_solution
            - real_rhs
        )
        != 0
        or cancel(
            imaginary_solution.diff(x)
            + lambda_coefficient*real_solution
            - imaginary_rhs
        )
        != 0
    ):
        raise ValueError(
            "rational-rotation coupled RDE reconstruction failed"
        )
    return CoupledRationalRDEResult(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        real_solution,
        imaginary_solution,
        None,
        real_denominator,
        real_degree_bound,
        gaussian_normalization_factor=scalar_result.normalization_factor,
        gaussian_normalized_coefficient=(
            scalar_result.normalized_coefficient
        ),
        gaussian_denominator_bound=scalar_result.denominator_bound,
    )


def solve_coupled_rational_rde(
    lambda_coefficient: Expr | Poly,
    real_right_hand_side: Expr | Poly,
    imaginary_right_hand_side: Expr | Poly,
    x: Symbol,
) -> CoupledRationalRDEResult:
    r"""Solve a real coupled RDE with rational rotation coefficient.

    This decides

    ``Dp - lambda*q = u`` and ``Dq + lambda*p = v``

    in ``QQ(x)`` for nonzero ``lambda``.  Polynomial ``lambda`` uses the
    direct real denominator theorem and two-coordinate leading recurrence.
    Rational ``lambda`` is transformed to

    ``Dz + i*lambda*z = u + i*v``

    over ``QQ_I(x)``.  Bronstein weak normalization, normal-denominator
    reduction, SPDE, and a forced polynomial recurrence decide that scalar
    equation, after which its solution is split exactly back into ``p,q``.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    lambda_expression, lambda_numerator, lambda_denominator = (
        _as_qq_rational(lambda_coefficient, x)
    )
    if lambda_numerator.is_zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs, real_numerator, real_denominator = _as_qq_rational(
        real_right_hand_side,
        x,
    )
    imaginary_rhs, imaginary_numerator, imaginary_denominator = (
        _as_qq_rational(imaginary_right_hand_side, x)
    )
    if lambda_denominator.degree() > 0:
        return _solve_coupled_rational_rotation(
            lambda_expression,
            real_rhs,
            imaginary_rhs,
            x,
        )
    lambda_coefficient = Poly(lambda_expression, x, domain=QQ)

    common_denominator = real_denominator.lcm(imaginary_denominator)
    common_real_numerator = (
        real_numerator*common_denominator.exquo(real_denominator)
    )
    common_imaginary_numerator = (
        imaginary_numerator
        * common_denominator.exquo(imaginary_denominator)
    )
    # At a finite pole, differentiation raises the order while multiplication
    # by polynomial lambda does not.  A right-hand-side pole of order e
    # therefore forces a solution pole of order e - 1.  For a common
    # denominator D, h = gcd(D, D') has precisely those exponents.  The test
    # D | h**2 rejects a simple pole before any coefficient solving.
    denominator_bound = common_denominator.gcd(
        common_denominator.diff()
    ).monic()
    quotient, denominator_remainder = (
        denominator_bound*denominator_bound
    ).div(common_denominator)
    if not denominator_remainder.is_zero:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            CoupledRDEObstruction.DENOMINATOR,
            denominator_bound,
            obstruction_polynomial=common_denominator,
            obstruction_remainder=denominator_remainder,
        )

    # Write p=P/h and q=Q/h and clear h**2:
    #
    # h*P' - h'*P - lambda*h*Q = U
    # h*Q' - h'*Q + lambda*h*P = V.
    transformed_real_rhs = quotient*common_real_numerator
    transformed_imaginary_rhs = quotient*common_imaginary_numerator
    rotation = lambda_coefficient*denominator_bound
    residual_degree = _maximum_polynomial_degree(
        transformed_real_rhs,
        transformed_imaginary_rhs,
    )
    if residual_degree is None:
        zero = S.Zero
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            zero,
            zero,
            None,
            denominator_bound,
            0,
        )

    degree_bound = residual_degree - rotation.degree()
    real_numerator_solution = Poly(0, x, domain=QQ)
    imaginary_numerator_solution = Poly(0, x, domain=QQ)
    real_residual = transformed_real_rhs
    imaginary_residual = transformed_imaginary_rhs
    if degree_bound < 0:
        return CoupledRationalRDEResult(
            lambda_coefficient,
            real_rhs,
            imaginary_rhs,
            None,
            None,
            CoupledRDEObstruction.DEGREE,
            denominator_bound,
            degree_bound,
            real_numerator_solution,
            imaginary_numerator_solution,
            real_residual,
            imaginary_residual,
        )

    current_bound = degree_bound
    while not (real_residual.is_zero and imaginary_residual.is_zero):
        residual_degree = _maximum_polynomial_degree(
            real_residual,
            imaginary_residual,
        )
        if residual_degree is None:
            raise ValueError("a nonzero coupled residual has no degree")
        candidate_degree = residual_degree - rotation.degree()
        if candidate_degree < 0 or candidate_degree > current_bound:
            if (
                denominator_bound*real_numerator_solution.diff()
                - denominator_bound.diff()*real_numerator_solution
                - rotation*imaginary_numerator_solution
                + real_residual
                != transformed_real_rhs
                or denominator_bound*imaginary_numerator_solution.diff()
                - denominator_bound.diff()*imaginary_numerator_solution
                + rotation*real_numerator_solution
                + imaginary_residual
                != transformed_imaginary_rhs
            ):
                raise ValueError("coupled RDE reduction invariant failed")
            return CoupledRationalRDEResult(
                lambda_coefficient,
                real_rhs,
                imaginary_rhs,
                None,
                None,
                CoupledRDEObstruction.NONZERO_REMAINDER,
                denominator_bound,
                degree_bound,
                real_numerator_solution,
                imaginary_numerator_solution,
                real_residual,
                imaginary_residual,
            )

        real_leading_residual = real_residual.nth(residual_degree)
        imaginary_leading_residual = imaginary_residual.nth(
            residual_degree
        )
        # The rotation terms have one degree more than the derivative terms
        # (at least), so their invertible 2-by-2 leading matrix uniquely fixes
        # the next coefficients of P and Q.
        real_term = Poly.from_dict(
            {
                (candidate_degree,): (
                    imaginary_leading_residual/rotation.LC()
                )
            },
            x,
            domain=QQ,
        )
        imaginary_term = Poly.from_dict(
            {
                (candidate_degree,): (
                    -real_leading_residual/rotation.LC()
                )
            },
            x,
            domain=QQ,
        )
        new_real_residual = (
            real_residual
            - denominator_bound*real_term.diff()
            + denominator_bound.diff()*real_term
            + rotation*imaginary_term
        )
        new_imaginary_residual = (
            imaginary_residual
            - denominator_bound*imaginary_term.diff()
            + denominator_bound.diff()*imaginary_term
            - rotation*real_term
        )
        new_degree = _maximum_polynomial_degree(
            new_real_residual,
            new_imaginary_residual,
        )
        if new_degree is not None and new_degree >= residual_degree:
            raise ValueError("coupled RDE residual degree did not decrease")

        real_numerator_solution += real_term
        imaginary_numerator_solution += imaginary_term
        real_residual = new_real_residual
        imaginary_residual = new_imaginary_residual
        current_bound = candidate_degree - 1

    real_solution = cancel(
        real_numerator_solution.as_expr()/denominator_bound.as_expr()
    )
    imaginary_solution = cancel(
        imaginary_numerator_solution.as_expr()
        / denominator_bound.as_expr()
    )
    lambda_expression = lambda_coefficient.as_expr()
    if (
        cancel(
            real_solution.diff(x)
            - lambda_expression*imaginary_solution
            - real_rhs
        )
        != 0
        or cancel(
            imaginary_solution.diff(x)
            + lambda_expression*real_solution
            - imaginary_rhs
        )
        != 0
    ):
        raise ValueError("coupled rational RDE reconstruction failed")
    return CoupledRationalRDEResult(
        lambda_coefficient,
        real_rhs,
        imaginary_rhs,
        real_solution,
        imaginary_solution,
        None,
        denominator_bound,
        degree_bound,
    )


def solve_hyperexponential_coupled_laurent_rde(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledHyperexponentialLaurentRDEResult:
    r"""Solve a coupled Laurent RDE directly at a hyperexponential level.

    This is the modewise form of Bronstein's ``CoupledDECancelExp``. If
    ``Dt = eta*t`` and both matrix coefficients belong to the lower field,
    the coefficient pair at power ``m`` satisfies a lower coupled system with
    diagonal coefficient ``a + m*eta``. No Gaussian tower is constructed by
    this recurrence.
    """
    extension = _hyperexponential_laurent_rde_extension(derivation)
    field = extension.field
    parent = extension.parent_field
    diagonal_element, diagonal_terms = (
        _hyperexponential_laurent_coefficients(
            diagonal_coefficient,
            extension,
            "diagonal coefficient",
        )
    )
    rotation_element, rotation_terms = (
        _hyperexponential_laurent_coefficients(
            rotation_coefficient,
            extension,
            "rotation coefficient",
        )
    )
    if any(power != 0 for power in diagonal_terms):
        raise NotImplementedError(
            "direct hyperexponential coupled cancellation requires the "
            "diagonal coefficient to belong to the lower field"
        )
    if any(power != 0 for power in rotation_terms):
        raise NotImplementedError(
            "direct hyperexponential coupled cancellation requires the "
            "rotation coefficient to belong to the lower field"
        )
    real_rhs, real_terms = _hyperexponential_laurent_coefficients(
        real_right_hand_side,
        extension,
        "real right-hand side",
    )
    imaginary_rhs, imaginary_terms = (
        _hyperexponential_laurent_coefficients(
            imaginary_right_hand_side,
            extension,
            "imaginary right-hand side",
        )
    )
    diagonal_lower = diagonal_terms.get(0, parent.zero)
    rotation_lower = rotation_terms.get(0, parent.zero)
    eta = parent.convert(extension.generator_derivative.nth(1))
    lower_derivation = derivation.lower()
    partial_real = field.zero
    partial_imaginary = field.zero
    real_residual = field.zero
    imaginary_residual = field.zero
    steps: list[CoupledHyperexponentialLaurentRDEStep] = []

    for power in sorted(set(real_terms) | set(imaginary_terms)):
        real_coefficient = real_terms.get(power, parent.zero)
        imaginary_coefficient = imaginary_terms.get(power, parent.zero)
        shifted_diagonal = (
            diagonal_lower + parent.convert(power)*eta
        )
        lower_result = _solve_general_coupled_risch_differential_equation(
            shifted_diagonal,
            rotation_lower,
            real_coefficient,
            imaginary_coefficient,
            lower_derivation,
        )
        steps.append(
            CoupledHyperexponentialLaurentRDEStep(
                power,
                real_coefficient,
                imaginary_coefficient,
                shifted_diagonal,
                lower_result,
            )
        )
        if lower_result.has_solution:
            if (
                lower_result.real_solution is None
                or lower_result.imaginary_solution is None
            ):
                raise ValueError(
                    "a solved lower coupled RDE lacks its solution pair"
                )
            partial_real += _hyperexponential_laurent_term(
                lower_result.real_solution,
                power,
                extension,
            )
            partial_imaginary += _hyperexponential_laurent_term(
                lower_result.imaginary_solution,
                power,
                extension,
            )
        else:
            real_residual += _hyperexponential_laurent_term(
                real_coefficient,
                power,
                extension,
            )
            imaginary_residual += _hyperexponential_laurent_term(
                imaginary_coefficient,
                power,
                extension,
            )

    has_obstruction = (
        real_residual != field.zero
        or imaginary_residual != field.zero
    )
    result = CoupledHyperexponentialLaurentRDEResult(
        derivation,
        diagonal_element,
        rotation_element,
        real_rhs,
        imaginary_rhs,
        None if has_obstruction else partial_real,
        None if has_obstruction else partial_imaginary,
        partial_real,
        partial_imaginary,
        real_residual,
        imaginary_residual,
        tuple(steps),
        CoupledRDEObstruction.LOWER_FIELD if has_obstruction else None,
    )
    if not result.reconstructs:
        raise ValueError(
            "direct hyperexponential coupled recurrence failed to reconstruct"
        )
    return result


def _hypertangent_extension(derivation: Derivation) -> MonomialExtension:
    """Validate a hypertangent top level for coupled polynomial solving."""
    if derivation.level < 1:
        raise TypeError(
            "hypertangent coupled RDE solving requires a nonconstant "
            "lower field"
        )
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise NotImplementedError(
            "hypertangent coupled RDE solving does not support algebraic "
            "tower levels"
        )
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent coupled RDE solving requires Dt = eta*(1+t**2)"
        )
    extension.require_preserved_constants(
        "hypertangent coupled RDE solving"
    )
    return extension


def _polynomial_at_i_coordinates(
    polynomial: Poly,
) -> tuple[FieldElement, FieldElement]:
    """Return the two real coordinates of ``polynomial(i)``."""
    domain = polynomial.domain
    real = domain.zero
    imaginary = domain.zero
    for (power,), coefficient in polynomial.terms():
        value = domain.convert(coefficient)
        phase = power % 4
        if phase == 0:
            real += value
        elif phase == 1:
            imaginary += value
        elif phase == 2:
            real -= value
        else:
            imaginary -= value
    return real, imaginary


def _zero_hypertangent_coupled_result(
    derivation: Derivation,
    mode: CoupledHypertangentPolynomialRDEMode,
    diagonal_coefficient: Poly,
    rotation_coefficient: FieldElement,
    real_right_hand_side: Poly,
    imaginary_right_hand_side: Poly,
    degree_bound: int,
    obstruction: CoupledRDEObstruction,
    *,
    boundary_result: CoupledRischDifferentialEquationResult | None = None,
    recursive_result: CoupledHypertangentPolynomialRDEResult | None = None,
) -> CoupledHypertangentPolynomialRDEResult:
    """Build a reconstructing failure certificate with zero partial solution."""
    zero = real_right_hand_side.zero
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        mode,
        diagonal_coefficient,
        rotation_coefficient,
        real_right_hand_side,
        imaginary_right_hand_side,
        degree_bound,
        None,
        None,
        zero,
        zero,
        real_right_hand_side,
        imaginary_right_hand_side,
        obstruction,
        boundary_result=boundary_result,
        recursive_result=recursive_result,
    )
    if not result.reconstructs:
        raise ValueError(
            "hypertangent coupled failure certificate did not reconstruct"
        )
    return result


def solve_hypertangent_coupled_cancellation_rde(
    base_diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
    degree_bound: int,
) -> CoupledHypertangentPolynomialRDEResult:
    r"""Implement Bronstein's ``CoupledDECancelTan`` recurrence.

    With ``Dt = eta*(1+t**2)``, this decides polynomial solutions of degree at
    most ``n`` for the cancellation-shaped system whose diagonal is
    ``b0 - n*eta*t`` and whose rotation coefficient is ``b2`` in the lower
    field. Evaluation at the place ``t = i`` creates one lower coupled
    system. Exact division by ``t - i`` is performed in real coordinates
    using ``t**2 + 1`` before recurring with bound ``n - 1``.
    """
    if (
        not isinstance(degree_bound, int)
        or isinstance(degree_bound, bool)
        or degree_bound < 0
    ):
        raise ValueError("the degree bound must be a nonnegative int")
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    eta = parent.convert(extension.generator_derivative.nth(2))
    base_diagonal = _coerce(parent, base_diagonal_coefficient)
    rotation = _coerce(parent, rotation_coefficient)
    real_rhs = _as_primitive_polynomial(
        real_right_hand_side,
        extension,
    )
    imaginary_rhs = _as_primitive_polynomial(
        imaginary_right_hand_side,
        extension,
    )
    diagonal = Poly(
        parent.to_sympy(base_diagonal)
        - degree_bound*parent.to_sympy(eta)*t,
        t,
        domain=parent,
    )
    maximum_rhs_degree = _maximum_polynomial_degree(real_rhs, imaginary_rhs)
    if (
        maximum_rhs_degree is not None
        and maximum_rhs_degree > degree_bound
    ):
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            CoupledRDEObstruction.DEGREE,
        )

    if degree_bound == 0:
        boundary_result = _solve_general_coupled_risch_differential_equation(
            base_diagonal,
            rotation,
            parent.convert(real_rhs.nth(0)),
            parent.convert(imaginary_rhs.nth(0)),
            derivation.lower(),
        )
        if not boundary_result.has_solution:
            return _zero_hypertangent_coupled_result(
                derivation,
                CoupledHypertangentPolynomialRDEMode.CANCELLATION,
                diagonal,
                rotation,
                real_rhs,
                imaginary_rhs,
                degree_bound,
                CoupledRDEObstruction.LOWER_FIELD,
                boundary_result=boundary_result,
            )
        if (
            boundary_result.real_solution is None
            or boundary_result.imaginary_solution is None
        ):
            raise ValueError(
                "a solved tangent boundary system lacks its solution pair"
            )
        real_solution = Poly(
            parent.to_sympy(boundary_result.real_solution),
            t,
            domain=parent,
        )
        imaginary_solution = Poly(
            parent.to_sympy(boundary_result.imaginary_solution),
            t,
            domain=parent,
        )
        zero = real_rhs.zero
        result = CoupledHypertangentPolynomialRDEResult(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            real_solution,
            imaginary_solution,
            real_solution,
            imaginary_solution,
            zero,
            zero,
            None,
            boundary_result=boundary_result,
        )
        if not result.reconstructs:
            raise ValueError(
                "tangent cancellation base case failed to reconstruct"
            )
        return result

    real_at_i, real_at_i_imaginary = _polynomial_at_i_coordinates(
        real_rhs
    )
    imaginary_at_i_real, imaginary_at_i = _polynomial_at_i_coordinates(
        imaginary_rhs
    )
    boundary_real = real_at_i - imaginary_at_i
    boundary_imaginary = (
        real_at_i_imaginary + imaginary_at_i_real
    )
    shifted_rotation = rotation - parent.convert(degree_bound)*eta
    boundary_result = _solve_general_coupled_risch_differential_equation(
        base_diagonal,
        shifted_rotation,
        boundary_real,
        boundary_imaginary,
        derivation.lower(),
    )
    if not boundary_result.has_solution:
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            CoupledRDEObstruction.LOWER_FIELD,
            boundary_result=boundary_result,
        )
    if (
        boundary_result.real_solution is None
        or boundary_result.imaginary_solution is None
    ):
        raise ValueError(
            "a solved tangent boundary system lacks its solution pair"
        )
    s1 = parent.convert(boundary_result.real_solution)
    s2 = parent.convert(boundary_result.imaginary_solution)
    n_eta = parent.convert(degree_bound)*eta
    real_numerator = (
        real_rhs
        - Poly(parent.to_sympy(boundary_real), t, domain=parent)
        + Poly(
            parent.to_sympy(n_eta)
            * (parent.to_sympy(s1)*t + parent.to_sympy(s2)),
            t,
            domain=parent,
        )
    )
    imaginary_numerator = (
        imaginary_rhs
        - Poly(parent.to_sympy(boundary_imaginary), t, domain=parent)
        + Poly(
            parent.to_sympy(n_eta)
            * (parent.to_sympy(s2)*t - parent.to_sympy(s1)),
            t,
            domain=parent,
        )
    )
    top_generator = Poly(t, t, domain=parent)
    special = Poly(t**2 + 1, t, domain=parent)
    reduced_real, real_remainder = (
        top_generator*real_numerator - imaginary_numerator
    ).div(special)
    reduced_imaginary, imaginary_remainder = (
        real_numerator + top_generator*imaginary_numerator
    ).div(special)
    if not real_remainder.is_zero or not imaginary_remainder.is_zero:
        raise ValueError(
            "tangent cancellation numerator is not divisible by t - i"
        )

    recursive_result = solve_hypertangent_coupled_cancellation_rde(
        base_diagonal,
        rotation + eta,
        reduced_real,
        reduced_imaginary,
        derivation,
        degree_bound - 1,
    )
    if not recursive_result.has_solution:
        return _zero_hypertangent_coupled_result(
            derivation,
            CoupledHypertangentPolynomialRDEMode.CANCELLATION,
            diagonal,
            rotation,
            real_rhs,
            imaginary_rhs,
            degree_bound,
            recursive_result.obstruction or CoupledRDEObstruction.LOWER_FIELD,
            boundary_result=boundary_result,
            recursive_result=recursive_result,
        )
    if (
        recursive_result.real_solution is None
        or recursive_result.imaginary_solution is None
    ):
        raise ValueError(
            "a solved recursive tangent system lacks its solution pair"
        )
    real_solution = (
        recursive_result.real_solution*top_generator
        + recursive_result.imaginary_solution
        + Poly(parent.to_sympy(s1), t, domain=parent)
    )
    imaginary_solution = (
        recursive_result.imaginary_solution*top_generator
        - recursive_result.real_solution
        + Poly(parent.to_sympy(s2), t, domain=parent)
    )
    if (
        real_solution.degree() > degree_bound
        or imaginary_solution.degree() > degree_bound
    ):
        raise ValueError(
            "tangent cancellation recurrence exceeded its degree bound"
        )
    zero = real_rhs.zero
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        CoupledHypertangentPolynomialRDEMode.CANCELLATION,
        diagonal,
        rotation,
        real_rhs,
        imaginary_rhs,
        degree_bound,
        real_solution,
        imaginary_solution,
        real_solution,
        imaginary_solution,
        zero,
        zero,
        None,
        boundary_result=boundary_result,
        recursive_result=recursive_result,
    )
    if not result.reconstructs:
        raise ValueError(
            "tangent cancellation recurrence failed to reconstruct"
        )
    return result


def _solve_hypertangent_coupled_no_cancellation(
    diagonal_coefficient: FieldElement,
    rotation_coefficient: FieldElement,
    real_right_hand_side: Poly,
    imaginary_right_hand_side: Poly,
    derivation: Derivation,
) -> CoupledHypertangentPolynomialRDEResult:
    """Solve the lower-coefficient nonlinear polynomial branch."""
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    t = extension.generator
    eta = parent.convert(extension.generator_derivative.nth(2))
    diagonal = Poly(
        parent.to_sympy(diagonal_coefficient),
        t,
        domain=parent,
    )
    zero = real_right_hand_side.zero
    real_solution = zero
    imaginary_solution = zero
    real_residual = real_right_hand_side
    imaginary_residual = imaginary_right_hand_side
    corrections: list[CoupledHypertangentPolynomialRDECorrection] = []
    maximum_degree = _maximum_polynomial_degree(
        real_right_hand_side,
        imaginary_right_hand_side,
    )
    degree_bound = max(0, (maximum_degree or 0) - 1)

    while True:
        residual_degree = _maximum_polynomial_degree(
            real_residual,
            imaginary_residual,
        )
        if residual_degree is None or residual_degree <= 0:
            break
        correction_degree = residual_degree - 1
        if correction_degree <= 0:
            result = CoupledHypertangentPolynomialRDEResult(
                derivation,
                CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
                diagonal,
                rotation_coefficient,
                real_right_hand_side,
                imaginary_right_hand_side,
                degree_bound,
                None,
                None,
                real_solution,
                imaginary_solution,
                real_residual,
                imaginary_residual,
                CoupledRDEObstruction.DEGREE,
                tuple(corrections),
            )
            if not result.reconstructs:
                raise ValueError(
                    "tangent no-cancellation obstruction did not reconstruct"
                )
            return result
        divisor = parent.convert(correction_degree)*eta
        real_coefficient = (
            parent.convert(real_residual.nth(residual_degree))/divisor
        )
        imaginary_coefficient = (
            parent.convert(imaginary_residual.nth(residual_degree))/divisor
        )
        real_correction = Poly.from_dict(
            {(correction_degree,): real_coefficient},
            t,
            domain=parent,
        )
        imaginary_correction = Poly.from_dict(
            {(correction_degree,): imaginary_coefficient},
            t,
            domain=parent,
        )
        new_real_residual = (
            real_residual
            - derivation.polynomial(real_correction)
            - diagonal*real_correction
            + imaginary_correction.mul_ground(rotation_coefficient)
        )
        new_imaginary_residual = (
            imaginary_residual
            - derivation.polynomial(imaginary_correction)
            - real_correction.mul_ground(rotation_coefficient)
            - diagonal*imaginary_correction
        )
        new_degree = _maximum_polynomial_degree(
            new_real_residual,
            new_imaginary_residual,
        )
        if new_degree is not None and new_degree >= residual_degree:
            raise ValueError(
                "tangent no-cancellation recurrence did not lower degree"
            )
        corrections.append(
            CoupledHypertangentPolynomialRDECorrection(
                correction_degree,
                real_coefficient,
                imaginary_coefficient,
                real_residual,
                imaginary_residual,
                new_real_residual,
                new_imaginary_residual,
            )
        )
        real_solution += real_correction
        imaginary_solution += imaginary_correction
        real_residual = new_real_residual
        imaginary_residual = new_imaginary_residual

    lower_result = _solve_general_coupled_risch_differential_equation(
        diagonal_coefficient,
        rotation_coefficient,
        parent.convert(real_residual.nth(0)),
        parent.convert(imaginary_residual.nth(0)),
        derivation.lower(),
    )
    if not lower_result.has_solution:
        result = CoupledHypertangentPolynomialRDEResult(
            derivation,
            CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
            diagonal,
            rotation_coefficient,
            real_right_hand_side,
            imaginary_right_hand_side,
            degree_bound,
            None,
            None,
            real_solution,
            imaginary_solution,
            real_residual,
            imaginary_residual,
            CoupledRDEObstruction.LOWER_FIELD,
            tuple(corrections),
            boundary_result=lower_result,
        )
        if not result.reconstructs:
            raise ValueError(
                "tangent lower-field obstruction did not reconstruct"
            )
        return result
    if (
        lower_result.real_solution is None
        or lower_result.imaginary_solution is None
    ):
        raise ValueError(
            "a solved tangent constant system lacks its solution pair"
        )
    real_solution += Poly(
        parent.to_sympy(lower_result.real_solution),
        t,
        domain=parent,
    )
    imaginary_solution += Poly(
        parent.to_sympy(lower_result.imaginary_solution),
        t,
        domain=parent,
    )
    result = CoupledHypertangentPolynomialRDEResult(
        derivation,
        CoupledHypertangentPolynomialRDEMode.NON_CANCELLATION,
        diagonal,
        rotation_coefficient,
        real_right_hand_side,
        imaginary_right_hand_side,
        degree_bound,
        real_solution,
        imaginary_solution,
        real_solution,
        imaginary_solution,
        zero,
        zero,
        None,
        tuple(corrections),
        boundary_result=lower_result,
    )
    if not result.reconstructs:
        raise ValueError(
            "tangent no-cancellation recurrence failed to reconstruct"
        )
    return result


def solve_hypertangent_coupled_polynomial_rde(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledHypertangentPolynomialRDEResult:
    r"""Decide implemented polynomial coupled branches for ``Dt=eta(1+t²)``.

    Lower-field matrix coefficients use the forced nonlinear leading
    recurrence. A diagonal of the form ``b0 - n*eta*t`` with positive integer
    ``n`` dispatches to Bronstein's direct cancellation recurrence.
    """
    extension = _hypertangent_extension(derivation)
    parent = extension.parent_field
    diagonal = _as_primitive_polynomial(
        diagonal_coefficient,
        extension,
    )
    rotation = _as_primitive_polynomial(
        rotation_coefficient,
        extension,
    )
    real_rhs = _as_primitive_polynomial(
        real_right_hand_side,
        extension,
    )
    imaginary_rhs = _as_primitive_polynomial(
        imaginary_right_hand_side,
        extension,
    )
    if not rotation.is_zero and rotation.degree() > 0:
        raise NotImplementedError(
            "hypertangent polynomial coupled solving currently requires "
            "the rotation coefficient to belong to the lower field"
        )
    rotation_lower = parent.convert(rotation.nth(0))
    if diagonal.is_zero or diagonal.degree() <= 0:
        return _solve_hypertangent_coupled_no_cancellation(
            parent.convert(diagonal.nth(0)),
            rotation_lower,
            real_rhs,
            imaginary_rhs,
            derivation,
        )
    if diagonal.degree() != 1:
        raise NotImplementedError(
            "the hypertangent diagonal coefficient is outside the "
            "implemented cancellation shapes"
        )
    eta = parent.convert(extension.generator_derivative.nth(2))
    cancellation_degree = _positive_integer_in_domain(
        -parent.convert(diagonal.nth(1))/eta,
        parent,
    )
    if cancellation_degree is None:
        raise NotImplementedError(
            "the hypertangent diagonal coefficient is not b0 - n*eta*t "
            "for a positive integer n"
        )
    return solve_hypertangent_coupled_cancellation_rde(
        parent.convert(diagonal.nth(0)),
        rotation_lower,
        real_rhs,
        imaginary_rhs,
        derivation,
        cancellation_degree,
    )


def _coupled_obstruction_from_scalar(
    obstruction: (
        RationalRDEObstruction
        | PrimitiveRDEObstruction
        | HyperexponentialRDEObstruction
        | HyperexponentialLaurentRDEObstruction
    ),
) -> CoupledRDEObstruction:
    """Map a scalar Gaussian RDE obstruction to the real coupled contract."""
    if obstruction in {
        RationalRDEObstruction.DENOMINATOR,
        PrimitiveRDEObstruction.DENOMINATOR,
        HyperexponentialRDEObstruction.DENOMINATOR,
    }:
        return CoupledRDEObstruction.DENOMINATOR
    if obstruction in {
        RationalRDEObstruction.SPDE_DEGREE,
        PrimitiveRDEObstruction.SPDE_DEGREE,
        HyperexponentialRDEObstruction.SPDE_DEGREE,
    }:
        return CoupledRDEObstruction.DEGREE
    return CoupledRDEObstruction.NONZERO_REMAINDER


def _require_supported_coupled_tower(
    derivation: Derivation,
) -> None:
    """Validate the monomial families understood by coupled recursion."""
    if not _is_standard_rational_derivation(
        Derivation(derivation.tower, 0)
    ):
        raise NotImplementedError(
            "recursive coupled RDEs require the standard QQ(x) base"
        )
    for level in range(1, derivation.level + 1):
        extension = derivation.tower.levels[level]
        if (
            not isinstance(extension, MonomialExtension)
            or extension.family
            not in {
                MonomialFamily.PRIMITIVE,
                MonomialFamily.HYPEREXPONENTIAL,
                MonomialFamily.HYPERTANGENT,
            }
        ):
            raise NotImplementedError(
                "recursive coupled RDEs currently support primitive, "
                "hyperexponential, and hypertangent monomial levels"
            )
        extension.require_preserved_constants("recursive coupled RDE solving")


def _coupled_tower_is_gaussian_safe(derivation: Derivation) -> bool:
    """Whether adjoining ``i`` preserves every active monomial family."""
    return all(
        isinstance(extension, MonomialExtension)
        and extension.family
        in {
            MonomialFamily.PRIMITIVE,
            MonomialFamily.HYPEREXPONENTIAL,
        }
        for extension in derivation.tower.levels[1:derivation.level + 1]
    )


def _has_direct_hyperexponential_laurent_shape(
    diagonal_coefficient: FieldElement,
    rotation_coefficient: FieldElement,
    real_right_hand_side: FieldElement,
    imaginary_right_hand_side: FieldElement,
    extension: MonomialExtension,
) -> bool:
    """Whether Chapter 8 modewise cancellation accepts all four inputs."""
    try:
        _, diagonal_terms = _hyperexponential_laurent_coefficients(
            diagonal_coefficient,
            extension,
            "diagonal coefficient",
        )
        _, rotation_terms = _hyperexponential_laurent_coefficients(
            rotation_coefficient,
            extension,
            "rotation coefficient",
        )
        _hyperexponential_laurent_coefficients(
            real_right_hand_side,
            extension,
            "real right-hand side",
        )
        _hyperexponential_laurent_coefficients(
            imaginary_right_hand_side,
            extension,
            "imaginary right-hand side",
        )
    except NotImplementedError:
        return False
    return (
        all(power == 0 for power in diagonal_terms)
        and all(power == 0 for power in rotation_terms)
    )


def _split_gaussian_tower_element(
    value: FieldElement,
    gaussian_field: Domain,
    real_field: Domain,
) -> tuple[FieldElement, FieldElement]:
    """Split a Gaussian tower element into exact real-field coordinates."""
    expression = gaussian_field.to_sympy(value)
    conjugate = expression.xreplace({I: -I})
    real_expression = cancel((expression + conjugate)/2)
    imaginary_expression = cancel((expression - conjugate)/(2*I))
    real_part = _coerce(real_field, real_expression)
    imaginary_part = _coerce(real_field, imaginary_expression)
    reconstructed = gaussian_field.from_sympy(
        real_field.to_sympy(real_part)
        + I*real_field.to_sympy(imaginary_part)
    )
    if not _field_element_is_zero(
        reconstructed - value,
        gaussian_field,
    ):
        raise ValueError("Gaussian tower coordinates failed to reconstruct")
    return real_part, imaginary_part


def _solve_general_coupled_risch_differential_equation(
    diagonal_coefficient: object,
    rotation_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledRischDifferentialEquationResult:
    """Solve the general real two-coordinate RDE over supported tower shapes."""
    if not isinstance(derivation, Derivation):
        raise TypeError("coupled RDE solving requires a Derivation")
    if derivation.level < 0:
        raise TypeError("coupled RDE solving requires a nonconstant field")
    _require_supported_coupled_tower(derivation)

    real_field = derivation.field
    diagonal_element = _coerce(real_field, diagonal_coefficient)
    rotation_element = _coerce(real_field, rotation_coefficient)
    real_rhs = _coerce(real_field, real_right_hand_side)
    imaginary_rhs = _coerce(real_field, imaginary_right_hand_side)

    if derivation.level > 0:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "coupled RDE solving does not support algebraic tower levels"
            )
        if (
            extension.family is MonomialFamily.HYPEREXPONENTIAL
            and _has_direct_hyperexponential_laurent_shape(
                diagonal_element,
                rotation_element,
                real_rhs,
                imaginary_rhs,
                extension,
            )
        ):
            direct_result = solve_hyperexponential_coupled_laurent_rde(
                diagonal_element,
                rotation_element,
                real_rhs,
                imaginary_rhs,
                derivation,
            )
            result = CoupledRischDifferentialEquationResult(
                derivation=derivation,
                diagonal_coefficient=diagonal_element,
                lambda_coefficient=rotation_element,
                real_right_hand_side=real_rhs,
                imaginary_right_hand_side=imaginary_rhs,
                real_solution=direct_result.real_solution,
                imaginary_solution=direct_result.imaginary_solution,
                obstruction=direct_result.obstruction,
                direct_result=direct_result,
            )
            if result.has_solution and not result.reconstructs:
                raise ValueError(
                    "direct hyperexponential coupled result failed to "
                    "reconstruct"
                )
            return result
        if extension.family is MonomialFamily.HYPERTANGENT:
            try:
                direct_result = solve_hypertangent_coupled_polynomial_rde(
                    diagonal_element,
                    rotation_element,
                    real_rhs,
                    imaginary_rhs,
                    derivation,
                )
            except NotImplementedError:
                # The terminal recurrence is complete only for its stated
                # polynomial shapes.  General rational cases are sent
                # through the certified Cayley normal form below.
                pass
            else:
                real_solution = (
                    _coerce(real_field, direct_result.real_solution.as_expr())
                    if direct_result.real_solution is not None
                    else None
                )
                imaginary_solution = (
                    _coerce(
                        real_field,
                        direct_result.imaginary_solution.as_expr(),
                    )
                    if direct_result.imaginary_solution is not None
                    else None
                )
                result = CoupledRischDifferentialEquationResult(
                    derivation=derivation,
                    diagonal_coefficient=diagonal_element,
                    lambda_coefficient=rotation_element,
                    real_right_hand_side=real_rhs,
                    imaginary_right_hand_side=imaginary_rhs,
                    real_solution=real_solution,
                    imaginary_solution=imaginary_solution,
                    obstruction=direct_result.obstruction,
                    direct_result=direct_result,
                )
                if result.has_solution and not result.reconstructs:
                    raise ValueError(
                        "direct hypertangent coupled result failed to "
                        "reconstruct"
                    )
                return result

    gaussian_normal_form = None
    if _coupled_tower_is_gaussian_safe(derivation):
        gaussian_derivation = _rebase_monomial_derivation(derivation, QQ_I)
    else:
        from .gaussian import build_hypertangent_gaussian_normal_form

        gaussian_normal_form = build_hypertangent_gaussian_normal_form(
            derivation,
        )
        gaussian_derivation = gaussian_normal_form.target_derivation
    gaussian_field = gaussian_derivation.field
    if gaussian_normal_form is None:
        gaussian_diagonal = _coerce(
            gaussian_field,
            real_field.to_sympy(diagonal_element),
        )
        gaussian_rotation = _coerce(
            gaussian_field,
            real_field.to_sympy(rotation_element),
        )
        gaussian_rhs = _coerce(
            gaussian_field,
            real_field.to_sympy(real_rhs)
            + I*real_field.to_sympy(imaginary_rhs),
        )
    else:
        gaussian_diagonal = gaussian_normal_form.forward(diagonal_element)
        gaussian_rotation = gaussian_normal_form.forward(rotation_element)
        gaussian_rhs = (
            gaussian_normal_form.forward(real_rhs)
            + gaussian_field.from_sympy(I)*gaussian_normal_form.forward(
                imaginary_rhs,
            )
        )
    scalar_result = solve_risch_differential_equation(
        gaussian_diagonal
        + gaussian_field.from_sympy(I)*gaussian_rotation,
        gaussian_rhs,
        gaussian_derivation,
    )
    if scalar_result.solution is None:
        if scalar_result.obstruction is None:
            raise ValueError("a failed scalar RDE lacks its obstruction")
        return CoupledRischDifferentialEquationResult(
            derivation=derivation,
            diagonal_coefficient=diagonal_element,
            lambda_coefficient=rotation_element,
            real_right_hand_side=real_rhs,
            imaginary_right_hand_side=imaginary_rhs,
            real_solution=None,
            imaginary_solution=None,
            obstruction=_coupled_obstruction_from_scalar(
                scalar_result.obstruction
            ),
            scalar_result=scalar_result,
            gaussian_normal_form=gaussian_normal_form,
        )

    if gaussian_normal_form is None:
        real_solution, imaginary_solution = _split_gaussian_tower_element(
            scalar_result.solution,
            gaussian_field,
            real_field,
        )
    else:
        real_solution, imaginary_solution = gaussian_normal_form.split_inverse(
            scalar_result.solution,
        )
    result = CoupledRischDifferentialEquationResult(
        derivation=derivation,
        diagonal_coefficient=diagonal_element,
        lambda_coefficient=rotation_element,
        real_right_hand_side=real_rhs,
        imaginary_right_hand_side=imaginary_rhs,
        real_solution=real_solution,
        imaginary_solution=imaginary_solution,
        obstruction=None,
        scalar_result=scalar_result,
        gaussian_normal_form=gaussian_normal_form,
    )
    if not result.reconstructs:
        raise ValueError("recursive coupled RDE reconstruction failed")
    return result


def solve_coupled_risch_differential_equation(
    lambda_coefficient: object,
    real_right_hand_side: object,
    imaginary_right_hand_side: object,
    derivation: Derivation,
) -> CoupledRischDifferentialEquationResult:
    r"""Solve ``Dp-lambda*q=u`` and ``Dq+lambda*p=v`` recursively.

    Primitive and non-Laurent hyperexponential cases use the equivalent scalar
    equation over an immutable ``QQ_I`` rebase. General rational
    hypertangent cases use a certified Cayley normal form whose top generator
    is hyperexponential, then map the scalar answer back to real coordinates.
    Hyperexponential Laurent cancellation and implemented hypertangent
    polynomial branches retain the direct real Chapter 8 recurrences.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("coupled RDE solving requires a Derivation")
    if derivation.level < 0:
        raise TypeError("coupled RDE solving requires a nonconstant field")
    field = derivation.field
    rotation = _coerce(field, lambda_coefficient)
    if rotation == field.zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    return _solve_general_coupled_risch_differential_equation(
        field.zero,
        rotation,
        real_right_hand_side,
        imaginary_right_hand_side,
        derivation,
    )


def solve_bounded_coupled_rde_matrix(
    lambda_coefficient: Expr | Poly,
    real_right_hand_side: Expr | Poly,
    imaginary_right_hand_side: Expr | Poly,
    x: Symbol,
    denominator: Expr | Poly,
    numerator_degree_bound: int,
) -> BoundedCoupledRDEMatrixResult:
    r"""Solve a bounded coupled rational ansatz by exact block linear algebra.

    This deliberately shares no denominator or leading-recurrence logic with
    :func:`solve_coupled_rational_rde`; it is a differential-test oracle for a
    caller-supplied common denominator and numerator degree bound.
    """
    if not isinstance(x, Symbol):
        raise TypeError("the differentiation variable must be a Symbol")
    if (
        not isinstance(numerator_degree_bound, int)
        or isinstance(numerator_degree_bound, bool)
        or numerator_degree_bound < 0
    ):
        raise ValueError("the numerator degree bound must be a nonnegative int")

    lambda_expression, lambda_numerator, _ = _as_qq_rational(
        lambda_coefficient,
        x,
    )
    if lambda_numerator.is_zero:
        raise ValueError("the coupled rotation coefficient must be nonzero")
    real_rhs, _, _ = _as_qq_rational(real_right_hand_side, x)
    imaginary_rhs, _, _ = _as_qq_rational(imaginary_right_hand_side, x)
    denominator = Poly(denominator, x, domain=QQ)
    if denominator.is_zero:
        raise ValueError("the ansatz denominator must be nonzero")
    denominator = denominator.monic()

    basis = [
        x**degree/denominator.as_expr()
        for degree in range(numerator_degree_bound + 1)
    ]
    real_columns: list[Expr] = []
    imaginary_columns: list[Expr] = []
    for candidate in basis:
        real_columns.append(cancel(candidate.diff(x)))
        imaginary_columns.append(cancel(lambda_expression*candidate))
    for candidate in basis:
        real_columns.append(cancel(-lambda_expression*candidate))
        imaginary_columns.append(cancel(candidate.diff(x)))

    rational_expressions = [
        *real_columns,
        *imaginary_columns,
        real_rhs,
        imaginary_rhs,
    ]
    common_denominator = Poly(1, x, domain=QQ)
    for expression in rational_expressions:
        _, _, expression_denominator = _as_qq_rational(expression, x)
        common_denominator = common_denominator.lcm(expression_denominator)
    cleared = [
        Poly(
            cancel(expression*common_denominator.as_expr()),
            x,
            domain=QQ,
        )
        for expression in rational_expressions
    ]
    unknown_count = 2*(numerator_degree_bound + 1)
    cleared_real_columns = cleared[:unknown_count]
    cleared_imaginary_columns = cleared[
        unknown_count:2*unknown_count
    ]
    cleared_real_rhs = cleared[-2]
    cleared_imaginary_rhs = cleared[-1]
    equation_degree = max(
        (
            polynomial.degree()
            for polynomial in cleared
            if not polynomial.is_zero
        ),
        default=0,
    )
    coefficient_rows = []
    rhs_rows = []
    for columns, rhs_polynomial in (
        (cleared_real_columns, cleared_real_rhs),
        (cleared_imaginary_columns, cleared_imaginary_rhs),
    ):
        for degree in range(equation_degree + 1):
            coefficient_rows.append(
                [
                    QQ.convert(column.nth(degree))
                    for column in columns
                ]
            )
            rhs_rows.append([QQ.convert(rhs_polynomial.nth(degree))])

    system = DomainMatrix(
        coefficient_rows,
        (len(coefficient_rows), unknown_count),
        QQ,
    )
    rhs = DomainMatrix(
        rhs_rows,
        (len(rhs_rows), 1),
        QQ,
    )
    coefficient_rank = system.rank()
    augmented = system.hstack(rhs)
    augmented_rank = augmented.rank()
    if augmented_rank > coefficient_rank:
        return BoundedCoupledRDEMatrixResult(
            denominator,
            numerator_degree_bound,
            None,
            None,
            coefficient_rank,
            augmented_rank,
        )

    reduced_matrix, pivots = augmented.rref()
    reduced_rows = reduced_matrix.to_list()
    coefficients = [QQ.zero for _ in range(unknown_count)]
    for row, pivot in enumerate(pivots):
        if pivot < unknown_count:
            coefficients[pivot] = reduced_rows[row][unknown_count]
    component_count = numerator_degree_bound + 1
    real_numerator = Poly.from_dict(
        {
            (degree,): value
            for degree, value in enumerate(coefficients[:component_count])
        },
        x,
        domain=QQ,
    )
    imaginary_numerator = Poly.from_dict(
        {
            (degree,): value
            for degree, value in enumerate(coefficients[component_count:])
        },
        x,
        domain=QQ,
    )
    real_solution = cancel(
        real_numerator.as_expr()/denominator.as_expr()
    )
    imaginary_solution = cancel(
        imaginary_numerator.as_expr()/denominator.as_expr()
    )
    if (
        cancel(
            real_solution.diff(x)
            - lambda_expression*imaginary_solution
            - real_rhs
        )
        != 0
        or cancel(
            imaginary_solution.diff(x)
            + lambda_expression*real_solution
            - imaginary_rhs
        )
        != 0
    ):
        raise ValueError("bounded coupled matrix RDE reconstruction failed")
    return BoundedCoupledRDEMatrixResult(
        denominator,
        numerator_degree_bound,
        real_solution,
        imaginary_solution,
        coefficient_rank,
        augmented_rank,
    )
