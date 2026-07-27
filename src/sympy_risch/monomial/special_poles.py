r"""Certified exact reduction at linear special poles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement, is_field_zero
from ..admission import assess_scalar_rde_admission
from ..rde import (
    RischDifferentialEquationResult,
    solve_risch_differential_equation,
)
from ..reduction.fractions import (
    _reduced_polynomial_fraction,
)
from ..special import SpecialPolynomialCertificate
from ..tower.core import (
    Derivation,
    MonomialExtension,
    _coerce,
)


class LinearSpecialPoleReductionStatus(Enum):
    """Outcome of reduction at one certified linear special factor."""

    REDUCED = "reduced"
    UNRESOLVED_RDE = "unresolved_rde"
    UNSUPPORTED_LOWER_RDE = "unsupported_lower_rde"


@dataclass(frozen=True, slots=True)
class LinearSpecialPoleStep:
    r"""One principal-part RDE at a certified linear special factor.

    At pole order ``m``, ``principal_coefficient`` is the coefficient of
    ``p**(-m)`` modulo lower powers of ``p``.  The stored lower-field RDE is

    .. math::

       D(q) - m\,\bar h q = r,

    where ``D(p) = h*p`` and ``bar h`` is ``h`` reduced modulo ``p``.  A
    returned ``correction`` is exactly ``q/p**m`` in the top field.
    """

    pole_order: int
    principal_coefficient: FieldElement
    cofactor_residue: FieldElement
    rde_result: RischDifferentialEquationResult | None
    correction: FieldElement | None

    def __post_init__(self) -> None:
        """Keep an RDE witness and its lifted correction consistent."""
        if not isinstance(self.pole_order, int) or isinstance(self.pole_order, bool):
            raise TypeError("a special-pole step needs an integer pole order")
        if self.pole_order <= 0:
            raise ValueError("a special-pole step needs a positive pole order")
        if self.rde_result is not None and not isinstance(
            self.rde_result,
            RischDifferentialEquationResult,
        ):
            raise TypeError("a special-pole step needs an RDE result")
        if self.rde_result is None:
            if self.correction is not None:
                raise ValueError(
                    "an unsupported special-pole RDE cannot carry a correction"
                )
            return
        if self.rde_result.has_solution != (self.correction is not None):
            raise ValueError(
                "a special-pole RDE solution and lifted correction must agree"
            )

    @property
    def is_reduced(self) -> bool:
        """Whether this principal part was cancelled exactly."""
        return self.correction is not None


@dataclass(frozen=True, slots=True)
class LinearSpecialPoleReduction:
    r"""Partial exact reduction at one caller-certified linear special factor.

    This is deliberately a local exactness operation, not an elementary
    integration decision.  A failed lower RDE is retained as data and leaves
    the current principal part in ``remainder``; no non-elementarity verdict
    follows from this result alone.
    """

    derivation: Derivation
    certificate: SpecialPolynomialCertificate
    integrand: FieldElement
    exact_part: FieldElement
    remainder: FieldElement
    steps: tuple[LinearSpecialPoleStep, ...]
    status: LinearSpecialPoleReductionStatus
    remaining_pole_order: int
    unsupported_detail: str | None = None

    def __post_init__(self) -> None:
        """Validate the immutable local-reduction certificate."""
        _, extension = _require_linear_special_pole_certificate(
            self.certificate,
            self.derivation,
        )
        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, LinearSpecialPoleStep) for step in self.steps
        ):
            raise TypeError("special-pole reduction steps must be an immutable tuple")
        if not isinstance(self.status, LinearSpecialPoleReductionStatus):
            raise TypeError("a special-pole reduction needs a valid status")
        if not isinstance(self.remaining_pole_order, int) or isinstance(
            self.remaining_pole_order,
            bool,
        ):
            raise TypeError("a special-pole remainder needs an integer pole order")
        if self.remaining_pole_order < 0:
            raise ValueError("a special-pole remainder cannot have negative order")
        if self.unsupported_detail is not None and not isinstance(
            self.unsupported_detail,
            str,
        ):
            raise TypeError("special-pole unsupported detail must be a string")
        lower_derivation = self.derivation.lower()
        lower_rde_admission = assess_scalar_rde_admission(lower_derivation)
        if lower_rde_admission.unsupported and any(
            step.rde_result is not None for step in self.steps
        ):
            raise ValueError("a special-pole RDE step requires an admitted lower tower")
        cofactor_residue = _coerce(
            extension.parent_field,
            self.certificate.cofactor.rem(self.certificate.factor).nth(0),
        )
        expected_exact_part = extension.field.zero
        working_remainder = self.integrand
        for index, step in enumerate(self.steps):
            numerator, denominator = _reduced_polynomial_fraction(
                working_remainder,
                extension,
            )
            actual_pole_order = _linear_factor_multiplicity(
                denominator,
                self.certificate.factor,
            )
            if step.pole_order != actual_pole_order:
                raise ValueError("a special-pole step has the wrong pole order")
            actual_principal_coefficient = _linear_special_principal_coefficient(
                numerator,
                denominator,
                self.certificate.factor,
                actual_pole_order,
            )
            if not _linear_special_pole_elements_equal(
                step.principal_coefficient,
                actual_principal_coefficient,
                extension.parent_field,
            ):
                raise ValueError(
                    "a special-pole step has the wrong principal coefficient"
                )
            if not _linear_special_pole_elements_equal(
                step.cofactor_residue,
                cofactor_residue,
                extension.parent_field,
            ):
                raise ValueError("a special-pole step has the wrong cofactor residue")

            if step.rde_result is not None:
                if not step.rde_result.derivation.has_same_position(
                    lower_derivation,
                ):
                    raise ValueError(
                        "a special-pole step belongs to another lower derivation"
                    )
                expected_rde_coefficient = -(
                    lower_derivation.field.convert(step.pole_order) * cofactor_residue
                )
                if not _linear_special_pole_elements_equal(
                    step.rde_result.coefficient,
                    expected_rde_coefficient,
                    lower_derivation.field,
                ) or not _linear_special_pole_elements_equal(
                    step.rde_result.right_hand_side,
                    step.principal_coefficient,
                    lower_derivation.field,
                ):
                    raise ValueError("a special-pole step has the wrong lower RDE")
                if step.rde_result.has_solution:
                    if (
                        step.rde_result.solution is None
                        or not step.rde_result.reconstructs
                    ):
                        raise ValueError(
                            "a special-pole step has an invalid lower RDE solution"
                        )
                    expected_correction = _coerce(
                        extension.field,
                        lower_derivation.field.to_sympy(step.rde_result.solution)
                        / self.certificate.factor.as_expr() ** step.pole_order,
                    )
                    if (
                        step.correction is None
                        or not _linear_special_pole_elements_equal(
                            step.correction,
                            expected_correction,
                            extension.field,
                        )
                    ):
                        raise ValueError(
                            "a special-pole step has the wrong lifted correction"
                        )

            if step.correction is None:
                if index != len(self.steps) - 1:
                    raise ValueError(
                        "only the final special-pole step may be unresolved"
                    )
                break
            expected_exact_part += step.correction
            working_remainder -= self.derivation(step.correction)

        if not _linear_special_pole_elements_equal(
            working_remainder,
            self.remainder,
            extension.field,
        ):
            raise ValueError("special-pole steps do not reconstruct the remainder")
        if not _linear_special_pole_elements_equal(
            expected_exact_part,
            self.exact_part,
            extension.field,
        ):
            raise ValueError("special-pole steps do not reconstruct the exact part")
        if not self.reconstructs:
            raise ValueError("special-pole reduction did not reconstruct its input")
        _, denominator = _reduced_polynomial_fraction(self.remainder, extension)
        actual_pole_order = _linear_factor_multiplicity(
            denominator,
            self.certificate.factor,
        )
        if actual_pole_order != self.remaining_pole_order:
            raise ValueError(
                "special-pole result records the wrong remaining pole order"
            )

        if self.status is LinearSpecialPoleReductionStatus.REDUCED:
            if self.remaining_pole_order != 0 or self.unsupported_detail is not None:
                raise ValueError("a reduced special-pole result has inconsistent state")
            if any(not step.is_reduced for step in self.steps):
                raise ValueError(
                    "a reduced special-pole result retains an unresolved step"
                )
            return

        if self.remaining_pole_order == 0 or not self.steps:
            raise ValueError("an unresolved special-pole result needs a remaining pole")
        if any(not step.is_reduced for step in self.steps[:-1]):
            raise ValueError("only the final special-pole step may be unresolved")
        final_step = self.steps[-1]
        if final_step.is_reduced:
            raise ValueError(
                "an unresolved special-pole result needs an unresolved step"
            )
        if self.status is LinearSpecialPoleReductionStatus.UNRESOLVED_RDE:
            if final_step.rde_result is None or self.unsupported_detail is not None:
                raise ValueError("an unresolved RDE result has inconsistent state")
            return
        if self.status is LinearSpecialPoleReductionStatus.UNSUPPORTED_LOWER_RDE:
            if final_step.rde_result is not None or not self.unsupported_detail:
                raise ValueError("an unsupported RDE result has inconsistent state")
            return
        raise ValueError("unknown linear special-pole reduction status")

    @property
    def reconstructs(self) -> bool:
        """Whether ``integrand = D(exact_part) + remainder`` holds exactly."""
        return _linear_special_pole_elements_equal(
            self.derivation(self.exact_part) + self.remainder,
            self.integrand,
            self.derivation.field,
        )

    @property
    def is_fully_reduced(self) -> bool:
        """Whether no pole at the certified factor remains."""
        return self.status is LinearSpecialPoleReductionStatus.REDUCED

    @property
    def unresolved_step(self) -> LinearSpecialPoleStep | None:
        """Return the retained lower-RDE boundary, when one stopped reduction."""
        return None if self.is_fully_reduced else self.steps[-1]


def _require_linear_special_pole_certificate(
    certificate: object,
    derivation: object,
) -> tuple[SpecialPolynomialCertificate, MonomialExtension]:
    """Require positive evidence for one active monic linear special factor."""
    if not isinstance(certificate, SpecialPolynomialCertificate):
        raise TypeError(
            "linear special-pole reduction requires a SpecialPolynomialCertificate"
        )
    if not isinstance(derivation, Derivation):
        raise TypeError("linear special-pole reduction requires a Derivation")
    if not certificate.derivation.has_same_position(derivation):
        raise ValueError("the special-pole certificate belongs to another derivation")
    if derivation.level < 0:
        raise TypeError("linear special-pole reduction requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("linear special-pole reduction requires a monomial level")
    if certificate.factor.degree() != 1:
        raise ValueError("linear special-pole reduction requires a linear factor")
    return certificate, extension


def _linear_factor_multiplicity(denominator: Poly, factor: Poly) -> int:
    """Return the exact exponent of a monic linear ``factor`` in ``denominator``."""
    multiplicity = 0
    quotient = denominator
    while True:
        next_quotient, remainder = quotient.div(factor)
        if not remainder.is_zero:
            return multiplicity
        multiplicity += 1
        quotient = next_quotient


def _linear_special_principal_coefficient(
    numerator: Poly,
    denominator: Poly,
    factor: Poly,
    pole_order: int,
) -> FieldElement:
    r"""Return the coefficient of ``factor**(-pole_order)`` in a principal part.

    If ``denominator = factor**m * s`` with ``gcd(factor, s)=1``, then the
    coefficient is ``numerator * s**(-1) mod factor``.  The linear-factor
    premise makes this residue an element of the lower field rather than an
    algebraic residue field.
    """
    cofactor = denominator.exquo(factor**pole_order)
    principal = (numerator * cofactor.invert(factor)).rem(factor)
    coefficient = _coerce(factor.domain, principal.nth(0))
    if _linear_special_pole_elements_equal(
        coefficient,
        factor.domain.zero,
        factor.domain,
    ):
        raise ValueError("a reduced special-pole fraction has zero principal part")
    return coefficient


def _linear_special_pole_elements_equal(
    left: FieldElement,
    right: FieldElement,
    field: Domain,
) -> bool:
    """Compare local-pole coefficients despite noncanonical field fractions."""
    difference = left - right
    return is_field_zero(difference) or cancel(field.to_sympy(difference)) == 0


def reduce_certified_linear_special_poles(
    integrand: object,
    derivation: Derivation,
    certificate: SpecialPolynomialCertificate,
) -> LinearSpecialPoleReduction:
    r"""Remove exact principal parts at one supplied linear special factor.

    Let ``p=t-a`` be the monic factor retained by ``certificate`` and write
    ``D(p)=h*p``.  For a pole of order ``m``, reduce ``h`` modulo ``p`` to
    ``hbar`` in the lower field.  If ``r`` is the coefficient of ``p**(-m)``,
    then

    .. math::

       D\left(\frac{q}{p^m}\right)
       \equiv \frac{D(q)-m\,\bar h q}{p^m}
       \pmod {p^{-(m-1)}}.

    The routine therefore solves the lower-field Risch differential equation
    ``D(q)-m*hbar*q=r``.  A solution removes the top principal part and the
    exact derivative is recomputed before the next order.  This last detail
    matters when ``h`` itself depends on ``t``: the difference between ``h``
    and ``hbar`` spills into the next lower pole order.

    The operation consumes only positive factor evidence.  If its lower RDE
    is unsupported or has no solution, it returns the exact work completed
    and an explicit boundary; it does *not* conclude that the original
    integrand has no elementary antiderivative.
    """
    certificate, extension = _require_linear_special_pole_certificate(
        certificate,
        derivation,
    )
    field = extension.field
    factor = certificate.factor
    cofactor_residue = _coerce(
        extension.parent_field,
        certificate.cofactor.rem(factor).nth(0),
    )
    element = _coerce(field, integrand)
    exact_part = field.zero
    remainder = element
    steps: list[LinearSpecialPoleStep] = []

    while True:
        numerator, denominator = _reduced_polynomial_fraction(remainder, extension)
        pole_order = _linear_factor_multiplicity(denominator, factor)
        if pole_order == 0:
            return LinearSpecialPoleReduction(
                derivation,
                certificate,
                element,
                exact_part,
                remainder,
                tuple(steps),
                LinearSpecialPoleReductionStatus.REDUCED,
                0,
            )

        principal_coefficient = _linear_special_principal_coefficient(
            numerator,
            denominator,
            factor,
            pole_order,
        )
        lower_derivation = derivation.lower()
        rde_coefficient = -(
            lower_derivation.field.convert(pole_order) * cofactor_residue
        )
        lower_rde_admission = assess_scalar_rde_admission(lower_derivation)
        if lower_rde_admission.unsupported:
            steps.append(
                LinearSpecialPoleStep(
                    pole_order,
                    principal_coefficient,
                    cofactor_residue,
                    None,
                    None,
                )
            )
            return LinearSpecialPoleReduction(
                derivation,
                certificate,
                element,
                exact_part,
                remainder,
                tuple(steps),
                LinearSpecialPoleReductionStatus.UNSUPPORTED_LOWER_RDE,
                pole_order,
                lower_rde_admission.detail,
            )
        try:
            rde_result = solve_risch_differential_equation(
                rde_coefficient,
                principal_coefficient,
                lower_derivation,
            )
        except NotImplementedError as error:
            steps.append(
                LinearSpecialPoleStep(
                    pole_order,
                    principal_coefficient,
                    cofactor_residue,
                    None,
                    None,
                )
            )
            return LinearSpecialPoleReduction(
                derivation,
                certificate,
                element,
                exact_part,
                remainder,
                tuple(steps),
                LinearSpecialPoleReductionStatus.UNSUPPORTED_LOWER_RDE,
                pole_order,
                str(error) or "lower-field RDE support is unavailable",
            )

        if not rde_result.has_solution:
            steps.append(
                LinearSpecialPoleStep(
                    pole_order,
                    principal_coefficient,
                    cofactor_residue,
                    rde_result,
                    None,
                )
            )
            return LinearSpecialPoleReduction(
                derivation,
                certificate,
                element,
                exact_part,
                remainder,
                tuple(steps),
                LinearSpecialPoleReductionStatus.UNRESOLVED_RDE,
                pole_order,
            )
        if not rde_result.reconstructs or rde_result.solution is None:
            raise ValueError("special-pole lower RDE solution did not reconstruct")

        correction = _coerce(
            field,
            lower_derivation.field.to_sympy(rde_result.solution)
            / factor.as_expr() ** pole_order,
        )
        new_remainder = remainder - derivation(correction)
        _, new_denominator = _reduced_polynomial_fraction(new_remainder, extension)
        remaining_pole_order = _linear_factor_multiplicity(
            new_denominator,
            factor,
        )
        if remaining_pole_order >= pole_order:
            raise ValueError("special-pole reduction did not lower the pole order")
        steps.append(
            LinearSpecialPoleStep(
                pole_order,
                principal_coefficient,
                cofactor_residue,
                rde_result,
                correction,
            )
        )
        exact_part += correction
        remainder = new_remainder
