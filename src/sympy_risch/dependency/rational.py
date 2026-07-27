r"""Rational parametric logarithmic-derivative decisions."""

from __future__ import annotations

from typing import cast

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.fields import FracElement
from sympy.polys.polytools import Poly, invert

from .._typing import FieldElement
from ..constant_fields import rational_constant_coordinates
from ..rational import (
    RationalLogDerivativeResult,
    _solve_rational_logarithmic_derivative_over_domain,
)
from ..tower.core import Derivation, MonomialExtension, _coerce
from ._common import (
    _constant_field_linear_ratio,
    _parametric_logarithmic_derivative_eta,
    _standard_rational_variable,
)
from .models import (
    ParametricLogarithmicDerivativeHeuristicBranch,
    ParametricLogarithmicDerivativeHeuristicCertificate,
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
    RationalParametricLogarithmicDerivativeCertificate,
    RationalParametricLogarithmicDerivativeResidue,
)


def _minimal_logarithmic_power(
    result: RationalLogDerivativeResult,
) -> int | None:
    r"""Return minimal ``n > 0`` making all logarithmic valuations integral."""
    return result.minimal_radical_power


def _rational_polynomial_parts(
    element: FracElement,
    variable: Symbol,
    domain: Domain,
) -> tuple[Poly, Poly, Poly]:
    """Return ``p, a, d`` with ``element = p + a/d`` over ``domain(x)``."""
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=domain,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=domain,
    )
    polynomial_part, proper_numerator = numerator.div(denominator)
    return polynomial_part, proper_numerator, denominator


def _rational_coefficient_ratio(
    left: Poly,
    right: Poly,
    degrees: range,
    constant_field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve ``coefficient(left, x**i) = c*coefficient(right, x**i)``.

    The boolean distinguishes an inconsistent system from a consistent one.
    A ``None`` ratio means that every compared coefficient vanishes, so the
    system is consistent but does not determine ``c``.  Although the
    coefficients may live in a finite extension of ``QQ``, the parameter
    ``c=m/n`` is required to belong to ``QQ``.
    """
    constraints = tuple(
        (
            constant_field.convert(left.nth(degree)),
            constant_field.convert(right.nth(degree)),
        )
        for degree in degrees
    )
    return _constant_field_linear_ratio(constraints, constant_field)


def _rational_linear_ratio(
    constraints: tuple[tuple[FieldElement, FieldElement], ...],
    constant_field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve the exact scalar system ``left = c*right`` with ``c`` in ``QQ``.

    ``(True, None)`` means that every equation is ``0 = c*0``, so the ratio
    remains free.  ``(False, None)`` means that no rational ratio satisfies
    all constraints.
    """
    return _constant_field_linear_ratio(constraints, constant_field)


def solve_parametric_logarithmic_derivative_heuristic(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Run Bronstein's Section 7.3 heuristic over a supported ``C(x)``.

    Given ``Dtheta/theta = eta``, seek integers ``n != 0`` and ``m`` and a
    nonzero ``v`` in ``QQ(x)`` such that

    ``n*f = Dv/v + m*eta``.

    Lemma 7.3.1 supplies necessary high-degree and denominator congruences
    that often determine the rational ratio ``m/n``.  Once a ratio is known,
    the exact rational logarithmic-derivative solver decides whether an
    integer multiple of the residual is ``Dv/v``.

    This routine intentionally preserves the heuristic's incompleteness.
    When the lemma does not determine ``m/n``, the result is ``INCONCLUSIVE``,
    not a false proof of nonsolvability.  Chapter 9 structure theorems are the
    complete fallback.
    """
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    field = lower_derivation.field
    constant_field = lower_derivation.tower.constants
    element = _coerce(field, f)

    try:
        variable = _standard_rational_variable(lower_derivation)
    except NotImplementedError as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    p, a, d = _rational_polynomial_parts(
        cast(FracElement, element),
        variable,
        constant_field,
    )
    q, b, e = _rational_polynomial_parts(
        cast(FracElement, eta),
        variable,
        constant_field,
    )
    degree_bound = 0  # Dx = 1, so max(0, degree(Dx) - 1) = 0.

    branch = ParametricLogarithmicDerivativeHeuristicBranch.POLYNOMIAL_PART
    modulus = None
    candidate_ratio = None

    def certificate(
        reason: str,
        *,
        ratio: FieldElement | None,
        logarithmic_power: int | None = None,
        logarithmic_result: RationalLogDerivativeResult | None = None,
    ) -> ParametricLogarithmicDerivativeHeuristicCertificate:
        return ParametricLogarithmicDerivativeHeuristicCertificate(
            variable,
            degree_bound,
            p,
            q,
            a,
            b,
            d,
            e,
            branch,
            modulus,
            ratio,
            logarithmic_power,
            logarithmic_result,
            reason,
        )

    if q.degree() > degree_bound:
        top_degree = max(p.degree(), q.degree())
        consistent, candidate_ratio = _rational_coefficient_ratio(
            p,
            q,
            range(degree_bound + 1, int(top_degree) + 1),
            constant_field,
        )
        if not consistent:
            obstruction = certificate(
                "polynomial coefficients above the degree bound are not proportional",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                certificate=obstruction,
            )
        if candidate_ratio is None:
            raise ValueError("a nonconstant eta polynomial part did not determine m/n")
    elif p.degree() > degree_bound:
        obstruction = certificate(
            "f has polynomial degree above the bound while eta does not",
            ratio=None,
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=obstruction,
        )
    else:
        branch = ParametricLogarithmicDerivativeHeuristicBranch.DENOMINATOR_CONGRUENCE
        common_denominator = d.lcm(e)

        # In QQ[x] every nonconstant irreducible polynomial is normal for
        # D=d/dx, so Bronstein's z = l_s*gcd(l_n, D(l_n)) reduces to gcd(l,l').
        modulus = common_denominator.gcd(common_denominator.diff())
        if modulus.degree() <= 0:
            trace = certificate(
                "Lemma 7.3.1 supplies no nonconstant denominator modulus",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                certificate=trace,
                detail=(
                    "Section 7.3 could not determine m/n; a structure-theorem "
                    "fallback is required"
                ),
            )

        scaled_f = (common_denominator.exquo(d) * (p * d + a)).rem(modulus)
        scaled_eta = (common_denominator.exquo(e) * (q * e + b)).rem(modulus)
        consistent, candidate_ratio = _rational_coefficient_ratio(
            scaled_f,
            scaled_eta,
            range(modulus.degree()),
            constant_field,
        )
        if not consistent:
            obstruction = certificate(
                "denominator congruences are not proportional",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
                certificate=obstruction,
            )
        if candidate_ratio is None:
            trace = certificate(
                "denominator congruences do not determine m/n",
                ratio=None,
            )
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.INCONCLUSIVE,
                certificate=trace,
                detail=(
                    "Section 7.3 denominator congruences leave m/n free; "
                    "a structure-theorem fallback is required"
                ),
            )

    # The coefficient comparison determines s=m/n.  Write it in lowest terms
    # as M/N with N>0, then decide whether some positive integer Q makes
    # Q*(N*f - M*eta) a rational logarithmic derivative.
    m = int(QQ.numer(candidate_ratio))
    n = int(QQ.denom(candidate_ratio))
    residual = n * element - m * eta
    logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
        field.to_sympy(residual),
        variable,
        constant_field,
    )
    logarithmic_power = _minimal_logarithmic_power(logarithmic_result)
    if logarithmic_power is None:
        obstruction = certificate(
            "the uniquely determined residual is not the logarithmic "
            "derivative of a rational radical",
            ratio=candidate_ratio,
            logarithmic_result=logarithmic_result,
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=obstruction,
        )

    if logarithmic_power > 1:
        logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
            logarithmic_power * field.to_sympy(residual),
            variable,
            constant_field,
        )
        if not logarithmic_result.has_solution:
            raise ValueError(
                "clearing logarithmic residues did not produce an argument"
            )

    solution_certificate = certificate(
        "coefficient comparison and radical logarithmic-derivative test "
        "produced a verified witness",
        ratio=candidate_ratio,
        logarithmic_power=logarithmic_power,
        logarithmic_result=logarithmic_result,
    )
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power * n,
        m=logarithmic_power * m,
        argument=logarithmic_result.argument,
        certificate=solution_certificate,
    )


def _rational_residue_coefficient(
    partial_numerator: Poly,
    factor: Poly,
    constant_field: Domain,
) -> FieldElement:
    r"""Return the ``factor.diff()`` coefficient at the leading pivot."""
    factor_degree = factor.degree()
    if factor_degree <= 0:
        raise ValueError("a residue factor must be nonconstant")

    derivative = factor.diff()
    pivot_degree = factor_degree - 1
    pivot = constant_field.convert(derivative.nth(pivot_degree))
    if pivot == constant_field.zero:
        raise ValueError("characteristic-zero factor derivative lost its leading term")
    return cast(
        FieldElement,
        constant_field.convert(partial_numerator.nth(pivot_degree)) / pivot,
    )


def _rational_residue_obstruction(
    partial_numerator: Poly,
    factor: Poly,
    constant_field: Domain,
) -> tuple[FieldElement, ...]:
    r"""Project a local numerator modulo the span of ``factor.diff()``."""
    factor_degree = factor.degree()
    if factor_degree <= 0:
        raise ValueError("a residue factor must be nonconstant")

    derivative = factor.diff()
    pivot_degree = factor_degree - 1
    multiple = _rational_residue_coefficient(
        partial_numerator,
        factor,
        constant_field,
    )
    return tuple(
        constant_field.convert(partial_numerator.nth(degree))
        - multiple * constant_field.convert(derivative.nth(degree))
        for degree in range(pivot_degree)
    )


def _rational_residue_rationality_constraints(
    f_residue: FieldElement,
    eta_residue: FieldElement,
    constant_field: Domain,
) -> tuple[tuple[FieldElement, FieldElement], ...]:
    r"""Constrain ``f_residue-c*eta_residue`` to lie in ``QQ``.

    A local logarithmic-derivative coefficient is a valuation after a
    positive integer scaling, hence it must be rational even after the base
    constants are enlarged to a finite extension of ``QQ``. Every non-rational
    power-basis coordinate therefore supplies a linear equation for the
    rational parameter ``c=m/n``.
    """
    if constant_field == QQ:
        return ()
    f_coordinates = rational_constant_coordinates(constant_field, f_residue)
    eta_coordinates = rational_constant_coordinates(constant_field, eta_residue)
    if len(f_coordinates) != len(eta_coordinates):
        raise ValueError("residue coordinates have inconsistent dimensions")
    return tuple(zip(f_coordinates[1:], eta_coordinates[1:], strict=True))


def _solve_rational_parametric_logarithmic_derivative_fallback(
    heuristic_result: ParametricLogarithmicDerivativeResult,
) -> ParametricLogarithmicDerivativeResult:
    r"""Complete an inconclusive Section 7.3 result over a rational base.

    At the heuristic's ``failed`` exit, both polynomial parts have degree at
    most zero and the common denominator is square-free.  For
    ``c = m/n``, the residual ``f - c*eta`` is a logarithmic derivative of a
    rational radical exactly when:

    * its constant polynomial part vanishes; and
    * at every irreducible denominator factor, its partial numerator is a
      rational multiple of the factor derivative.

    Removing each factor-derivative direction leaves an exact scalar linear
    system for the rational parameter ``c``. The coefficient field may be a
    finite extension of ``QQ``; a non-rational quotient is an inconsistency,
    never a candidate parameter.
    """
    if not heuristic_result.is_inconclusive:
        raise ValueError(
            "the rational fallback requires an inconclusive heuristic result"
        )
    heuristic_certificate = heuristic_result.certificate
    if not isinstance(
        heuristic_certificate,
        ParametricLogarithmicDerivativeHeuristicCertificate,
    ):
        raise TypeError("the rational fallback requires a Section 7.3 certificate")

    extension = heuristic_result.extension
    lower_derivation = heuristic_result.lower_derivation
    field = lower_derivation.field
    constant_field = lower_derivation.tower.constants
    variable = heuristic_certificate.variable
    eta = heuristic_result.eta

    p = heuristic_certificate.f_polynomial_part
    q = heuristic_certificate.eta_polynomial_part
    a = heuristic_certificate.f_proper_numerator
    b = heuristic_certificate.eta_proper_numerator
    d = heuristic_certificate.f_denominator
    e = heuristic_certificate.eta_denominator

    if p.degree() > 0 or q.degree() > 0:
        raise ValueError(
            "an inconclusive Section 7.3 result has a nonconstant polynomial part"
        )

    common_denominator = d.lcm(e)
    factorization = tuple(common_denominator.factor_list()[1])
    if any(multiplicity != 1 for _, multiplicity in factorization):
        raise ValueError(
            "an inconclusive Section 7.3 result has a repeated denominator"
        )

    common_f_numerator = common_denominator.exquo(d) * a
    common_eta_numerator = common_denominator.exquo(e) * b
    constraints: list[tuple[FieldElement, FieldElement]] = [
        (
            constant_field.convert(p.nth(0)),
            constant_field.convert(q.nth(0)),
        )
    ]
    residue_certificates: list[RationalParametricLogarithmicDerivativeResidue] = []

    for factor, _ in factorization:
        cofactor = common_denominator.exquo(factor)
        inverse_cofactor = invert(cofactor, factor)
        f_partial_numerator = (common_f_numerator * inverse_cofactor).rem(factor)
        eta_partial_numerator = (common_eta_numerator * inverse_cofactor).rem(factor)
        f_residue = _rational_residue_coefficient(
            f_partial_numerator,
            factor,
            constant_field,
        )
        eta_residue = _rational_residue_coefficient(
            eta_partial_numerator,
            factor,
            constant_field,
        )
        f_obstruction = _rational_residue_obstruction(
            f_partial_numerator,
            factor,
            constant_field,
        )
        eta_obstruction = _rational_residue_obstruction(
            eta_partial_numerator,
            factor,
            constant_field,
        )
        constraints.extend(zip(f_obstruction, eta_obstruction, strict=True))
        rationality_constraints = _rational_residue_rationality_constraints(
            f_residue,
            eta_residue,
            constant_field,
        )
        constraints.extend(rationality_constraints)
        residue_certificates.append(
            RationalParametricLogarithmicDerivativeResidue(
                factor,
                f_partial_numerator,
                eta_partial_numerator,
                f_obstruction,
                eta_obstruction,
                f_residue,
                eta_residue,
                rationality_constraints,
            )
        )

    linear_constraints = tuple(constraints)
    residues = tuple(residue_certificates)
    consistent, candidate_ratio = _rational_linear_ratio(
        linear_constraints,
        constant_field,
    )
    if not consistent:
        certificate = RationalParametricLogarithmicDerivativeCertificate(
            heuristic_result,
            common_denominator,
            residues,
            linear_constraints,
            None,
            None,
            None,
            "the rational structure constraints on m/n are inconsistent",
        )
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            heuristic_result.f,
            ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
            certificate=certificate,
        )

    # If every obstruction coordinate vanishes identically, f itself is a
    # radical logarithmic derivative and m=0 is the simplest witness.
    if candidate_ratio is None:
        candidate_ratio = QQ.zero

    m = int(QQ.numer(candidate_ratio))
    n = int(QQ.denom(candidate_ratio))
    residual = n * heuristic_result.f - m * eta
    logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
        field.to_sympy(residual),
        variable,
        constant_field,
    )
    logarithmic_power = _minimal_logarithmic_power(logarithmic_result)
    if logarithmic_power is None:
        raise ValueError(
            "rational structure constraints produced a nonlogarithmic residual"
        )
    if logarithmic_power > 1:
        logarithmic_result = _solve_rational_logarithmic_derivative_over_domain(
            logarithmic_power * field.to_sympy(residual),
            variable,
            constant_field,
        )
        if not logarithmic_result.has_solution:
            raise ValueError(
                "clearing fallback logarithmic residues did not produce an argument"
            )

    certificate = RationalParametricLogarithmicDerivativeCertificate(
        heuristic_result,
        common_denominator,
        residues,
        linear_constraints,
        candidate_ratio,
        logarithmic_power,
        logarithmic_result,
        "the rational structure constraints produced a verified witness",
    )
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        heuristic_result.f,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=logarithmic_power * n,
        m=logarithmic_power * m,
        argument=logarithmic_result.argument,
        certificate=certificate,
    )
