r"""PLogD dispatch across primitive, hyperexponential, and tangent towers."""

from __future__ import annotations

from sympy.polys.polytools import Poly

from ..relations.linear import solve_logarithmic_relation_system
from ..tower.core import Derivation, MonomialExtension, MonomialFamily, _coerce
from ._common import (
    _parametric_logarithmic_derivative_eta,
    _tower_contains_hyperexponential_level,
    _tower_contains_hypertangent_level,
)
from .models import (
    HypertangentGaussianParametricLogarithmicDerivativeCertificate,
    MonomialRelationSystemParametricCertificate,
    ParametricLogarithmicDerivativeOutcome,
    ParametricLogarithmicDerivativeResult,
)
from .primitive import _solve_primitive_parametric_logarithmic_derivative
from .rational import (
    _solve_rational_parametric_logarithmic_derivative_fallback,
    solve_parametric_logarithmic_derivative_heuristic,
)


def _solve_monomial_relation_system_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Solve PLogD by the complete finite QQ relation system.

    A hyperexponential lower level introduces its own special direction
    ``Dt/t``.  Therefore the lower descent is generally multi-parameter and
    cannot be represented by the one-ratio primitive recurrence.  The
    relation system keeps all such directions simultaneously.
    """
    field = lower_derivation.field
    element = _coerce(field, f)
    try:
        system = solve_logarithmic_relation_system(
            (
                element,
                _parametric_logarithmic_derivative_eta(
                    extension,
                    lower_derivation,
                ),
            ),
            lower_derivation,
        )
    except NotImplementedError as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    for relation in system.relations:
        n, eta_coefficient = relation.source_coefficients
        if n == 0:
            continue
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.SOLUTION,
            n=n,
            m=-eta_coefficient,
            argument=relation.argument,
            certificate=MonomialRelationSystemParametricCertificate(
                system,
                "the complete monomial relation system produced a verified witness",
            ),
        )

    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.PROVED_NO_SOLUTION,
        certificate=MonomialRelationSystemParametricCertificate(
            system,
            "no rational relation-system kernel vector has a nonzero f coordinate",
        ),
    )


def _solve_hypertangent_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Solve a real PLogD through an immutable Cayley normal form.

    If the Gaussian witness is ``V=P+iQ``, its logarithmic derivative is
    real.  Thus each nonzero real coordinate separately satisfies the same
    first-order equation.  Selecting ``P`` or ``Q`` recovers a real witness
    without unnecessarily doubling the relation.
    """
    field = lower_derivation.field
    element = _coerce(field, f)
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    try:
        from ..gaussian import build_hypertangent_gaussian_normal_form

        normal_form = build_hypertangent_gaussian_normal_form(
            lower_derivation,
        )
    except (NotImplementedError, ValueError) as exc:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
            detail=str(exc),
        )

    target_extension = MonomialExtension(
        Poly(
            normal_form.target_field.to_sympy(normal_form.forward(eta))
            * extension.generator,
            extension.generator,
            domain=normal_form.target_field,
        ),
    )
    target_result = solve_parametric_logarithmic_derivative(
        normal_form.forward(element),
        target_extension,
        normal_form.target_derivation,
    )
    certificate = HypertangentGaussianParametricLogarithmicDerivativeCertificate(
        normal_form,
        target_result,
        "the Cayley Gaussian normal form preserves the parametric relation",
    )
    if not target_result.has_solution:
        return ParametricLogarithmicDerivativeResult(
            extension,
            lower_derivation,
            element,
            target_result.outcome,
            certificate=(certificate if target_result.proves_no_solution else None),
            detail=target_result.detail,
        )
    if (
        target_result.n is None
        or target_result.m is None
        or target_result.argument is None
    ):
        raise ValueError("a solved Gaussian parametric result lacks a witness")
    real_part, imaginary_part = normal_form.split_inverse(
        target_result.argument,
    )
    argument = real_part if real_part != field.zero else imaginary_part
    if argument == field.zero:
        raise ValueError("a nonzero Gaussian witness has no real coordinate")
    return ParametricLogarithmicDerivativeResult(
        extension,
        lower_derivation,
        element,
        ParametricLogarithmicDerivativeOutcome.SOLUTION,
        n=target_result.n,
        m=target_result.m,
        argument=argument,
        certificate=certificate,
    )


def solve_parametric_logarithmic_derivative(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> ParametricLogarithmicDerivativeResult:
    r"""Decide the parametric logarithmic derivative in supported fields.

    Over ``C(x)`` for an admitted finite algebraic constant field ``C/QQ``,
    the Section 7.3 coefficient heuristic is the fast path and a complete
    rational structure fallback handles its inconclusive exit. The integer
    ratio ``m/n`` and all radical-logarithmic valuation relations remain
    rational, even though field coefficients lie in ``C``.

    Over a recursive primitive tower, the solver compares positive-degree
    polynomial coordinates, repeated principal parts, and simple residues
    modulo ``C*Dp`` at each level. If the current rational-coordinate system
    leaves ``m/n`` free, the same complete decision descends to the lower
    component.
    """
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    field = lower_derivation.field
    element = _coerce(field, f)

    has_hypertangent_level = _tower_contains_hypertangent_level(
        lower_derivation,
    )
    has_hyperexponential_level = _tower_contains_hyperexponential_level(
        lower_derivation,
    )
    if has_hypertangent_level or has_hyperexponential_level:
        # These two witnesses do not need any structural theorem about the
        # lower tower. In particular, an unavailable lower solver must not
        # turn ``D(1)/1`` or ``Dtheta/theta`` into an unsupported (and later
        # potentially negative) parametric query.
        if element == field.zero:
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.SOLUTION,
                n=1,
                m=0,
                argument=field.one,
            )
        if element == eta:
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.SOLUTION,
                n=1,
                m=1,
                argument=field.one,
            )

    if has_hypertangent_level:
        return _solve_hypertangent_parametric_logarithmic_derivative(
            element,
            extension,
            lower_derivation,
        )

    if has_hyperexponential_level:
        from ..constant_proofs import first_nonpreserved_constant_level

        evidence = first_nonpreserved_constant_level(lower_derivation)
        if evidence is not None:
            suffix = f": {evidence.detail}" if evidence.detail else ""
            return ParametricLogarithmicDerivativeResult(
                extension,
                lower_derivation,
                element,
                ParametricLogarithmicDerivativeOutcome.UNSUPPORTED,
                detail=(
                    "parametric logarithmic derivatives require "
                    "Const(K(t)) = Const(K) at every lower monomial level"
                    f"{suffix}"
                ),
            )
        return _solve_monomial_relation_system_parametric_logarithmic_derivative(
            element,
            extension,
            lower_derivation,
        )

    if lower_derivation.level >= 1:
        lower_extension = lower_derivation.tower.levels[lower_derivation.level]
        if (
            isinstance(lower_extension, MonomialExtension)
            and lower_extension.family is MonomialFamily.PRIMITIVE
        ):
            return _solve_primitive_parametric_logarithmic_derivative(
                f,
                extension,
                lower_derivation,
            )

    heuristic_result = solve_parametric_logarithmic_derivative_heuristic(
        f,
        extension,
        lower_derivation,
    )
    if not heuristic_result.is_inconclusive:
        return heuristic_result
    return _solve_rational_parametric_logarithmic_derivative_fallback(
        heuristic_result,
    )
