r"""Dependency analysis and certified extension factories."""

from __future__ import annotations

from dataclasses import replace
from math import lcm

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.polytools import Poly, cancel, invert

from .._typing import FieldElement
from ..constant_fields import (
    finite_constant_field_profile,
    gaussian_constant_field,
)
from ..tower.core import (
    ConstantFieldStatus,
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
    _rebase_monomial_derivation,
)
from ._common import (
    _certified_lower_fundamental_signature,
    _standard_rational_variable,
    _tower_contains_hypertangent_level,
    _uncertified_lower_tower_detail,
)
from .models import (
    HyperexponentialDependency,
    HyperexponentialDependencyKind,
    HypertangentDependency,
    PrimitiveExtensionAnalysis,
    TranscendentalExtensionAnalysis,
)


def _gaussian_logarithmic_relation(
    logarithmic_derivative: Expr,
    variable: Symbol,
) -> tuple[int, Expr] | None:
    r"""Find minimal ``n>0`` with ``n*f=Dv/v`` in ``QQ_I(x)``.

    A rational logarithmic derivative has no polynomial or repeated-pole
    part, and its residue at every irreducible denominator factor is an
    integer.  Rational residues become integral after multiplication by their
    denominator least common multiple.
    """
    field = QQ_I.frac_field(variable)
    element = field.from_sympy(logarithmic_derivative)
    numerator = Poly.from_dict(
        element.numer.to_dict(),
        variable,
        domain=QQ_I,
    )
    denominator = Poly.from_dict(
        element.denom.to_dict(),
        variable,
        domain=QQ_I,
    )
    polynomial_part, proper_numerator = numerator.div(denominator)
    if not polynomial_part.is_zero:
        return None
    if proper_numerator.is_zero:
        return 1, S.One
    if denominator.gcd(denominator.diff()).degree() > 0:
        return None

    residues: list[tuple[Poly, FieldElement]] = []
    power = 1
    for factor, multiplicity in denominator.factor_list()[1]:
        if multiplicity != 1:
            return None
        cofactor = denominator.exquo(factor)
        local_denominator = (cofactor * factor.diff()).rem(factor)
        residue = (proper_numerator * invert(local_denominator, factor)).rem(factor)
        if residue.degree() > 0:
            return None
        value = QQ_I.convert(residue.nth(0))
        if value.y != 0:
            return None
        rational_value = QQ.convert(value.x)
        residues.append((factor, rational_value))
        power = lcm(power, int(QQ.denom(rational_value)))

    relation_argument = S.One
    for factor, residue in residues:
        multiplicity = QQ.convert(power) * residue
        if QQ.denom(multiplicity) != 1:
            raise ValueError("Gaussian logarithmic relation did not clear residues")
        relation_argument *= factor.as_expr() ** int(QQ.numer(multiplicity))
    relation_argument = cancel(relation_argument)
    if (
        cancel(
            relation_argument.diff(variable) / relation_argument
            - power * logarithmic_derivative
        )
        != 0
    ):
        raise ValueError("Gaussian logarithmic dependency reconstruction failed")
    return power, relation_argument


def analyze_hypertangent_extension(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> HypertangentDependency:
    r"""Classify ``Dt=eta*(1+t**2)`` over a supported lower tower.

    The Cayley coordinate has logarithmic derivative ``2*i*eta``. Over the
    rational base the direct Gaussian residue test remains the fast path.
    Over recursively primitive towers, the tower is immutably rebased over
    ``C(i)`` and the recursive radical logarithmic-derivative decision is
    applied there.
    """
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "hypertangent dependency analysis requires a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError("hypertangent dependency analysis requires a Derivation")
    if extension.family is not MonomialFamily.HYPERTANGENT:
        raise ValueError(
            "hypertangent dependency analysis requires Dt = eta*(1 + t**2)"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError("extension parent field does not match the lower derivation")

    constants = lower_derivation.tower.constants
    profile = finite_constant_field_profile(constants)
    if profile is None or (not profile.is_real and constants != QQ_I):
        raise NotImplementedError(
            "hypertangent dependency analysis requires a provably real finite "
            "constant field or QQ_I",
        )

    eta = extension.parent_field.convert(extension.generator_derivative.nth(2))
    if lower_derivation.level == 0 and constants in {QQ, QQ_I}:
        variable = _standard_rational_variable(lower_derivation)
        relation = _gaussian_logarithmic_relation(
            2 * I * extension.parent_field.to_sympy(eta),
            variable,
        )
        if relation is None:
            kind = HyperexponentialDependencyKind.TRANSCENDENTAL
            minimal_power = None
            relation_argument = None
        else:
            minimal_power, relation_argument = relation
            if minimal_power == 1:
                kind = HyperexponentialDependencyKind.RATIONAL
            else:
                kind = HyperexponentialDependencyKind.ALGEBRAIC
    else:
        normal_form = None
        if _tower_contains_hypertangent_level(lower_derivation):
            from ..gaussian import build_hypertangent_gaussian_normal_form

            normal_form = build_hypertangent_gaussian_normal_form(
                lower_derivation,
            )
            gaussian_derivation = normal_form.target_derivation
        else:
            gaussian_derivation = _rebase_monomial_derivation(
                lower_derivation,
                gaussian_constant_field(constants),
            )
        gaussian_field = gaussian_derivation.field
        eta_expression = extension.parent_field.to_sympy(eta)
        if normal_form is not None:
            eta_expression = gaussian_field.to_sympy(
                normal_form.forward(eta),
            )
        gaussian_eta = gaussian_field.from_sympy(
            2 * I * eta_expression,
        )
        from ..monomial import solve_radical_logarithmic_derivative

        radical_result = solve_radical_logarithmic_derivative(
            gaussian_eta,
            gaussian_derivation,
        )
        minimal_power = radical_result.minimal_power
        relation_argument = (
            (
                normal_form.inverse_expression(radical_result.argument)
                if normal_form is not None
                else gaussian_field.to_sympy(radical_result.argument)
            )
            if radical_result.argument is not None
            else None
        )
        if radical_result.proves_no_solution:
            kind = HyperexponentialDependencyKind.TRANSCENDENTAL
        elif minimal_power == 1:
            kind = HyperexponentialDependencyKind.RATIONAL
        else:
            kind = HyperexponentialDependencyKind.ALGEBRAIC
    return HypertangentDependency(
        extension,
        eta,
        kind,
        minimal_power,
        relation_argument,
    )


def certify_primitive_extension(
    generator_derivative: Poly,
    lower_derivation: Derivation,
) -> PrimitiveExtensionAnalysis:
    r"""Construct ``K(t)``, certifying whether it preserves constants.

    The proposed derivative must have degree zero in ``t`` and its coefficient
    domain must equal the field of ``lower_derivation``.  The lower
    exact-derivative decision then gives the complete primitive criterion:

    - ``Dt = D(v)`` gives the new constant ``t - v`` and status ``ENLARGED``;
    - a proved non-exact ``Dt`` gives status ``PRESERVED``;
    - an uncertified or unsupported lower tower gives status ``UNKNOWN``.

    Invalid derivative laws and mismatched parent fields are input errors, not
    unknown mathematical outcomes, and are therefore rejected.
    """
    if not isinstance(lower_derivation, Derivation):
        raise TypeError("primitive extension certification requires a Derivation")
    extension = MonomialExtension(generator_derivative)
    if extension.family is not MonomialFamily.PRIMITIVE:
        raise ValueError(
            "primitive extension certification requires Dt in the lower field"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError("extension parent field does not match the lower derivation")

    lower_fundamental_signature = _certified_lower_fundamental_signature(
        lower_derivation,
    )
    # ``Dx=1`` is the supported base primitive extension over constants.  At
    # every higher level, lower replay evidence is a theorem prerequisite.
    detail = (
        None
        if lower_derivation.level < 0
        else _uncertified_lower_tower_detail(lower_derivation)
    )
    if detail is not None:
        return PrimitiveExtensionAnalysis(
            extension,
            None,
            detail,
            lower_fundamental_signature,
        )

    eta = _coerce(
        extension.parent_field,
        extension.generator_derivative.nth(0),
    )
    # Local import avoids the module cycle: primitive exact reduction uses
    # dependency decisions from this module.
    from ..exact import solve_exact_derivative

    try:
        exact_result = solve_exact_derivative(eta, lower_derivation)
    except NotImplementedError as exc:
        return PrimitiveExtensionAnalysis(
            extension,
            None,
            str(exc),
            lower_fundamental_signature,
        )

    status = (
        ConstantFieldStatus.ENLARGED
        if exact_result.has_solution
        else ConstantFieldStatus.PRESERVED
    )
    certified_extension = replace(
        extension,
        constant_field_status=status,
    )
    return PrimitiveExtensionAnalysis(
        certified_extension,
        exact_result,
        lower_fundamental_signature=lower_fundamental_signature,
    )


def analyze_hyperexponential_extension(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> HyperexponentialDependency:
    r"""Classify a proposed ``Dt = eta*t`` over a supported lower tower.

    The recursive logarithmic-derivative test detects both direct and radical
    dependencies.  Its ``minimal_power`` is the least positive ``n`` for
    which ``n*eta = Du/u``.

    The ordinary exact-derivative test is independent: ``eta = Dv`` records
    the presentation ``t = c*exp(v)`` for a genuinely transcendental
    candidate.
    """
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "hyperexponential dependency analysis requires a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError("hyperexponential dependency analysis requires a Derivation")
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError("hyperexponential dependency analysis requires Dt = eta*t")
    if extension.parent_field != lower_derivation.field:
        raise ValueError("extension parent field does not match the lower derivation")

    eta = extension.parent_field.convert(extension.generator_derivative.nth(1))
    # Local imports avoid the module cycle: primitive monomial reductions use
    # the dependency analyses defined in this module.
    from ..exact import solve_exact_derivative
    from ..monomial import solve_radical_logarithmic_derivative

    radical_result = solve_radical_logarithmic_derivative(
        eta,
        lower_derivation,
    )
    minimal_power = radical_result.minimal_power
    relation_argument = None
    if radical_result.proves_no_solution:
        kind = HyperexponentialDependencyKind.TRANSCENDENTAL
    elif minimal_power == 1:
        kind = HyperexponentialDependencyKind.RATIONAL
        relation_argument = lower_derivation.field.to_sympy(
            radical_result.argument,
        )
    else:
        kind = HyperexponentialDependencyKind.ALGEBRAIC
        relation_argument = lower_derivation.field.to_sympy(
            radical_result.argument,
        )

    recursive_exact_result = solve_exact_derivative(
        eta,
        lower_derivation,
    )
    exponential_exponent = (
        lower_derivation.field.to_sympy(
            recursive_exact_result.antiderivative,
        )
        if recursive_exact_result.has_solution
        else None
    )
    logarithmic_result = (
        radical_result.certificate if lower_derivation.level == 0 else radical_result
    )
    exact_result = (
        recursive_exact_result.certificate
        if lower_derivation.level == 0
        else recursive_exact_result
    )

    return HyperexponentialDependency(
        extension,
        eta,
        kind,
        minimal_power,
        relation_argument,
        logarithmic_result,
        exact_result,
        exponential_exponent,
    )


def _validate_transcendental_extension_proposal(
    generator_derivative: Poly,
    lower_derivation: Derivation,
    family: MonomialFamily,
    derivative_description: str,
) -> MonomialExtension:
    """Construct and validate a proposed exponential or tangent extension."""
    if not isinstance(lower_derivation, Derivation):
        raise TypeError("transcendental extension certification requires a Derivation")
    if not isinstance(generator_derivative, Poly):
        raise TypeError("transcendental extension certification requires a SymPy Poly")
    extension = MonomialExtension(generator_derivative)
    if extension.family is not family:
        raise ValueError(
            f"transcendental extension certification requires {derivative_description}"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError("extension parent field does not match the lower derivation")
    return extension


def certify_hyperexponential_extension(
    generator_derivative: Poly,
    lower_derivation: Derivation,
) -> TranscendentalExtensionAnalysis:
    r"""Certify constants for a proposal ``Dtheta = eta*theta``.

    The complete supported radical-logarithmic derivative decision determines
    whether ``theta**n/v`` would be a new constant. A true relation yields
    ``ENLARGED``; otherwise the hyperexponential structure theorem certifies
    ``PRESERVED``. Unsupported lower fields remain explicitly ``UNKNOWN``.
    """
    extension = _validate_transcendental_extension_proposal(
        generator_derivative,
        lower_derivation,
        MonomialFamily.HYPEREXPONENTIAL,
        "Dtheta = eta*theta",
    )
    lower_fundamental_signature = _certified_lower_fundamental_signature(
        lower_derivation,
    )
    detail = _uncertified_lower_tower_detail(lower_derivation)
    if detail is not None:
        return TranscendentalExtensionAnalysis(
            extension,
            None,
            detail,
            lower_fundamental_signature,
        )
    try:
        dependency = analyze_hyperexponential_extension(
            extension,
            lower_derivation,
        )
    except NotImplementedError as exc:
        return TranscendentalExtensionAnalysis(
            extension,
            None,
            str(exc),
            lower_fundamental_signature,
        )
    certified_extension = replace(
        extension,
        constant_field_status=dependency.inferred_constant_field_status,
    )
    return TranscendentalExtensionAnalysis(
        certified_extension,
        replace(dependency, extension=certified_extension),
        lower_fundamental_signature=lower_fundamental_signature,
    )


def certify_hypertangent_extension(
    generator_derivative: Poly,
    lower_derivation: Derivation,
) -> TranscendentalExtensionAnalysis:
    r"""Certify constants for a proposal ``Dt = eta*(1+t**2)``.

    Cayley transforms the candidate to a hyperexponential generator over the
    Gaussian constant field. The resulting radical relation is therefore the
    exact constant-field criterion for the real hypertangent proposal.
    """
    extension = _validate_transcendental_extension_proposal(
        generator_derivative,
        lower_derivation,
        MonomialFamily.HYPERTANGENT,
        "Dt = eta*(1 + t**2)",
    )
    lower_fundamental_signature = _certified_lower_fundamental_signature(
        lower_derivation,
    )
    detail = _uncertified_lower_tower_detail(lower_derivation)
    if detail is not None:
        return TranscendentalExtensionAnalysis(
            extension,
            None,
            detail,
            lower_fundamental_signature,
        )
    try:
        dependency = analyze_hypertangent_extension(
            extension,
            lower_derivation,
        )
    except NotImplementedError as exc:
        return TranscendentalExtensionAnalysis(
            extension,
            None,
            str(exc),
            lower_fundamental_signature,
        )
    certified_extension = replace(
        extension,
        constant_field_status=dependency.inferred_constant_field_status,
    )
    return TranscendentalExtensionAnalysis(
        certified_extension,
        replace(dependency, extension=certified_extension),
        lower_fundamental_signature=lower_fundamental_signature,
    )
