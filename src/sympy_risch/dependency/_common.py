r"""Shared validation and lower-tower helpers for dependency decisions."""

from __future__ import annotations

from operator import index as integer_index
from typing import SupportsIndex

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly

from .._typing import FieldElement
from ..constant_fields import is_supported_rational_constant_field
from ..reduction.polynomials import exact_field_equal
from ..relations.signature import FundamentalSignature, build_fundamental_signature
from ..tower.core import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)


def _parametric_logarithmic_derivative_eta(
    extension: MonomialExtension,
    lower_derivation: Derivation,
) -> FieldElement:
    r"""Validate ``Dtheta = eta*theta`` over the supplied lower field."""
    if not isinstance(extension, MonomialExtension):
        raise TypeError(
            "parametric logarithmic derivatives require a monomial extension"
        )
    if not isinstance(lower_derivation, Derivation):
        raise TypeError("parametric logarithmic derivatives require a Derivation")
    if extension.family is not MonomialFamily.HYPEREXPONENTIAL:
        raise ValueError(
            "parametric logarithmic derivatives require Dtheta = eta*theta"
        )
    if extension.parent_field != lower_derivation.field:
        raise ValueError("extension parent field does not match the lower derivation")
    return _coerce(
        lower_derivation.field,
        extension.generator_derivative.nth(1),
    )


def _integer_parameter(value: object, name: str) -> int:
    """Return an exact integer parameter, rejecting booleans and rationals."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an integer")
    try:
        return int(integer_index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def verify_parametric_logarithmic_derivative_witness(
    f: object,
    extension: MonomialExtension,
    lower_derivation: Derivation,
    n: int,
    m: int,
    argument: object,
) -> bool:
    r"""Verify one exact witness for Bronstein's equation (7.37).

    For ``Dtheta/theta = eta``, a witness consists of integers ``n != 0`` and
    ``m`` and a nonzero ``v`` in the lower field satisfying

    ``n*f = Dv/v + m*eta``.

    Native SymPy domain equality is tried first. A symbolic exact-cancellation
    fallback handles equal algebraic fraction-field units whose stored
    denominators are not canonical; numerical equality is never used.
    """
    eta = _parametric_logarithmic_derivative_eta(
        extension,
        lower_derivation,
    )
    n_integer = _integer_parameter(n, "n")
    m_integer = _integer_parameter(m, "m")
    if n_integer == 0:
        return False

    field = lower_derivation.field
    element = _coerce(field, f)
    witness_argument = _coerce(field, argument)
    if witness_argument == field.zero:
        return False

    return exact_field_equal(
        n_integer * element,
        lower_derivation(witness_argument) / witness_argument + m_integer * eta,
        field,
    )


def _uncertified_lower_tower_detail(
    lower_derivation: Derivation,
) -> str | None:
    """Explain why a proposed level lacks replayed lower-field evidence."""
    if lower_derivation.level < 0:
        return (
            "transcendental extension certification requires a supported "
            "nonconstant lower differential field"
        )
    # Local import avoids the cycle: constant-field replay delegates its
    # family decisions to the factories in this module.
    from ..constant_proofs import first_nonpreserved_constant_level

    evidence = first_nonpreserved_constant_level(lower_derivation)
    if evidence is None:
        return None
    suffix = f": {evidence.detail}" if evidence.detail else ""
    return (
        "the lower tower contains a monomial level without replayed "
        f"constant preservation at level {evidence.derivation.level} "
        f"({evidence.status.value}){suffix}"
    )


def _certified_lower_fundamental_signature(
    lower_derivation: Derivation,
) -> FundamentalSignature | None:
    r"""Return P/E/T evidence only for an already-certified lower tower.

    The relation signature deliberately accepts a standard rational base even
    when callers have not annotated its status.  Extension factories instead
    replay every lower monomial criterion, so caller metadata never becomes a
    substitute for the no-new-constants premise.
    """
    if _uncertified_lower_tower_detail(lower_derivation) is not None:
        return None
    signature = build_fundamental_signature(lower_derivation)
    return signature if signature.supported else None


def _standard_rational_variable(derivation: Derivation) -> Symbol:
    """Return ``x`` for the standard derivation on a supported ``C(x)``.

    This helper needs only the exact rational algorithms over ``C(x)``. The
    relation compilers that require rational coordinates keep their stricter
    finite-constant-field admission.
    """
    if derivation.level != 0:
        raise NotImplementedError(
            "rational-base dependency analysis requires a standard C(x) base",
        )
    constants = derivation.tower.constants
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or not is_supported_rational_constant_field(constants)
        or extension.parent_field != constants
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=constants)
    ):
        raise NotImplementedError(
            "rational-base dependency analysis requires C(x) with Dx = 1 "
            "over a supported exact rational constant field C",
        )
    return extension.generator


def _constant_field_linear_ratio(
    constraints: tuple[tuple[FieldElement, FieldElement], ...],
    field: Domain,
) -> tuple[bool, FieldElement | None]:
    r"""Solve ``left = c*right`` for one ``c in QQ`` over ``field``."""
    ratio = None
    for left, right in constraints:
        left_element = _coerce(field, left)
        right_element = _coerce(field, right)
        if right_element == field.zero:
            if left_element != field.zero:
                return False, None
            continue

        quotient = left_element / right_element
        try:
            current = QQ.from_sympy(field.to_sympy(quotient))
        except CoercionFailed:
            return False, None
        if ratio is None:
            ratio = current
        elif current != ratio:
            return False, None
    return True, ratio


def _tower_contains_hyperexponential_level(derivation: Derivation) -> bool:
    """Whether a lower tower needs the multi-direction relation fallback."""
    return any(
        isinstance(level, MonomialExtension)
        and level.family is MonomialFamily.HYPEREXPONENTIAL
        for level in derivation.tower.levels[: derivation.level + 1]
    )


def _tower_contains_hypertangent_level(derivation: Derivation) -> bool:
    """Whether a real lower tower must first enter Cayley normal form."""
    return any(
        isinstance(level, MonomialExtension)
        and level.family is MonomialFamily.HYPERTANGENT
        for level in derivation.tower.levels[: derivation.level + 1]
    )
