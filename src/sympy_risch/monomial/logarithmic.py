r"""Recursive radical logarithmic-derivative decisions."""

# The dispatcher must initialize before the family modules below import it;
# its late imports complete the public certificate union after definition.
# ruff: noqa: I001

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sympy.polys.polytools import Poly, cancel

from .._typing import FieldElement
from ..constant_fields import is_supported_finite_constant_field
from ..rational import (
    RationalLogDerivativeResult,
    _solve_rational_logarithmic_derivative_over_domain,
)
from ..tower.core import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    _coerce,
)

if TYPE_CHECKING:
    from .hyperexponential import HyperexponentialLogarithmicDerivativeSignature
    from .hypertangent import HypertangentLogarithmicDerivativeSignature
    from .primitive import PrimitiveLogarithmicDerivativeSignature


@dataclass(frozen=True, slots=True)
class RadicalLogarithmicDerivativeResult:
    r"""Recursive decision for ``n*f = D(v)/v`` in a supported tower.

    ``minimal_power`` is the least positive integer ``n`` for which such a
    nonzero ``v`` exists.  The certificate is the complete rational result at
    the base level or the monomial signature at the current level. Unsupported
    recursive subproblems raise :class:`NotImplementedError` instead of
    constructing this result, so a returned nonsolution is a proved negative.
    """

    derivation: Derivation
    integrand: FieldElement
    minimal_power: int | None
    argument: FieldElement | None
    certificate: (
        RationalLogDerivativeResult
        | PrimitiveLogarithmicDerivativeSignature
        | HyperexponentialLogarithmicDerivativeSignature
        | HypertangentLogarithmicDerivativeSignature
    )

    @property
    def has_solution(self) -> bool:
        """Whether some positive integer multiple is logarithmic."""
        return self.minimal_power is not None

    @property
    def proves_no_solution(self) -> bool:
        """Whether the completed recursive decision proves nonsolvability."""
        return not self.has_solution

    @property
    def reconstructs(self) -> bool:
        """Whether the returned radical-logarithmic identity is exact."""
        if not self.has_solution or self.argument is None:
            return False
        field = self.derivation.field
        argument = _coerce(field, self.argument)
        return (
            self.minimal_power * _coerce(field, self.integrand)
            == self.derivation(argument) / argument
        )


def solve_radical_logarithmic_derivative(
    integrand: object,
    derivation: Derivation,
) -> RadicalLogarithmicDerivativeResult:
    r"""Decide ``n*f = D(v)/v`` recursively in a supported tower.

    The base case is the complete rational logarithmic-derivative decision in
    ``C(x)`` for ``C = QQ`` or ``QQ_I``. Primitive levels use their principal
    part signature. A hyperexponential level separates normal logarithmic
    factors and descends through its special generator exponent.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("radical logarithmic derivatives require a Derivation")

    field = derivation.field
    element = _coerce(field, integrand)
    if derivation.level == 0:
        extension = derivation.tower.levels[0]
        if not isinstance(extension, MonomialExtension):
            raise TypeError("the rational logarithmic base must be monomial")
        if is_supported_finite_constant_field(
            extension.parent_field
        ) and extension.generator_derivative == Poly(
            1,
            extension.generator,
            domain=extension.parent_field,
        ):
            variable = extension.generator
        else:
            raise NotImplementedError(
                "radical logarithmic derivatives require C(x) with Dx = 1 "
                "over an effective finite constant field C"
            )
        rational_result = _solve_rational_logarithmic_derivative_over_domain(
            field.to_sympy(element),
            variable,
            extension.parent_field,
        )
        minimal_power = rational_result.minimal_radical_power
        argument = None
        if minimal_power is not None:
            scaled_result = _solve_rational_logarithmic_derivative_over_domain(
                cancel(minimal_power * field.to_sympy(element)),
                variable,
                extension.parent_field,
            )
            if not scaled_result.has_solution:
                raise ValueError(
                    "clearing rational logarithmic residues did not produce an argument"
                )
            argument = field.from_sympy(scaled_result.argument)
        result = RadicalLogarithmicDerivativeResult(
            derivation,
            element,
            minimal_power,
            argument,
            rational_result,
        )
    else:
        extension = derivation.tower.levels[derivation.level]
        if not isinstance(extension, MonomialExtension):
            raise NotImplementedError(
                "radical logarithmic derivatives do not support algebraic tower levels"
            )
        if extension.family is MonomialFamily.PRIMITIVE:
            from .primitive import primitive_logarithmic_derivative_signature

            signature = primitive_logarithmic_derivative_signature(
                element,
                derivation,
            )
        elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
            from .hyperexponential import (
                hyperexponential_logarithmic_derivative_signature,
            )

            signature = hyperexponential_logarithmic_derivative_signature(
                element,
                derivation,
            )
        elif extension.family is MonomialFamily.HYPERTANGENT:
            from .hypertangent import hypertangent_logarithmic_derivative_signature

            signature = hypertangent_logarithmic_derivative_signature(
                element,
                derivation,
            )
        else:
            raise NotImplementedError(
                "radical logarithmic derivatives currently support rational, "
                "primitive, hyperexponential, and hypertangent tower levels"
            )
        result = RadicalLogarithmicDerivativeResult(
            derivation,
            element,
            signature.minimal_power,
            signature.relation_argument,
            signature,
        )

    if result.has_solution and not result.reconstructs:
        raise ValueError("radical logarithmic derivative witness did not reconstruct")
    return result


# The family signatures recurse through this dispatcher.  At this point its
# class and function are defined, so these imports complete the concrete,
# runtime-resolvable certificate union without an import-time cycle.
from .hyperexponential import HyperexponentialLogarithmicDerivativeSignature  # noqa: E402
from .hypertangent import HypertangentLogarithmicDerivativeSignature  # noqa: E402
from .primitive import PrimitiveLogarithmicDerivativeSignature  # noqa: E402
