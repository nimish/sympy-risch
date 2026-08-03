r"""Recognition of the standard rational base ``C(x)``."""

from __future__ import annotations

from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from ..constant_fields import is_supported_rational_constant_field
from ..tower.core import Derivation, MonomialExtension


def _standard_rational_base_variable(derivation: Derivation) -> Symbol:
    """Return ``x`` when ``derivation`` is standard ``d/dx`` on ``C(x)``."""
    if derivation.level != 0:
        raise NotImplementedError(
            "this reduction currently requires the lower field C(x)",
        )
    extension = derivation.tower.levels[0]
    constants = derivation.tower.constants
    if (
        not isinstance(extension, MonomialExtension)
        or not is_supported_rational_constant_field(constants)
        or extension.parent_field != constants
        or extension.generator_derivative
        != Poly(1, extension.generator, domain=constants)
    ):
        raise NotImplementedError(
            "this reduction currently requires C(x) with Dx = 1 over an "
            "supported exact rational constant field C",
        )
    return extension.generator
