r"""Recognition of the standard rational base ``QQ(x)``."""

from __future__ import annotations

from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ..tower.core import Derivation, MonomialExtension


def _standard_rational_base_variable(derivation: Derivation) -> Symbol:
    """Return ``x`` when ``derivation`` is the standard ``d/dx`` on ``QQ(x)``."""
    if derivation.level != 0:
        raise NotImplementedError(
            "this reduction currently requires the lower field QQ(x)"
        )
    extension = derivation.tower.levels[0]
    if (
        not isinstance(extension, MonomialExtension)
        or extension.parent_field != QQ
        or extension.generator_derivative != Poly(1, extension.generator, domain=QQ)
    ):
        raise NotImplementedError("this reduction currently requires QQ(x) with Dx = 1")
    return extension.generator
