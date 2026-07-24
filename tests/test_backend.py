"""Project-wide arithmetic backend contract."""

from sympy.external.gmpy import GROUND_TYPES
from sympy.polys.domains import QQ


def test_python_flint_is_the_active_ground_type_backend() -> None:
    assert GROUND_TYPES == "flint"
    assert QQ.dtype.__module__.startswith("flint.")
