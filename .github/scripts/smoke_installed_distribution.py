"""Verify that an installed distribution contains the public package contract."""

from importlib.metadata import metadata
from importlib.resources import files
from pathlib import Path

from sympy import cancel, symbols
from sympy.external.gmpy import GROUND_TYPES
from sympy.polys.domains import QQ

import sympy_risch


def main() -> None:
    package_path = Path(sympy_risch.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(
            f"sympy_risch was not imported from an installed artifact: {package_path}"
        )

    requires_python = metadata("sympy-risch")["Requires-Python"]
    if requires_python != ">=3.13":
        raise RuntimeError(f"unexpected Requires-Python metadata: {requires_python}")

    typing_marker = files("sympy_risch").joinpath("py.typed")
    if not typing_marker.is_file():
        raise RuntimeError("the installed package does not contain py.typed")

    if GROUND_TYPES != "flint" or not QQ.dtype.__module__.startswith("flint."):
        raise RuntimeError("python-flint is not the active SymPy ground-type backend")

    x = symbols("x")
    integrand = 1 / (x + 1)
    antiderivative = sympy_risch.integrate_rational(integrand, x)
    if cancel(antiderivative.diff(x) - integrand) != 0:
        raise RuntimeError("installed-package rational integration smoke test failed")


if __name__ == "__main__":
    main()
