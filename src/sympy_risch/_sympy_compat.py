"""Temporary compatibility fixes for the supported released SymPy."""

from __future__ import annotations

import re
from typing import Protocol, Self, cast

import sympy
from sympy.polys.domains.field import Field


class _FieldElement(Protocol):
    """Multiplicative operation needed by the no-ring LCM fallback."""

    def __mul__(self, other: Self, /) -> Self: ...


class _MarkedPatch(Protocol):
    """Callable marker used to make installation idempotent."""

    _sympy_risch_no_ring_fast_path: bool


def _sympy_minor_version() -> tuple[int, int]:
    """Return the numeric major/minor prefix of the installed SymPy."""
    match = re.match(r"^(\d+)\.(\d+)", sympy.__version__)
    if match is None:
        raise RuntimeError(f"unrecognized SymPy version {sympy.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def install_field_no_ring_fast_path() -> bool:
    r"""Avoid ``DomainError`` control flow for fields without a ring.

    SymPy 1.14's :class:`~sympy.polys.domains.field.Field` asks for an
    associated ring in ``gcd`` and ``lcm``, then catches ``DomainError`` when
    none exists. Algebraic coefficient arithmetic can execute that fallback
    thousands of times.

    This is the generic upstream fix from SymPy commit ``48bf879232``. Remove
    this compatibility patch when the minimum SymPy version contains that
    change.
    """
    if _sympy_minor_version() >= (1, 15):
        return False
    if getattr(Field.gcd, "_sympy_risch_no_ring_fast_path", False):
        return False

    original_gcd = Field.gcd
    original_lcm = Field.lcm

    def gcd(
        self: Field,
        a: _FieldElement,
        b: _FieldElement,
    ) -> _FieldElement:
        if not self.has_assoc_Ring:
            return cast(_FieldElement, self.one)
        return cast(_FieldElement, original_gcd(self, a, b))

    def lcm(
        self: Field,
        a: _FieldElement,
        b: _FieldElement,
    ) -> _FieldElement:
        if not self.has_assoc_Ring:
            return a * b
        return cast(_FieldElement, original_lcm(self, a, b))

    cast(_MarkedPatch, gcd)._sympy_risch_no_ring_fast_path = True
    cast(_MarkedPatch, lcm)._sympy_risch_no_ring_fast_path = True
    Field.gcd = gcd
    Field.lcm = lcm
    return True


install_field_no_ring_fast_path()


__all__ = ["install_field_no_ring_fast_path"]
