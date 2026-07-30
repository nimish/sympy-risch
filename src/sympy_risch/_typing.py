"""Shared static types for exact differential-field arithmetic.

SymPy's :class:`~sympy.polys.domains.domain.Domain` is intentionally dynamic:
``Domain.convert`` returns the backend-specific element type selected at
runtime.  For this project that can be a python-flint ground element, a
``FracElement``, or a finite-extension element.  ``FieldElement`` describes
the arithmetic all of those native values must support without collapsing
them to ``Any``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sympy.core.expr import Expr


@runtime_checkable
class SupportsAsExpr(Protocol):
    """Exact value that can expose an equivalent SymPy expression."""

    def as_expr(self) -> Expr: ...


class FieldElement(Protocol):
    """Native element of one exact field in a differential tower."""

    def __add__(self, other: object, /) -> FieldElement: ...

    def __radd__(self, other: object, /) -> FieldElement: ...

    def __sub__(self, other: object, /) -> FieldElement: ...

    def __rsub__(self, other: object, /) -> FieldElement: ...

    def __mul__(self, other: object, /) -> FieldElement: ...

    def __rmul__(self, other: object, /) -> FieldElement: ...

    def __truediv__(self, other: object, /) -> FieldElement: ...

    def __rtruediv__(self, other: object, /) -> FieldElement: ...

    def __pow__(self, exponent: int, /) -> FieldElement: ...

    def __neg__(self) -> FieldElement: ...


type ExactInput = Expr | int | SupportsAsExpr
"""Public exact input accepted by expression conversion boundaries."""

type FieldInput = ExactInput | FieldElement
"""Public exact input accepted when the owning differential field is known."""


def is_field_zero(value: FieldElement) -> bool:
    """Return whether an exact native field element is zero.

    Do not compare a :class:`FieldElement` directly with the Python integer
    ``0``.  SymPy's fraction-field and finite-extension elements intentionally
    need not compare equal to an integer from a different domain.  Subtracting
    an element from itself instead constructs zero in its own native domain.

    Callers that already have the owning ``Domain`` can equivalently compare
    against ``field.zero``.  This helper is for small result objects that
    retain elements but not their domain.
    """
    return value == value - value
