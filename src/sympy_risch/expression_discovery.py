r"""Discover differential-field generators in a SymPy expression.

This module is deliberately independent of the native tower backend.  It
answers the first question in automatic Risch integration: which
subexpressions may require field generators, and in what dependency order?
Actual algebraic-dependence and constant-field decisions belong to the tower
compiler, where the relevant lower differential field is available.

The discovery pass recognizes rational arithmetic, rational powers, and the
primitive/exponential/tangent families.  A rational power ``u**(p/q)`` is an
algebraic proposal satisfying ``Y**q = u**p``; this records the selected
source branch without claiming that the displayed polynomial is already
minimal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import tan


class ExpressionGeneratorKind(Enum):
    """Structural family proposed by one source subexpression."""

    ALGEBRAIC = "algebraic"
    PRIMITIVE = "primitive"
    HYPEREXPONENTIAL = "hyperexponential"
    HYPERTANGENT = "hypertangent"


class ExpressionDiscoveryOutcome(Enum):
    """Whether the source grammar could be analyzed."""

    DISCOVERED = "discovered"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ExpressionTowerCandidate:
    """One proposed generator and its direct extension dependencies."""

    source: Expr
    kind: ExpressionGeneratorKind
    dependencies: tuple[Expr, ...]
    algebraic_exponent: Rational | None = None

    def __post_init__(self) -> None:
        if self.kind is ExpressionGeneratorKind.ALGEBRAIC:
            exponent = self.algebraic_exponent
            if exponent is None or exponent.q == 1:
                raise ValueError(
                    "an algebraic power candidate needs a nonintegral rational exponent"
                )
        elif self.algebraic_exponent is not None:
            raise ValueError(
                "only algebraic power candidates carry an algebraic exponent"
            )


@dataclass(frozen=True, slots=True)
class ExpressionTowerDiscovery:
    """Immutable generator DAG in bottom-up topological order."""

    expression: Expr
    variable: Symbol
    outcome: ExpressionDiscoveryOutcome
    candidates: tuple[ExpressionTowerCandidate, ...]
    detail: str

    @property
    def discovered(self) -> bool:
        """Whether every source node belongs to the supported grammar."""
        return self.outcome is ExpressionDiscoveryOutcome.DISCOVERED


class _UnsupportedDiscovery(Exception):
    """Internal early exit for syntax outside the discovery grammar."""


class _CandidateDiscoverer:
    """One-pass DAG builder with expression equality as common-subtree sharing."""

    def __init__(self, variable: Symbol) -> None:
        self.variable = variable
        self.candidates: list[ExpressionTowerCandidate] = []
        self.by_source: dict[Expr, ExpressionTowerCandidate] = {}

    def visit(self, expression: Expr) -> tuple[Expr, ...]:
        """Return maximal candidate roots in ``expression``'s subtree."""
        existing = self.by_source.get(expression)
        if existing is not None:
            return (existing.source,)
        if expression == self.variable or expression.is_Rational is True:
            return ()
        if expression.is_number:
            raise _UnsupportedDiscovery(
                f"{expression} is not a rational constant in QQ"
            )
        if isinstance(expression, (Add, Mul)):
            return self._merge_roots(
                root for argument in expression.args for root in self.visit(argument)
            )
        if isinstance(expression, Pow):
            dependencies = self.visit(expression.base)
            if isinstance(expression.exp, Integer):
                return dependencies
            if isinstance(expression.exp, Rational):
                return self._add(
                    expression,
                    ExpressionGeneratorKind.ALGEBRAIC,
                    dependencies,
                    expression.exp,
                )
            raise _UnsupportedDiscovery(f"{expression} has a non-rational exponent")
        if expression.func in {log, exp, tan} and len(expression.args) == 1:
            argument = expression.args[0]
            if not isinstance(argument, Expr):
                raise _UnsupportedDiscovery(
                    f"{expression.func.__name__} has a non-expression argument"
                )
            kind = {
                log: ExpressionGeneratorKind.PRIMITIVE,
                exp: ExpressionGeneratorKind.HYPEREXPONENTIAL,
                tan: ExpressionGeneratorKind.HYPERTANGENT,
            }[expression.func]
            return self._add(expression, kind, self.visit(argument))
        raise _UnsupportedDiscovery(
            f"{expression.func.__name__} is outside rational arithmetic, "
            "rational powers, and the supported log/exp/tan families"
        )

    def _add(
        self,
        source: Expr,
        kind: ExpressionGeneratorKind,
        dependencies: tuple[Expr, ...],
        exponent: Rational | None = None,
    ) -> tuple[Expr, ...]:
        candidate = ExpressionTowerCandidate(
            source,
            kind,
            dependencies,
            exponent,
        )
        self.by_source[source] = candidate
        self.candidates.append(candidate)
        return (source,)

    @staticmethod
    def _merge_roots(roots: Iterable[Expr]) -> tuple[Expr, ...]:
        """Deduplicate roots while retaining deterministic traversal order."""
        return tuple(dict.fromkeys(roots))


def discover_expression_tower(
    expression: Expr,
    variable: Symbol,
) -> ExpressionTowerDiscovery:
    r"""Discover algebraic and P/E/T generator candidates bottom-up.

    This operation does not yet assert algebraic independence.  Its output is
    a dependency graph for the relation-aware compiler, so it remains useful
    even when the current native integration backend cannot execute the final
    mixed tower.
    """
    if not isinstance(expression, Expr):
        raise TypeError("tower discovery requires a SymPy Expr")
    if not isinstance(variable, Symbol):
        raise TypeError("the integration variable must be a SymPy Symbol")
    foreign_symbols = expression.free_symbols - {variable}
    if foreign_symbols:
        names = ", ".join(str(symbol) for symbol in sorted(foreign_symbols, key=str))
        return ExpressionTowerDiscovery(
            expression,
            variable,
            ExpressionDiscoveryOutcome.UNSUPPORTED,
            (),
            f"the expression contains symbols other than {variable}: {names}",
        )

    discoverer = _CandidateDiscoverer(variable)
    try:
        discoverer.visit(expression)
    except _UnsupportedDiscovery as failure:
        return ExpressionTowerDiscovery(
            expression,
            variable,
            ExpressionDiscoveryOutcome.UNSUPPORTED,
            (),
            str(failure),
        )
    return ExpressionTowerDiscovery(
        expression,
        variable,
        ExpressionDiscoveryOutcome.DISCOVERED,
        tuple(discoverer.candidates),
        "the expression generator dependency graph was discovered",
    )


__all__ = [
    "ExpressionDiscoveryOutcome",
    "ExpressionGeneratorKind",
    "ExpressionTowerCandidate",
    "ExpressionTowerDiscovery",
    "discover_expression_tower",
]
