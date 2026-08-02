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
from math import lcm
from typing import cast

from sympy import preorder_traversal
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

    def __init__(self, variable: Symbol, constants: frozenset[Symbol]) -> None:
        self.variable = variable
        self.constants = constants
        self.candidates: list[ExpressionTowerCandidate] = []
        self.by_source: dict[Expr, ExpressionTowerCandidate] = {}

    def visit(self, expression: Expr) -> tuple[Expr, ...]:
        """Return maximal candidate roots in ``expression``'s subtree."""
        existing = self.by_source.get(expression)
        if existing is not None:
            return (existing.source,)
        if (
            expression == self.variable
            or expression in self.constants
            or expression.is_Rational is True
        ):
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


def _rational_power_relation(
    source: Pow, generator_source: Pow
) -> tuple[int, int] | None:
    r"""Return ``(k, m)`` when ``source = base**m * generator_source**k``.

    Exponents are compared modulo integers because integer powers of the base
    already belong to the lower field.  No root absent from the source is
    introduced.
    """
    if source.base != generator_source.base:
        return None
    source_exponent = source.exp
    generator_exponent = generator_source.exp
    if not isinstance(source_exponent, Rational) or not isinstance(
        generator_exponent, Rational
    ):
        return None
    if generator_exponent.q % source_exponent.q:
        return None
    target_numerator = source_exponent.p * (generator_exponent.q // source_exponent.q)
    power = (
        target_numerator
        * pow(generator_exponent.p % generator_exponent.q, -1, generator_exponent.q)
    ) % generator_exponent.q
    base_power = source_exponent - power * generator_exponent
    if base_power.q != 1:
        return None
    return power, int(base_power)


def _rational_power_representatives(expression: Expr) -> dict[Pow, Pow]:
    """Choose existing powers that generate dependent powers of the same base."""
    groups: dict[Expr, list[Pow]] = {}
    for node in preorder_traversal(expression):
        if (
            isinstance(node, Pow)
            and isinstance(node.exp, Rational)
            and not isinstance(node.exp, Integer)
        ):
            groups.setdefault(node.base, []).append(node)

    representatives: dict[Pow, Pow] = {}
    for powers in groups.values():
        unique = tuple(dict.fromkeys(powers))
        exponents = {power: cast(Rational, power.exp) for power in unique}
        common_denominator = lcm(*(int(exponents[power].q) for power in unique))
        eligible = [
            power for power in unique if int(exponents[power].q) == common_denominator
        ]
        if not eligible:
            representatives.update((power, power) for power in unique)
            continue
        representative = min(
            eligible,
            key=lambda power: (
                abs(int(exponents[power].p)),
                exponents[power].p < 0,
            ),
        )
        representatives.update(
            (power, representative)
            if _rational_power_relation(power, representative) is not None
            else (power, power)
            for power in unique
        )
    return representatives


def _normalize_dependent_rational_powers(
    expression: Expr,
    candidates: list[ExpressionTowerCandidate],
) -> tuple[ExpressionTowerCandidate, ...]:
    """Collapse dependent rational-power proposals and restore topological order."""
    representatives = _rational_power_representatives(expression)
    retained: dict[Expr, ExpressionTowerCandidate] = {}
    for candidate in candidates:
        if (
            candidate.kind is ExpressionGeneratorKind.ALGEBRAIC
            and representatives.get(cast(Pow, candidate.source), candidate.source)
            != candidate.source
        ):
            continue
        dependencies = tuple(
            dict.fromkeys(
                representatives.get(cast(Pow, dependency), dependency)
                for dependency in candidate.dependencies
            )
        )
        retained[candidate.source] = ExpressionTowerCandidate(
            candidate.source,
            candidate.kind,
            dependencies,
            candidate.algebraic_exponent,
        )

    ordered: list[ExpressionTowerCandidate] = []
    emitted: set[Expr] = set()

    def emit(candidate: ExpressionTowerCandidate) -> None:
        if candidate.source in emitted:
            return
        for dependency in candidate.dependencies:
            dependency_candidate = retained.get(dependency)
            if dependency_candidate is not None:
                emit(dependency_candidate)
        emitted.add(candidate.source)
        ordered.append(candidate)

    for candidate in retained.values():
        emit(candidate)
    return tuple(ordered)


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
    constant_candidates = expression.free_symbols - {variable}
    unsupported_constant = next(
        (
            candidate
            for candidate in constant_candidates
            if not isinstance(candidate, Symbol)
        ),
        None,
    )
    if unsupported_constant is not None:
        return ExpressionTowerDiscovery(
            expression,
            variable,
            ExpressionDiscoveryOutcome.UNSUPPORTED,
            (),
            f"{unsupported_constant} is not a symbolic constant supported by "
            "tower discovery",
        )
    constants = cast(frozenset[Symbol], frozenset(constant_candidates))
    discoverer = _CandidateDiscoverer(variable, constants)
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
        _normalize_dependent_rational_powers(expression, discoverer.candidates),
        "the expression generator dependency graph was discovered",
    )


__all__ = [
    "ExpressionDiscoveryOutcome",
    "ExpressionGeneratorKind",
    "ExpressionTowerCandidate",
    "ExpressionTowerDiscovery",
    "discover_expression_tower",
]
