from sympy import Rational, exp, log, sin, sqrt, symbols, tan

from sympy_risch.expression_discovery import (
    ExpressionDiscoveryOutcome,
    ExpressionGeneratorKind,
    discover_expression_tower,
)


def _kinds_and_sources(expression, variable):
    discovery = discover_expression_tower(expression, variable)
    assert discovery.discovered
    return [(candidate.kind, candidate.source) for candidate in discovery.candidates]


def test_discovers_algebraic_power_candidate() -> None:
    x = symbols("x")

    discovery = discover_expression_tower((x + 1) ** Rational(2, 3), x)

    assert discovery.discovered
    (candidate,) = discovery.candidates
    assert candidate.kind is ExpressionGeneratorKind.ALGEBRAIC
    assert candidate.algebraic_exponent == Rational(2, 3)
    assert candidate.dependencies == ()


def test_discovers_both_mixed_tower_directions() -> None:
    x = symbols("x")

    assert _kinds_and_sources(sqrt(x + log(x)), x) == [
        (ExpressionGeneratorKind.PRIMITIVE, log(x)),
        (ExpressionGeneratorKind.ALGEBRAIC, sqrt(x + log(x))),
    ]
    assert _kinds_and_sources(exp(sqrt(x)), x) == [
        (ExpressionGeneratorKind.ALGEBRAIC, sqrt(x)),
        (ExpressionGeneratorKind.HYPEREXPONENTIAL, exp(sqrt(x))),
    ]


def test_discovers_deep_mixed_dependencies_in_topological_order() -> None:
    x = symbols("x")
    radical = sqrt(x)
    logarithm = log(1 + radical)
    expression = tan(logarithm) + sqrt(exp(x) + 1)

    discovery = discover_expression_tower(expression, x)

    assert discovery.discovered
    positions = {
        candidate.source: index for index, candidate in enumerate(discovery.candidates)
    }
    assert positions[radical] < positions[logarithm] < positions[tan(logarithm)]
    assert positions[exp(x)] < positions[sqrt(exp(x) + 1)]
    by_source = {candidate.source: candidate for candidate in discovery.candidates}
    assert by_source[logarithm].dependencies == (radical,)
    assert by_source[tan(logarithm)].dependencies == (logarithm,)
    assert by_source[sqrt(exp(x) + 1)].dependencies == (exp(x),)


def test_common_subexpressions_share_one_candidate() -> None:
    x = symbols("x")
    radical = sqrt(x)
    expression = exp(radical) + tan(radical)

    discovery = discover_expression_tower(expression, x)

    assert discovery.discovered
    assert [candidate.source for candidate in discovery.candidates].count(radical) == 1


def test_reports_unsupported_source_grammar_without_partial_graph() -> None:
    x, y = symbols("x y")

    extra_symbol = discover_expression_tower(log(x) + y, x)
    unsupported_function = discover_expression_tower(log(x) + sin(x), x)

    assert extra_symbol.outcome is ExpressionDiscoveryOutcome.UNSUPPORTED
    assert extra_symbol.candidates == ()
    assert unsupported_function.outcome is ExpressionDiscoveryOutcome.UNSUPPORTED
    assert unsupported_function.candidates == ()
