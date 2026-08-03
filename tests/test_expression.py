"""Expression-level compilation and integration over certified P/E/T towers."""

from dataclasses import replace
from typing import cast

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sympy import (
    Abs,
    IndexedBase,
    Poly,
    Rational,
    Symbol,
    cancel,
    diff,
    exp,
    log,
    sin,
    sqrt,
    symbols,
    tan,
)
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import RootSum

from sympy_risch._typing import FieldElement
from sympy_risch.dependency import certify_primitive_extension
from sympy_risch.elementary import (
    IntegrationObstruction,
    IntegrationObstructionKind,
    integrate_elementary,
)
from sympy_risch.expression import (
    ExpressionCompilation,
    ExpressionCompilationOutcome,
    ExpressionIntegralResult,
    ExpressionIntegrationOutcome,
    ExpressionUnsupportedReason,
    compile_expression,
    integrate_expression,
)
from sympy_risch.tower import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
)

x, a, e, t = symbols("x a e t")


def test_compiler_rejects_a_noncommutative_integration_variable() -> None:
    noncommutative_x = Symbol("noncommutative_x", commutative=False)

    with pytest.raises(TypeError, match="commutative"):
        compile_expression(noncommutative_x, noncommutative_x)


def test_compiler_rejects_an_identically_zero_integration_variable() -> None:
    zero_x = Symbol("zero_x", zero=True)

    with pytest.raises(ValueError, match="known to be zero"):
        compile_expression(zero_x, zero_x)


def test_compiler_rejects_an_infinite_integration_variable() -> None:
    infinite_x = Symbol("infinite_x", finite=False)

    with pytest.raises(ValueError, match="known infinite"):
        compile_expression(infinite_x, infinite_x)


def test_compilation_round_trips_a_rational_expression() -> None:
    expression = (x + 1) / (x**2 + 1)

    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.reason is None
    assert compilation.tower is not None
    assert compilation.element is not None
    assert len(compilation.tower.levels) == 1
    assert compilation.generator_substitutions == ()
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    native_derivative = compilation.tower.derivation()(compilation.element)
    assert (
        cancel(compilation.back_substitute(native_derivative) - diff(expression, x))
        == 0
    )


def test_logarithmic_expression_compiles_and_integrates() -> None:
    expression = 1 / (x * log(x))

    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert len(compilation.tower.levels) == 2
    assert tuple(
        source for _generator, source in compilation.generator_substitutions
    ) == (log(x),)
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == log(log(x))
    assert cancel(diff(result.antiderivative, x) - expression) == 0


def test_rational_cubic_integral_preserves_its_rootsum_presentation() -> None:
    expression = 1 / (x**3 + x + 1)

    result = integrate_expression(expression, x)

    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative is not None
    assert result.antiderivative.has(RootSum)
    assert cancel(diff(result.antiderivative, x) - expression) == 0


def test_hyperexponential_expression_compiles_and_integrates() -> None:
    expression = exp(x) / (1 + exp(x))

    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert len(compilation.tower.levels) == 2
    assert tuple(
        source for _generator, source in compilation.generator_substitutions
    ) == (exp(x),)
    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == log(1 + exp(x))
    assert cancel(diff(result.antiderivative, x) - expression) == 0


def test_nested_hypertangent_expression_compiles_and_integrates() -> None:
    expression = (1 + tan(log(x)) ** 2) / x

    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert len(compilation.tower.levels) == 3
    assert tuple(
        source for _generator, source in compilation.generator_substitutions
    ) == (
        log(x),
        tan(log(x)),
    )
    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == tan(log(x))
    assert cancel(diff(result.antiderivative, x) - expression) == 0


def test_non_elementary_expression_keeps_the_native_obstruction() -> None:
    result = integrate_expression(exp(x**2), x)

    assert result.outcome is ExpressionIntegrationOutcome.NON_ELEMENTARY
    assert result.reason is None
    assert result.native_result is not None
    assert not result.native_result.is_elementary
    assert result.antiderivative is None


@pytest.mark.parametrize(
    ("expression", "reason"),
    (
        (sin(x), ExpressionUnsupportedReason.UNSUPPORTED_FUNCTION),
        (Abs(x), ExpressionUnsupportedReason.BRANCH_SENSITIVE_FORM),
        (
            log(0, evaluate=False),
            ExpressionUnsupportedReason.SINGULAR_FUNCTION_ARGUMENT,
        ),
        (log(2), ExpressionUnsupportedReason.CONSTANT_EXTENSION),
    ),
)
def test_compiler_returns_structured_boundary_outcomes(expression, reason) -> None:
    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.outcome is ExpressionCompilationOutcome.UNSUPPORTED
    assert compilation.reason is reason
    assert compilation.tower is None
    assert compilation.element is None
    assert result.outcome is ExpressionIntegrationOutcome.UNSUPPORTED
    assert result.reason is reason
    assert result.native_result is None
    assert result.antiderivative is None


def test_compiler_infers_algebraically_independent_constant_parameters() -> None:
    expression = 1 / (x + a)

    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert compilation.tower.constants == QQ.frac_field(a)
    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == log(a + x)
    assert cancel(diff(result.antiderivative, x) - expression) == 0


def test_compiler_does_not_misclassify_a_declared_algebraic_parameter() -> None:
    algebraic_parameter = Symbol("alpha", algebraic=True)

    compilation = compile_expression(x + algebraic_parameter, x)

    assert compilation.outcome is ExpressionCompilationOutcome.UNSUPPORTED
    assert compilation.reason is ExpressionUnsupportedReason.CONSTANT_FIELD


def test_compiler_rejects_non_symbolic_free_parameters_structurally() -> None:
    compilation = compile_expression(x + IndexedBase("indexed")[0], x)

    assert compilation.outcome is ExpressionCompilationOutcome.UNSUPPORTED
    assert compilation.reason is ExpressionUnsupportedReason.CONSTANT_FIELD


def test_wide_rational_parameter_source_uses_one_native_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sympy_risch._expression_compilation import _ExpressionCompiler

    parameters = symbols("a0:8")
    expression = sum(1 / (x + parameter) for parameter in parameters)
    converted: list[object] = []
    original = _ExpressionCompiler._from_sympy

    def counting_conversion(self, source):
        converted.append(source)
        return original(self, source)

    def reject_recursive_source_derivative(*_args: object) -> None:
        raise AssertionError("a rational source should not rebuild its derivative")

    monkeypatch.setattr(_ExpressionCompiler, "_from_sympy", counting_conversion)
    monkeypatch.setattr(
        "sympy_risch._expression_compilation.source_value_and_derivative_in_native",
        reject_recursive_source_derivative,
    )

    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert converted == [expression]


@pytest.mark.parametrize(
    "expression",
    (
        log(exp(x), evaluate=False),
        exp(log(x), evaluate=False),
        log(x) + log(x**2),
        exp(x) + exp(2 * x),
        tan(x) + tan(2 * x),
        tan(x) + tan(x + 1),
    ),
)
def test_dependent_function_calls_are_not_silently_rewritten(expression) -> None:
    compilation = compile_expression(expression, x)

    assert compilation.outcome is ExpressionCompilationOutcome.UNSUPPORTED
    assert compilation.reason is ExpressionUnsupportedReason.DEPENDENT_EXTENSION
    assert compilation.tower is None
    assert compilation.generator_substitutions == ()


def test_repeated_identical_function_calls_share_one_generator() -> None:
    expression = 2 * log(x)

    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert len(compilation.tower.levels) == 2
    assert len(compilation.generator_substitutions) == 1
    assert cancel(compilation.reconstructed_expression() - expression) == 0


@pytest.mark.parametrize(
    "expression",
    (
        sqrt(x) + 1 / sqrt(x),
        x ** Rational(1, 3) + x ** Rational(2, 3) + x ** Rational(-4, 3),
    ),
)
def test_dependent_rational_powers_share_one_algebraic_level(expression) -> None:
    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert compilation.element is not None
    assert len(compilation.tower.levels) == 2
    assert len(compilation.generator_substitutions) == 1
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    assert (
        cancel(
            compilation.back_substitute(
                compilation.tower.derivation()(compilation.element)
            )
            - diff(expression, x)
        )
        == 0
    )


@pytest.mark.parametrize(
    ("expression", "degree"),
    (
        (sqrt(x) + x ** Rational(1, 3), 6),
        (sqrt(x) + sqrt(x + 1), 4),
    ),
)
def test_independent_rational_powers_use_one_primitive_extension(
    expression,
    degree,
) -> None:
    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.reason is None
    assert compilation.tower is not None
    assert compilation.element is not None
    assert len(compilation.tower.levels) == 2
    assert isinstance(compilation.tower.levels[-1], AlgebraicExtension)
    assert compilation.tower.levels[-1].field.rank == degree
    assert len(compilation.generator_substitutions) == 1
    assert len(compilation.algebraic_presentations) == 1
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    assert (
        cancel(
            compilation.back_substitute(
                compilation.tower.derivation()(compilation.element)
            )
            - diff(expression, x)
        )
        == 0
    )


@pytest.mark.parametrize(
    "expression",
    (
        sqrt(x),
        (x + 1) ** Rational(2, 3),
        sqrt(x + log(x)),
        sqrt(exp(x) + 1),
        exp(log(x) / 2, evaluate=False),
    ),
)
def test_compiler_constructs_pure_and_mixed_algebraic_towers(expression) -> None:
    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert isinstance(compilation.tower.levels[-1], AlgebraicExtension)
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    assert (
        cancel(
            compilation.back_substitute(
                compilation.tower.derivation()(compilation.element)
            )
            - diff(expression, x)
        )
        == 0
    )


def test_expression_integrator_handles_a_discovered_simple_radical() -> None:
    result = integrate_expression(sqrt(x), x)

    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == 2 * x ** Rational(3, 2) / 3
    assert cancel(diff(result.antiderivative, x) - sqrt(x)) == 0


def test_finite_power_exponential_is_reclassified_as_algebraic() -> None:
    expression = exp(log(x) / 2, evaluate=False)

    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert isinstance(compilation.tower.levels[-1], AlgebraicExtension)
    assert result.outcome is ExpressionIntegrationOutcome.ELEMENTARY
    assert result.antiderivative == 2 * x ** Rational(3, 2) / 3


@pytest.mark.parametrize(
    "expression",
    (
        sqrt(x + log(x)),
        sqrt(exp(x) + 1),
    ),
)
def test_mixed_tower_discovery_is_separate_from_integration_support(
    expression,
) -> None:
    compilation = compile_expression(expression, x)
    result = integrate_expression(expression, x)

    assert compilation.compiled
    assert result.outcome is ExpressionIntegrationOutcome.UNSUPPORTED
    assert result.reason is ExpressionUnsupportedReason.INTEGRATION_BACKEND


def _forged_tower() -> DifferentialTower:
    """Return an intentionally misdeclared primitive level for constructor tests."""
    constants = DifferentialTower(QQ)
    base = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        constants.derivation(),
    )
    rational = constants.extend(base.extension)
    return rational.extend(
        MonomialExtension(
            Poly(1, t, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )


def test_compilation_constructor_rejects_forged_source_mapping() -> None:
    tower = _forged_tower()

    with pytest.raises(ValueError, match="source substitution"):
        ExpressionCompilation(
            x**2,
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "forged mapping",
            tower,
            tower.field().from_sympy(t),
            ((t, x**2),),
        )


def test_compilation_constructor_rejects_forged_derivative_law() -> None:
    tower = _forged_tower()

    with pytest.raises(ValueError, match="independently certified"):
        ExpressionCompilation(
            log(x),
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "forged derivative law",
            tower,
            tower.field().from_sympy(t),
            ((t, log(x)),),
        )


def test_compilation_constructor_replays_valid_unknown_metadata() -> None:
    base = DifferentialTower(QQ).extend(
        MonomialExtension(Poly(1, x, domain=QQ)),
    )
    tower = base.extend(
        MonomialExtension(Poly(1 / x, t, domain=base.field())),
    )

    compilation = ExpressionCompilation(
        log(x),
        x,
        ExpressionCompilationOutcome.COMPILED,
        None,
        "replayed unknown metadata",
        tower,
        tower.field().from_sympy(t),
        ((t, log(x)),),
    )

    assert compilation.compiled
    assert compilation.tower is tower


def test_compilation_constructor_replays_nested_dependency_certification() -> None:
    constants = DifferentialTower(QQ)
    base = certify_primitive_extension(
        Poly(1, x, domain=QQ),
        constants.derivation(),
    )
    rational = constants.extend(base.extension)
    exponential = rational.extend(
        MonomialExtension(
            Poly(e, e, domain=rational.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    forged = exponential.extend(
        MonomialExtension(
            Poly(1, t, domain=exponential.field()),
            ConstantFieldStatus.PRESERVED,
        )
    )
    source = log(exp(x), evaluate=False)

    with pytest.raises(ValueError, match="independently certified"):
        ExpressionCompilation(
            source,
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "forged nested dependency",
            forged,
            forged.field().from_sympy(t),
            ((e, exp(x)), (t, source)),
        )


def test_compilation_constructor_rejects_a_malformed_tower() -> None:
    with pytest.raises(TypeError, match="DifferentialTower"):
        ExpressionCompilation(
            x,
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "malformed tower",
            cast(DifferentialTower, object()),
            cast(FieldElement, object()),
        )


def test_compilation_constructor_rejects_the_wrong_parameter_field() -> None:
    rational = compile_expression(x, x)
    assert rational.tower is not None

    with pytest.raises(ValueError, match="wrong inferred constant field"):
        ExpressionCompilation(
            x + a,
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "forged rational constant field",
            rational.tower,
            rational.tower.field().from_sympy(x),
        )


def test_compilation_constructor_rejects_an_integration_only_reason() -> None:
    with pytest.raises(ValueError, match="integration-backend reason"):
        ExpressionCompilation(
            x,
            x,
            ExpressionCompilationOutcome.UNSUPPORTED,
            ExpressionUnsupportedReason.INTEGRATION_BACKEND,
            "forged compiler boundary",
        )


def test_compilation_constructor_rejects_an_unused_generator() -> None:
    logarithmic = compile_expression(log(x), x)
    assert logarithmic.compiled
    assert logarithmic.tower is not None

    with pytest.raises(ValueError, match="must occur"):
        ExpressionCompilation(
            x,
            x,
            ExpressionCompilationOutcome.COMPILED,
            None,
            "unused generator",
            logarithmic.tower,
            logarithmic.tower.field().from_sympy(x),
            logarithmic.generator_substitutions,
        )


def test_integration_constructor_rejects_a_tampered_presentation() -> None:
    compilation = compile_expression(x, x)
    assert compilation.compiled
    assert compilation.tower is not None
    assert compilation.element is not None
    native_result = integrate_elementary(
        compilation.element,
        compilation.tower.derivation(),
    )

    with pytest.raises(ValueError, match="public presentation"):
        ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.ELEMENTARY,
            None,
            "tampered presentation",
            native_result,
            x,
        )


def test_integration_constructor_rejects_a_forged_non_elementary_claim() -> None:
    compilation = compile_expression(x, x)
    assert compilation.compiled
    assert compilation.tower is not None
    assert compilation.element is not None
    native_result = integrate_elementary(
        compilation.element,
        compilation.tower.derivation(),
    )
    fake = replace(
        native_result,
        obstruction=IntegrationObstruction(
            0,
            x,
            MonomialFamily.PRIMITIVE,
            (IntegrationObstructionKind.CONSTANT_FIELD,),
            compilation.tower.field().zero,
            None,
        ),
    )

    with pytest.raises(ValueError, match="nonzero remainder"):
        ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.NON_ELEMENTARY,
            None,
            "forged non-elementary claim",
            fake,
        )


def test_integration_constructor_rejects_a_forged_boundary_reason() -> None:
    compilation = compile_expression(x, x)

    with pytest.raises(ValueError, match="integration backend"):
        ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.UNSUPPORTED,
            ExpressionUnsupportedReason.CONSTANT_FIELD,
            "forged boundary",
        )


def test_integration_constructor_retains_the_compilation_boundary_reason() -> None:
    compilation = compile_expression(sin(x), x)
    assert compilation.reason is ExpressionUnsupportedReason.UNSUPPORTED_FUNCTION

    with pytest.raises(ValueError, match="retain its boundary reason"):
        ExpressionIntegralResult(
            compilation,
            ExpressionIntegrationOutcome.UNSUPPORTED,
            ExpressionUnsupportedReason.CONSTANT_FIELD,
            "mismatched boundary",
        )


@given(
    a_coefficient=st.integers(-4, 4),
    b_coefficient=st.integers(-4, 4),
    c_coefficient=st.integers(-4, 4),
    d_coefficient=st.integers(-4, 4),
)
@settings(max_examples=30, deadline=None)
def test_generated_rational_functions_round_trip_and_differentiate(
    a_coefficient: int,
    b_coefficient: int,
    c_coefficient: int,
    d_coefficient: int,
) -> None:
    assume(c_coefficient != 0 or d_coefficient != 0)
    expression = (a_coefficient * x + b_coefficient) / (
        c_coefficient * x + d_coefficient
    )

    compilation = compile_expression(expression, x)

    assert compilation.compiled
    assert compilation.tower is not None
    assert compilation.element is not None
    assert cancel(compilation.reconstructed_expression() - expression) == 0
    native_derivative = compilation.tower.derivation()(compilation.element)
    assert (
        cancel(compilation.back_substitute(native_derivative) - diff(expression, x))
        == 0
    )


@given(
    slope=st.integers(-4, 4),
    intercept=st.integers(-4, 4),
)
@settings(max_examples=24, deadline=None)
def test_generated_single_pet_calls_compile_and_obey_the_chain_rule(
    slope: int,
    intercept: int,
) -> None:
    assume(slope != 0)
    argument = slope * x + intercept

    for function in (log, exp, tan):
        expression = function(argument)
        compilation = compile_expression(expression, x)

        assert compilation.compiled
        assert compilation.tower is not None
        assert compilation.element is not None
        assert len(compilation.tower.levels) == 2
        native_derivative = compilation.tower.derivation()(compilation.element)
        assert (
            cancel(compilation.back_substitute(native_derivative) - diff(expression, x))
            == 0
        )
