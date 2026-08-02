r"""Independent semantic validation for compiled expression towers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sympy import cancel, diff, preorder_traversal
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import tan
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly

from ._typing import FieldElement
from .dependency import (
    certify_hyperexponential_extension,
    certify_hypertangent_extension,
    certify_primitive_extension,
)
from .expression_discovery import _rational_power_relation
from .tower.core import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
)

if TYPE_CHECKING:
    from ._expression_compilation import ExpressionCompilation


def _source_value_and_derivative_in_native(
    expression: Expr,
    compilation: ExpressionCompilation,
    source_generators: dict[Expr, Symbol],
    algebraic_representatives: dict[Pow, Pow],
    cache: dict[Expr, tuple[FieldElement, FieldElement]],
) -> tuple[FieldElement, FieldElement]:
    """Evaluate one source subtree and its derivative in a single traversal."""
    cached = cache.get(expression)
    if cached is not None:
        return cached

    tower, _element = compilation._require_compiled()
    field = tower.field()
    generator = source_generators.get(expression)
    native_generator = (
        cast(FieldElement, field.from_sympy(generator))
        if generator is not None
        else None
    )
    if (
        isinstance(expression, Pow)
        and isinstance(expression.exp, Rational)
        and not isinstance(expression.exp, Integer)
    ):
        base, base_derivative = _source_value_and_derivative_in_native(
            expression.base,
            compilation,
            source_generators,
            algebraic_representatives,
            cache,
        )
        if base == field.zero:
            raise ValueError("an algebraic power source has a singular base")
        value = native_generator
        if value is None:
            representative = algebraic_representatives.get(expression)
            relation = (
                _rational_power_relation(expression, representative)
                if representative is not None
                else None
            )
            representative_generator = source_generators.get(representative)
            if relation is None or representative_generator is None:
                raise ValueError("an algebraic power source has no native generator")
            power, base_power = relation
            value = cast(
                FieldElement,
                base**base_power * field.from_sympy(representative_generator) ** power,
            )
        result = (
            value,
            cast(
                FieldElement,
                field.convert(expression.exp) * value * base_derivative / base,
            ),
        )
    elif expression == compilation.variable:
        result = (cast(FieldElement, field.from_sympy(expression)), field.one)
    elif expression.is_Rational is True:
        result = (cast(FieldElement, field.from_sympy(expression)), field.zero)
    elif isinstance(expression, Add):
        value = field.zero
        derivative = field.zero
        for term in expression.args:
            term_value, term_derivative = _source_value_and_derivative_in_native(
                term,
                compilation,
                source_generators,
                algebraic_representatives,
                cache,
            )
            value += term_value
            derivative += term_derivative
        result = (cast(FieldElement, value), cast(FieldElement, derivative))
    elif isinstance(expression, Mul):
        value = field.one
        derivative = field.zero
        for factor in expression.args:
            factor_value, factor_derivative = _source_value_and_derivative_in_native(
                factor,
                compilation,
                source_generators,
                algebraic_representatives,
                cache,
            )
            derivative = derivative * factor_value + value * factor_derivative
            value *= factor_value
        result = (cast(FieldElement, value), cast(FieldElement, derivative))
    elif isinstance(expression, Pow) and isinstance(expression.exp, Integer):
        base, base_derivative = _source_value_and_derivative_in_native(
            expression.base,
            compilation,
            source_generators,
            algebraic_representatives,
            cache,
        )
        exponent = int(expression.exp)
        result = (
            base**exponent,
            cast(
                FieldElement,
                field.zero
                if exponent == 0
                else exponent * base ** (exponent - 1) * base_derivative,
            ),
        )
    elif expression.func in {log, exp, tan} and len(expression.args) == 1:
        if native_generator is None:
            raise ValueError("a source function has no native generator")
        argument_expression = expression.args[0]
        if not isinstance(argument_expression, Expr):
            raise ValueError("a source function has a non-expression argument")
        argument, argument_derivative = _source_value_and_derivative_in_native(
            argument_expression,
            compilation,
            source_generators,
            algebraic_representatives,
            cache,
        )
        if expression.func is log:
            derivative = argument_derivative / argument
        elif expression.func is exp:
            derivative = argument_derivative * native_generator
        else:
            derivative = argument_derivative * (field.one + native_generator**2)
        result = (native_generator, cast(FieldElement, derivative))
    else:
        raise ValueError("the source contains syntax outside the compiled grammar")

    cache[expression] = result
    return result


def source_value_and_derivative_in_native(
    expression: Expr,
    compilation: ExpressionCompilation,
    source_generators: dict[Expr, Symbol],
    algebraic_representatives: dict[Pow, Pow],
) -> tuple[FieldElement, FieldElement]:
    """Evaluate an accepted source tree and derivative in one traversal."""
    return _source_value_and_derivative_in_native(
        expression,
        compilation,
        source_generators,
        algebraic_representatives,
        {},
    )


def _locally_equal(left: Expr, right: Expr) -> bool:
    """Compare without applying additional transcendental identities."""
    try:
        return cancel(left - right) == 0
    except (NotImplementedError, TypeError, ValueError):
        return False


def validate_compiled_extension_sources(
    compilation: ExpressionCompilation,
    substitutions: dict[Symbol, Expr],
    algebraic_representatives: dict[Pow, Pow],
) -> None:
    """Replay every source extension through its checked construction."""
    tower, _element = compilation._require_compiled()
    if not tower.levels:
        raise ValueError("a compiled expression needs the QQ(x) base level")

    constants = DifferentialTower(QQ)
    base_analysis = certify_primitive_extension(
        Poly(1, compilation.variable, domain=QQ),
        constants.derivation(),
    )
    base_extension = tower.levels[0]
    if (
        not isinstance(base_extension, MonomialExtension)
        or base_extension.generator_derivative
        != base_analysis.extension.generator_derivative
    ):
        raise ValueError("the compiled base is not the certified standard QQ(x) tower")

    level_indices = {
        extension.generator: index for index, extension in enumerate(tower.levels)
    }
    expected_generators = set(level_indices) - {tower.levels[0].generator}
    if set(substitutions) != expected_generators:
        raise ValueError(
            "compiled source substitutions must cover exactly the non-base generators"
        )

    source_generators = {
        source: generator for generator, source in substitutions.items()
    }
    if len(source_generators) != len(substitutions):
        raise ValueError("each source call needs at most one native generator")
    expression_nodes = set(preorder_traversal(compilation.expression))
    lower_replacements: dict[Expr, Symbol] = {}
    source_evaluation_cache: dict[Expr, tuple[FieldElement, FieldElement]] = {}
    ordered_substitutions = sorted(
        substitutions.items(),
        key=lambda item: level_indices[item[0]],
    )
    for generator, source in ordered_substitutions:
        index = level_indices[generator]
        extension = tower.levels[index]
        if source.free_symbols & expected_generators:
            raise ValueError("a source substitution cannot contain native generators")
        is_algebraic_power = (
            isinstance(source, Pow)
            and isinstance(source.exp, Rational)
            and not isinstance(source.exp, Integer)
        )
        is_supported_function = source.func in {log, exp, tan} and len(source.args) == 1
        if (
            not is_algebraic_power and not is_supported_function
        ) or source.free_symbols - {compilation.variable}:
            raise ValueError("a source substitution is outside the accepted grammar")
        if source not in expression_nodes:
            raise ValueError(
                "every source substitution must occur in the compiled expression"
            )

        lower_derivation = tower.derivation(index - 1)
        native_argument_expression = source.args[0].xreplace(lower_replacements)
        try:
            argument = cast(
                FieldElement,
                lower_derivation.field.from_sympy(native_argument_expression),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "a source extension argument does not belong to its lower tower"
            ) from exc
        argument_derivative = lower_derivation(argument)
        if isinstance(extension, AlgebraicExtension):
            if is_algebraic_power:
                exponent = cast(Rational, source.exp)
                relation_value = argument ** int(exponent.p)
                relation_degree = int(exponent.q)
            elif source.func is exp:
                relation_degree = extension.minimal_polynomial.degree()
                powered_source = exp(relation_degree * source.args[0])
                native_powered_source = powered_source.xreplace(lower_replacements)
                try:
                    relation_value = lower_derivation.field.from_sympy(
                        native_powered_source
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "an algebraic exponential relation does not belong to "
                        "its lower tower"
                    ) from exc
            else:
                raise ValueError(
                    "an algebraic compiled level needs a rational-power or "
                    "finite-power exponential source"
                )
            expected_modulus = Poly(
                generator**relation_degree
                - lower_derivation.field.to_sympy(relation_value),
                generator,
                domain=lower_derivation.field,
            )
            if extension.minimal_polynomial != expected_modulus:
                raise ValueError(
                    "an algebraic source does not match its defining polynomial"
                )
        elif not isinstance(extension, MonomialExtension):
            raise ValueError("the compiled tower contains an unsupported level")
        elif source.func is log:
            if argument == lower_derivation.field.zero:
                raise ValueError("a logarithmic source extension has zero argument")
            eta = argument_derivative / argument
            analysis = certify_primitive_extension(
                Poly(
                    lower_derivation.field.to_sympy(eta),
                    generator,
                    domain=lower_derivation.field,
                ),
                lower_derivation,
            )
        elif source.func is exp:
            analysis = certify_hyperexponential_extension(
                Poly(
                    lower_derivation.field.to_sympy(argument_derivative) * generator,
                    generator,
                    domain=lower_derivation.field,
                ),
                lower_derivation,
            )
        else:
            analysis = certify_hypertangent_extension(
                Poly(
                    lower_derivation.field.to_sympy(argument_derivative)
                    * (1 + generator**2),
                    generator,
                    domain=lower_derivation.field,
                ),
                lower_derivation,
            )
        if isinstance(extension, MonomialExtension) and (
            analysis.status is not ConstantFieldStatus.PRESERVED
            or analysis.extension.generator_derivative != extension.generator_derivative
        ):
            raise ValueError(
                "a source extension is not independently certified with "
                "preserved constants"
            )

        native_generator = cast(FieldElement, tower.field().from_sympy(generator))
        _source_value, source_derivative = _source_value_and_derivative_in_native(
            source,
            compilation,
            source_generators,
            algebraic_representatives,
            source_evaluation_cache,
        )
        if tower.derivation()(native_generator) != source_derivative:
            raise ValueError(
                "a native generator derivative disagrees with its source function"
            )
        presented_derivative = compilation.back_substitute_expression(
            tower.field().to_sympy(tower.derivation()(native_generator)),
        )
        if not _locally_equal(
            presented_derivative,
            diff(source, compilation.variable),
        ):
            raise ValueError(
                "a native generator derivative fails its back-substituted "
                "local chain-rule check"
            )
        lower_replacements[source] = generator


__all__ = [
    "source_value_and_derivative_in_native",
    "validate_compiled_extension_sources",
]
