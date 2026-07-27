r"""Independent semantic validation for compiled expression towers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sympy import cancel, diff
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
from .tower.core import (
    AlgebraicExtension,
    ConstantFieldStatus,
    DifferentialTower,
    MonomialExtension,
)

if TYPE_CHECKING:
    from ._expression_compilation import ExpressionCompilation


def source_value_in_native(
    expression: Expr,
    compilation: ExpressionCompilation,
    source_generators: dict[Expr, Symbol],
) -> FieldElement:
    """Evaluate an accepted source tree in the compiled native top field."""
    tower, _element = compilation._require_compiled()
    field = tower.field()
    generator = source_generators.get(expression)
    if generator is not None:
        return cast(FieldElement, field.from_sympy(generator))
    if expression == compilation.variable or expression.is_Rational is True:
        return cast(FieldElement, field.from_sympy(expression))
    if isinstance(expression, Add):
        value = field.zero
        for term in expression.args:
            value += source_value_in_native(term, compilation, source_generators)
        return cast(FieldElement, value)
    if isinstance(expression, Mul):
        value = field.one
        for factor in expression.args:
            value *= source_value_in_native(factor, compilation, source_generators)
        return cast(FieldElement, value)
    if isinstance(expression, Pow) and isinstance(expression.exp, Integer):
        return source_value_in_native(
            expression.base,
            compilation,
            source_generators,
        ) ** int(expression.exp)
    raise ValueError("the source contains syntax outside the compiled grammar")


def source_derivative_in_native(
    expression: Expr,
    compilation: ExpressionCompilation,
    source_generators: dict[Expr, Symbol],
) -> FieldElement:
    """Apply the source-tree chain rule without inventing new function calls."""
    tower, _element = compilation._require_compiled()
    field = tower.field()
    generator = source_generators.get(expression)
    if (
        generator is not None
        and isinstance(expression, Pow)
        and isinstance(expression.exp, Rational)
        and not isinstance(expression.exp, Integer)
    ):
        base = source_value_in_native(
            expression.base,
            compilation,
            source_generators,
        )
        if base == field.zero:
            raise ValueError("an algebraic power source has a singular base")
        native_generator = cast(FieldElement, field.from_sympy(generator))
        return (
            field.convert(expression.exp)
            * native_generator
            * source_derivative_in_native(
                expression.base,
                compilation,
                source_generators,
            )
            / base
        )
    if expression == compilation.variable:
        return cast(FieldElement, field.one)
    if expression.is_Rational is True:
        return cast(FieldElement, field.zero)
    if isinstance(expression, Add):
        value = field.zero
        for term in expression.args:
            value += source_derivative_in_native(
                term,
                compilation,
                source_generators,
            )
        return cast(FieldElement, value)
    if isinstance(expression, Mul):
        factors = tuple(
            source_value_in_native(factor, compilation, source_generators)
            for factor in expression.args
        )
        value = field.zero
        for index, _factor in enumerate(factors):
            term = source_derivative_in_native(
                expression.args[index],
                compilation,
                source_generators,
            )
            for other_index, other_factor in enumerate(factors):
                if other_index != index:
                    term *= other_factor
            value += term
        return cast(FieldElement, value)
    if isinstance(expression, Pow) and isinstance(expression.exp, Integer):
        exponent = int(expression.exp)
        if exponent == 0:
            return cast(FieldElement, field.zero)
        base = source_value_in_native(
            expression.base,
            compilation,
            source_generators,
        )
        return (
            exponent
            * base ** (exponent - 1)
            * source_derivative_in_native(
                expression.base,
                compilation,
                source_generators,
            )
        )
    if expression.func not in {log, exp, tan} or len(expression.args) != 1:
        raise ValueError("the source contains an unsupported function call")
    if generator is None:
        raise ValueError("a source function has no native generator")
    argument_expression = expression.args[0]
    if not isinstance(argument_expression, Expr):
        raise ValueError("a source function has a non-expression argument")
    argument = source_value_in_native(
        argument_expression,
        compilation,
        source_generators,
    )
    argument_derivative = source_derivative_in_native(
        argument_expression,
        compilation,
        source_generators,
    )
    native_generator = cast(FieldElement, field.from_sympy(generator))
    if expression.func is log:
        return argument_derivative / argument
    if expression.func is exp:
        return argument_derivative * native_generator
    return argument_derivative * (field.one + native_generator**2)


def _locally_equal(left: Expr, right: Expr) -> bool:
    """Compare without applying additional transcendental identities."""
    try:
        return cancel(left - right) == 0
    except (NotImplementedError, TypeError, ValueError):
        return False


def validate_compiled_extension_sources(
    compilation: ExpressionCompilation,
    substitutions: dict[Symbol, Expr],
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
    for generator, source in substitutions.items():
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
        if not compilation.expression.has(source):
            raise ValueError(
                "every source substitution must occur in the compiled expression"
            )

        lower_derivation = tower.derivation(index - 1)
        lower_replacements = {
            prior_source: prior_generator
            for prior_generator, prior_source in substitutions.items()
            if level_indices[prior_generator] < index
        }
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
        source_derivative = source_derivative_in_native(
            source,
            compilation,
            source_generators,
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


__all__ = [
    "source_derivative_in_native",
    "source_value_in_native",
    "validate_compiled_extension_sources",
]
