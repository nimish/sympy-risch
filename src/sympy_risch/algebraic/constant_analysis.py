r"""Proof-carrying constant-field analysis for algebraic extensions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import cast

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.polys.agca.extensions import ExtensionElement
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.polytools import Poly

from ..tower import AlgebraicExtension, Derivation, MonomialExtension
from ._field import (
    _base_polynomial_ring,
    _embed_parent,
    _evaluate_constant_polynomial,
    _extension_element,
    _same_univariate_polynomial_coefficients,
)
from .basis import SimpleRadicalData, recognize_simple_radical


class AlgebraicConstantFieldStatus(Enum):
    r"""Certified status of the constants of one algebraic function field.

    ``REBASING_REQUIRED`` says that this analysis has exhibited a new
    algebraic constant and an exact embedding of its generated constant field
    into the presented function field.  It deliberately does *not* claim
    that the current tower backend can already rewrite the full extension over
    that larger field.
    """

    PRESERVED = "preserved"
    REBASING_REQUIRED = "rebasing_required"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AlgebraicConstantEmbedding:
    r"""An exact embedding of a proposed algebraic constant field into ``L``.

    The embedding sends the primitive generator selected by SymPy's
    ``constant_domain`` to ``generator_image`` in ``source_extension``.  It
    is a certificate for the *constant subfield* only; turning it into a
    re-based function-field presentation can lower the algebraic degree and
    is a separate operation.
    """

    source_extension: AlgebraicExtension
    constant_domain: AlgebraicField
    minimal_polynomial: Poly
    generator_image: ExtensionElement

    def __post_init__(self) -> None:
        if not isinstance(self.source_extension, AlgebraicExtension):
            raise TypeError("a constant embedding needs an algebraic extension")
        if not isinstance(self.constant_domain, AlgebraicField):
            raise TypeError("a constant embedding needs an algebraic number field")
        if not isinstance(self.minimal_polynomial, Poly):
            raise TypeError("a constant embedding needs a SymPy Poly")
        _, constants, _ = _base_polynomial_ring(self.source_extension)
        if self.minimal_polynomial.domain != constants:
            raise ValueError(
                "the constant polynomial must be over the displayed constants"
            )
        if self.minimal_polynomial.degree() < 2:
            raise ValueError("a rebasing certificate needs a nontrivial polynomial")
        if self.minimal_polynomial.is_irreducible is not True:
            raise ValueError("the constant polynomial must be irreducible")
        target_polynomial = self.constant_domain.ext.minpoly
        if not _same_univariate_polynomial_coefficients(
            target_polynomial,
            self.minimal_polynomial,
        ):
            raise ValueError(
                "the algebraic-number domain must realize the stated polynomial"
            )
        image = _extension_element(self.source_extension, self.generator_image)
        object.__setattr__(
            self,
            "generator_image",
            image,
        )
        if (
            _evaluate_constant_polynomial(
                self.source_extension,
                self.minimal_polynomial,
                image,
            )
            != self.source_extension.field.zero
        ):
            raise ValueError(
                "the proposed constant image does not satisfy its minimal polynomial"
            )

    def embed(self, value: object) -> ExtensionElement:
        r"""Map an exact proposed constant into the source function field."""
        constant = self.constant_domain.convert(value)
        root = cast(Expr, self.constant_domain.ext.root)
        expression = cast(Expr, self.constant_domain.to_sympy(constant))
        image = cast(
            Expr,
            self.source_extension.field.to_sympy(self.generator_image),
        )
        return _extension_element(
            self.source_extension,
            expression.xreplace({root: image}),
        )


@dataclass(frozen=True, slots=True)
class AlgebraicConstantFieldAnalysis:
    r"""Sound constant-field analysis for one supported algebraic extension.

    The current theorem-backed scope is a relation already over ``C`` or a
    simple radical over the standard rational differential field ``C(x)``,
    ``Dx = 1``. An unsupported result is a boundary of this analyzer, not
    evidence that the displayed constants are correct.
    """

    extension: AlgebraicExtension
    derivation: Derivation
    status: AlgebraicConstantFieldStatus
    radical: SimpleRadicalData | None
    exponent_gcd: int | None
    embedding: AlgebraicConstantEmbedding | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.extension, AlgebraicExtension):
            raise TypeError("an algebraic constant analysis needs an extension")
        if not isinstance(self.derivation, Derivation):
            raise TypeError("an algebraic constant analysis needs a derivation")
        if not isinstance(self.status, AlgebraicConstantFieldStatus):
            raise TypeError("the constant-field status has the wrong type")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("an algebraic constant analysis needs a detail")
        if self.derivation.level < 0 or (
            self.derivation.tower.levels[self.derivation.level] is not self.extension
        ):
            raise ValueError("the derivation must be bound to the extension")
        if self.radical is not None and self.radical.extension is not self.extension:
            raise ValueError("the radical data belongs to another extension")
        if self.exponent_gcd is not None and self.exponent_gcd < 1:
            raise ValueError("a radical exponent gcd must be positive")
        if self.embedding is not None and (
            self.embedding.source_extension is not self.extension
        ):
            raise ValueError("the constant embedding belongs to another extension")

        base_detail = _algebraic_constant_base_detail(
            self.extension,
            self.derivation,
        )
        if self.status is AlgebraicConstantFieldStatus.PRESERVED:
            if base_detail is not None:
                raise ValueError(base_detail)
            recognized = recognize_simple_radical(self.extension)
            if recognized != self.radical:
                raise ValueError("a preserved result needs recognized radical data")
            if recognized is None:
                raise ValueError("a preserved result needs a geometric radical proof")
            exponent_gcd, _witness, _polynomial = _radical_constant_certificate(
                recognized,
            )
            if self.exponent_gcd != exponent_gcd or exponent_gcd != 1:
                raise ValueError("a preserved result needs a geometric radical proof")
            if self.embedding is not None:
                raise ValueError("a preserved result cannot carry a new constant")
        elif self.status is AlgebraicConstantFieldStatus.REBASING_REQUIRED:
            if base_detail is not None:
                raise ValueError(base_detail)
            if self.embedding is None:
                raise ValueError("a rebasing result needs a constant embedding")
            if self.radical is None:
                constant_relation = _constant_minimal_polynomial(self.extension)
                if constant_relation is None:
                    raise ValueError(
                        "a non-radical rebasing result needs a constant relation"
                    )
                if self.exponent_gcd is not None:
                    raise ValueError(
                        "a non-radical rebasing result cannot carry a radical gcd"
                    )
                if self.embedding.generator_image != self.extension.field.generator:
                    raise ValueError(
                        "a constant-relation certificate must map its generator"
                    )
                if not _same_univariate_polynomial_coefficients(
                    self.embedding.minimal_polynomial,
                    constant_relation,
                ):
                    raise ValueError(
                        "the constant relation and embedding polynomial disagree"
                    )
            else:
                recognized = recognize_simple_radical(self.extension)
                if recognized != self.radical:
                    raise ValueError("a rebasing result needs recognized radical data")
                if recognized is None:
                    raise ValueError("a rebasing result needs radical data")
                exponent_gcd, witness, polynomial = _radical_constant_certificate(
                    recognized,
                )
                if self.exponent_gcd != exponent_gcd or exponent_gcd == 1:
                    raise ValueError("a radical rebasing result needs a nontrivial gcd")
                if self.embedding.generator_image != witness:
                    raise ValueError("the radical constant witness is inconsistent")
                if not _same_univariate_polynomial_coefficients(
                    self.embedding.minimal_polynomial,
                    polynomial,
                ):
                    raise ValueError("the radical constant polynomial is inconsistent")
        elif self.embedding is not None:
            raise ValueError("an unsupported result cannot carry a certificate")

    @property
    def preserved(self) -> bool:
        """Whether the declared constants are proved exact in this scope."""
        return self.status is AlgebraicConstantFieldStatus.PRESERVED

    @property
    def rebasing_required(self) -> bool:
        """Whether a new algebraic constant has been exhibited exactly."""
        return self.status is AlgebraicConstantFieldStatus.REBASING_REQUIRED

    @property
    def unsupported(self) -> bool:
        """Whether this analyzer cannot decide the presented extension."""
        return self.status is AlgebraicConstantFieldStatus.UNSUPPORTED

    @property
    def constant_witness(self) -> ExtensionElement | None:
        """Return the displayed new constant when rebasing is required."""
        if self.embedding is None:
            return None
        return self.embedding.generator_image

    @property
    def constant_minimal_polynomial(self) -> Poly | None:
        """Return the minimal polynomial of the displayed new constant."""
        if self.embedding is None:
            return None
        return self.embedding.minimal_polynomial

    @property
    def proposed_constant_domain(self) -> AlgebraicField | None:
        """Return the exact constant field embedded by this analysis."""
        if self.embedding is None:
            return None
        return self.embedding.constant_domain


def _radical_constant_certificate(
    radical: SimpleRadicalData,
) -> tuple[int, ExtensionElement, Poly]:
    """Return the exponent gcd, constant witness, and its polynomial."""
    extension = radical.extension
    _, constants, variable = _base_polynomial_ring(extension)
    unit, factors = radical.normalized_radicand.sqf_list()
    exponent_gcd = radical.degree
    for _factor, multiplicity in factors:
        exponent_gcd = gcd(exponent_gcd, multiplicity)

    factor_root = Poly(1, variable, domain=constants)
    for factor, multiplicity in factors:
        factor_root *= factor ** (multiplicity // exponent_gcd)
    witness = cast(
        ExtensionElement,
        radical.normalized_generator ** (radical.degree // exponent_gcd)
        / _embed_parent(extension, factor_root),
    )
    constant = constants.convert(unit)
    indeterminate = Dummy("constant")
    constant_minimal_polynomial = Poly(
        indeterminate**exponent_gcd - constants.to_sympy(constant),
        indeterminate,
        domain=constants,
    )
    return exponent_gcd, witness, constant_minimal_polynomial


def _algebraic_constant_base_detail(
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> str | None:
    """Return why the present constant theorem does not apply, if any."""
    if derivation.level != 1:
        return (
            "constant analysis currently supports one algebraic extension "
            "directly over C(x)"
        )

    base = derivation.tower.levels[0]
    if not isinstance(base, MonomialExtension):
        return "the lower level is not a rational monomial extension"

    try:
        parent, constants, variable = _base_polynomial_ring(extension)
    except NotImplementedError as exc:
        return str(exc)

    if base.field != parent or base.parent_field != constants:
        return "the algebraic parent is not the active rational lower field"
    if base.generator != variable:
        return "the rational base generator does not match the algebraic parent"
    if base.generator_derivative != Poly(1, variable, domain=constants):
        return "constant analysis currently requires the standard derivative Dx = 1"
    return None


def _constant_minimal_polynomial(
    extension: AlgebraicExtension,
) -> Poly | None:
    """Return the defining polynomial over ``C`` when it has no ``x`` terms."""
    parent, constants, _variable = _base_polynomial_ring(extension)
    coefficients: dict[tuple[int, ...], object] = {}
    for monomial, coefficient in extension.minimal_polynomial.terms():
        parent_coefficient = parent.convert(coefficient)
        numerator_terms = parent_coefficient.numer.to_dict()
        denominator_terms = parent_coefficient.denom.to_dict()
        if any(monomial != (0,) for monomial in numerator_terms) or any(
            monomial != (0,) for monomial in denominator_terms
        ):
            return None
        numerator = numerator_terms.get((0,), constants.zero)
        denominator = denominator_terms.get((0,), constants.one)
        coefficients[monomial] = constants.convert(numerator) / constants.convert(
            denominator
        )
    return Poly.from_dict(coefficients, extension.generator, domain=constants)


def _build_algebraic_constant_embedding(
    extension: AlgebraicExtension,
    minimal_polynomial: Poly,
    witness: ExtensionElement,
) -> tuple[AlgebraicConstantEmbedding | None, str | None]:
    """Build the exact embedded constant field, or name the backend boundary."""
    if minimal_polynomial.is_irreducible is not True:
        return (
            None,
            "the exhibited constant has no supported irreducible minimal "
            "polynomial over the displayed constants",
        )
    try:
        proposed_candidate = minimal_polynomial.domain.alg_field_from_poly(
            minimal_polynomial,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        return (
            None,
            "the new constant is certified, but SymPy cannot construct its "
            f"exact embedding: {exc}",
        )
    if not isinstance(proposed_candidate, AlgebraicField):
        return (
            None,
            "the new constant is certified, but SymPy did not return an "
            "algebraic-number constant domain",
        )

    try:
        embedding = AlgebraicConstantEmbedding(
            extension,
            proposed_candidate,
            minimal_polynomial,
            witness,
        )
        if embedding.embed(proposed_candidate.unit) != witness:
            return None, "SymPy could not realize the proposed constant embedding"
        if embedding.embed(proposed_candidate.one) != extension.field.one:
            return None, "SymPy could not realize the proposed constant embedding"
    except (NotImplementedError, TypeError, ValueError) as exc:
        return (
            None,
            "the new constant is certified, but SymPy cannot realize its "
            f"exact embedding: {exc}",
        )
    return embedding, None


def _unsupported_algebraic_constant_analysis(
    extension: AlgebraicExtension,
    derivation: Derivation,
    detail: str,
    *,
    radical: SimpleRadicalData | None = None,
    exponent_gcd: int | None = None,
) -> AlgebraicConstantFieldAnalysis:
    """Build the explicit non-decision result used by this narrow analyzer."""
    return AlgebraicConstantFieldAnalysis(
        extension,
        derivation,
        AlgebraicConstantFieldStatus.UNSUPPORTED,
        radical,
        exponent_gcd,
        None,
        detail,
    )


def analyze_algebraic_constant_field(
    extension: AlgebraicExtension,
    derivation: Derivation,
) -> AlgebraicConstantFieldAnalysis:
    r"""Analyze constants in one supported algebraic function field.

    For a normalized radical presentation

    ``L = C(x)(z),  z**n = c * product(f_i**e_i)``,

    the exponent gcd ``d = gcd(n, e_1, ...)`` controls geometricity.  If
    ``d = 1``, the binomial remains irreducible over an algebraic closure of
    ``C`` and ``Const(L) = C``.  If ``d > 1``,

    ``z**(n/d) / product(f_i**(e_i/d))``

    is a constant whose ``d``-th power is ``c``.  The latter case returns an
    exact embedded constant subfield and requires a later presentation rebase;
    it does not claim the current finite-extension level can simply be copied
    over the larger constants.

    A defining polynomial already over ``C`` gives an immediate rebasing
    certificate with its generator as witness. A result of
    :attr:`AlgebraicConstantFieldStatus.UNSUPPORTED` is never a preservation
    claim. The first implementation intentionally covers only a single
    algebraic level over the standard rational base ``C(x)``, ``Dx = 1``.
    """
    if not isinstance(extension, AlgebraicExtension):
        raise TypeError("constant analysis requires an algebraic extension")
    if not isinstance(derivation, Derivation):
        raise TypeError("constant analysis requires a Derivation")
    if derivation.level < 0 or (
        derivation.tower.levels[derivation.level] is not extension
    ):
        raise ValueError("the derivation must be bound to the supplied extension")

    base_detail = _algebraic_constant_base_detail(extension, derivation)
    if base_detail is not None:
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            base_detail,
        )

    constant_relation = _constant_minimal_polynomial(extension)
    if constant_relation is not None:
        witness = extension.field.generator
        if derivation(witness) != extension.field.zero:
            raise ValueError("a displayed algebraic constant is not annihilated by D")
        embedding, detail = _build_algebraic_constant_embedding(
            extension,
            constant_relation,
            witness,
        )
        if embedding is None:
            if detail is None:
                raise ValueError("a failed constant embedding needs a reason")
            return _unsupported_algebraic_constant_analysis(
                extension,
                derivation,
                detail,
            )
        return AlgebraicConstantFieldAnalysis(
            extension,
            derivation,
            AlgebraicConstantFieldStatus.REBASING_REQUIRED,
            None,
            None,
            embedding,
            "the defining polynomial is constant, so its algebraic generator "
            "belongs in the exact constant field",
        )

    radical = recognize_simple_radical(extension)
    if radical is None:
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            "the extension is not a recognized simple radical",
        )

    exponent_gcd, witness, constant_minimal_polynomial = _radical_constant_certificate(
        radical
    )

    if exponent_gcd == 1:
        return AlgebraicConstantFieldAnalysis(
            extension,
            derivation,
            AlgebraicConstantFieldStatus.PRESERVED,
            radical,
            exponent_gcd,
            None,
            "the normalized radical is geometrically irreducible by the "
            "binomial exponent-gcd criterion",
        )

    constant = constant_minimal_polynomial.domain.convert(
        -constant_minimal_polynomial.nth(0),
    )
    embedded_constant = _extension_element(
        extension,
        constant_minimal_polynomial.domain.to_sympy(constant),
    )
    if witness**exponent_gcd != embedded_constant:
        raise ValueError("the radical constant witness failed its power identity")
    if derivation(witness) != extension.field.zero:
        raise ValueError("the radical constant witness is not annihilated by D")

    embedding, detail = _build_algebraic_constant_embedding(
        extension,
        constant_minimal_polynomial,
        witness,
    )
    if embedding is None:
        if detail is None:
            raise ValueError("a failed constant embedding needs a reason")
        return _unsupported_algebraic_constant_analysis(
            extension,
            derivation,
            detail,
            radical=radical,
            exponent_gcd=exponent_gcd,
        )

    return AlgebraicConstantFieldAnalysis(
        extension,
        derivation,
        AlgebraicConstantFieldStatus.REBASING_REQUIRED,
        radical,
        exponent_gcd,
        embedding,
        "a nontrivial algebraic constant was certified; the function-field "
        "presentation must be rebased before global algebraic decisions",
    )
