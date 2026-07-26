r"""Certified Möbius normal forms for quadratic monomial extensions.

Let ``K(t)`` be a preserved-constants monomial extension with quadratic
generator derivative.  If two distinct linear factors ``t-a`` and ``t-b``
are certified special, then

.. math::

   u = \frac{t-a}{t-b}

is hyperexponential.  Indeed, the two Darboux cofactors have the same
``t``-linear term, so their difference is in ``K`` and

.. math::

   \frac{D u}{u} = \frac{D(t-a)}{t-a} - \frac{D(t-b)}{t-b} \in K.

This module records that exact differential-field isomorphism rather than
mutating a user's tower.  It does *not* search for special factors: callers
must supply checked :class:`~sympy_risch.special.SpecialPolynomialCertificate`
objects. The exact real hyperbolic form is a scaled instance of this
construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains.domain import Domain
from sympy.polys.polytools import Poly, cancel

from ._typing import FieldElement, is_field_zero
from .constant_proofs import require_preserved_constant_field
from .special import SpecialPolynomialCertificate, certify_special_polynomial
from .tower import (
    Derivation,
    DifferentialTower,
    MonomialExtension,
    _coerce,
    _domain_symbols,
)

_HYPERBOLIC_LAW_MESSAGE = "hyperbolic Möbius normal form requires Dt = eta*(1 - t**2)"
_RICCATI_LAW_MESSAGE = (
    "quadratic Möbius normal form requires a preserved-constants "
    "quadratic monomial with two distinct linear special factors"
)


def _field_elements_equal(
    left: FieldElement,
    right: FieldElement,
    field: Domain,
) -> bool:
    """Compare exact native field elements without backend representation bias."""
    difference = left - right
    return is_field_zero(difference) or cancel(field.to_sympy(difference)) == 0


def _hyperbolic_mobius_coefficient(
    extension: MonomialExtension,
) -> FieldElement | None:
    """Return ``eta`` exactly when ``Dt = eta*(1-t**2)``."""
    derivative = extension.generator_derivative
    if derivative.degree() != 2:
        return None
    parent = extension.parent_field
    eta = parent.convert(derivative.nth(0))
    if (
        is_field_zero(eta)
        or not is_field_zero(parent.convert(derivative.nth(1)))
        or not is_field_zero(parent.convert(derivative.nth(2)) + eta)
    ):
        return None
    return eta


def is_hyperbolic_mobius_law(extension: MonomialExtension) -> bool:
    r"""Whether a monomial has the exact law ``Dt=eta*(1-t**2)``."""
    return _hyperbolic_mobius_coefficient(extension) is not None


@dataclass(frozen=True, slots=True)
class RiccatiMobiusNormalForm:
    r"""Immutable ``u=(t-a)/(t-b)`` normal form for a quadratic monomial.

    ``numerator_certificate`` and ``denominator_certificate`` retain the
    checked monic linear special polynomials ``t-a`` and ``t-b`` that define
    the chosen coordinate.  Reversing their order replaces ``u`` by ``1/u``
    and hence negates the target logarithmic derivative; both choices are
    valid but intentionally remain distinct normal forms.
    """

    source_derivation: Derivation
    target_derivation: Derivation
    source_generator: Symbol
    target_generator: Symbol
    numerator_certificate: SpecialPolynomialCertificate
    denominator_certificate: SpecialPolynomialCertificate

    def __post_init__(self) -> None:
        """Keep the stored special-factor proof tied to this source tower."""
        if not isinstance(self.source_derivation, Derivation) or not isinstance(
            self.target_derivation,
            Derivation,
        ):
            raise TypeError("a Riccati Möbius normal form needs two derivations")
        if not isinstance(self.source_generator, Symbol) or not isinstance(
            self.target_generator,
            Symbol,
        ):
            raise TypeError("a Riccati Möbius normal form needs Symbol generators")
        if not isinstance(
            self.numerator_certificate,
            SpecialPolynomialCertificate,
        ) or not isinstance(
            self.denominator_certificate,
            SpecialPolynomialCertificate,
        ):
            raise TypeError("a Riccati Möbius normal form needs special certificates")
        if not self.numerator_certificate.derivation.has_same_position(
            self.source_derivation,
        ) or not self.denominator_certificate.derivation.has_same_position(
            self.source_derivation,
        ):
            raise ValueError(
                "Riccati Möbius special factors belong to another derivation"
            )
        if self.numerator_factor.degree() != 1 or self.denominator_factor.degree() != 1:
            raise ValueError("Riccati Möbius special factors must be linear")
        if self.numerator_factor == self.denominator_factor:
            raise ValueError("Riccati Möbius special factors must be distinct")
        if self.source_derivation.level < 0:
            raise ValueError("a Riccati Möbius normal form needs a monomial source")
        source_extension = self.source_derivation.tower.levels[
            self.source_derivation.level
        ]
        if (
            not isinstance(source_extension, MonomialExtension)
            or source_extension.generator != self.source_generator
        ):
            raise ValueError("Riccati Möbius source generator does not match its tower")
        if self.target_derivation.level != self.source_derivation.level:
            raise ValueError("Riccati Möbius towers must have the same active level")
        target_extension = self.target_derivation.tower.levels[
            self.target_derivation.level
        ]
        if (
            not isinstance(target_extension, MonomialExtension)
            or target_extension.generator != self.target_generator
        ):
            raise ValueError("Riccati Möbius target generator does not match its tower")

    @property
    def source_field(self) -> Domain:
        """Return the field containing the original quadratic generator."""
        return self.source_derivation.field

    @property
    def target_field(self) -> Domain:
        """Return the field containing the hyperexponential coordinate."""
        return self.target_derivation.field

    @property
    def numerator_factor(self) -> Poly:
        """Return the certified numerator factor ``t-a``."""
        return self.numerator_certificate.factor

    @property
    def denominator_factor(self) -> Poly:
        """Return the certified denominator factor ``t-b``."""
        return self.denominator_certificate.factor

    @property
    def numerator_root(self) -> FieldElement:
        """Return ``a`` for the stored factor ``t-a``."""
        return -self.numerator_factor.nth(0)

    @property
    def denominator_root(self) -> FieldElement:
        """Return ``b`` for the stored factor ``t-b``."""
        return -self.denominator_factor.nth(0)

    @property
    def forward_generator_expression(self) -> Expr:
        r"""Return the target expression for ``t=(b*u-a)/(u-1)``."""
        parent = self.numerator_factor.domain
        numerator_root = parent.to_sympy(self.numerator_root)
        denominator_root = parent.to_sympy(self.denominator_root)
        return cancel(
            (denominator_root * self.target_generator - numerator_root)
            / (self.target_generator - 1)
        )

    @property
    def inverse_generator_expression(self) -> Expr:
        r"""Return the source expression for ``u=(t-a)/(t-b)``."""
        parent = self.numerator_factor.domain
        numerator_root = parent.to_sympy(self.numerator_root)
        denominator_root = parent.to_sympy(self.denominator_root)
        return cancel(
            (self.source_generator - numerator_root)
            / (self.source_generator - denominator_root)
        )

    def forward_expression(self, expression: Expr) -> Expr:
        """Rewrite one source expression in the target coordinate."""
        return cancel(
            expression.xreplace(
                {self.source_generator: self.forward_generator_expression},
            )
        )

    def inverse_expression(self, value: object) -> Expr:
        """Rewrite one target native element as a source expression."""
        target_value = _coerce(self.target_field, value)
        return cancel(
            self.target_field.to_sympy(target_value).xreplace(
                {self.target_generator: self.inverse_generator_expression},
            )
        )

    def forward(self, value: object) -> FieldElement:
        """Map a source native field element into the target field."""
        source_value = _coerce(self.source_field, value)
        return _coerce(
            self.target_field,
            self.forward_expression(self.source_field.to_sympy(source_value)),
        )

    def inverse(self, value: object) -> FieldElement:
        """Map a target native field element back into the source field."""
        return _coerce(self.source_field, self.inverse_expression(value))

    def verifies_isomorphism(self) -> bool:
        """Whether the two maps invert each other on their generators."""
        source_generator = _coerce(self.source_field, self.source_generator)
        target_generator = _coerce(self.target_field, self.target_generator)
        return _field_elements_equal(
            self.inverse(self.forward(source_generator)),
            source_generator,
            self.source_field,
        ) and _field_elements_equal(
            self.forward(self.inverse(target_generator)),
            target_generator,
            self.target_field,
        )

    def verifies_derivation(self) -> bool:
        r"""Whether the Möbius map intertwines the two derivations.

        A differential-field homomorphism is determined by the tower
        generators.  The lower levels are identical, and the final equality
        checks the certified quadratic coordinate change.
        """
        for extension in self.source_derivation.tower.levels[
            : self.source_derivation.level + 1
        ]:
            source_generator = _coerce(self.source_field, extension.generator)
            expected = self.forward(self.source_derivation(source_generator))
            actual = self.target_derivation(self.forward(source_generator))
            if not _field_elements_equal(actual, expected, self.target_field):
                return False
        return True


@dataclass(frozen=True, slots=True)
class HyperbolicMobiusNormalForm:
    r"""Immutable conjugacy from ``Dt=eta*(1-t**2)`` to ``Du=2*eta*u``.

    The lower tower is shared structurally by the source and target immutable
    towers.  Only the active top level changes: the source generator is
    replaced by the fresh target generator.  ``forward`` and ``inverse``
    accept and return native SymPy-domain elements, not expression wrappers.
    """

    source_derivation: Derivation
    target_derivation: Derivation
    source_generator: Symbol
    target_generator: Symbol

    @property
    def source_field(self) -> Domain:
        """Return the field containing the original hyperbolic generator."""
        return self.source_derivation.field

    @property
    def target_field(self) -> Domain:
        """Return the field containing the hyperexponential coordinate."""
        return self.target_derivation.field

    @property
    def forward_generator_expression(self) -> Expr:
        r"""Return the target expression for ``t = (u-1)/(u+1)``."""
        return cancel((self.target_generator - 1) / (self.target_generator + 1))

    @property
    def inverse_generator_expression(self) -> Expr:
        r"""Return the source expression for ``u = (1+t)/(1-t)``."""
        return cancel((1 + self.source_generator) / (1 - self.source_generator))

    def forward_expression(self, expression: Expr) -> Expr:
        """Rewrite one source expression in the target coordinate."""
        return cancel(
            expression.xreplace(
                {self.source_generator: self.forward_generator_expression},
            )
        )

    def inverse_expression(self, value: object) -> Expr:
        """Rewrite one target native element as a source expression."""
        target_value = _coerce(self.target_field, value)
        return cancel(
            self.target_field.to_sympy(target_value).xreplace(
                {self.target_generator: self.inverse_generator_expression},
            )
        )

    def forward(self, value: object) -> FieldElement:
        """Map a source native field element into the target field."""
        source_value = _coerce(self.source_field, value)
        return _coerce(
            self.target_field,
            self.forward_expression(self.source_field.to_sympy(source_value)),
        )

    def inverse(self, value: object) -> FieldElement:
        """Map a target native field element back into the source field."""
        return _coerce(self.source_field, self.inverse_expression(value))

    def verifies_derivation(self) -> bool:
        r"""Whether ``D_target(forward(v)) = forward(D_source(v))`` on generators.

        A differential-field homomorphism is determined by the tower
        generators.  The lower levels are identical, and the final equality
        checks the Möbius law for the active source generator.
        """
        for extension in self.source_derivation.tower.levels[
            : self.source_derivation.level + 1
        ]:
            source_generator = _coerce(self.source_field, extension.generator)
            expected = self.forward(self.source_derivation(source_generator))
            actual = self.target_derivation(self.forward(source_generator))
            if not _field_elements_equal(actual, expected, self.target_field):
                return False
        return True


def _validate_hyperbolic_source(
    derivation: Derivation,
    target_generator: Symbol,
) -> tuple[MonomialExtension, FieldElement]:
    """Validate the exact monomial law and return its coefficient ``eta``."""
    if not isinstance(derivation, Derivation):
        raise TypeError("hyperbolic Möbius normal form requires a Derivation")
    if derivation.level < 0:
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)
    if not isinstance(target_generator, Symbol):
        raise TypeError("the target generator must be a SymPy Symbol")
    if target_generator in _domain_symbols(derivation.field):
        raise ValueError("the target generator must be fresh over the source field")

    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)

    eta = _hyperbolic_mobius_coefficient(extension)
    if eta is None:
        raise ValueError(_HYPERBOLIC_LAW_MESSAGE)
    return extension, eta


def _validate_riccati_certificate(
    certificate: SpecialPolynomialCertificate,
    derivation: Derivation,
    *,
    role: str,
) -> None:
    """Require one checked monic linear special factor at the active level."""
    if not isinstance(certificate, SpecialPolynomialCertificate):
        raise TypeError(f"the {role} factor must be a SpecialPolynomialCertificate")
    if not certificate.derivation.has_same_position(derivation):
        raise ValueError(f"the {role} special factor belongs to another derivation")
    if certificate.factor.degree() != 1:
        raise ValueError(
            f"the {role} special factor must be monic and linear in the active generator"
        )


def _validate_riccati_source(
    derivation: Derivation,
    target_generator: Symbol,
    numerator_certificate: SpecialPolynomialCertificate,
    denominator_certificate: SpecialPolynomialCertificate,
) -> tuple[MonomialExtension, FieldElement]:
    """Validate the quadratic theorem hypotheses and return ``Du/u``."""
    if not isinstance(derivation, Derivation):
        raise TypeError("quadratic Möbius normal form requires a Derivation")
    if derivation.level < 0:
        raise ValueError(_RICCATI_LAW_MESSAGE)
    if not isinstance(target_generator, Symbol):
        raise TypeError("the target generator must be a SymPy Symbol")
    if target_generator in _domain_symbols(derivation.field):
        raise ValueError("the target generator must be fresh over the source field")

    extension = derivation.tower.levels[derivation.level]
    if (
        not isinstance(extension, MonomialExtension)
        or extension.generator_derivative.degree() != 2
    ):
        raise ValueError(_RICCATI_LAW_MESSAGE)
    _validate_riccati_certificate(
        numerator_certificate,
        derivation,
        role="numerator",
    )
    _validate_riccati_certificate(
        denominator_certificate,
        derivation,
        role="denominator",
    )
    if numerator_certificate.factor == denominator_certificate.factor:
        raise ValueError("quadratic Möbius normal form needs distinct special factors")

    logarithmic_direction = (
        numerator_certificate.cofactor - denominator_certificate.cofactor
    )
    if logarithmic_direction.is_zero or logarithmic_direction.degree() > 0:
        raise ValueError(
            "the special-factor quotient does not have a lower-field "
            "logarithmic derivative"
        )
    direction = extension.parent_field.convert(logarithmic_direction.nth(0))
    if is_field_zero(direction):
        raise ValueError(
            "distinct special factors would make the Möbius coordinate a new constant"
        )
    return extension, direction


def _build_riccati_mobius_normal_form(
    derivation: Derivation,
    target_generator: Symbol,
    numerator_certificate: SpecialPolynomialCertificate,
    denominator_certificate: SpecialPolynomialCertificate,
    *,
    require_preserved_constants: bool,
) -> RiccatiMobiusNormalForm:
    """Build the shared target construction for generic and hyperbolic forms."""
    source_extension, direction = _validate_riccati_source(
        derivation,
        target_generator,
        numerator_certificate,
        denominator_certificate,
    )
    target_tower = DifferentialTower(
        derivation.tower.constants,
        derivation.tower.levels[: derivation.level],
    )
    target_parent = target_tower.field()
    target_derivative = Poly(
        target_parent.to_sympy(target_parent.convert(direction)) * target_generator,
        target_generator,
        domain=target_parent,
    )
    # The checked special-factor quotient gives an exact source-to-target
    # differential-field isomorphism.  Replay the *hyperexponential* target
    # rather than trusting metadata on the generic quadratic source: target
    # preservation proves source preservation through that isomorphism.
    from .dependency import certify_hyperexponential_extension

    target_analysis = certify_hyperexponential_extension(
        target_derivative,
        target_tower.derivation(),
    )
    target_extension = target_analysis.extension
    target_tower = target_tower.extend(target_extension)
    normal_form = RiccatiMobiusNormalForm(
        derivation,
        target_tower.derivation(),
        source_extension.generator,
        target_generator,
        numerator_certificate,
        denominator_certificate,
    )
    if not normal_form.verifies_isomorphism():
        raise ValueError("quadratic Möbius normal-form maps are not inverse")
    if not normal_form.verifies_derivation():
        raise ValueError("quadratic Möbius normal form does not intertwine derivations")
    if require_preserved_constants:
        require_preserved_constant_field(
            target_tower.derivation(), "quadratic Möbius normal form"
        )
    return normal_form


def build_riccati_mobius_normal_form(
    derivation: Derivation,
    target_generator: Symbol,
    numerator_certificate: SpecialPolynomialCertificate,
    denominator_certificate: SpecialPolynomialCertificate,
) -> RiccatiMobiusNormalForm:
    r"""Build the certified quadratic normal form ``u=(t-a)/(t-b)``.

    The active extension must have ``deg_t(Dt)=2``.  Its preservation premise
    is independently replayed after the checked coordinate change: the
    certificates must certify distinct monic linear special factors ``t-a``
    and ``t-b``, and the resulting hyperexponential target must preserve
    constants.  No factor search or no-special conclusion is performed here.
    """
    return _build_riccati_mobius_normal_form(
        derivation,
        target_generator,
        numerator_certificate,
        denominator_certificate,
        require_preserved_constants=True,
    )


def build_hyperbolic_mobius_normal_form(
    derivation: Derivation,
    target_generator: Symbol,
) -> HyperbolicMobiusNormalForm:
    r"""Build the exact ``Dt=eta*(1-t**2)`` to ``Du=2*eta*u`` normal form.

    ``target_generator`` must be fresh over the source field.  The rational
    Möbius coordinate preserves the constant-field question, so the target's
    status is derived by replaying its hyperexponential criterion rather than
    copied from source metadata. The checked source factors remain part of
    this normal-form proof; the target carries no special-polynomial metadata.
    """
    source_extension, _ = _validate_hyperbolic_source(
        derivation,
        target_generator,
    )
    parent = source_extension.parent_field
    numerator_certificate = certify_special_polynomial(
        Poly(source_extension.generator + 1, source_extension.generator, domain=parent),
        derivation,
    )
    denominator_certificate = certify_special_polynomial(
        Poly(source_extension.generator - 1, source_extension.generator, domain=parent),
        derivation,
    )
    generic_normal_form = _build_riccati_mobius_normal_form(
        derivation,
        target_generator,
        numerator_certificate,
        denominator_certificate,
        # The coordinate map remains exact when constant-field evidence is
        # unknown; integration separately requires the target's P/E/T
        # admission. The generic Riccati builder requires the stronger
        # preserved-constants premise.
        require_preserved_constants=False,
    )
    normal_form = HyperbolicMobiusNormalForm(
        derivation,
        generic_normal_form.target_derivation,
        source_extension.generator,
        target_generator,
    )
    if not normal_form.verifies_derivation():
        raise ValueError(
            "hyperbolic Möbius normal form does not intertwine derivations"
        )
    return normal_form
