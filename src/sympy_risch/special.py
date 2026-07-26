r"""Certificates for concrete special polynomials in monomial fields.

A nonconstant polynomial ``p`` in ``K[t]`` is *special* when ``p`` divides
``D(p)``.  Equivalently, it is a Darboux polynomial for the tower derivation:

.. math::

   D(p) = a p, \qquad a \in K[t].

Finding every irreducible special polynomial of a general polynomial monomial
is a separate structure-theorem problem.  This module intentionally does not
claim to solve that problem.  It only turns explicitly supplied candidates and
the finite candidates visible in ``Dt`` into checked certificates.

There is one complete family result already available here.  If ``Dt`` lies
in ``K`` and the primitive extension ``K(t)`` is certified to preserve
constants, then it has no nonconstant special polynomial.  The degree
argument proving this is recorded explicitly below.  Every other empty search
means ``UNKNOWN``, not ``ABSENT``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sympy.core.expr import Expr
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from .constant_proofs import require_preserved_constant_field
from .tower import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
)


class SpecialPolynomialStatus(Enum):
    r"""Outcome of a derivation-bound special-polynomial analysis.

    Unlike the former extension metadata, this status is derived from the
    certificates retained by :class:`SpecialPolynomialAnalysis` and cannot
    authorize an algorithm by itself.
    """

    UNKNOWN = "unknown"
    ABSENT = "absent"
    PRESENT = "present"


@dataclass(frozen=True, slots=True)
class SpecialPolynomialCertificate:
    r"""A checked identity ``D(p) = a*p`` for one monic nonconstant ``p``.

    ``factor`` need not be irreducible: a product of special polynomials is
    special as well.  Consumers that need individual irreducible factors can
    factor this certified polynomial over its parent field.  The certificate
    itself makes no claim that every special polynomial has been found.
    """

    derivation: Derivation
    factor: Poly
    cofactor: Poly

    def __post_init__(self) -> None:
        """Reject a forged or mismatched Darboux identity eagerly."""
        _top_monomial_extension(self.derivation)
        if not isinstance(self.factor, Poly) or not isinstance(self.cofactor, Poly):
            raise TypeError("a special-polynomial certificate needs SymPy Poly data")
        if self.factor.is_zero or self.factor.degree() <= 0:
            raise ValueError(
                "a special-polynomial certificate needs a nonconstant factor"
            )
        self.derivation.polynomial(self.factor)
        self.derivation.polynomial(self.cofactor)
        if self.factor != self.factor.monic():
            raise ValueError("a special-polynomial certificate factor must be monic")
        if not self.reconstructs:
            raise ValueError("a special-polynomial certificate must reconstruct")

    @property
    def reconstructs(self) -> bool:
        """Whether the stored Darboux identity holds exactly."""
        return (
            self.derivation.polynomial(self.factor) - self.factor * self.cofactor
        ).is_zero


@dataclass(frozen=True, slots=True)
class PrimitiveNoSpecialPolynomialCertificate:
    r"""Certificate that a primitive preserved-constants level has no specials.

    Let ``p`` be a nonconstant special polynomial in ``K[t]`` and suppose
    ``Dt`` lies in ``K``.  Since ``deg_t(Dp) <= deg_t(p)``, the Darboux
    cofactor ``Dp/p`` lies in ``K``.  Dividing ``p`` by its leading coefficient
    gives a monic polynomial ``q`` with ``Dq = 0``: its derivative has degree
    strictly less than ``deg_t(q)``, while a nonzero multiple of ``q`` does
    not.  Thus ``q`` is a nonconstant constant of ``K(t)``, contradicting
    ``Const(K(t)) = Const(K)``.

    The primitive constant-field premise is independently replayed against
    the exact lower derivation.  A caller-supplied ``PRESERVED`` enum is never
    accepted as evidence.
    """

    derivation: Derivation

    def __post_init__(self) -> None:
        """Require exactly the hypotheses of the primitive theorem."""
        extension = _top_monomial_extension(self.derivation)
        if extension.family is not MonomialFamily.PRIMITIVE:
            raise ValueError(
                "a primitive no-special certificate requires Dt in the lower field"
            )
        require_preserved_constant_field(
            self.derivation,
            "primitive no-special certification",
        )


@dataclass(frozen=True, slots=True)
class SpecialPolynomialAnalysis:
    r"""Conservative special-factor evidence for one monomial level.

    The analyzer checks the nonconstant factors already visible in ``Dt`` and
    any caller-supplied candidates.  It returns ``PRESENT`` as soon as one
    certificate is found.  A primitive level with certified preserved
    constants instead carries a complete no-special certificate and returns
    ``ABSENT``.  Other empty searches return ``UNKNOWN``.
    """

    derivation: Derivation
    examined_candidates: tuple[Poly, ...]
    certificates: tuple[SpecialPolynomialCertificate, ...]
    absence_certificate: PrimitiveNoSpecialPolynomialCertificate | None = None

    def __post_init__(self) -> None:
        """Keep positive and negative structural evidence disjoint and checked."""
        _top_monomial_extension(self.derivation)
        if not isinstance(self.examined_candidates, tuple) or not all(
            isinstance(candidate, Poly) for candidate in self.examined_candidates
        ):
            raise TypeError(
                "examined special-polynomial candidates must be a Poly tuple"
            )
        for candidate in self.examined_candidates:
            self.derivation.polynomial(candidate)
        if not isinstance(self.certificates, tuple) or not all(
            isinstance(certificate, SpecialPolynomialCertificate)
            for certificate in self.certificates
        ):
            raise TypeError(
                "special-polynomial certificates must be an immutable tuple"
            )
        if any(
            not certificate.derivation.has_same_position(self.derivation)
            for certificate in self.certificates
        ):
            raise ValueError(
                "a special-polynomial certificate belongs to another derivation"
            )
        if self.absence_certificate is not None:
            if not isinstance(
                self.absence_certificate,
                PrimitiveNoSpecialPolynomialCertificate,
            ):
                raise TypeError(
                    "a special-polynomial absence certificate has the wrong type"
                )
            if not self.absence_certificate.derivation.has_same_position(
                self.derivation,
            ):
                raise ValueError(
                    "a special-polynomial absence certificate belongs to another derivation"
                )
            if self.certificates:
                raise ValueError(
                    "a special polynomial contradicts the primitive no-special certificate"
                )

    @property
    def status(self) -> SpecialPolynomialStatus:
        """Return the checked positive, primitive-negative, or unknown outcome."""
        if self.certificates:
            return SpecialPolynomialStatus.PRESENT
        if self.absence_certificate is not None:
            return SpecialPolynomialStatus.ABSENT
        return SpecialPolynomialStatus.UNKNOWN

    @property
    def has_special_polynomials(self) -> bool:
        """Whether at least one special polynomial was certified."""
        return bool(self.certificates)

    @property
    def reconstructs(self) -> bool:
        """Whether every stored special-factor certificate reconstructs."""
        return all(certificate.reconstructs for certificate in self.certificates)


def _top_monomial_extension(derivation: Derivation) -> MonomialExtension:
    """Return the monomial extension at the active derivation level."""
    if not isinstance(derivation, Derivation):
        raise TypeError("special-polynomial analysis requires a Derivation")
    if derivation.level < 0:
        raise TypeError("special-polynomial analysis requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("special-polynomial analysis requires a monomial level")
    return extension


def certify_primitive_no_special_polynomials(
    derivation: Derivation,
) -> PrimitiveNoSpecialPolynomialCertificate:
    r"""Certify ``S^{irr}(K[t]) = emptyset`` for a primitive level.

    This is the complete primitive special-polynomial theorem, conditional on
    replayed evidence that the level preserves constants.  It intentionally
    does not extend to hyperexponential, hypertangent, or general polynomial
    monomials, where a non-generator Darboux factor needs additional theory.
    """
    return PrimitiveNoSpecialPolynomialCertificate(derivation)


def certify_special_polynomial(
    candidate: Poly,
    derivation: Derivation,
) -> SpecialPolynomialCertificate:
    r"""Verify that a supplied ``candidate`` is a nonconstant special polynomial.

    The polynomial is normalized to monic form.  A failed divisibility test is
    an input error, rather than negative evidence about other candidates.
    """
    _top_monomial_extension(derivation)
    if not isinstance(candidate, Poly):
        raise TypeError("a special-polynomial candidate must be a SymPy Poly")
    if candidate.is_zero or candidate.degree() <= 0:
        raise ValueError("a special-polynomial candidate must be nonconstant")

    # This validates that the candidate has the active generator and exact
    # parent field before normalization or division.
    derivation.polynomial(candidate)
    factor = candidate.monic()
    derivative = derivation.polynomial(factor)
    quotient, remainder = derivative.div(factor)
    if not remainder.is_zero:
        raise ValueError("the candidate is not a special polynomial")
    certificate = SpecialPolynomialCertificate(derivation, factor, quotient)
    if not certificate.reconstructs:
        raise ValueError("special-polynomial certificate did not reconstruct")
    return certificate


def _generator_derivative_candidates(
    extension: MonomialExtension,
) -> tuple[Poly, ...]:
    r"""Return finite candidate factors visible in ``Dt``.

    This captures the canonical hyperexponential factor ``t`` and the
    hypertangent factor ``1+t**2`` automatically.  For a general polynomial
    monomial it may discover further factors of ``Dt`` that happen to be
    special, but it is intentionally not an exhaustive Darboux search.
    """
    derivative = extension.generator_derivative
    if extension.family is MonomialFamily.PRIMITIVE:
        return ()
    if extension.family is MonomialFamily.HYPEREXPONENTIAL:
        return (
            Poly(
                extension.generator, extension.generator, domain=extension.parent_field
            ),
        )
    if extension.family is MonomialFamily.HYPERTANGENT:
        return (
            Poly(
                1 + extension.generator**2,
                extension.generator,
                domain=extension.parent_field,
            ),
        )

    # Factoring an arbitrary polynomial over an arbitrary exact parent field
    # is an opportunistic discovery pass, not a prerequisite for a sound
    # analysis.  If SymPy has no factorization algorithm for that field, the
    # correct result is incomplete evidence rather than an implementation
    # error or an absence conclusion.
    try:
        factors = derivative.factor_list()[1]
    except (NotImplementedError, PolynomialError):
        return ()
    return tuple(factor.monic() for factor, _ in factors if factor.degree() > 0)


def analyze_special_polynomials(
    derivation: Derivation,
    *,
    candidates: tuple[Poly, ...] = (),
) -> SpecialPolynomialAnalysis:
    r"""Collect verified concrete special-polynomial evidence.

    The analyzer examines factors of the active generator derivative and then
    caller-provided candidates.  Caller-provided candidates are promises to
    verify: an invalid candidate raises :class:`ValueError`.  In contrast, a
    visible factor of ``Dt`` that is not special is simply discarded.

    This gives a sound structure-theorem interface: it can certify ``PRESENT``
    and expose usable factors, and it certifies ``ABSENT`` for primitive levels
    whose constants are already certified preserved.  A general
    special-polynomial/Darboux decision remains outside this routine.
    """
    extension = _top_monomial_extension(derivation)
    if not isinstance(candidates, tuple):
        raise TypeError("special-polynomial candidates must be a tuple of Poly")

    examined: list[Poly] = []
    certificates: list[SpecialPolynomialCertificate] = []
    seen: set[Expr] = set()

    def add_candidate(candidate: Poly, *, required: bool) -> None:
        if not isinstance(candidate, Poly):
            raise TypeError("a special-polynomial candidate must be a SymPy Poly")
        if candidate.is_zero or candidate.degree() <= 0:
            if required:
                raise ValueError("a special-polynomial candidate must be nonconstant")
            return
        factor = candidate.monic()
        key = factor.as_expr()
        if key in seen:
            return
        seen.add(key)
        examined.append(factor)
        try:
            certificate = certify_special_polynomial(factor, derivation)
        except (NotImplementedError, ValueError):
            if required:
                raise
            return
        certificates.append(certificate)

    for candidate in _generator_derivative_candidates(extension):
        add_candidate(candidate, required=False)
    for candidate in candidates:
        add_candidate(candidate, required=True)

    absence_certificate = None
    if extension.family is MonomialFamily.PRIMITIVE:
        try:
            absence_certificate = certify_primitive_no_special_polynomials(derivation)
        except (NotImplementedError, ValueError):
            # A failed constant-field replay leaves the general search
            # incomplete.  It is not evidence that special polynomials exist.
            absence_certificate = None
    result = SpecialPolynomialAnalysis(
        derivation,
        tuple(examined),
        tuple(certificates),
        absence_certificate,
    )
    if not result.reconstructs:
        raise ValueError("special-polynomial analysis did not reconstruct")
    return result
