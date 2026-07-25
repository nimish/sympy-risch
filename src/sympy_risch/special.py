r"""Certificates for concrete special polynomials in monomial fields.

A nonconstant polynomial ``p`` in ``K[t]`` is *special* when ``p`` divides
``D(p)``.  Equivalently, it is a Darboux polynomial for the tower derivation:

.. math::

   D(p) = a p, \qquad a \in K[t].

Finding every irreducible special polynomial of a general polynomial monomial
is a separate structure-theorem problem.  This module intentionally does not
claim to solve that problem.  It only turns explicitly supplied candidates and
the finite candidates visible in ``Dt`` into checked certificates.  Therefore
an empty analysis means ``UNKNOWN``, never ``ABSENT``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sympy.core.expr import Expr
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from .tower import (
    Derivation,
    MonomialExtension,
    MonomialFamily,
    SpecialPolynomialStatus,
)


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

    @property
    def reconstructs(self) -> bool:
        """Whether the stored Darboux identity holds exactly."""
        return (
            self.derivation.polynomial(self.factor) - self.factor * self.cofactor
        ).is_zero


@dataclass(frozen=True, slots=True)
class SpecialPolynomialAnalysis:
    r"""Conservative special-factor evidence for one monomial level.

    The analyzer checks the nonconstant factors already visible in ``Dt`` and
    any caller-supplied candidates.  It returns ``PRESENT`` as soon as one
    certificate is found; otherwise it returns ``UNKNOWN``.  In particular,
    lack of a certificate never proves that the field has no nonconstant
    special polynomials.
    """

    derivation: Derivation
    examined_candidates: tuple[Poly, ...]
    certificates: tuple[SpecialPolynomialCertificate, ...]

    @property
    def status(self) -> SpecialPolynomialStatus:
        """Return only positive evidence, never an absence claim."""
        if self.certificates:
            return SpecialPolynomialStatus.PRESENT
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

    This gives a sound first structure-theorem interface: it can certify
    ``PRESENT`` and expose usable factors, but it cannot certify ``ABSENT``.
    The latter requires a complete special-polynomial/Darboux decision
    procedure for the active monomial family.
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
        except ValueError:
            if required:
                raise
            return
        certificates.append(certificate)

    for candidate in _generator_derivative_candidates(extension):
        add_candidate(candidate, required=False)
    for candidate in candidates:
        add_candidate(candidate, required=True)

    result = SpecialPolynomialAnalysis(
        derivation,
        tuple(examined),
        tuple(certificates),
    )
    if not result.reconstructs:
        raise ValueError("special-polynomial analysis did not reconstruct")
    return result
