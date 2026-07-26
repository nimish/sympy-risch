r"""Replayable constant-field evidence for monomial tower levels.

The no-new-constants premise in the monomial structure theorems concerns a
*position in a differential tower*, not merely a derivative polynomial.  In
particular, a bare ``MonomialExtension(..., PRESERVED)`` is not evidence:
the same derivative law can be dependent or independent over different lower
derivations.

This module is the single internal boundary for that premise.  It replays the
implemented primitive, hyperexponential, hypertangent, and hyperbolic
constant-field criteria against an immutable :class:`~sympy_risch.tower.Derivation`.
Consumers use :func:`require_preserved_constant_field` rather than trusting
caller-supplied status metadata.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ, QQ_I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly

from .tower.core import (
    ConstantFieldStatus,
    Derivation,
    DifferentialTower,
    MonomialExtension,
    MonomialFamily,
)


@dataclass(frozen=True, slots=True)
class ConstantFieldEvidence:
    r"""Independently replayed status for one active monomial tower level.

    ``status`` is established from the derivative law and the recursively
    checked lower derivation.  It deliberately does not use
    ``extension.constant_field_status`` as a premise.  The latter remains
    useful cached/display metadata, but cannot authorize a structure theorem.
    """

    derivation: Derivation
    status: ConstantFieldStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.derivation, Derivation):
            raise TypeError("constant-field evidence requires a Derivation")
        if self.derivation.level < 0 or not isinstance(
            self.derivation.tower.levels[self.derivation.level],
            MonomialExtension,
        ):
            raise TypeError("constant-field evidence requires a monomial level")
        if not isinstance(self.status, ConstantFieldStatus):
            raise TypeError("constant-field evidence needs a ConstantFieldStatus")
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("constant-field evidence detail must be a string or None")

    @property
    def extension(self) -> MonomialExtension:
        """Return the exact tower level certified by this evidence."""
        extension = self.derivation.tower.levels[self.derivation.level]
        assert isinstance(extension, MonomialExtension)
        return extension

    @property
    def preserves_constants(self) -> bool:
        """Whether the replay proves ``Const(K(t)) = Const(K)``."""
        return self.status is ConstantFieldStatus.PRESERVED


def _is_hyperbolic_mobius_law(extension: MonomialExtension) -> bool:
    r"""Recognize ``Dt = eta*(1-t**2)`` without importing normalizers.

    Keeping this tiny shape test here prevents a dependency cycle through the
    Möbius-normal-form module, while the actual proof is replayed through its
    hyperexponential coordinate below.
    """
    derivative = extension.generator_derivative
    if derivative.degree() != 2:
        return False
    parent = extension.parent_field
    eta = parent.convert(derivative.nth(0))
    return (
        eta != parent.zero
        and parent.convert(derivative.nth(1)) == parent.zero
        and parent.convert(derivative.nth(2)) == -eta
    )


def replay_constant_field(derivation: Derivation) -> ConstantFieldEvidence:
    r"""Replay the implemented constant-field criterion at ``derivation``.

    Every call returns evidence bound to the caller's exact immutable tower
    position. The bounded helper cache below stores status/detail plus an
    identity guard; it must never cache ``ConstantFieldEvidence`` itself
    because value-equal towers can still have distinct extension identities.
    """
    status, detail = _replay_constant_status(derivation)
    return ConstantFieldEvidence(derivation, status, detail)


# This bounds retained user towers. A prefix deeper than this can recompute
# evicted statuses, but never loses the identity/provenance check.
_REPLAY_STATUS_CACHE_SIZE = 256
_ReplayStatus = tuple[ConstantFieldStatus, str | None]
_ReplayStatusCacheEntry = tuple[DifferentialTower, ConstantFieldStatus, str | None]
_PrefixFailureCacheEntry = tuple[DifferentialTower, int | None]
_replay_status_cache: OrderedDict[tuple[int, int], _ReplayStatusCacheEntry] = (
    OrderedDict()
)
_prefix_failure_cache: OrderedDict[tuple[int, int], _PrefixFailureCacheEntry] = (
    OrderedDict()
)
_replay_status_cache_lock = RLock()


def _replay_constant_status(derivation: Derivation) -> _ReplayStatus:
    r"""Return bounded replay data without provenance objects.

    The cache is keyed by the *identity* of the immutable tower and its
    active level, then checks that identity again on lookup. A structural
    ``Derivation`` key is not sound: distinct but value-equal towers need
    evidence bound to their own exact position. The cache keeps only a
    bounded number of towers and only status/detail values, never evidence.

    A replay of level ``n`` asks certification factories only about level
    ``n-1``; their lower-prefix checks therefore recurse strictly downward.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("constant-field replay requires a Derivation")
    if derivation.level < 0:
        raise TypeError("constant-field replay requires a monomial level")
    extension = derivation.tower.levels[derivation.level]
    if not isinstance(extension, MonomialExtension):
        raise TypeError("constant-field replay requires a monomial level")

    key = (id(derivation.tower), derivation.level)
    with _replay_status_cache_lock:
        cached = _replay_status_cache.get(key)
        if cached is not None:
            cached_tower, status, detail = cached
            if cached_tower is derivation.tower:
                _replay_status_cache.move_to_end(key)
                return status, detail
            # An id can only be reused after eviction, but keep the identity
            # check explicit: it is the cache's provenance boundary.
            del _replay_status_cache[key]

    result = _replay_constant_status_uncached(derivation, extension)

    with _replay_status_cache_lock:
        cached = _replay_status_cache.get(key)
        if cached is not None and cached[0] is derivation.tower:
            _replay_status_cache.move_to_end(key)
            return cached[1], cached[2]
        _replay_status_cache[key] = (derivation.tower, *result)
        _replay_status_cache.move_to_end(key)
        if len(_replay_status_cache) > _REPLAY_STATUS_CACHE_SIZE:
            _replay_status_cache.popitem(last=False)
    return result


def _replay_constant_status_uncached(
    derivation: Derivation,
    extension: MonomialExtension,
) -> _ReplayStatus:
    """Run one criterion after an identity-safe cache miss."""

    descended = _replay_gaussian_constant_descent(derivation)
    if descended is not None:
        return descended

    lower_derivation = derivation.lower()
    # Local imports avoid the dependency cycle: the proposal factories use
    # lower-prefix replay evidence from this module.
    from .dependency import (
        certify_hyperexponential_extension,
        certify_hypertangent_extension,
        certify_primitive_extension,
    )

    if extension.family is MonomialFamily.PRIMITIVE:
        analysis = certify_primitive_extension(
            extension.generator_derivative,
            lower_derivation,
        )
    elif extension.family is MonomialFamily.HYPEREXPONENTIAL:
        analysis = certify_hyperexponential_extension(
            extension.generator_derivative,
            lower_derivation,
        )
    elif extension.family is MonomialFamily.HYPERTANGENT:
        analysis = certify_hypertangent_extension(
            extension.generator_derivative,
            lower_derivation,
        )
    elif _is_hyperbolic_mobius_law(extension):
        # Under u=(t+1)/(t-1), Dt=eta*(1-t**2) becomes Du/u=2*eta.
        # This rational differential-field isomorphism preserves the exact
        # constant-field question, so replay the hyperexponential proposal.
        probe = Dummy("constant_field_probe")
        eta = extension.parent_field.convert(extension.generator_derivative.nth(0))
        probe_derivative = Poly(
            2 * extension.parent_field.to_sympy(eta) * probe,
            probe,
            domain=extension.parent_field,
        )
        analysis = certify_hyperexponential_extension(
            probe_derivative,
            lower_derivation,
        )
    else:
        return (
            ConstantFieldStatus.UNKNOWN,
            "constant-field replay supports only primitive, hyperexponential, "
            "hypertangent, and hyperbolic monomial laws",
        )

    return analysis.status, analysis.detail


def _replay_gaussian_constant_descent(
    derivation: Derivation,
) -> _ReplayStatus | None:
    r"""Replay a rational ``QQ(i)`` coefficient extension over ``QQ``.

    A tower whose derivative laws all descend coefficientwise from ``QQ(i)``
    to ``QQ`` is the exact finite constant extension ``L(i)`` of that rational
    tower ``L``.  Once replay proves ``Const(L) = QQ``, every element of
    ``L(i)`` has a unique ``a + i*b`` form and ``D(a + i*b) = 0`` implies
    ``Da = Db = 0``.  Thus replaying the descended tower is positive evidence
    for the Gaussian tower, rather than trust in copied status metadata.

    This deliberately handles only literal coefficient descent.  Cayley
    targets with an essential ``I`` coefficient need their own explicit
    coordinate proof and fall through to the family criterion below.
    """
    if derivation.tower.constants != QQ_I:
        return None

    descended_tower = DifferentialTower(QQ)
    try:
        for source in derivation.tower.levels[: derivation.level + 1]:
            if not isinstance(source, MonomialExtension):
                return None
            derivative = Poly(
                source.generator_derivative.as_expr(),
                source.generator,
                domain=descended_tower.field(),
            )
            descended_tower = descended_tower.extend(MonomialExtension(derivative))
    except (CoercionFailed, TypeError, ValueError):
        return None

    status, detail = _replay_constant_status(descended_tower.derivation())
    if status is ConstantFieldStatus.UNKNOWN:
        return None
    suffix = f": {detail}" if detail else ""
    return (
        status,
        "replayed by exact coefficient descent along QQ(i)/QQ" + suffix,
    )


def _cache_key(derivation: Derivation) -> tuple[int, int]:
    """Return the identity-safe bounded-cache key for one tower position."""
    return id(derivation.tower), derivation.level


def _cached_prefix_failure_level(
    derivation: Derivation,
) -> tuple[bool, int | None]:
    """Return a cached first non-preserved level for a prefix when present."""
    key = _cache_key(derivation)
    with _replay_status_cache_lock:
        cached = _prefix_failure_cache.get(key)
        if cached is None:
            return False, None
        cached_tower, failure_level = cached
        if cached_tower is derivation.tower:
            _prefix_failure_cache.move_to_end(key)
            return True, failure_level
        del _prefix_failure_cache[key]
    return False, None


def _store_prefix_failure_level(
    derivation: Derivation,
    failure_level: int | None,
) -> None:
    """Store only the index of a prefix result, never a provenance object."""
    key = _cache_key(derivation)
    with _replay_status_cache_lock:
        _prefix_failure_cache[key] = (derivation.tower, failure_level)
        _prefix_failure_cache.move_to_end(key)
        if len(_prefix_failure_cache) > _REPLAY_STATUS_CACHE_SIZE:
            _prefix_failure_cache.popitem(last=False)


def _prefix_derivations(derivation: Derivation) -> tuple[Derivation, ...]:
    """Return existing derivation objects from the base upward."""
    prefix: list[Derivation] = []
    current = derivation
    while current.level >= 0:
        prefix.append(current)
        current = current.lower()
    return tuple(reversed(prefix))


def first_nonpreserved_constant_level(
    derivation: Derivation,
) -> ConstantFieldEvidence | None:
    r"""Return the first replayed non-preserved monomial level in a prefix.

    Prefix results are cached bottom-up by tower identity and active level.
    This prevents each certification factory from rescanning its entire lower
    prefix while still returning fresh caller-bound evidence when a failed
    level is requested.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError("constant-prefix replay requires a Derivation")
    prefix = _prefix_derivations(derivation)
    cached, failure_level = _cached_prefix_failure_level(derivation)
    if cached:
        if failure_level is None:
            return None
        return replay_constant_field(prefix[failure_level])

    for current in prefix:
        cached, failure_level = _cached_prefix_failure_level(current)
        if cached:
            if failure_level is None:
                continue
            return replay_constant_field(prefix[failure_level])
        extension = current.tower.levels[current.level]
        if not isinstance(extension, MonomialExtension):
            _store_prefix_failure_level(current, None)
            continue
        evidence = replay_constant_field(current)
        if not evidence.preserves_constants:
            _store_prefix_failure_level(current, current.level)
            return evidence
        _store_prefix_failure_level(current, None)
    return None


def require_preserved_constant_field(
    derivation: Derivation,
    algorithm: str,
) -> ConstantFieldEvidence:
    r"""Require independently replayed preservation at the active level.

    A bare extension is intentionally insufficient: it does not determine the
    lower derivation required by the structure theorem.
    """
    if not isinstance(derivation, Derivation):
        raise TypeError(f"{algorithm} requires a Derivation")
    # Internal theorem gates need a fresh derivation-bound evidence object,
    # but do not need to re-enter the public replay wrapper while recursive
    # exact solving repeatedly checks already-cached lower prefixes.
    status, detail = _replay_constant_status(derivation)
    evidence = ConstantFieldEvidence(derivation, status, detail)
    if evidence.preserves_constants:
        return evidence
    suffix = f": {evidence.detail}" if evidence.detail else ""
    if evidence.status is ConstantFieldStatus.UNKNOWN:
        raise NotImplementedError(
            f"{algorithm} requires Const(K(t)) = Const(K); independent "
            f"constant-field replay is unavailable{suffix}"
        )
    raise ValueError(
        f"{algorithm} requires Const(K(t)) = Const(K); independent "
        f"constant-field replay is {evidence.status.value}{suffix}"
    )


__all__ = [
    "ConstantFieldEvidence",
    "first_nonpreserved_constant_level",
    "replay_constant_field",
    "require_preserved_constant_field",
]
