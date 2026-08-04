from dataclasses import replace
from itertools import product

import pytest
from sympy import Poly, symbols
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

import sympy_risch.algebraic.compositum as compositum_module
from sympy_risch.algebraic.compositum import (
    normalize_primitive_element,
    trager_compositum,
)
from sympy_risch.algebraic.presentation import AmbiguousCompositumError
from sympy_risch.tower import (
    AlgebraicExtension,
    DifferentialTower,
    MonomialExtension,
)

x, a, b, c, z = symbols("x a b c z")


def rational_function_field():
    return (
        DifferentialTower(QQ).extend(MonomialExtension(Poly(1, x, domain=QQ))).field()
    )


def tensor_characteristic_polynomial(sources, coefficients, generator):
    """Independent dense-matrix oracle for small compositum examples."""
    parent = sources[0].parent_field
    degrees = tuple(source.field.rank for source in sources)
    basis = tuple(product(*(range(degree) for degree in degrees)))
    indices = {monomial: index for index, monomial in enumerate(basis)}
    rows = [[parent.zero for _ in basis] for _ in basis]
    for column, exponents in enumerate(basis):
        for source_index, (source, scalar) in enumerate(
            zip(sources, coefficients, strict=True)
        ):
            next_power = exponents[source_index] + 1
            if next_power < source.field.rank:
                target = list(exponents)
                target[source_index] = next_power
                rows[indices[tuple(target)]][column] += scalar
                continue
            for power in range(source.field.rank):
                coefficient = -scalar * parent.convert(
                    source.minimal_polynomial.nth(power)
                )
                if coefficient != parent.zero:
                    target = list(exponents)
                    target[source_index] = power
                    rows[indices[tuple(target)]][column] += coefficient
    matrix = DomainMatrix(rows, (len(basis), len(basis)), parent)
    return Poly.from_list(matrix.charpoly(), generator, domain=parent).monic()


def test_primitive_presentation_recovers_independent_radicals() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**3 - x, b, domain=parent)),
    )

    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 6
    assert presentation.target.field.generator == sum(presentation.source_images)
    embedded_x = presentation.target.from_power_basis_coordinates(
        (
            parent.from_sympy(x),
            *(parent.zero for _ in range(presentation.tensor_rank - 1)),
        )
    )
    assert presentation.source_images[0] ** 2 == embedded_x
    assert presentation.source_images[1] ** 3 == embedded_x


def test_primitive_presentation_uses_one_trager_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )

    calls = 0
    original = compositum_module._trager_norm

    def counting_trager_norm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(compositum_module, "_trager_norm", counting_trager_norm)
    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 4
    assert calls == 1


def test_trager_compositum_naturally_tries_a_second_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = (
        AlgebraicExtension(Poly(a**2 - 2, a, domain=QQ)),
        AlgebraicExtension(Poly(b**2 - 2, b, domain=QQ)),
    )
    attempts = 0
    original = compositum_module._trager_norm

    def counting_trager_norm(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        compositum_module,
        "_trager_norm",
        counting_trager_norm,
    )

    decomposition = trager_compositum(sources, z)

    assert attempts == 2
    assert len(decomposition.components) == 2
    assert all(
        component.primitive_coefficients == (QQ.one, QQ(2))
        for component in decomposition.components
    )
    assert {
        component.target.minimal_polynomial for component in decomposition.components
    } == {Poly(z**2 - 18, z, domain=QQ), Poly(z**2 - 2, z, domain=QQ)}


def test_trager_compositum_merges_sources_pairwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
        AlgebraicExtension(Poly(c**2 - x - 2, c, domain=parent)),
    )
    ranks: list[tuple[int, int]] = []
    original = compositum_module._trager_norm

    def recording_trager_norm(left, right, generator, scalar):
        ranks.append((left.field.rank, right.field.rank))
        return original(left, right, generator, scalar)

    monkeypatch.setattr(
        compositum_module,
        "_trager_norm",
        recording_trager_norm,
    )
    presentation = normalize_primitive_element(sources, z)

    assert presentation.tensor_rank == 8
    assert ranks == [(2, 2), (4, 2)]


def test_trager_compositum_exposes_dependent_field_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - 4 * x, b, domain=parent)),
    )

    calls = 0
    original = compositum_module._trager_norm

    def counting_trager_norm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(compositum_module, "_trager_norm", counting_trager_norm)
    decomposition = trager_compositum(sources, z)

    assert calls == 1
    assert not decomposition.is_field
    assert decomposition.tensor_rank == 4
    assert len(decomposition.components) == 2
    assert {
        component.target.minimal_polynomial for component in decomposition.components
    } == {
        Poly(z**2 - 9 * x, z, domain=parent),
        Poly(z**2 - x, z, domain=parent),
    }
    for component in decomposition.components:
        first, second = component.source_images
        assert second == 2 * first or second == -2 * first

    with pytest.raises(AmbiguousCompositumError) as exc_info:
        normalize_primitive_element(sources, z)
    assert exc_info.value.decomposition == decomposition


def test_trager_modulus_agrees_with_dense_tensor_oracle() -> None:
    sources = (
        AlgebraicExtension(Poly(a**2 - 2, a, domain=QQ)),
        AlgebraicExtension(Poly(b**3 - 3, b, domain=QQ)),
    )

    presentation = normalize_primitive_element(sources, z)

    expected_modulus = tensor_characteristic_polynomial(
        sources,
        presentation.primitive_coefficients,
        z,
    )
    assert presentation.target.minimal_polynomial == expected_modulus


def test_primitive_presentation_rejects_a_forged_source_image() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )
    presentation = normalize_primitive_element(sources, z)

    with pytest.raises(ValueError, match="defining relation"):
        replace(
            presentation,
            source_images=(
                presentation.target.field.one,
                presentation.source_images[1],
            ),
        )


def test_primitive_presentation_rejects_a_foreign_parent() -> None:
    parent = rational_function_field()
    other_parent = QQ.frac_field(symbols("u"))
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - 2, b, domain=other_parent)),
    )

    with pytest.raises(ValueError, match="same parent field"):
        normalize_primitive_element(sources, z)


def test_primitive_presentation_rejects_a_parent_generator() -> None:
    parent = rational_function_field()
    sources = (
        AlgebraicExtension(Poly(a**2 - x, a, domain=parent)),
        AlgebraicExtension(Poly(b**2 - x - 1, b, domain=parent)),
    )

    with pytest.raises(ValueError, match="already occurs in the parent field"):
        normalize_primitive_element(sources, x)
