r"""Real descent of parametric RDE spaces through the Cayley isomorphism."""

from __future__ import annotations

from collections.abc import Sequence

from ..gaussian import HypertangentGaussianNormalForm
from ._parametric_linear import (
    ParametricRDESolution,
    canonicalize_directions,
)


def descend_hypertangent_parametric_basis(
    target_basis: Sequence[ParametricRDESolution],
    normal_form: HypertangentGaussianNormalForm,
) -> tuple[ParametricRDESolution, ...]:
    r"""Descend a complete Gaussian PRDE basis to its real source tower.

    The source equation has coefficients in the fixed field of Gaussian
    conjugation.  Consequently, real and imaginary parts of a basis of its
    complexified solution space span the complete source solution space.
    Splitting includes the parameter coordinates: retaining only real target
    parameters would lose valid combinations whose target basis is not itself
    conjugation invariant.
    """
    directions = []
    for vector in target_basis:
        split_parameters = tuple(
            normal_form.split_constant(parameter) for parameter in vector.parameters
        )
        real_solution, imaginary_solution = normal_form.split_inverse(vector.solution)
        directions.extend(
            (
                ParametricRDESolution(
                    tuple(real for real, _imaginary in split_parameters),
                    real_solution,
                ),
                ParametricRDESolution(
                    tuple(imaginary for _real, imaginary in split_parameters),
                    imaginary_solution,
                ),
            )
        )

    return canonicalize_directions(
        directions,
        normal_form.source_field,
        normal_form.source_derivation.tower.constants,
    )


__all__ = ["descend_hypertangent_parametric_basis"]
