from importlib.metadata import version
from typing import get_type_hints

import sympy_risch
import sympy_risch.algebraic as algebraic
import sympy_risch.algebraic_geometry as algebraic_geometry
import sympy_risch.dependency as dependency
import sympy_risch.monomial as monomial
import sympy_risch.rde as rde
import sympy_risch.relations as relations
import sympy_risch.tower as tower
from sympy_risch import __version__
from sympy_risch.dependency import models as dependency_models
from sympy_risch.monomial import primitive as monomial_primitive
from sympy_risch.rde import models as rde_models


def test_package_import() -> None:
    assert __version__ == version("sympy-risch")


def test_public_exports_are_defined_once() -> None:
    assert len(sympy_risch.__all__) == len(set(sympy_risch.__all__))
    assert all(hasattr(sympy_risch, name) for name in sympy_risch.__all__)


def test_rde_facade_keeps_the_scalar_solver_surface() -> None:
    assert all(hasattr(rde, name) for name in rde.__all__)
    assert (
        rde.PolynomialRischDifferentialEquationResult
        is rde_models.PolynomialRischDifferentialEquationResult
    )
    assert (
        rde.solve_risch_differential_equation
        is sympy_risch.solve_risch_differential_equation
    )


def test_dependency_facade_keeps_the_parametric_solver_surface() -> None:
    assert all(hasattr(dependency, name) for name in dependency.__all__)
    assert (
        dependency.ParametricLogarithmicDerivativeResult
        is dependency_models.ParametricLogarithmicDerivativeResult
    )
    assert (
        dependency.solve_parametric_logarithmic_derivative
        is sympy_risch.solve_parametric_logarithmic_derivative
    )


def test_monomial_facade_keeps_the_family_reducer_surface() -> None:
    assert all(hasattr(monomial, name) for name in monomial.__all__)
    assert (
        monomial.PrimitiveLogarithmicDerivativeSignature
        is monomial_primitive.PrimitiveLogarithmicDerivativeSignature
    )
    assert monomial.solve_radical_logarithmic_derivative is (
        sympy_risch.solve_radical_logarithmic_derivative
    )


def test_monomial_public_annotations_are_runtime_resolvable() -> None:
    for name in monomial.__all__:
        value = getattr(monomial, name)
        if isinstance(value, type):
            get_type_hints(value)


def test_algebraic_facade_has_an_intentional_advanced_surface() -> None:
    assert set(algebraic.__all__) == {
        "AlgebraicAllPlaceResidueResult",
        "AlgebraicAllPlaceResidueStatus",
        "AlgebraicConstantEmbedding",
        "AlgebraicConstantFieldAnalysis",
        "AlgebraicConstantFieldStatus",
        "AlgebraicExactnessCertificate",
        "AlgebraicExactnessResult",
        "AlgebraicExactnessStatus",
        "AlgebraicFraction",
        "AlgebraicIntegratingFactorCertificate",
        "AlgebraicIntegratingFactorRDEResult",
        "AlgebraicIntegratingFactorRDEStatus",
        "AlgebraicIntegratingFactorResult",
        "AlgebraicIntegratingFactorStatus",
        "AlgebraicLocalDifferentialResidue",
        "AlgebraicPlaceResidue",
        "AlgebraicResidueProfile",
        "AlgebraicResidueProfileFactor",
        "AlgebraicResidueProfileStatus",
        "IntegralBasis",
        "RelativeSimpleRadicalData",
        "RelativeSimpleRadicalHermiteMode",
        "RelativeSimpleRadicalHermiteReduction",
        "RelativeSimpleRadicalIntegrationCertificate",
        "RelativeSimpleRadicalRDEComponent",
        "RelativeSimpleRadicalRDEResult",
        "RelativeSimpleRadicalRDEStatus",
        "SimpleRadicalData",
        "TragerAlgebraicResidueResult",
        "analyze_algebraic_constant_field",
        "build_algebraic_residue_profile",
        "collect_algebraic_all_place_residues",
        "find_algebraic_integrating_factor",
        "hermite_reduce_algebraic",
        "hermite_reduce_relative_simple_radical",
        "lazy_hermite_reduce_algebraic",
        "recognize_simple_radical",
        "recognize_relative_simple_radical",
        "solve_algebraic_exactness",
        "solve_algebraic_integrating_factor_rde",
        "solve_relative_simple_radical_exact_derivative",
        "solve_relative_simple_radical_rde",
        "simple_radical_integral_basis",
        "trager_algebraic_place_residues",
        "trager_algebraic_residue_polynomial",
    }
    assert all(hasattr(algebraic, name) for name in algebraic.__all__)
    assert not hasattr(algebraic, "UnramifiedFinitePlace")
    assert not hasattr(algebraic, "unramified_finite_places")


def test_algebraic_geometry_facade_has_an_intentional_advanced_surface() -> None:
    assert set(algebraic_geometry.__all__) == {
        "AlgebraicDifferential",
        "AlgebraicDivisor",
        "AlgebraicConstantFieldExtension",
        "AlgebraicFractionalIdeal",
        "AlgebraicFunctionField",
        "AlgebraicOrder",
        "AlgebraicPrincipalityDecision",
        "AlgebraicPrincipalityStatus",
        "AlgebraicTorsionSearch",
        "AlgebraicTorsionStatus",
        "FiniteAlgebraicPlace",
        "HessDivisorIdeals",
        "InfiniteAlgebraicPlace",
        "IntegralClosureResult",
        "LocalLaurentExpansion",
        "ReducedAlgebraicDivisorClass",
        "integral_closure",
        "hess_divisor_ideals",
        "polynomial_order",
        "reduce_algebraic_divisor_class",
        "search_algebraic_divisor_torsion",
        "decide_algebraic_principality",
        "extend_algebraic_constants",
    }
    assert all(hasattr(algebraic_geometry, name) for name in algebraic_geometry.__all__)
    assert not hasattr(algebraic_geometry, "AlgebraicPlace")
    assert not hasattr(algebraic_geometry, "IntegralClosureStep")
    assert not hasattr(algebraic_geometry, "OrderMaximalityCertificate")
    assert not hasattr(algebraic_geometry, "ReciprocalFunctionField")


def test_algebraic_geometry_public_annotations_are_runtime_resolvable() -> None:
    for name in algebraic_geometry.__all__:
        value = getattr(algebraic_geometry, name)
        if isinstance(value, type):
            get_type_hints(value)


def test_algebraic_integration_public_annotations_are_runtime_resolvable() -> None:
    get_type_hints(sympy_risch.AlgebraicConstantBaseChangeIntegrationResult)
    get_type_hints(sympy_risch.AlgebraicExactnessResult)
    get_type_hints(sympy_risch.AlgebraicIntegratingFactorResult)
    get_type_hints(sympy_risch.AlgebraicIntegratingFactorRDEResult)
    get_type_hints(sympy_risch.AlgebraicIntegrationResult)
    get_type_hints(sympy_risch.AlgebraicLogarithmicTerm)
    get_type_hints(sympy_risch.ElementaryIntegralResult)


def test_tower_facade_has_an_intentional_advanced_surface() -> None:
    assert set(tower.__all__) == {
        "AlgebraicExtension",
        "CoefficientDerivation",
        "ConstantFieldStatus",
        "Derivation",
        "DifferentialExtension",
        "DifferentialTower",
        "MonomialExtension",
        "MonomialFamily",
        "TowerRepresentationAnalysis",
        "TowerRepresentationOutcome",
        "TowerRepresentationProposal",
        "TowerRepresentationReason",
        "analyze_tower_representation",
        "rebase_derivation_constants",
    }
    assert all(hasattr(tower, name) for name in tower.__all__)


def test_tower_public_annotations_are_runtime_resolvable() -> None:
    for name in tower.__all__:
        value = getattr(tower, name)
        if isinstance(value, type):
            get_type_hints(value)


def test_relations_facade_has_an_intentional_advanced_surface() -> None:
    assert set(relations.__all__) == {
        "FundamentalDirection",
        "FundamentalSignature",
        "FundamentalSignatureOutcome",
        "FundamentalSignatureReason",
        "HyperexponentialFundamentalDirection",
        "HypertangentFundamentalDirection",
        "LogarithmicRelationSystemResult",
        "LogarithmicRelationWitness",
        "PrimitiveFundamentalDirection",
        "build_fundamental_signature",
        "solve_logarithmic_relation_system",
    }
    assert all(hasattr(relations, name) for name in relations.__all__)


def test_relations_public_annotations_are_runtime_resolvable() -> None:
    for name in relations.__all__:
        value = getattr(relations, name)
        if isinstance(value, type):
            get_type_hints(value)


def test_admission_exports_are_public() -> None:
    expected = {
        "TowerAdmission",
        "TowerAdmissionOutcome",
        "TowerAdmissionReason",
        "assess_elementary_admission",
        "assess_scalar_rde_admission",
    }

    assert expected <= set(sympy_risch.__all__)


def test_expression_frontend_exports_are_public() -> None:
    expected = {
        "ExpressionCompilation",
        "ExpressionCompilationOutcome",
        "ExpressionIntegrationOutcome",
        "ExpressionIntegralResult",
        "ExpressionUnsupportedReason",
        "compile_expression",
        "integrate_expression",
    }

    assert expected <= set(sympy_risch.__all__)


def test_riccati_mobius_exports_are_public() -> None:
    expected = {
        "RiccatiMobiusElementaryCertificate",
        "RiccatiMobiusNormalForm",
        "build_riccati_mobius_normal_form",
        "integrate_riccati_mobius_normal_form",
    }

    assert expected <= set(sympy_risch.__all__)


def test_certified_special_pole_exports_are_public() -> None:
    expected = {
        "LinearSpecialPoleReduction",
        "LinearSpecialPoleReductionStatus",
        "LinearSpecialPoleStep",
        "reduce_certified_linear_special_poles",
    }

    assert expected <= set(sympy_risch.__all__)
