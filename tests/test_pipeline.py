import pytest

from alchemind import DiscoveryPipeline
from alchemind.generation import GenerationConfig
from alchemind.utils.chem import is_valid_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def pipeline():
    # Small, fast configuration for tests.
    return DiscoveryPipeline(gen_config=GenerationConfig(
        population_size=25, generations=3, seed=7))


def test_pipeline_runs_end_to_end(pipeline):
    result = pipeline.run(ASPIRIN, objective="qed", n=8)
    assert result.n_generated > 0
    assert result.n_valid > 0
    # Every returned candidate must be valid, novel, and validated.
    for c in result.candidates:
        assert is_valid_smiles(c.smiles)
        assert c.valid and c.novel and c.passed_validation
        assert c.smiles != ASPIRIN


def test_pipeline_ranking_is_sorted(pipeline):
    result = pipeline.run(ASPIRIN, objective="qed", n=8)
    scores = [c.objective_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_solubility_objective(pipeline):
    result = pipeline.run(ASPIRIN, objective="solubility", n=5)
    assert all(c.predicted_logS is not None for c in result.candidates)


def test_pipeline_rejects_unknown_objective(pipeline):
    with pytest.raises(ValueError):
        pipeline.run(ASPIRIN, objective="nonsense")


def test_pipeline_include_svg(pipeline):
    result = pipeline.run(ASPIRIN, objective="qed", n=3, include_svg=True)
    if result.candidates:
        assert result.candidates[0].svg is not None
        assert "<svg" in result.candidates[0].svg
