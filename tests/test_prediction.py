import numpy as np

from alchemind.prediction import (
    PropertyPredictor,
    SolubilityModel,
    descriptor_vector,
    molecular_properties,
)
from alchemind.prediction.solubility import esol_estimate

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_descriptor_vector_shape_and_validity():
    vec = descriptor_vector(ASPIRIN)
    assert vec is not None and vec.shape == (10,)
    assert descriptor_vector("xyz!!") is None


def test_molecular_properties_has_qed():
    props = molecular_properties(ASPIRIN)
    assert props is not None
    assert 0.0 <= props["QED"] <= 1.0
    assert props["MolWt"] > 150


def test_esol_estimate_reasonable():
    # Aspirin measured logS ~ -1.7; empirical estimate should be in a sane range.
    est = esol_estimate(ASPIRIN)
    assert est is not None and -6 < est < 2


def test_solubility_model_train_and_predict():
    smiles = ["CCO", "c1ccccc1", "CCCCCCCC", "CC(=O)O", ASPIRIN]
    targets = [1.1, -1.64, -5.2, 1.22, -1.72]
    model = SolubilityModel().fit(smiles, targets)
    assert model.is_trained
    pred = model.predict(ASPIRIN)
    assert isinstance(pred, float)


def test_property_predictor_fallback_without_model():
    predictor = PropertyPredictor(solubility_model=SolubilityModel(model=None))
    out = predictor.predict(ASPIRIN)
    assert out.valid
    assert out.predicted_logS is not None
    assert out.qed is not None
