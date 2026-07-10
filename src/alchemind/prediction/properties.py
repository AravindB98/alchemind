"""Aggregate property predictor combining descriptors, drug-likeness, and the
trained solubility model into a single object per molecule."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from .descriptors import molecular_properties
from .solubility import SolubilityModel


@dataclass
class PredictedProperties:
    smiles: str
    valid: bool
    properties: Dict[str, float] = field(default_factory=dict)
    predicted_logS: Optional[float] = None
    qed: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class PropertyPredictor:
    """One entry point for all property predictions on a molecule."""

    def __init__(self, solubility_model: Optional[SolubilityModel] = None):
        self.solubility = solubility_model or SolubilityModel.load()

    def predict(self, smiles: str) -> PredictedProperties:
        props = molecular_properties(smiles)
        if props is None:
            return PredictedProperties(smiles=smiles, valid=False)
        return PredictedProperties(
            smiles=smiles,
            valid=True,
            properties=props,
            predicted_logS=self.solubility.predict(smiles),
            qed=props.get("QED"),
        )
