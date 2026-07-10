"""Aqueous-solubility (logS) prediction.

A RandomForest regressor is trained on RDKit descriptors (see
``scripts/train_solubility.py``). When no trained model file is present, the
predictor falls back to the Delaney "ESOL" linear estimate so the system works
out of the box — better once you train, usable immediately.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import config
from ..utils.chem import mol_from_smiles
from .descriptors import descriptor_vector


def esol_estimate(smiles: str) -> Optional[float]:
    """Delaney (2004) ESOL empirical logS estimate. Zero-training baseline."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    ap = (aromatic * 6) / heavy if heavy else 0.0  # aromatic proportion proxy
    return round(0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap, 3)


class SolubilityModel:
    """Trainable logS regressor with a graceful ESOL fallback."""

    def __init__(self, model=None):
        self._model = model

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # -- training / IO ------------------------------------------------------
    def fit(self, smiles: List[str], targets: List[float], n_estimators: int = 300):
        from sklearn.ensemble import RandomForestRegressor

        X, y = [], []
        for smi, t in zip(smiles, targets):
            vec = descriptor_vector(smi)
            if vec is not None:
                X.append(vec)
                y.append(t)
        if not X:
            raise ValueError("No valid molecules to train on.")
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=config.RANDOM_SEED, n_jobs=-1
        )
        model.fit(np.vstack(X), np.asarray(y))
        self._model = model
        return self

    def save(self, path: Path = config.SOLUBILITY_MODEL) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: Path = config.SOLUBILITY_MODEL) -> "SolubilityModel":
        import joblib

        path = Path(path)
        if not path.exists():
            return cls(model=None)
        return cls(model=joblib.load(path))

    # -- inference ----------------------------------------------------------
    def predict(self, smiles: str) -> Optional[float]:
        """Predicted logS (mol/L). Uses the trained model if available."""
        if self._model is None:
            return esol_estimate(smiles)
        vec = descriptor_vector(smiles)
        if vec is None:
            return None
        return round(float(self._model.predict(vec.reshape(1, -1))[0]), 3)
