"""Molecular descriptors: the numeric feature vector for ML models and the
human-readable property panel shown in the UI."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from ..utils.chem import mol_from_smiles

# Ordered descriptor set used as ML features (keep names/order stable).
DESCRIPTOR_NAMES: List[str] = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
]


def descriptor_vector(smiles: str) -> Optional[np.ndarray]:
    """Return the fixed-length descriptor feature vector, or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
    ]
    return np.asarray(values, dtype=float)


def molecular_properties(smiles: str) -> Optional[Dict[str, float]]:
    """Return a labeled dict of interpretable molecular properties."""
    vec = descriptor_vector(smiles)
    if vec is None:
        return None
    mol = mol_from_smiles(smiles)
    props = {name: round(float(v), 3) for name, v in zip(DESCRIPTOR_NAMES, vec)}
    props["QED"] = round(float(QED.qed(mol)), 3)
    return props
