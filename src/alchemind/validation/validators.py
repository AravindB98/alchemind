"""Validation layer: the gate between "the generator dreamed up a string" and
"this is a credible, novel, plausibly-makeable molecule."

Checks performed:
* validity        — parses and sanitizes under RDKit.
* novelty         — not present in (and not near-identical to) a reference set
                    of known compounds.
* drug-likeness   — Lipinski Rule-of-Five compliance.
* synthesizability — a fast heuristic ease-of-synthesis score (1 easy … 10 hard).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from .. import config
from ..utils.chem import canonical_smiles, mol_from_smiles, tanimoto


# --------------------------------------------------------------------------- #
# Drug-likeness
# --------------------------------------------------------------------------- #
def lipinski_report(smiles: str) -> Optional[Dict[str, object]]:
    """Lipinski Rule-of-Five: MW<=500, logP<=5, HBD<=5, HBA<=10."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MolWt": round(mw, 2),
        "MolLogP": round(logp, 2),
        "NumHDonors": hbd,
        "NumHAcceptors": hba,
        "violations": violations,
        "passes": violations <= 1,
    }


# --------------------------------------------------------------------------- #
# Synthesizability (fast heuristic; higher = harder to make)
# --------------------------------------------------------------------------- #
def synthesizability_score(smiles: str) -> Optional[float]:
    """Heuristic synthetic-accessibility score on a 1 (easy) – 10 (hard) scale.

    This is a lightweight, dependency-free approximation in the spirit of Ertl &
    Schuffenhauer's SA score: it penalizes size, ring complexity, stereochemistry,
    spiro/bridgehead atoms and macrocycles. It is intended for *relative* ranking
    of candidates, not as a substitute for a full retrosynthetic assessment.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ring_info = mol.GetRingInfo()
    macrocycles = sum(1 for r in ring_info.AtomRings() if len(r) > 8)

    size_penalty = math.log1p(max(heavy - 10, 0)) * 0.6
    complexity = (
        0.5 * n_stereo
        + 0.9 * n_spiro
        + 0.9 * n_bridge
        + 1.2 * macrocycles
    )
    score = 1.0 + size_penalty + complexity
    return round(min(max(score, 1.0), 10.0), 2)


# --------------------------------------------------------------------------- #
# Report + validator
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    smiles: str
    valid: bool
    novel: bool = False
    max_similarity_to_known: float = 0.0
    nearest_known: Optional[str] = None
    drug_likeness: Optional[Dict[str, object]] = None
    synthesizability: Optional[float] = None
    passed: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class MoleculeValidator:
    """Validates generated molecules against validity/novelty/quality gates."""

    def __init__(
        self,
        reference_smiles: Optional[List[str]] = None,
        novelty_threshold: float = 0.4,
        max_sa_score: float = 6.0,
        require_lipinski: bool = False,
    ):
        self.novelty_threshold = novelty_threshold
        self.max_sa_score = max_sa_score
        self.require_lipinski = require_lipinski
        refs = reference_smiles if reference_smiles is not None else self._load_reference()
        self._reference = [c for c in (canonical_smiles(s) for s in refs) if c]
        self._reference_set = set(self._reference)

    @staticmethod
    def _load_reference() -> List[str]:
        path: Path = config.REFERENCE_SMILES_FILE
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]

    def _novelty(self, smiles: str):
        canon = canonical_smiles(smiles)
        if canon in self._reference_set:
            return False, 1.0, canon
        best_sim, best_ref = 0.0, None
        for ref in self._reference:
            sim = tanimoto(canon, ref)
            if sim > best_sim:
                best_sim, best_ref = sim, ref
        # Novel if it is not a known compound and not a near-duplicate of one.
        novel = best_sim < (1.0 - self.novelty_threshold)
        return novel, round(best_sim, 3), best_ref

    def validate(self, smiles: str) -> ValidationReport:
        canon = canonical_smiles(smiles)
        if canon is None:
            return ValidationReport(smiles=smiles, valid=False,
                                    reasons=["invalid SMILES"])
        novel, sim, nearest = self._novelty(canon)
        lip = lipinski_report(canon)
        sa = synthesizability_score(canon)

        reasons: List[str] = []
        if not novel:
            reasons.append(f"too similar to known compound ({sim:.2f})")
        if sa is not None and sa > self.max_sa_score:
            reasons.append(f"hard to synthesize (SA={sa})")
        if self.require_lipinski and lip and not lip["passes"]:
            reasons.append("fails Lipinski Rule-of-Five")

        return ValidationReport(
            smiles=canon,
            valid=True,
            novel=novel,
            max_similarity_to_known=sim,
            nearest_known=nearest,
            drug_likeness=lip,
            synthesizability=sa,
            passed=len(reasons) == 0,
            reasons=reasons,
        )
