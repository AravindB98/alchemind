"""Low-level cheminformatics helpers built on RDKit."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity

# RDKit is noisy about parse failures; we handle those explicitly.
RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES into an RDKit Mol, or return None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """True if the SMILES parses and sanitizes cleanly."""
    return mol_from_smiles(smiles) is not None


@lru_cache(maxsize=8192)
def canonical_smiles(smiles: str) -> Optional[str]:
    """Return the canonical SMILES, or None if the input is invalid."""
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """ECFP-style Morgan fingerprint bit vector, or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Tanimoto similarity between two molecules (0.0 if either is invalid)."""
    fa = morgan_fingerprint(smiles_a)
    fb = morgan_fingerprint(smiles_b)
    if fa is None or fb is None:
        return 0.0
    return float(TanimotoSimilarity(fa, fb))


def to_svg(smiles: str, size: int = 320) -> Optional[str]:
    """Render a molecule to an inline SVG string for the web UI."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def to_png_bytes(smiles: str, size: int = 320) -> Optional[bytes]:
    """Render a molecule to PNG bytes (useful for reports)."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(size, size))
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
