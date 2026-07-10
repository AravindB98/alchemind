"""Core mutation and crossover operators for de novo molecule generation.

These operate directly on RDKit molecules and always return *valid, sanitized*
SMILES (invalid intermediates are discarded), so downstream layers never see
garbage. Two complementary operators are provided:

* ``mutate``    — atom/bond level edits (add, replace, delete, add ring bond).
* ``crossover`` — BRICS fragment recombination between two parent molecules.

Together they drive the evolutionary search in :mod:`alchemind.generation.genetic`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import BRICS, RWMol

from ..utils.chem import canonical_smiles, mol_from_smiles

# Organic atoms the mutator is allowed to introduce, with a common valence.
_ADDABLE_ATOMS = ["C", "N", "O", "F", "S", "Cl"]


@dataclass
class GenerationConfig:
    population_size: int = 60
    generations: int = 8
    mutation_rate: float = 0.5
    elite_fraction: float = 0.2
    max_heavy_atoms: int = 50
    seed: int = 42


def _sanitized_smiles(mol: Chem.Mol) -> Optional[str]:
    """Sanitize a (possibly edited) mol and return canonical SMILES or None."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return canonical_smiles(Chem.MolToSmiles(mol))


def _add_atom(mol: RWMol, rng: random.Random) -> Optional[str]:
    candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetImplicitValence() > 0]
    if not candidates:
        return None
    anchor = rng.choice(candidates)
    new_idx = mol.AddAtom(Chem.Atom(rng.choice(_ADDABLE_ATOMS)))
    mol.AddBond(anchor, new_idx, Chem.BondType.SINGLE)
    return _sanitized_smiles(mol.GetMol())


def _replace_atom(mol: RWMol, rng: random.Random) -> Optional[str]:
    atoms = [a for a in mol.GetAtoms() if a.GetSymbol() in _ADDABLE_ATOMS]
    if not atoms:
        return None
    atom = rng.choice(atoms)
    choices = [s for s in _ADDABLE_ATOMS if s != atom.GetSymbol()]
    atom.SetAtomicNum(Chem.Atom(rng.choice(choices)).GetAtomicNum())
    return _sanitized_smiles(mol.GetMol())


def _delete_atom(mol: RWMol, rng: random.Random) -> Optional[str]:
    terminal = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1]
    if not terminal or mol.GetNumHeavyAtoms() <= 3:
        return None
    mol.RemoveAtom(rng.choice(terminal))
    return _sanitized_smiles(mol.GetMol())


def _add_ring_bond(mol: RWMol, rng: random.Random) -> Optional[str]:
    open_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetImplicitValence() > 0]
    if len(open_atoms) < 2:
        return None
    i, j = rng.sample(open_atoms, 2)
    if mol.GetBondBetweenAtoms(i, j) is not None:
        return None
    mol.AddBond(i, j, Chem.BondType.SINGLE)
    return _sanitized_smiles(mol.GetMol())


_OPERATORS = [_add_atom, _replace_atom, _delete_atom, _add_ring_bond]


def mutate(smiles: str, n: int = 5, rng: Optional[random.Random] = None) -> List[str]:
    """Return up to ``n`` distinct valid mutations of ``smiles``."""
    rng = rng or random.Random()
    parent = mol_from_smiles(smiles)
    if parent is None:
        return []
    out: set[str] = set()
    parent_canon = canonical_smiles(smiles)
    for _ in range(n * 4):
        if len(out) >= n:
            break
        op = rng.choice(_OPERATORS)
        candidate = op(RWMol(parent), rng)
        if candidate and candidate != parent_canon:
            out.add(candidate)
    return list(out)


def crossover(smiles_a: str, smiles_b: str, n: int = 5,
              rng: Optional[random.Random] = None) -> List[str]:
    """Recombine two molecules into novel children using BRICS fragmentation."""
    rng = rng or random.Random()
    ma, mb = mol_from_smiles(smiles_a), mol_from_smiles(smiles_b)
    if ma is None or mb is None:
        return []
    fragments: set[str] = set()
    for m in (ma, mb):
        fragments.update(BRICS.BRICSDecompose(m))
    frag_mols = [Chem.MolFromSmiles(f) for f in fragments]
    frag_mols = [f for f in frag_mols if f is not None]
    if not frag_mols:
        return []
    out: set[str] = set()
    try:
        builder = BRICS.BRICSBuild(frag_mols, scrambleReagents=True, maxDepth=2)
        for _ in range(n * 6):
            try:
                child = next(builder)
            except StopIteration:
                break
            smi = _sanitized_smiles(child)
            if smi:
                out.add(smi)
            if len(out) >= n:
                break
    except Exception:
        return list(out)
    return list(out)
