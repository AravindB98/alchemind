"""The Alchemind discovery loop: generate -> predict -> validate -> rank.

This is the single entry point that ties every layer together. Give it a seed
molecule and an objective ("solubility", "qed", or "drug_likeness") and it
evolves novel candidates that optimize that objective, then returns only the
ones that pass validation, ranked by a composite score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..generation import GenerationConfig, GeneticGenerator
from ..prediction import PropertyPredictor
from ..validation import MoleculeValidator
from ..utils.chem import to_svg

# Objective functions map a fully-scored candidate to a scalar "goodness".
OBJECTIVES: Dict[str, Callable[["Candidate"], float]] = {
    # More soluble is better (higher logS).
    "solubility": lambda c: (c.predicted_logS if c.predicted_logS is not None else -10),
    # More drug-like is better.
    "qed": lambda c: (c.qed or 0.0),
    # Drug-like AND easy to synthesize.
    "drug_likeness": lambda c: (c.qed or 0.0) - 0.05 * (c.synthesizability or 10),
}


@dataclass
class Candidate:
    smiles: str
    valid: bool = False
    novel: bool = False
    qed: Optional[float] = None
    predicted_logS: Optional[float] = None
    synthesizability: Optional[float] = None
    similarity_to_known: float = 0.0
    passed_validation: bool = False
    properties: Dict[str, float] = field(default_factory=dict)
    objective_score: float = 0.0
    svg: Optional[str] = None

    def to_dict(self, include_svg: bool = True) -> dict:
        d = asdict(self)
        if not include_svg:
            d.pop("svg", None)
        return d


@dataclass
class DiscoveryResult:
    objective: str
    seeds: List[str]
    candidates: List[Candidate]
    n_generated: int
    n_valid: int
    n_novel: int

    def to_dict(self, include_svg: bool = True) -> dict:
        return {
            "objective": self.objective,
            "seeds": self.seeds,
            "n_generated": self.n_generated,
            "n_valid": self.n_valid,
            "n_novel": self.n_novel,
            "candidates": [c.to_dict(include_svg=include_svg) for c in self.candidates],
        }


class DiscoveryPipeline:
    """End-to-end de novo discovery orchestrator."""

    def __init__(
        self,
        predictor: Optional[PropertyPredictor] = None,
        validator: Optional[MoleculeValidator] = None,
        gen_config: Optional[GenerationConfig] = None,
    ):
        self.predictor = predictor or PropertyPredictor()
        self.validator = validator or MoleculeValidator()
        self.generator = GeneticGenerator(gen_config)

    # -- scoring ------------------------------------------------------------
    def _score_candidate(self, smiles: str, objective: str) -> Candidate:
        pred = self.predictor.predict(smiles)
        report = self.validator.validate(smiles)
        cand = Candidate(
            smiles=smiles,
            valid=pred.valid and report.valid,
            novel=report.novel,
            qed=pred.qed,
            predicted_logS=pred.predicted_logS,
            synthesizability=report.synthesizability,
            similarity_to_known=report.max_similarity_to_known,
            passed_validation=report.passed,
            properties=pred.properties,
        )
        cand.objective_score = OBJECTIVES[objective](cand)
        return cand

    def _fitness(self, objective: str):
        """Fitness used to steer evolution: objective, gated by validity/novelty
        and softly penalized for poor synthesizability."""
        def fn(smiles: str) -> float:
            cand = self._score_candidate(smiles, objective)
            if not cand.valid:
                return -1e6
            score = cand.objective_score
            if not cand.novel:
                score -= 1.0
            if cand.synthesizability and cand.synthesizability > 6.0:
                score -= 0.5 * (cand.synthesizability - 6.0)
            return score
        return fn

    # -- public API ---------------------------------------------------------
    def run(
        self,
        seed_smiles: Sequence[str] | str,
        objective: str = "qed",
        n: int = 20,
        include_svg: bool = False,
        only_valid_novel: bool = True,
    ) -> DiscoveryResult:
        if objective not in OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{objective}'. Choose from {list(OBJECTIVES)}."
            )
        seeds = [seed_smiles] if isinstance(seed_smiles, str) else list(seed_smiles)

        # 1-2. Generate + optimize via the genetic generator (predict/validate
        #      happen inside the fitness function).
        evolved = self.generator.generate(seeds, self._fitness(objective),
                                           n=max(n * 3, 30))

        # 3. Fully score the surviving population.
        candidates = [self._score_candidate(smi, objective) for smi, _ in evolved]

        n_generated = len(candidates)
        n_valid = sum(c.valid for c in candidates)
        n_novel = sum(c.valid and c.novel for c in candidates)

        # 4. Filter + rank.
        pool = candidates
        if only_valid_novel:
            pool = [c for c in candidates if c.valid and c.novel and c.passed_validation]
        pool.sort(key=lambda c: c.objective_score, reverse=True)
        top = pool[:n]

        if include_svg:
            for c in top:
                c.svg = to_svg(c.smiles)

        return DiscoveryResult(
            objective=objective,
            seeds=seeds,
            candidates=top,
            n_generated=n_generated,
            n_valid=n_valid,
            n_novel=n_novel,
        )
