"""Evolutionary (genetic-algorithm) de novo molecule generator.

Given one or more seed molecules and a fitness function, evolve a population
of novel structures using mutation and BRICS crossover. This is a graph-based
generator in the spirit of Jensen (2019), "A graph-based genetic algorithm and
generative model for de novo molecular design" — no GPU or pretrained weights
required, so it runs anywhere immediately.
"""
from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

from ..utils.chem import canonical_smiles
from .base import GenerationConfig, crossover, mutate

FitnessFn = Callable[[str], float]


class GeneticGenerator:
    """Evolve novel molecules that maximize a fitness function."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._rng = random.Random(self.config.seed)

    # -- population helpers -------------------------------------------------
    def _seed_population(self, seeds: Sequence[str]) -> List[str]:
        pop: set[str] = set()
        for s in seeds:
            c = canonical_smiles(s)
            if c:
                pop.add(c)
        # Expand initial seeds by mutation until we reach the population size.
        seeds_list = list(pop) or []
        guard = 0
        while len(pop) < self.config.population_size and seeds_list and guard < 2000:
            guard += 1
            parent = self._rng.choice(seeds_list)
            for child in mutate(parent, n=3, rng=self._rng):
                pop.add(child)
                if len(pop) >= self.config.population_size:
                    break
        return list(pop)

    def _breed(self, population: List[str]) -> List[str]:
        children: set[str] = set()
        for parent in population:
            if self._rng.random() < self.config.mutation_rate:
                children.update(mutate(parent, n=2, rng=self._rng))
        # Crossover among random pairs.
        if len(population) >= 2:
            for _ in range(len(population) // 2):
                a, b = self._rng.sample(population, 2)
                children.update(crossover(a, b, n=2, rng=self._rng))
        return [c for c in children
                if canonical_smiles(c) is not None
                and _heavy_ok(c, self.config.max_heavy_atoms)]

    # -- public API ---------------------------------------------------------
    def generate(
        self,
        seeds: Sequence[str],
        fitness_fn: FitnessFn,
        n: int = 20,
    ) -> List[Tuple[str, float]]:
        """Return the top-``n`` (smiles, fitness) pairs discovered.

        Args:
            seeds: starting SMILES to evolve from.
            fitness_fn: maps a SMILES to a score (higher = better).
            n: number of top molecules to return.
        """
        population = self._seed_population(seeds)
        scored = {s: fitness_fn(s) for s in population}

        for _ in range(self.config.generations):
            children = self._breed(list(scored.keys()))
            for c in children:
                if c not in scored:
                    scored[c] = fitness_fn(c)
            # Elitist truncation selection.
            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            keep = max(self.config.population_size, n)
            scored = dict(ranked[:keep])

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]


def _heavy_ok(smiles: str, max_heavy: int) -> bool:
    from ..utils.chem import mol_from_smiles

    m = mol_from_smiles(smiles)
    return m is not None and m.GetNumHeavyAtoms() <= max_heavy
