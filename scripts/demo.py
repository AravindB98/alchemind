#!/usr/bin/env python3
"""End-to-end demo: invent soluble analogs of aspirin and print the top hits."""
from __future__ import annotations

from alchemind import DiscoveryPipeline

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def main() -> None:
    pipe = DiscoveryPipeline()
    print("Seed:", ASPIRIN)
    for objective in ("qed", "solubility"):
        result = pipe.run(ASPIRIN, objective=objective, n=5)
        print(f"\n=== Objective: {objective} ===")
        print(f"generated={result.n_generated} valid={result.n_valid} novel={result.n_novel}")
        for i, c in enumerate(result.candidates, 1):
            print(f"{i}. {c.smiles}  score={c.objective_score:.3f} "
                  f"QED={c.qed} logS={c.predicted_logS} SA={c.synthesizability}")


if __name__ == "__main__":
    main()
