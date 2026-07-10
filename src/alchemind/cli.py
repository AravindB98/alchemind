"""Command-line interface: `alchemind discover ...` and `alchemind predict ...`."""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import OBJECTIVES, DiscoveryPipeline
from .prediction import PropertyPredictor
from .validation import lipinski_report, synthesizability_score


def _predict(args) -> int:
    pred = PropertyPredictor().predict(args.smiles)
    out = pred.to_dict()
    out["drug_likeness"] = lipinski_report(args.smiles)
    out["synthesizability"] = synthesizability_score(args.smiles)
    print(json.dumps(out, indent=2))
    return 0 if pred.valid else 1


def _discover(args) -> int:
    pipe = DiscoveryPipeline()
    result = pipe.run(args.seed, objective=args.objective, n=args.n)
    print(
        f"Generated {result.n_generated} molecules "
        f"({result.n_valid} valid, {result.n_novel} novel). "
        f"Top {len(result.candidates)} by '{result.objective}':\n"
    )
    for i, c in enumerate(result.candidates, 1):
        print(
            f"{i:2d}. {c.smiles}\n"
            f"    score={c.objective_score:.3f}  QED={c.qed}  "
            f"logS={c.predicted_logS}  SA={c.synthesizability}  "
            f"novel={c.novel}"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alchemind", description="Alchemind CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Predict properties of a molecule.")
    p.add_argument("smiles")
    p.set_defaults(func=_predict)

    d = sub.add_parser("discover", help="Invent novel molecules from a seed.")
    d.add_argument("seed", help="Seed SMILES.")
    d.add_argument("--objective", default="qed", choices=list(OBJECTIVES))
    d.add_argument("--n", type=int, default=10)
    d.set_defaults(func=_discover)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
