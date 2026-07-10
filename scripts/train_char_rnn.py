#!/usr/bin/env python3
"""Train the optional char-RNN SMILES generator (requires `pip install torch`).

    python scripts/train_char_rnn.py --epochs 40 --sample 15

Trains on the SMILES in the bundled solubility dataset (swap in a larger corpus
for real use), then samples a few novel molecules to demonstrate the model.
"""
from __future__ import annotations

import argparse

import pandas as pd

from alchemind import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.SOLUBILITY_DATASET))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--out", default=str(config.MODELS_DIR / "char_rnn.pt"))
    args = ap.parse_args()

    try:
        from alchemind.generation.char_rnn import CharRNNGenerator
    except ImportError as e:
        raise SystemExit(str(e))

    smiles = pd.read_csv(args.dataset)["smiles"].tolist()
    print(f"Training char-RNN on {len(smiles)} SMILES for {args.epochs} epochs…")
    gen = CharRNNGenerator().train(smiles, epochs=args.epochs)
    gen.save(args.out)
    print(f"Saved -> {args.out}\nSampled molecules:")
    for smi in gen.sample(n=args.sample):
        print(" ", smi)


if __name__ == "__main__":
    main()
