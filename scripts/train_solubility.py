#!/usr/bin/env python3
"""Train the aqueous-solubility (logS) RandomForest model.

Uses the bundled sample dataset by default. Pass --dataset to point at a larger
CSV (e.g. the full Delaney/ESOL set) with columns `smiles,log_solubility`.

    python scripts/train_solubility.py
    python scripts/train_solubility.py --dataset path/to/delaney.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from alchemind import config
from alchemind.prediction import SolubilityModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(config.SOLUBILITY_DATASET))
    ap.add_argument("--out", default=str(config.SOLUBILITY_MODEL))
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    smiles = df["smiles"].tolist()
    targets = df["log_solubility"].astype(float).tolist()
    print(f"Loaded {len(df)} molecules from {args.dataset}")

    model = SolubilityModel().fit(smiles, targets)

    # Quick cross-validated sanity check (R^2).
    from alchemind.prediction.descriptors import descriptor_vector

    X = np.vstack([descriptor_vector(s) for s in smiles if descriptor_vector(s) is not None])
    y = np.asarray([t for s, t in zip(smiles, targets) if descriptor_vector(s) is not None])
    from sklearn.ensemble import RandomForestRegressor

    cv = cross_val_score(
        RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_SEED),
        X, y, cv=min(5, len(y)), scoring="r2",
    )
    print(f"5-fold CV R^2: mean={cv.mean():.3f} (+/- {cv.std():.3f})")

    path = model.save(args.out)
    print(f"Saved trained model -> {path}")


if __name__ == "__main__":
    main()
