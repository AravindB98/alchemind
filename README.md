# Alchemind

**Invent, predict, and validate novel chemical compounds.** Alchemind is an end-to-end
de novo molecular discovery platform: give it a seed molecule and an objective, and it
*evolves brand-new molecules*, predicts their properties with ML, and filters them
through validity / novelty / synthesizability gates — then serves the results through a
REST API and a web UI.

```
  seed molecule ──▶ GENERATE ──▶ PREDICT ──▶ VALIDATE ──▶ RANK ──▶ novel candidates
                   (evolve new)  (ML props)  (real/novel/  (by objective)
                                              makeable?)
```

> **Codename:** *Heisenberg* · **Project:** Alchemind (*alchemy* — inventing new matter — + *mind*).

---

## Why this exists

Modern drug/material discovery increasingly runs a generative loop: propose new structures,
score them, keep the good and novel ones, repeat. Alchemind implements that full loop in a
small, readable, production-shaped codebase — the same pattern used by platforms like
REINVENT and graph-based genetic algorithms, packaged as a service.

## What it does

| Layer | What it does | Key tech |
|-------|--------------|----------|
| **Generation** | Invents novel molecules by evolving a seed with mutation + BRICS crossover (graph-based GA). Optional neural char-RNN SMILES generator. | RDKit, (optional) PyTorch |
| **Prediction** | Predicts aqueous solubility (logS) with a trained RandomForest; computes QED drug-likeness, MolWt, logP, TPSA, H-bond donors/acceptors, and more. | scikit-learn, RDKit |
| **Validation** | Gates each molecule on RDKit validity, **novelty** vs a reference set of known compounds, Lipinski Rule-of-Five, and a heuristic **synthesizability** score. | RDKit |
| **Pipeline** | Orchestrates generate → predict → validate → rank against a chosen objective. | — |
| **Serving** | FastAPI REST API (`/discover`, `/predict`, `/health`) + single-page web UI with 2D molecule rendering. | FastAPI, Uvicorn |

## Quickstart

```bash
git clone https://github.com/AravindB98/alchemind.git
cd alchemind
pip install -e ".[dev]"

# (optional) train the solubility model on the bundled sample dataset
python scripts/train_solubility.py

# invent 5 novel, drug-like analogs of aspirin from the CLI
alchemind discover "CC(=O)Oc1ccccc1C(=O)O" --objective qed --n 5

# analyze a single molecule
alchemind predict "CC(=O)Oc1ccccc1C(=O)O"
```

### Run the API + web UI

```bash
uvicorn alchemind.api:app --reload
# open http://localhost:8000        (web UI)
# open http://localhost:8000/docs   (interactive API docs)
```

### Docker

```bash
docker compose up --build
# → http://localhost:8000
```

## API

```bash
# Invent molecules
curl -X POST localhost:8000/discover -H 'Content-Type: application/json' \
  -d '{"seed_smiles":"CC(=O)Oc1ccccc1C(=O)O","objective":"solubility","n":10}'

# Predict properties
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

Objectives: `qed` (drug-likeness), `solubility` (aqueous logS), `drug_likeness`
(drug-like **and** easy to synthesize).

## How the generation works

The default generator is a **graph-based genetic algorithm** (no GPU/pretraining needed):

1. **Seed & expand** — canonicalize seeds and grow an initial population by mutation.
2. **Mutate** — atom additions, element swaps, terminal deletions, and ring-closing bonds,
   each re-sanitized so only valid molecules survive.
3. **Crossover** — fragment parents with **BRICS** and recombine fragments into children.
4. **Select** — a composite fitness (objective, gated by validity/novelty, penalized for poor
   synthesizability) drives elitist truncation selection across generations.

An optional **char-RNN** (`alchemind.generation.char_rnn`, `pip install alchemind[deep]`)
learns the SMILES language and samples molecules neurally — the deep-learning counterpart.

## Project layout

```
src/alchemind/
├── generation/   # de novo generators (genetic GA + optional char-RNN)
├── prediction/   # descriptors, solubility model, aggregate predictor
├── validation/   # validity, novelty, Lipinski, synthesizability
├── pipeline/     # the generate→predict→validate→rank orchestrator
├── api/          # FastAPI app + Pydantic schemas
├── web/          # single-page UI
└── data/         # reference compounds + sample solubility dataset
scripts/          # train_solubility.py, train_char_rnn.py, demo.py
tests/            # pytest suite (generation, prediction, validation, pipeline, API)
```

## Testing & quality

```bash
pytest --cov=alchemind      # full suite with coverage
ruff check src tests        # lint
```

CI (GitHub Actions) runs the test matrix on Python 3.10/3.11 and builds the Docker image on
every push.

## Scientific notes & honest limitations

- The bundled `solubility_sample.csv` is a small illustrative subset; point
  `train_solubility.py` at the full Delaney/ESOL dataset for a production-grade model.
- The synthesizability score is a fast **heuristic** for *relative* ranking, not a substitute
  for full retrosynthetic analysis (e.g. AiZynthFinder).
- Novelty is assessed against a bundled reference set; swap in ChEMBL/PubChem for a stricter
  definition of "known."
- Generated molecules are computational hypotheses — they are **not** validated for safety,
  stability, or real-world synthesizability, and nothing here is a recommendation to
  synthesize any compound.

## Roadmap (Phase 2+)

- RAG over chemistry literature + a compound↔target↔disease knowledge graph
- Graph neural network property models (ChemBERTa / GNN) with a benchmark leaderboard
- Retrosynthesis-aware synthesizability scoring
- LLM agent front-end that plans multi-step discovery campaigns

## License

MIT © 2026 Aravind Balaji
