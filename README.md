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

## Contributing

Contributions are welcome and appreciated — whether it's a bug fix, a new property
model, a smarter generator, better docs, or just a typo. Here's how to get involved:

1. **⭐ Star the repo** — if Alchemind is useful or interesting to you, please
   [star it](https://github.com/AravindB98/alchemind). It helps others discover the project.
2. **🍴 Fork the repo** — click **Fork** (top-right on GitHub) to create your own copy under
   your account.
3. **Clone your fork and set up the dev environment:**
   ```bash
   git clone https://github.com/<your-username>/alchemind.git
   cd alchemind
   pip install -e ".[dev]"
   ```
4. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-improvement
   ```
5. **Make your change**, keeping the codebase style consistent. Add or update tests for any
   new behaviour under `tests/`.
6. **Run the checks locally** — please make sure they pass before opening a PR:
   ```bash
   ruff check src tests      # lint
   pytest --cov=alchemind    # tests + coverage
   ```
7. **Commit** with a clear, descriptive message:
   ```bash
   git commit -m "Add <what you changed>"
   ```
8. **Push to your fork** and **open a Pull Request** against `main`:
   ```bash
   git push origin feature/my-improvement
   ```
   Then open a PR on GitHub describing *what* you changed and *why*.

### Contribution ideas

- New property predictors (toxicity, bioavailability, binding affinity)
- Alternative generators (VAE, diffusion, reinforcement learning)
- A larger, cited reference/training dataset and benchmark
- Retrosynthesis-aware synthesizability scoring
- UI/UX improvements and molecule visualization

### Guidelines

- Keep PRs focused — one logical change per PR is easier to review.
- Every new feature should come with tests; CI (GitHub Actions) must be green.
- Be respectful and constructive in issues and reviews. By contributing you agree your work
  is licensed under the project's MIT license.

Found a bug or have an idea? [Open an issue](https://github.com/AravindB98/alchemind/issues) —
and don't forget to ⭐ **star** and 🍴 **fork** the repo to support the project!

## License

MIT © 2026 Aravind Balaji

---

## 🧒 Explain Like I'm 5

Imagine LEGO for molecules: a computer that invents brand-new chemical compounds by snapping atomic pieces together, then double-checks its inventions are stable, safe, and actually makeable. Alchemind is that end-to-end pipeline — invent, predict properties, validate.

## 🧰 Tech Stack

Python · RDKit · generative molecular models · property prediction · validation pipeline

## 🌍 Real-Life Applications

- Early-stage drug discovery — generating candidate molecules cheaply
- Materials science — hunting for compounds with target properties
- Chemistry education — visualizing how structure drives properties
