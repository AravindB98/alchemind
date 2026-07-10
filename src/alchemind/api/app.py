"""FastAPI application exposing the Alchemind discovery pipeline as a REST API,
and serving the single-page web UI."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .. import __version__, config
from ..pipeline import OBJECTIVES, DiscoveryPipeline
from ..prediction import PropertyPredictor
from ..utils.chem import is_valid_smiles, to_svg
from ..validation import MoleculeValidator, lipinski_report, synthesizability_score
from .schemas import (
    DiscoverRequest,
    DiscoverResponse,
    PredictRequest,
    PredictResponse,
)

# Instantiate heavy objects once at startup (model + reference set loaded here).
_predictor = PropertyPredictor()
_validator = MoleculeValidator()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Alchemind API",
        version=__version__,
        description="Invent, predict, and validate novel chemical compounds.",
    )
    pipeline = DiscoveryPipeline(predictor=_predictor, validator=_validator)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "solubility_model_trained": _predictor.solubility.is_trained,
            "objectives": list(OBJECTIVES),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not is_valid_smiles(req.smiles):
            raise HTTPException(status_code=422, detail="Invalid SMILES.")
        pred = _predictor.predict(req.smiles)
        return PredictResponse(
            smiles=pred.smiles,
            valid=pred.valid,
            properties=pred.properties,
            predicted_logS=pred.predicted_logS,
            qed=pred.qed,
            drug_likeness=lipinski_report(req.smiles),
            synthesizability=synthesizability_score(req.smiles),
            svg=to_svg(req.smiles),
        )

    @app.post("/discover", response_model=DiscoverResponse)
    def discover(req: DiscoverRequest) -> DiscoverResponse:
        if not is_valid_smiles(req.seed_smiles):
            raise HTTPException(status_code=422, detail="Invalid seed SMILES.")
        if req.objective not in OBJECTIVES:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown objective. Choose from {list(OBJECTIVES)}.",
            )
        result = pipeline.run(
            req.seed_smiles,
            objective=req.objective,
            n=req.n,
            include_svg=req.include_svg,
        )
        return DiscoverResponse(**result.to_dict(include_svg=req.include_svg))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_file = config.WEB_DIR / "index.html"
        if index_file.exists():
            return index_file.read_text()
        return "<h1>Alchemind</h1><p>UI not found. See /docs for the API.</p>"

    return app


app = create_app()
