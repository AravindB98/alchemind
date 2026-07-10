"""Pydantic request/response models for the Alchemind API."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DiscoverRequest(BaseModel):
    seed_smiles: str = Field(..., description="Seed molecule as SMILES.",
                             examples=["CC(=O)Oc1ccccc1C(=O)O"])
    objective: str = Field("qed", description="One of: solubility, qed, drug_likeness.")
    n: int = Field(12, ge=1, le=100, description="Number of candidates to return.")
    include_svg: bool = Field(True, description="Return 2D SVG depictions.")


class PredictRequest(BaseModel):
    smiles: str = Field(..., examples=["CC(=O)Oc1ccccc1C(=O)O"])


class CandidateModel(BaseModel):
    smiles: str
    valid: bool
    novel: bool
    qed: Optional[float] = None
    predicted_logS: Optional[float] = None
    synthesizability: Optional[float] = None
    similarity_to_known: float = 0.0
    passed_validation: bool = False
    properties: Dict[str, float] = {}
    objective_score: float = 0.0
    svg: Optional[str] = None


class DiscoverResponse(BaseModel):
    objective: str
    seeds: List[str]
    n_generated: int
    n_valid: int
    n_novel: int
    candidates: List[CandidateModel]


class PredictResponse(BaseModel):
    smiles: str
    valid: bool
    properties: Dict[str, float] = {}
    predicted_logS: Optional[float] = None
    qed: Optional[float] = None
    drug_likeness: Optional[dict] = None
    synthesizability: Optional[float] = None
    svg: Optional[str] = None
