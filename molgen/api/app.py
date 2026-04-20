"""
app.py

FastAPI backend for MolGen.

Endpoints:
    POST /generate          — generate molecules from property targets
    POST /predict_properties — predict properties for a SMILES string
    GET  /visualize/{smiles} — return SVG of 2D molecular structure
    GET  /health             — model status
"""

from contextlib import asynccontextmanager
from typing import Optional
import base64
import io
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from molgen.api.inference import MolGenInference
from molgen.api.visualizer import smiles_to_svg, smiles_to_image_b64


# ── Global model state ─────────────────────────────────────────────────────────

inference: Optional[MolGenInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    global inference
    encoder_path  = os.environ.get("ENCODER_CKPT",  "checkpoints/encoder_best.pt")
    diffusion_path= os.environ.get("DIFFUSION_CKPT", "checkpoints/diffusion_phase3_best.pt")

    print(f"Loading encoder from:   {encoder_path}")
    print(f"Loading diffusion from: {diffusion_path}")

    inference = MolGenInference(
        encoder_ckpt_path=encoder_path,
        diffusion_ckpt_path=diffusion_path,
    )
    print("Models loaded. API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="MolGen API",
    description="Conditional molecular generation with graph diffusion networks",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    target_logp:      float = Field(2.0,  ge=-5,  le=10,  description="Target LogP")
    target_qed:       float = Field(0.7,  ge=0.0, le=1.0, description="Target QED")
    target_sa:        float = Field(3.0,  ge=1.0, le=10,  description="Target SA Score")
    target_homo_lumo: float = Field(0.27, ge=0.0, le=1.0, description="Target HOMO-LUMO gap (eV)")
    n_molecules:      int   = Field(10,   ge=1,   le=50,  description="Number of molecules to generate")
    guidance_scale:   float = Field(2.0,  ge=0.0, le=10,  description="CFG guidance scale")


class MoleculeResult(BaseModel):
    smiles:        str
    image_b64:     Optional[str]   # base64 PNG
    svg:           Optional[str]   # SVG string
    predicted_logp:      Optional[float]
    predicted_qed:       Optional[float]
    predicted_sa:        Optional[float]
    predicted_homo_lumo: Optional[float]
    valid:         bool


class GenerateResponse(BaseModel):
    molecules:     list[MoleculeResult]
    n_requested:   int
    n_valid:       int
    generation_time_s: float


class PredictRequest(BaseModel):
    smiles: str


class PredictResponse(BaseModel):
    smiles:        str
    is_valid:      bool
    logp:          Optional[float]
    qed:           Optional[float]
    sa_score:      Optional[float]
    homo_lumo_gap: Optional[float]
    svg:           Optional[str]


class HealthResponse(BaseModel):
    status:          str
    encoder_loaded:  bool
    diffusion_loaded:bool
    device:          str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if models are loaded and API is ready."""
    if inference is None:
        return HealthResponse(
            status="loading",
            encoder_loaded=False,
            diffusion_loaded=False,
            device="unknown",
        )
    return HealthResponse(
        status="ready",
        encoder_loaded=inference.encoder is not None,
        diffusion_loaded=inference.denoiser is not None,
        device=str(inference.device),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate novel molecules matching the target property profile.

    Returns up to n_molecules valid SMILES with 2D structure images
    and predicted property scores.
    """
    if inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    import time
    t0 = time.time()

    target_properties = {
        "logp":      request.target_logp,
        "qed":       request.target_qed,
        "sa_score":  request.target_sa,
        "homo_lumo": request.target_homo_lumo,
    }

    results = inference.generate(
        target_properties=target_properties,
        n_molecules=request.n_molecules,
        guidance_scale=request.guidance_scale,
    )

    elapsed = time.time() - t0
    n_valid = sum(1 for r in results if r["valid"])

    molecules = []
    for r in results:
        molecules.append(MoleculeResult(
            smiles=r["smiles"],
            image_b64=smiles_to_image_b64(r["smiles"]) if r["valid"] else None,
            svg=smiles_to_svg(r["smiles"]) if r["valid"] else None,
            predicted_logp=r.get("predicted_logp"),
            predicted_qed=r.get("predicted_qed"),
            predicted_sa=r.get("predicted_sa"),
            predicted_homo_lumo=r.get("predicted_homo_lumo"),
            valid=r["valid"],
        ))

    return GenerateResponse(
        molecules=molecules,
        n_requested=request.n_molecules,
        n_valid=n_valid,
        generation_time_s=round(elapsed, 2),
    )


@app.post("/predict_properties", response_model=PredictResponse)
async def predict_properties(request: PredictRequest):
    """
    Predict molecular properties for a given SMILES string.
    Uses the trained GNN encoder for prediction.
    """
    if inference is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    from molgen.evaluation.metrics import is_valid

    if not is_valid(request.smiles):
        return PredictResponse(
            smiles=request.smiles,
            is_valid=False,
            logp=None, qed=None, sa_score=None, homo_lumo_gap=None,
            svg=None,
        )

    props = inference.predict_properties(request.smiles)

    return PredictResponse(
        smiles=request.smiles,
        is_valid=True,
        logp=props.get("logp"),
        qed=props.get("qed"),
        sa_score=props.get("sa_score"),
        homo_lumo_gap=props.get("homo_lumo"),
        svg=smiles_to_svg(request.smiles),
    )


@app.get("/visualize/{smiles}")
async def visualize(smiles: str):
    """
    Return SVG of 2D molecular structure for a given SMILES string.
    The SMILES must be URL-encoded if it contains special characters.
    """
    from fastapi.responses import Response
    from molgen.evaluation.metrics import is_valid
    import urllib.parse

    smiles = urllib.parse.unquote(smiles)

    if not is_valid(smiles):
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    svg = smiles_to_svg(smiles)
    if svg is None:
        raise HTTPException(status_code=500, detail="Could not generate SVG.")

    return Response(content=svg, media_type="image/svg+xml")