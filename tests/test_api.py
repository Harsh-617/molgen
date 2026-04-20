"""
Tests for the FastAPI backend.

Uses httpx AsyncClient to test endpoints without needing a running server.

Run with: pytest tests/test_api.py -v
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from molgen.api.app import app
import molgen.api.app as app_module
from molgen.api.inference import MolGenInference


@pytest_asyncio.fixture
async def client():
    """
    Async test client with models pre-loaded.
    Bypasses lifespan by setting the inference global directly —
    ASGITransport doesn't trigger FastAPI lifespan events.
    """
    # Load models before the test client starts
    app_module.inference = MolGenInference()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    # Clean up after tests
    app_module.inference = None


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health_returns_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    async def test_health_has_required_fields(self, client):
        data = (await client.get("/health")).json()
        for field in ["status", "encoder_loaded", "diffusion_loaded", "device"]:
            assert field in data, f"Missing field: {field}"

    async def test_health_status_is_ready(self, client):
        data = (await client.get("/health")).json()
        assert data["status"] == "ready"

    async def test_health_encoder_loaded(self, client):
        data = (await client.get("/health")).json()
        assert data["encoder_loaded"] is True


# ── /predict_properties ────────────────────────────────────────────────────────

class TestPredictProperties:
    async def test_valid_smiles_returns_200(self, client):
        r = await client.post("/predict_properties", json={"smiles": "CCO"})
        assert r.status_code == 200

    async def test_valid_smiles_has_all_fields(self, client):
        data = (await client.post(
            "/predict_properties", json={"smiles": "CCO"}
        )).json()
        for field in ["smiles", "is_valid", "logp", "qed",
                      "sa_score", "homo_lumo_gap", "svg"]:
            assert field in data, f"Missing field: {field}"

    async def test_valid_smiles_is_valid_true(self, client):
        data = (await client.post(
            "/predict_properties", json={"smiles": "CCO"}
        )).json()
        assert data["is_valid"] is True

    async def test_valid_smiles_properties_are_floats(self, client):
        data = (await client.post(
            "/predict_properties", json={"smiles": "c1ccccc1"}
        )).json()
        assert isinstance(data["logp"],          float)
        assert isinstance(data["qed"],           float)
        assert isinstance(data["sa_score"],      float)
        assert isinstance(data["homo_lumo_gap"], float)

    async def test_valid_smiles_returns_svg(self, client):
        data = (await client.post(
            "/predict_properties", json={"smiles": "CCO"}
        )).json()
        assert data["svg"] is not None
        assert data["svg"].startswith("<?xml")

    async def test_invalid_smiles_is_valid_false(self, client):
        data = (await client.post(
            "/predict_properties", json={"smiles": "not_a_molecule$$"}
        )).json()
        assert data["is_valid"] is False
        assert data["logp"] is None

    async def test_multiple_molecules(self, client):
        """Different molecules should give different property predictions."""
        ethanol = (await client.post(
            "/predict_properties", json={"smiles": "CCO"}
        )).json()
        benzene = (await client.post(
            "/predict_properties", json={"smiles": "c1ccccc1"}
        )).json()
        assert ethanol["logp"] != benzene["logp"]


# ── /visualize ─────────────────────────────────────────────────────────────────

class TestVisualize:
    async def test_valid_smiles_returns_svg(self, client):
        r = await client.get("/visualize/CCO")
        assert r.status_code == 200
        assert "svg" in r.headers["content-type"]

    async def test_svg_content_is_valid_xml(self, client):
        r = await client.get("/visualize/CCO")
        assert r.text.startswith("<?xml")
        assert "<svg" in r.text

    async def test_invalid_smiles_returns_400(self, client):
        r = await client.get("/visualize/not_valid_$$$$")
        assert r.status_code == 400

    async def test_benzene_returns_svg(self, client):
        import urllib.parse
        smiles = urllib.parse.quote("c1ccccc1")
        r = await client.get(f"/visualize/{smiles}")
        assert r.status_code == 200


# ── /generate (smoke test only — full generation needs diffusion checkpoint) ───

class TestGenerate:
    async def test_generate_returns_200(self, client):
        r = await client.post("/generate", json={
            "target_logp": 2.0,
            "target_qed":  0.7,
            "target_sa":   3.0,
            "target_homo_lumo": 0.27,
            "n_molecules": 2,
        })
        assert r.status_code == 200

    async def test_generate_has_required_fields(self, client):
        data = (await client.post("/generate", json={
            "target_logp": 2.0, "target_qed": 0.7,
            "target_sa": 3.0, "target_homo_lumo": 0.27,
            "n_molecules": 2,
        })).json()
        for field in ["molecules", "n_requested", "n_valid", "generation_time_s"]:
            assert field in data, f"Missing field: {field}"

    async def test_generate_n_requested_matches(self, client):
        data = (await client.post("/generate", json={
            "target_logp": 2.0, "target_qed": 0.7,
            "target_sa": 3.0, "target_homo_lumo": 0.27,
            "n_molecules": 3,
        })).json()
        assert data["n_requested"] == 3
        assert len(data["molecules"]) == 3

    async def test_generate_molecule_has_smiles(self, client):
        data = (await client.post("/generate", json={
            "target_logp": 2.0, "target_qed": 0.7,
            "target_sa": 3.0, "target_homo_lumo": 0.27,
            "n_molecules": 1,
        })).json()
        mol = data["molecules"][0]
        assert "smiles" in mol
        assert "valid"  in mol

    async def test_generate_rejects_too_many(self, client):
        """Requesting more than 50 molecules should fail validation."""
        r = await client.post("/generate", json={
            "target_logp": 2.0, "target_qed": 0.7,
            "target_sa": 3.0, "target_homo_lumo": 0.27,
            "n_molecules": 100,   # max is 50
        })
        assert r.status_code == 422   # Pydantic validation error