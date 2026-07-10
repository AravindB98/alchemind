from fastapi.testclient import TestClient

from alchemind.api import app

client = TestClient(app)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "qed" in body["objectives"]


def test_predict_endpoint():
    r = client.post("/predict", json={"smiles": ASPIRIN})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["qed"] is not None
    assert body["drug_likeness"]["passes"] is True
    assert body["svg"].startswith("<") or "<svg" in body["svg"]


def test_predict_invalid_smiles():
    r = client.post("/predict", json={"smiles": "nope[[["})
    assert r.status_code == 422


def test_discover_endpoint():
    r = client.post("/discover", json={
        "seed_smiles": ASPIRIN, "objective": "qed", "n": 5, "include_svg": False})
    assert r.status_code == 200
    body = r.json()
    assert body["objective"] == "qed"
    assert body["n_generated"] >= 0
    for c in body["candidates"]:
        assert c["valid"] and c["novel"]


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Alchemind" in r.text
