from alchemind.validation import (
    MoleculeValidator,
    lipinski_report,
    synthesizability_score,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_lipinski_passes_for_drug():
    rep = lipinski_report(ASPIRIN)
    assert rep is not None and rep["passes"] is True
    assert rep["violations"] == 0


def test_synthesizability_scale():
    sa = synthesizability_score(ASPIRIN)
    assert sa is not None and 1.0 <= sa <= 10.0
    # A large fused polycyclic should score as harder than aspirin.
    hard = synthesizability_score("C1CC2CC3CC1CC(C2)(C3)C4CC5CC(C4)CC5")
    assert hard is None or hard >= sa


def test_known_compound_is_not_novel():
    v = MoleculeValidator(reference_smiles=[ASPIRIN])
    report = v.validate(ASPIRIN)
    assert report.valid
    assert report.novel is False
    assert report.max_similarity_to_known == 1.0


def test_distinct_molecule_is_novel():
    v = MoleculeValidator(reference_smiles=[ASPIRIN])
    report = v.validate("CCOC(=O)c1ccc(N)cc1OCC")
    assert report.valid
    assert report.novel is True


def test_invalid_smiles_reported():
    v = MoleculeValidator(reference_smiles=[ASPIRIN])
    report = v.validate("not_valid[[[")
    assert report.valid is False
    assert "invalid SMILES" in report.reasons
