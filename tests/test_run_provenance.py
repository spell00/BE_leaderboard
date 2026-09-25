import hashlib
import json
from pathlib import Path

from src.run_provenance import capture_run_provenance


def test_capture_run_provenance_copies_sources_and_hashes_data(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "scripts" / "run.py").write_text("VALUE = 1\n")
    (repo / "src" / "model.py").write_text("WIDTH = 2048\n")
    (repo / "tests" / "test_model.py").write_text("def test_ok(): pass\n")
    (repo / ".env").write_text("SECRET=do-not-copy\n")
    bernn = tmp_path / "bernn"
    bernn.mkdir()
    (bernn / "trainer.py").write_text("LABEL_ORDER = 'identity'\n")
    data = tmp_path / "dataset.npz"
    data.write_bytes(b"exact-data-state")

    snapshot, manifest = capture_run_provenance(
        repo_root=repo,
        output_dir=tmp_path / "output",
        dataset_files=[data],
        argv=["python", "scripts/run.py"],
        bernn_root=bernn,
        capture_environment=False,
    )

    assert (snapshot / "repository/scripts/run.py").read_text() == "VALUE = 1\n"
    assert (snapshot / "repository/src/model.py").read_text() == "WIDTH = 2048\n"
    code_log = Path(manifest["training_code_log"])
    assert (code_log / "scripts/run.py").read_text() == "VALUE = 1\n"
    assert (code_log / "src/model.py").read_text() == "WIDTH = 2048\n"
    assert len(manifest["training_code_files"]) == 2
    assert (snapshot / "installed_packages/bernn/trainer.py").is_file()
    assert not (snapshot / "repository/.env").exists()
    assert manifest["dataset_files"][0]["sha256"] == hashlib.sha256(
        b"exact-data-state"
    ).hexdigest()
    saved = json.loads((snapshot / "manifest.json").read_text())
    assert saved["launch_id"] == manifest["launch_id"]
    assert json.loads((snapshot.parents[1] / "latest.json").read_text())[
        "launch_id"
    ] == manifest["launch_id"]
