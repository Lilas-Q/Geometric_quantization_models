import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_preview_provenance():
    metadata = json.loads((ROOT / "assets/lhfm-i-cifar10-150k.json").read_text())
    image = ROOT / "assets/lhfm-i-cifar10-150k.png"
    assert hashlib.sha256(image.read_bytes()).hexdigest() == metadata["image_sha256"]
    assert metadata["checkpoint_step"] == 150000
    assert metadata["sampling"]["transport_multiplier"] == 1.0
    readme = (ROOT / "README.md").read_text()
    assert "## Sampling preview" in readme
    assert "```math" in readme
    for target in re.findall(r'src="([^"]+)"|\]\(([^)]+)\)', readme):
        path = target[0] or target[1]
        assert (ROOT / path).is_file(), path


def test_video_assets_and_result_cutoff():
    metadata = json.loads((ROOT / "assets/lhfm-v-moving-mnist-1250k.json").read_text())
    for name, digest in metadata["files"].items():
        assert hashlib.sha256((ROOT / "assets" / name).read_bytes()).hexdigest() == digest
    result = json.loads((ROOT / "results/lhfm_v_moving_mnist_1250k.json").read_text())
    history = json.loads((ROOT / "results/lhfm_v_validation_history.json").read_text())
    assert metadata["checkpoint_step"] == result["checkpoint_step"] == 1250000
    assert metadata["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert result["test"]["count"] == 10000
    assert result["test"]["test_used_for_selection"] is False
    assert all(row["step"] <= 1250000 for row in history["rows"])
    assert max(row["step"] for row in history["rows"]) == 1250000
    for key in ("mse", "mae"):
        raw = result["test"]["overall"][key + "_raw"]
        spatial = result["test"]["overall"][key + "_spatial_sum_raw"]
        assert abs(raw * 4096 - spatial) < 1e-10
