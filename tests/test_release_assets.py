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
    assert re.findall(r"^## (.*)$", readme, flags=re.MULTILINE) == ["Core formulation", "Sampling preview"]
    for target in re.findall(r'src="([^"]+)"|\]\(([^)]+)\)', readme):
        path = target[0] or target[1]
        assert (ROOT / path).is_file(), path
