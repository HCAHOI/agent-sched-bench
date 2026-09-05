import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/baselines/cachewise_official.sh"


def test_cachewise_official_adapter_contract() -> None:
    subprocess.run(["bash", "-n", SCRIPT], check=True)
    script_text = SCRIPT.read_text()
    help_text = subprocess.run(
        ["bash", SCRIPT, "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "181c435" in help_text
    assert "official released tool-duration predictor" in help_text
    assert "not the full" in help_text
    assert "modified vLLM scheduler" in help_text
    assert 'if test ! -x "$venv/bin/python"; then' in script_text
