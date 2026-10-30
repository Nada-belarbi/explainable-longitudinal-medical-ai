import subprocess
import sys
from pathlib import Path


def test_unsafe_rekey_entrypoint_retired():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable,
        str(root / 'src/nacc_progression/workstream2/rekey_member1_private.py'),
    ],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Retired NACCID-only' in result.stderr
