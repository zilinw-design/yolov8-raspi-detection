"""Project structure and import verification (Part B — 电赛测量)."""

def test_part_b_files():
    from pathlib import Path
    assert Path("src/phase3/screen_measure.py").exists(), "screen_measure.py not found"
    assert Path("src/phase3/pinhole_measure.py").exists(), "pinhole_measure.py not found"
    assert Path("src/phase3/test_detect.py").exists(), "test_detect.py not found"

def test_governance_files_present():
    from pathlib import Path
    for f in ["Project Brief.md", "Security Boundary.md", "Project Invariants.md",
              "Constraints And Priority.md", "File Registry.md"]:
        assert Path("ai/governance").joinpath(f).exists(), f"missing: {f}"
