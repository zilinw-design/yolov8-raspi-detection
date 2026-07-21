"""Project structure and import verification."""

def test_part_b_files():
    from pathlib import Path
    assert Path("src/phase3/test_detect.py").exists()

def test_governance_files_present():
    from pathlib import Path
    for f in ["Project Brief.md", "Security Boundary.md",
              "Constraints And Priority.md", "File Registry.md"]:
        assert Path("ai/governance").joinpath(f).exists()
