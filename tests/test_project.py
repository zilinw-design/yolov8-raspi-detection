"""Project structure and import verification."""

def test_project_root_exists():
    """Verify we are in the correct project directory."""
    from pathlib import Path
    assert Path("src/task1_basic/detect_pi.py").exists(), "detect_pi.py not found"
    assert Path("src/task1_basic/detect_camera.py").exists(), "detect_camera.py not found"

def test_governance_files_present():
    """Verify required governance files exist."""
    from pathlib import Path
    for f in ["Project Brief.md", "Security Boundary.md", "Project Invariants.md",
              "Constraints And Priority.md", "File Registry.md"]:
        assert Path("ai/governance").joinpath(f).exists(), f"missing: {f}"
