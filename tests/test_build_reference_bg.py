import json

from scripts.build_reference_bg import _merged_manifest_entries
from src.reference_bg import ReferenceEntry


def _entry(camera_id: str, path: str) -> ReferenceEntry:
    return ReferenceEntry(
        camera_id=camera_id,
        daylight=False,
        era=None,
        start="2024-01-01T00:00:00Z",
        end="2024-03-31T00:00:00Z",
        clip_count=10,
        path=path,
    )


def test_single_camera_rebuild_preserves_other_manifest_entries(tmp_path):
    existing = [_entry("cam05", "cam05/old.png"), _entry("cam12", "cam12/old.png")]
    (tmp_path / "manifest.json").write_text(
        json.dumps([entry.__dict__ for entry in existing])
    )

    merged = _merged_manifest_entries(
        tmp_path,
        [_entry("cam12", "cam12/new.png")],
        "cam12",
    )

    assert [(entry.camera_id, entry.path) for entry in merged] == [
        ("cam05", "cam05/old.png"),
        ("cam12", "cam12/new.png"),
    ]


def test_full_rebuild_replaces_the_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps([_entry("cam05", "cam05/old.png").__dict__])
    )
    built = [_entry("cam12", "cam12/new.png")]

    assert _merged_manifest_entries(tmp_path, built, None) == built
