from types import SimpleNamespace

from scripts import watch
from scripts.watch import render_bot_debug_video, write_heartbeat
from src.config import load_cameras_config, load_thresholds_config


def test_write_heartbeat_is_atomic_and_creates_parent(tmp_path):
    path = tmp_path / "live" / "watcher.heartbeat"
    write_heartbeat(path)
    assert path.read_text() == "ok\n"
    assert not path.with_suffix(".heartbeat.part").exists()


def test_bot_debug_render_is_cached_by_production_inputs(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    calls = []

    def fake_render(_source, _zone, *, out_path, motion_thresholds, **_kwargs):
        calls.append(motion_thresholds)
        output = type(source)(out_path)
        output.write_bytes(b"debug")
        return str(output)

    monkeypatch.setattr(watch, "reference_for", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(watch, "render_clip", fake_render)
    clip = SimpleNamespace(
        camera_id="cam01",
        message_id=42,
        timestamp="2026-09-19T18:30:00Z",
    )
    kwargs = {
        "cameras": load_cameras_config("config/cameras.yaml"),
        "thresholds": load_thresholds_config("config/thresholds.yaml"),
        "reference_entries": [],
        "output_root": tmp_path / "debug",
    }

    first = render_bot_debug_video(clip, source, **kwargs)
    second = render_bot_debug_video(clip, source, **kwargs)

    assert first == second
    assert first is not None and first.read_bytes() == b"debug"
    assert len(calls) == 1
    assert calls[0].compensate_warmup is False
