from scripts.watch import write_heartbeat


def test_write_heartbeat_is_atomic_and_creates_parent(tmp_path):
    path = tmp_path / "live" / "watcher.heartbeat"
    write_heartbeat(path)
    assert path.read_text() == "ok\n"
    assert not path.with_suffix(".heartbeat.part").exists()
