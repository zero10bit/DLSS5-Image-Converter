"""The harness's scratch planes are removed once a conversion has its result.

A contract at the 7680-pixel Boost edge is ~800 MB across colour, depth,
motion and output, and nothing reads them after read_output. They used to be
left in the scratch folder for the life of the install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlss5_converter import evaluator, paths, pipeline, runtime
from dlss5_converter.settings import AppSettings


def _settings() -> AppSettings:
    settings = AppSettings()
    settings.detail.mode = "preserve"  # no Boost: keeps the fake planes tiny
    settings.evaluation.frames = 1
    return settings


def _prepared() -> pipeline.Prepared:
    rng = np.random.default_rng(0)
    source = rng.random((8, 8, 3), dtype=np.float32)
    depth = rng.random((8, 8), dtype=np.float32)
    return pipeline.Prepared(source=source, inverse_depth=depth)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    folder = tmp_path / "scratch"
    folder.mkdir()  # the real scratch_dir() creates it on demand
    monkeypatch.setattr(paths, "scratch_dir", lambda: folder)
    status = runtime.RuntimeStatus(harness=tmp_path / "dlss5_eval.exe")
    monkeypatch.setattr(runtime, "detect", lambda *_: status)
    monkeypatch.setattr(runtime, "stage_runtime", lambda *_: tmp_path)
    monkeypatch.setattr(runtime, "write_addon_config", lambda *_: tmp_path / "ReShade.ini")
    return folder


def test_planes_are_removed_after_a_conversion(scratch, monkeypatch):
    seen: list[Path] = []

    def fake_run_frames(_exe, *, width, height, out_path, **kwargs):
        # Everything the real run leaves behind: the harness's output plus the
        # planes convert() wrote for it.
        seen.extend(p for p in scratch.iterdir())
        np.full((height, width, 4), 0.5, np.float16).tofile(out_path)

    monkeypatch.setattr(evaluator, "run_frames", fake_run_frames)
    pipeline.convert("unused.png", _settings(), None, prepared=_prepared())

    assert {p.name for p in seen} == {"colour.bin", "depth.bin", "motion.bin"}
    assert list(scratch.iterdir()) == []


def test_planes_are_removed_when_the_harness_fails(scratch, monkeypatch):
    def failing_run_frames(*_args, **_kwargs):
        raise evaluator.HarnessError("ERROR the runtime said no")

    monkeypatch.setattr(evaluator, "run_frames", failing_run_frames)
    with pytest.raises(evaluator.HarnessError):
        pipeline.convert("unused.png", _settings(), None, prepared=_prepared())

    assert list(scratch.iterdir()) == []


def test_discard_tolerates_missing_and_locked_planes(tmp_path):
    """Best-effort by design: a leftover must never fail a finished conversion."""
    present = tmp_path / "a.bin"
    present.write_bytes(b"x")
    missing = tmp_path / "never-written.bin"
    with open(present, "rb"):  # Windows refuses to unlink a file that is open
        pipeline._discard_planes(present, missing)
    pipeline._discard_planes(present, missing)
    assert not present.exists()


class _FakeHarness:
    """Stands in for evaluator.Harness: records the protocol, writes a result."""

    calls: list[str] = []

    def __init__(self, _exe, *, width, height, depth_path, motion_path, neural, frames, use_shmem=False):
        self.width, self.height = width, height
        self.depth_path, self.motion_path = depth_path, motion_path
        _FakeHarness.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _FakeHarness.calls.append("exit")

    def colour_buffer(self, shape):
        return np.empty(shape, np.float16)

    def set_depth(self, path):
        assert Path(path).stat().st_size == self.width * self.height * 4
        _FakeHarness.calls.append("depth")

    def reset_history(self):
        _FakeHarness.calls.append("reset")

    def commit_colour(self, plane, path, offset):
        _FakeHarness.calls.append("frame")

    def write(self, out_path):
        np.full((self.height, self.width, 4), 0.25, np.float16).tofile(out_path)
        _FakeHarness.calls.append("write")


class _FakeEngine:
    def load(self, *_a, **_k):
        return "cpu"

    def infer(self, image, **_k):
        return np.full(image.shape[:2], 0.5, np.float32)


def test_sequence_runs_every_frame_on_one_harness_and_cleans_up(scratch, monkeypatch, tmp_path):
    import cv2

    monkeypatch.setattr(evaluator, "Harness", _FakeHarness)
    frames = []
    for i in range(3):
        path = tmp_path / f"f_{i:03d}.png"
        cv2.imwrite(str(path), np.full((64, 96, 3), 90 + i, np.uint8))
        frames.append(path)
    settings = _settings()
    settings.evaluation.frames = 2

    out = tmp_path / "out"
    done = list(pipeline.convert_sequence(frames, settings, _FakeEngine(), out))

    assert [f.index for f in done] == [0, 1, 2]
    assert all(f.output.is_file() for f in done)
    assert done[0].image.shape == (64, 96, 3)
    # Per frame: new depth, history reset, one FRAME per jitter offset, one WRITE.
    per_frame = ["depth", "reset", "frame", "frame", "write"]
    assert _FakeHarness.calls == per_frame * 3 + ["exit"]
    assert list(scratch.iterdir()) == []


class _RefusingEngine:
    """A depth engine that must never be touched."""

    def load(self, *_a, **_k):
        raise AssertionError("depth model loaded for a still with estimation off")

    def infer(self, *_a, **_k):
        raise AssertionError("depth inferred for a still with estimation off")


def test_stills_skip_depth_estimation_when_asked(tmp_path):
    import cv2

    path = tmp_path / "still.png"
    cv2.imwrite(str(path), np.full((64, 96, 3), 120, np.uint8))
    settings = AppSettings()
    settings.depth.estimate_for_stills = False
    prepared = pipeline.prepare(path, settings, _RefusingEngine())
    assert prepared.inverse_depth.shape == (64, 96)
    assert float(prepared.inverse_depth.min()) == float(prepared.inverse_depth.max()) == 0.5


def test_stills_estimate_depth_by_default(tmp_path):
    # On by default since the point-cloud reveal is built from the plane.
    import cv2

    path = tmp_path / "still.png"
    cv2.imwrite(str(path), np.full((64, 96, 3), 120, np.uint8))
    settings = AppSettings()
    assert settings.depth.estimate_for_stills
    prepared = pipeline.prepare(path, settings, _FakeEngine())
    assert float(prepared.inverse_depth.mean()) == 0.5  # _FakeEngine's constant
    with pytest.raises(AssertionError, match="loaded"):
        pipeline.prepare(path, settings, _RefusingEngine())


def test_batch_skips_depth_when_asked(scratch, monkeypatch, tmp_path):
    import cv2

    monkeypatch.setattr(evaluator, "Harness", _FakeHarness)
    images = []
    for i in range(2):
        path = tmp_path / f"b_{i}.png"
        cv2.imwrite(str(path), np.full((64, 96, 3), 100 + i, np.uint8))
        images.append(path)
    settings = _settings()
    settings.depth.estimate_for_stills = False
    done = list(pipeline.convert_batch(images, settings, _RefusingEngine(), tmp_path / "out"))
    assert [d.error for d in done] == ["", ""]
    assert list(scratch.iterdir()) == []
