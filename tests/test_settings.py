from __future__ import annotations

import json

from dlss5_converter.settings import AppSettings


def test_round_trip(tmp_path):
    settings = AppSettings()
    settings.neural.skin = 0.9
    settings.evaluation.frames = 12
    path = tmp_path / "settings.json"
    settings.save(path)

    loaded = AppSettings.load(path)
    assert loaded.neural.skin == 0.9
    assert loaded.evaluation.frames == 12


def test_unknown_keys_do_not_break_loading(tmp_path):
    """A file written by a newer build must not stop an older one starting."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"neural": {"intensity": 0.2, "unheard_of": 7}, "future_section": {}}),
        encoding="utf-8",
    )
    loaded = AppSettings.load(path)
    assert loaded.neural.intensity == 0.2
    assert loaded.neural.skin == AppSettings().neural.skin


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert AppSettings.load(path).neural.intensity == AppSettings().neural.intensity


def test_missing_file_falls_back_to_defaults(tmp_path):
    assert AppSettings.load(tmp_path / "nope.json").evaluation.frames == 1


def test_out_of_range_neural_values_are_clamped_on_load(tmp_path):
    """An older build's slider ran to 2.0; a stored 2.0 must come back as 1.0,
    and a paper white of 0 must come back at the floor, so the file, the chip
    readout and the add-on ini agree from the first frame."""
    from dlss5_converter.settings import NR_INTENSITY_MAX, NR_PAPER_WHITE_MIN

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"neural": {"intensity": 2.0, "paper_white": 0.0, "skin": 9}}))
    loaded = AppSettings.load(path)
    assert loaded.neural.intensity == NR_INTENSITY_MAX
    assert loaded.neural.paper_white == NR_PAPER_WHITE_MIN
    assert loaded.neural.skin == 2.0
