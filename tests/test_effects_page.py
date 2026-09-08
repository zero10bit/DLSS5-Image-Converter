"""The Effects tab restores saved toggles without firing its change callback.

Every launch with an effect left enabled (a LUT, say) used to call
MainWindow._effects_changed from inside MainWindow.__init__, before the
effects page existed, log an AttributeError and show a crash notice on the
next start.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dlss5_converter.app import EffectsPage  # noqa: E402
from dlss5_converter.settings import EffectsSettings  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_restoring_enabled_effects_does_not_fire_on_change(qt_app):
    calls: list[str] = []
    settings = EffectsSettings(lut_enabled=True, lut_name="look.cube", sharpen_enabled=True)
    page = EffectsPage(settings, lambda: calls.append("changed"))
    assert calls == []
    assert page._groups["lut_enabled"].isChecked()
    assert page._groups["sharpen_enabled"].isChecked()
    assert not page._groups["bloom_enabled"].isChecked()


def test_toggling_after_construction_still_reports(qt_app):
    calls: list[str] = []
    settings = EffectsSettings()
    page = EffectsPage(settings, lambda: calls.append("changed"))
    page._groups["vignette_enabled"].setChecked(True)
    assert settings.vignette_enabled is True
    assert calls == ["changed"]
