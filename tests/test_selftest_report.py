"""Issue #11: the frozen exe is GUI-subsystem, so `--selftest 2> report.txt`
handed it a stderr that failed on the first write. The self-test now writes
report.txt itself and treats stderr as best-effort."""

from __future__ import annotations

import io

from dlss5_converter import selftest


class _BrokenStream(io.TextIOBase):
    """What a redirected stderr looks like to a windowed exe."""

    def write(self, _text):
        raise OSError(22, "Invalid argument")


def test_lines_survive_a_broken_stderr(monkeypatch):
    monkeypatch.setattr(selftest.sys, "stderr", _BrokenStream())
    selftest._REPORT.clear()
    selftest._line("hello")
    selftest._line("")
    assert selftest._REPORT == ["hello", ""]


def test_the_report_is_written_beside_the_app(tmp_path, monkeypatch):
    monkeypatch.setattr(selftest, "report_path", lambda: tmp_path / "report.txt")
    selftest._REPORT.clear()
    selftest._line("PASS")
    written = selftest._write_report()
    assert written == tmp_path / "report.txt"
    assert written.read_text(encoding="utf-8") == "PASS\n"


def test_run_selftest_always_writes_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(selftest, "report_path", lambda: tmp_path / "report.txt")

    def boom():
        selftest._line("partial")
        raise RuntimeError("a check blew up")

    monkeypatch.setattr(selftest, "_run_selftest", boom)
    try:
        selftest.run_selftest()
    except RuntimeError:
        pass
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert text.startswith("partial\n")
