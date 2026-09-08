# TODO

Findings from the 2026-09-08 audit, grouped by what they needed. Everything is
done and in the working tree (uncommitted), including the depth finding that
came out of measuring the others.

## Fix (bugs) — all done

- [x] Effects tab fired `_effects_changed` during `MainWindow.__init__` whenever a
      saved effect was enabled (toggle connected before the saved state was
      restored). Logged an `AttributeError` and showed the "crashed" dialog on
      every launch with a LUT ticked. `app.py`, `tests/test_effects_page.py`.
- [x] Intensity slider offered 0..2 while the runtime clamps at 1.0.
      `NR_INTENSITY_MAX` in `settings.py`, `runtime.py`, `app.py`.
- [x] Skin slider was a no-op: the runtime only applies skin structure inside
      its character mask and the app never wrote `NRAutoMask=1`. `runtime.py`.
- [x] Still, sequence, batch and video paths left ~800 MB of contract planes in
      `engine\scratch` after every run. `_discard_planes` in `pipeline.py`,
      `tests/test_scratch_cleanup.py`.
- [x] `--selftest` aborted (0xC0000409) at its "window" step, and the GUI test
      files died the same way. Root cause: `MainWindow.__init__` schedules the
      first-run setup, which starts a depth-model download thread when the
      configured model is missing; destroying the window with that thread alive
      aborts the process. `MainWindow(startup=False)` builds a window with no
      deferred startup work; the self-test and the UI tests use it, and the
      self-test now closes the window (real teardown) before deleting it.
      `--selftest` on the installed build: PASS, including a real conversion.
- [x] Onboarding started inside the self-test's throwaway window on a fresh
      install and then reached into the deleted window. Same fix as above.
- [x] `crashlog._notify` called every unhandled exception a crash. Now titled
      "hit an error", says the app is still running, points at the log.
- [x] A `settings.json` from an older build could carry `intensity: 2.0` or a
      paper white of 0; `clamp_neural` in `AppSettings.load` migrates them.
      `tests/test_settings.py`.
- [x] Paper-white slider allowed 0. `NR_PAPER_WHITE_MIN` (0.1) on the slider and
      in `write_addon_config`.
- [x] `tests/test_main_window.py` queued every window's deletion with
      `deleteLater()` and never pumped events, so a dozen windows died at once
      inside the next test that called `qWait`. The fixture now closes, deletes
      and pumps. Full suite: 283 passed, 12 skipped.

## Rework (behaviour that measured wrong or misleading) — all done

- [x] Detail Boost ×4 halves the neural change (mean Δ vs source 8.0 → 4.4);
      tooltips and README say so. The presets in the installed app now use
      Preserve instead of Boost, Skin 1.0 instead of the 0.65 tuned while Skin
      was dead, and their README carries a dated correction block.
- [x] Style blurbs were backwards. Measured on three 1080p frames: Default 8–11,
      Natural 13–15, Cinematic 9 (mean |Δ|). Natural is the strong, darker look;
      Cinematic the gentlest. Tooltip and `settings.py` rewritten from the data.
- [x] `frames`: more passes do not make the pass stronger (mean Δ identical at
      1, 10, 20) but the result settles — 1 frame differs from 10 by ~1.5 levels,
      4 by ~0.8. Default 8 stands; the comment now says why.
- [x] Unwritten add-on keys measured: `NRGlobalTone`, `NRDepthMode`,
      `NRUICorrection`, `NRDiffuseWhiteNits` all byte-identical on a DLAA still.
      Documented in `runtime.py` and ROADMAP; none exposed.
- [x] Base/Large depth models: the download worker fetched PyTorch weights the
      ONNX engine cannot open, so a settings file asking for Base pulled 400 MB
      and then showed "Could not download the depth model". Now a missing
      Base/Large falls back to the bundled Small with a status-bar line saying
      how to export it; the models README says the same; onboarding no longer
      waits on it.

## Refactor / tooling — all done

- [x] `_effects_changed` saved `settings.json` on every slider tick; coalesced
      onto a 400 ms timer (closeEvent still saves unconditionally).
- [x] `_grain` seeded from the frame geometry so preview and saved file match.
- [x] `build_native.ps1` retries with NMake under `vcvars64.bat` when CMake has no
      generator for the installed Visual Studio (CMake 4.1 with Build Tools 18).
      Exercised: builds from a clean `native\build`.
- [x] `build_release.ps1` takes `-Release <folder>` and preserves `luts`,
      `presets`, `test`, `settings.json`, `crash.log`, `source` and any
      dot-folder, so it can upgrade an installed copy in place. It also writes
      `LICENSE.txt` and `TROUBLESHOOTING.txt`. Verified against a seeded folder.
- [x] `convert_sequence`'s per-frame body extracted into
      `_convert_sequence_frame` with a `_SequenceRun` context; an end-to-end test
      with a fake harness checks the DEPTH/reset/FRAME/WRITE protocol per frame.
- [x] ROADMAP: joint strength table marked superseded by the isolated one.
- [x] Nothing in the app or scripts touches `test\` or `output\` (the release
      script guard above is what enforces it).

## Found while measuring — resolved

- [x] **The depth plane does not influence the neural result on this path.**
      Re-tested with motion vectors present (which *do* change the output, so
      the temporal path is live): real and noise depth still byte-identical.
      There is no way to make it matter on this runtime. Stills now skip depth
      estimation by default (`DepthSettings.estimate_for_stills`, off), the
      batch path too; a Settings checkbox brings it back for the Depth view.
      Sequences and video unchanged. Tests in `tests/test_scratch_cleanup.py`.
      Original note:
      Six depth planes for one frame — Depth Anything's estimate, its inverse,
      flat near, flat far, flat mid, uniform noise — gave byte-identical output.
      Depth is bound (`pInDepth`, DepthInverted flag), so it reaches the
      runtime; with zero motion vectors and a static frame the DLSSNR 310.8
      snippet simply does not use it as a guide. Consequences worth acting on:
      a still-image conversion could skip depth estimation entirely (no model
      load, no inference, no 100–400 MB model), and the "load-bearing trick"
      paragraph in CLAUDE.md is wrong for this build. Not changed yet because
      the Depth view, onboarding and the sequence tab (renderer depth) all
      assume the depth engine exists; it needs a design decision, not a patch.
      Re-measure first on any newer `nvngx_dlssnr.dll`.

## Final audit (2026-09-08, after everything above)

- [x] `ruff --select F,E9,B006,B008` over the package, tests and scripts: six
      pre-existing unused imports and one unused variable removed. Clean.
- [x] All five PowerShell scripts parse.
- [x] README and TROUBLESHOOTING still described the PyTorch-era first launch
      (a 1.8 GB PyTorch download, a 400 MB model, a `pytorch\` folder, CUDA
      picking the GPU). Rewritten for the ONNX/DirectML build: nothing is
      downloaded except PyAV on first use of the Video tab.
- [x] Full suite 286 passed, 12 skipped. Installed build: `--selftest` PASS
      (window step, SDR and HDR conversions), GUI launch and close clean, no
      new exceptions logged, scratch folder empty afterwards.
