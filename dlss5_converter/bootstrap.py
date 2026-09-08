"""Fetch optional runtime pieces on first use instead of shipping them.

Only PyAV lives here now: its wheels bundle a GPL FFmpeg, so having the user's
machine fetch it from PyPI on first use of the Video tab keeps this project
clear of redistributing that binary. Depth runs on ONNX Runtime with the Small
model bundled, so the PyTorch download this module was written for is gone.

Nothing here is a package manager. The wheel is a zip; it is downloaded,
extracted, and put on ``sys.path``. No pip, no network resolution, no version
solving — a pinned URL and an unzip, so the failure modes are "no network" and
"disk full" rather than anything dependency-shaped.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import ssl
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

#: What actually goes wrong on first run is almost never a dead connection — it
#: is something sitting between the app and the download. Said plainly so a
#: stuck user has a checklist instead of a spinning bar. Ordered by how often
#: each is the culprit in the wild (a VPN, then filtered DNS, then AV/proxy).
NETWORK_HELP = (
    "This is almost always something between the app and the download rather "
    "than a real outage:\n"
    "  • Turn off any VPN — this is the most common cause.\n"
    "  • Switch your DNS to 1.1.1.1 (Cloudflare) or 8.8.8.8 (Google).\n"
    "  • Disable a proxy, or pause antivirus HTTPS/SSL scanning.\n"
    "Then start it again — the download resumes where it stopped, so nothing is "
    "lost."
)

#: Network exceptions we translate into NETWORK_HELP rather than a raw trace.
_NETWORK_ERRORS = (URLError, TimeoutError, socket.timeout, ConnectionError, ssl.SSLError)

from . import paths

#: Progress is reported to the UI over queued cross-thread signals, and each one
#: makes the GUI thread take the GIL to run its slot. Fire them per chunk (many
#: hundreds a second at download and unpack speeds) and the GUI thread spends its
#: time blocked on that handoff instead of pumping Windows messages — the window
#: goes "Not Responding" even though the work is progressing fine on the worker.
#: Coalescing to ~20 updates a second keeps the bar smooth and the window alive.
_PROGRESS_INTERVAL = 0.05


class _Throttle:
    """Rate-limit progress callbacks to at most one per interval.

    ``force=True`` always fires — used for the final update so the bar lands
    exactly on the total rather than a beat short.
    """

    def __init__(self, on_bytes: "BytesProgress | None", interval: float = _PROGRESS_INTERVAL) -> None:
        self._on_bytes = on_bytes
        self._interval = interval
        self._last = 0.0

    def __call__(self, done: int, total: int, force: bool = False) -> None:
        if self._on_bytes is None:
            return
        now = time.monotonic()
        if force or now - self._last >= self._interval:
            self._on_bytes(done, total)
            self._last = now

#: PyAV carries a full FFmpeg (H.264/H.265, NVENC, AAC muxing) that OpenCV's
#: prebuilt build does not. Downloaded on first use of the Video tab rather than
#: shipped: its wheels bundle a GPL FFmpeg, so having the user's machine fetch
#: it from PyPI keeps this project clear of redistributing that binary.
AV_VERSION = "18.1.0"
AV_APPROX_BYTES = 35_000_000

BytesProgress = Callable[[int, int], None]
TextProgress = Callable[[str], None]


def _remote_size(url: str) -> int:
    """The download's size, or 0 if the server won't say.

    Never raises: a HEAD blocked by a VPN or filtered DNS must not be the thing
    that fails first with an opaque error. Returning 0 lets the real download in
    _download attempt the transfer and surface the friendly NETWORK_HELP message
    (or resume) rather than dying on a size probe.
    """
    request = Request(url, headers={"User-Agent": "dlss5-converter"}, method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length") or 0)
    except (*_NETWORK_ERRORS, ValueError):
        return 0


def _download(url: str, destination: Path, on_bytes: BytesProgress | None) -> None:
    """Fetch `url` to `destination`, resuming a partial file if one is there.

    Resume matters more than it looks: this is nearly two gigabytes, the people
    running it are on domestic connections, and without a Range request every
    transient failure restarts from zero.
    """
    expected = _remote_size(url)
    have = destination.stat().st_size if destination.exists() else 0

    if expected and have == expected:
        if on_bytes:
            on_bytes(expected, expected)
        return
    if have > expected > 0:
        # Longer than the file it claims to be; start over rather than guess.
        destination.unlink()
        have = 0

    headers = {"User-Agent": "dlss5-converter"}
    if have:
        headers["Range"] = f"bytes={have}-"

    request = Request(url, headers=headers)
    # A 30 s socket timeout so a connection that goes silent mid-transfer — the
    # classic VPN/DNS stall that used to look like a frozen app — raises within
    # half a minute instead of hanging until the user gives up and kills it.
    try:
        with urlopen(request, timeout=30) as response:
            resuming = response.status == 206
            if not resuming:
                have = 0
            total = expected or (
                int(response.headers.get("Content-Length") or AV_APPROX_BYTES) + have
            )
            done = have
            report = _Throttle(on_bytes)
            with open(destination, "ab" if resuming else "wb") as handle:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    report(done, total)
            report(done, total, force=True)
    except _NETWORK_ERRORS as error:
        raise OSError(
            f"The download stalled after {have // 1_048_576} MB.\n\n{NETWORK_HELP}"
        ) from error

    if done < total * 0.99:
        raise OSError(
            f"The download stopped early ({done // 1_048_576} of "
            f"{total // 1_048_576} MB).\n\n{NETWORK_HELP}"
        )


#: Files at or above this are streamed in chunks rather than extracted in one
#: opaque call, so the bar keeps moving *through* them. It is the CUDA DLLs this
#: matters for — cuDNN and cuBLAS in a modern torch wheel are hundreds of MB to
#: over a gigabyte each, and extracting one with no feedback (plus the antivirus
#: scan Windows runs as it is written) is exactly what looks like a frozen app.
_STREAM_ABOVE = 16 * 1024 * 1024
_CHUNK = 4 * 1024 * 1024


def _safe_parts(name: str) -> list[str]:
    """The path components of a zip entry, with traversal segments dropped.

    zip entries always use forward slashes; ``''``, ``.`` and ``..`` are removed
    so a crafted name can never escape the target, matching what ZipFile.extract
    does for the members it handles.
    """
    return [part for part in name.split("/") if part not in ("", ".", "..")]


def _os_path(path: Path) -> str:
    r"""A path string safe to open on Windows however long it is.

    A modern CUDA torch tree nests well past the classic 260-character MAX_PATH
    - ``torch/include/...`` headers and long cuDNN/cuBLAS names - and on a machine
    that has not enabled long-path support the extract fails part way through,
    which is the "crashes/freezes at the end of the download" people hit. The
    ``\\?\`` extended-length prefix opts a single path out of MAX_PATH with no
    system setting and no admin, so the app fixes this itself rather than telling
    users to edit the registry. Non-Windows is returned untouched.
    """
    if os.name != "nt":
        return str(path)
    full = os.path.abspath(str(path))
    if full.startswith("\\\\?\\"):
        return full
    if full.startswith("\\\\"):
        # UNC (\\server\share) becomes \\?\UNC\server\share.
        return "\\\\?\\UNC\\" + full[2:]
    return "\\\\?\\" + full


def _extract(
    archive: Path,
    target: Path,
    on_bytes: BytesProgress | None,
    on_text: TextProgress | None = None,
) -> None:
    report = _Throttle(on_bytes)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        total = sum(member.file_size for member in members) or 1
        done = 0
        for member in members:
            parts = _safe_parts(member.filename)
            if member.is_dir():
                os.makedirs(_os_path(target.joinpath(*parts)), exist_ok=True)
                continue
            if not parts:
                continue

            # Every file is written through _os_path, not just the big ones and
            # not via ZipFile.extract, so the long-path prefix covers the whole
            # tree - the deep header paths that overflow MAX_PATH are ordinary
            # small files, and those were the ones ZipFile.extract used to fail
            # on. Big files are still named up front so a slow, antivirus-scanned
            # gigabyte reads as progress rather than a hang.
            if member.file_size >= _STREAM_ABOVE and on_text:
                size_mb = member.file_size / 1_048_576
                on_text(f"Unpacking {Path(member.filename).name} ({size_mb:.0f} MB)…")
            dest = target.joinpath(*parts)
            os.makedirs(_os_path(dest.parent), exist_ok=True)
            with bundle.open(member) as source, open(_os_path(dest), "wb") as sink:
                while True:
                    chunk = source.read(_CHUNK)
                    if not chunk:
                        break
                    sink.write(chunk)
                    done += len(chunk)
                    report(done, total)
        report(done, total, force=True)  # land the bar exactly on the total


# -- PyAV, for the Video tab -------------------------------------------------


def av_runtime_dir() -> Path:
    """Where the downloaded PyAV lives, beside the app's other data."""
    return paths.data_dir() / "pyav"


def activate_av() -> None:
    """Put a downloaded PyAV on the import path, DLLs included.

    PyAV's compiled modules link against FFmpeg libraries the wheel bundles in
    an ``av.libs`` folder on Windows; that folder has to be a legal DLL search
    path before ``import av`` will load.
    """
    target = av_runtime_dir()
    if not target.is_dir():
        return
    entry = str(target)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    if hasattr(os, "add_dll_directory"):
        for libs in (target / "av.libs", target / "av"):
            if libs.is_dir():
                try:
                    os.add_dll_directory(str(libs))
                except OSError:
                    pass


def av_is_ready() -> bool:
    activate_av()
    try:
        return importlib.util.find_spec("av") is not None
    except (ImportError, ValueError):
        return False


def _resolve_av_wheel() -> tuple[str, int]:
    """Ask PyPI for the cp312 win_amd64 wheel URL and size for our version.

    Resolved at download time rather than pinned, because the hashed file path
    on files.pythonhosted.org is not something to hard-code and keep correct.
    """
    import json

    api = f"https://pypi.org/pypi/av/{AV_VERSION}/json"
    request = Request(api, headers={"User-Agent": "dlss5-converter"})
    with urlopen(request, timeout=30) as response:
        data = json.load(response)

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    windows = [
        entry for entry in data.get("urls", [])
        if entry.get("filename", "").endswith("win_amd64.whl")
        # Free-threaded builds (a trailing "t" on the interpreter tag) are a
        # different ABI and would fail to load in the ordinary interpreter.
        and "t-win_amd64" not in entry["filename"]
    ]
    # Prefer the stable-ABI (abi3) wheel: PyAV ships one built for the oldest
    # supported Python that loads on every newer one, so it is the safe pick
    # regardless of which 3.x the app is frozen against. Fall back to an exact
    # version-tagged wheel if a release ever drops the abi3 one.
    for match in (lambda n: "abi3" in n, lambda n: tag in n):
        for entry in windows:
            if match(entry["filename"]):
                return entry["url"], int(entry.get("size") or AV_APPROX_BYTES)
    raise OSError(f"No compatible PyAV {AV_VERSION} wheel for this Python on PyPI.")


def install_av(
    on_bytes: BytesProgress | None = None,
    on_text: TextProgress | None = None,
) -> None:
    """Download and unpack PyAV. Raises on failure. Same staging as install()."""
    target = av_runtime_dir()
    staging = target.with_name(target.name + ".partial")
    archive = target.with_name(target.name + ".whl.part")
    if staging.is_dir():
        shutil.rmtree(_os_path(staging), ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if on_text:
            on_text("Finding the video component...")
        url, _size = _resolve_av_wheel()
        if on_text:
            on_text(f"Downloading video support (PyAV {AV_VERSION})...")
        _download(url, archive, on_bytes)
        if on_text:
            on_text("Unpacking…")
        _extract(archive, staging, on_bytes, on_text)
        if not (staging / "av" / "__init__.py").is_file():
            raise OSError("The downloaded video component is missing its av package.")
        if target.is_dir():
            shutil.rmtree(_os_path(target), ignore_errors=True)
        staging.rename(target)
    finally:
        shutil.rmtree(_os_path(staging), ignore_errors=True)
    archive.unlink(missing_ok=True)
    activate_av()
