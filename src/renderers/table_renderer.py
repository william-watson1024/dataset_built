from __future__ import annotations

import atexit
import base64
import hashlib
import html
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


Table = list[list[str]]


def normalize_table(table: Iterable[Iterable[Any]]) -> Table:
    """Normalize a table without changing row/column order or cell meaning."""
    if isinstance(table, (str, bytes)):
        raise TypeError("table must be a two-dimensional iterable, not a string")
    normalized: Table = []
    for row_index, row in enumerate(table):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"table row {row_index} must be an iterable of cells")
        normalized.append(["" if cell is None else str(cell).strip() for cell in row])
    if not normalized:
        raise ValueError("table must contain at least one row")
    width = max(len(row) for row in normalized)
    if width == 0:
        raise ValueError("table must contain at least one cell")
    return [row + [""] * (width - len(row)) for row in normalized]


def table_hash(table: Iterable[Iterable[Any]]) -> str:
    """Return a stable SHA-256 hash for the normalized table."""
    normalized = normalize_table(table)
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _FirefoxBiDi:
    """Small stdlib-only Firefox BiDi client used for persistent screenshots."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.profile = Path(tempfile.mkdtemp(prefix="firefox-profile-", dir=str(work_dir)))
        self.port = self._free_port()
        self.process = subprocess.Popen(
            [
                "firefox", "--headless", "--new-instance", "--no-remote",
                "--profile", str(self.profile),
                "--remote-debugging-port", str(self.port),
                "--remote-allow-hosts", "localhost", "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "MOZ_HEADLESS": "1"},
        )
        self.socket = self._connect()
        self.socket.settimeout(120)
        self._next_id = 1
        self._lock = threading.Lock()
        self._handshake()
        self.context = self._command("browsingContext.create", {"type": "tab"})["result"]["context"]

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                key = base64.b64encode(os.urandom(16)).decode("ascii")
                request = (
                    f"GET /session HTTP/1.1\r\nHost: localhost:{self.port}\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                    "Sec-WebSocket-Protocol: webDriverBidi\r\n\r\n"
                ).encode("ascii")
                sock.sendall(request)
                response = sock.recv(4096)
                if b"101 Switching Protocols" not in response:
                    sock.close()
                    raise RuntimeError("Firefox BiDi websocket handshake failed")
                return sock
            except (OSError, RuntimeError):
                time.sleep(0.2)
        raise RuntimeError("Firefox BiDi did not become available within 30 seconds")

    def _handshake(self) -> None:
        result = self._command("session.new", {"capabilities": {}})
        if result.get("type") != "success":
            raise RuntimeError(f"Firefox BiDi session.new failed: {result}")

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        mask = os.urandom(4)
        length = len(data)
        if length < 126:
            header = bytes((0x81, 0x80 | length))
        elif length < 65536:
            header = bytes((0x81, 0x80 | 126)) + length.to_bytes(2, "big")
        else:
            header = bytes((0x81, 0x80 | 127)) + length.to_bytes(8, "big")
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        self.socket.sendall(header + mask + masked)

    def _receive(self) -> dict[str, Any]:
        header = self._read_exact(2)
        first, second = header
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(self._read_exact(2), "big")
        elif length == 127:
            length = int.from_bytes(self._read_exact(8), "big")
        mask = self._read_exact(4) if second & 0x80 else None
        data = self._read_exact(length)
        if mask:
            data = bytes(value ^ mask[index % 4] for index, value in enumerate(data))
        if opcode == 0x9:
            self._send_raw(0xA, data)
            return self._receive()
        if opcode == 0x8:
            raise RuntimeError("Firefox BiDi websocket closed")
        return json.loads(data.decode("utf-8"))

    def _send_raw(self, opcode: int, data: bytes) -> None:
        mask = os.urandom(4)
        length = len(data)
        header = bytes((0x80 | opcode, 0x80 | length))
        self.socket.sendall(header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(data)))

    def _read_exact(self, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Firefox BiDi websocket closed unexpectedly")
            data += chunk
        return data

    def _command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            command_id = self._next_id
            self._next_id += 1
            self._send({"id": command_id, "method": method, "params": params})
            while True:
                response = self._receive()
                if response.get("id") == command_id:
                    if response.get("type") == "error":
                        raise RuntimeError(f"Firefox BiDi {method} failed: {response}")
                    return response

    def screenshot(self, html_uri: str, width: int, height: int) -> bytes:
        self._command(
            "browsingContext.setViewport",
            {"context": self.context, "viewport": {"width": width, "height": height}, "devicePixelRatio": 1},
        )
        self._command("browsingContext.navigate", {"context": self.context, "url": html_uri, "wait": "complete"})
        result = self._command("browsingContext.captureScreenshot", {"context": self.context, "origin": "document"})
        return base64.b64decode(result["result"]["data"])

    def close(self) -> None:
        try:
            self.socket.close()
        except Exception:
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


@dataclass(frozen=True)
class RenderResult:
    table_hash: str
    output_path: Path
    html_path: Path | None
    warnings: list[str]
    width: int
    height: int


class TableRenderer:
    """Render a normalized table to a deterministic HTML-backed PNG."""

    def __init__(
        self,
        browser: str = "firefox",
        max_width: int = 2400,
        max_height: int = 32000,
        cell_max_width: int = 520,
    ) -> None:
        self.browser = browser
        self.max_width = max_width
        self.max_height = max_height
        self.cell_max_width = cell_max_width
        self._bidi: _FirefoxBiDi | None = None
        self._bidi_lock = threading.Lock()
        atexit.register(self.close)

    def close(self) -> None:
        if self._bidi is not None:
            self._bidi.close()
            self._bidi = None

    def render(
        self,
        table: Iterable[Iterable[Any]],
        output_path: str | Path,
        html_path: str | Path | None = None,
    ) -> RenderResult:
        normalized = normalize_table(table)
        digest = table_hash(normalized)
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        warnings, width, height = self._estimate(normalized)
        document = self.to_html(normalized, digest)

        if output.is_file() and output.stat().st_size > 0:
            return RenderResult(digest, output, Path(html_path) if html_path else None, warnings, width, height)

        persisted_html = Path(html_path) if html_path else None
        if persisted_html:
            persisted_html.parent.mkdir(parents=True, exist_ok=True)
            persisted_html.write_text(document, encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="table-render-", dir=str(output.parent)) as temp_dir:
            source = Path(temp_dir) / "table.html"
            source.write_text(document, encoding="utf-8")
            if self.browser == "firefox_bidi":
                with self._bidi_lock:
                    if self._bidi is None:
                        self._bidi = _FirefoxBiDi(output.parent)
                    output.write_bytes(self._bidi.screenshot(source.as_uri(), width, height))
            else:
                profile = Path(temp_dir) / "profile"
                profile.mkdir()
                command = [
                    self.browser,
                    "--headless",
                    "--new-instance",
                    "--no-remote",
                    "--profile",
                    str(profile),
                    "--window-size",
                    f"{width},{height}",
                    "--screenshot",
                    str(output),
                    source.as_uri(),
                ]
                env = os.environ.copy()
                env.setdefault("MOZ_HEADLESS", "1")
                completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)
                if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                    details = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(f"browser rendering failed for {digest}: {details[-1000:]}")

        return RenderResult(digest, output, persisted_html, warnings, width, height)

    def to_html(self, table: Iterable[Iterable[Any]], digest: str | None = None) -> str:
        normalized = normalize_table(table)
        title = f"Table {digest}" if digest else "Table"
        rows: list[str] = []
        for row_index, row in enumerate(normalized):
            tag = "th" if row_index == 0 else "td"
            cells = "".join(f"<{tag}>{html.escape(cell, quote=True)}</{tag}>" for cell in row)
            rows.append(f"<tr>{cells}</tr>")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  body {{ display: inline-block; padding: 16px; color: #000; font-family: Arial, "Liberation Sans", sans-serif; font-size: 14px; line-height: 1.35; }}
  table {{ border-collapse: collapse; table-layout: auto; background: #fff; }}
  th, td {{ border: 1px solid #222; padding: 7px 9px; text-align: left; vertical-align: top; white-space: pre-wrap; overflow-wrap: anywhere; max-width: {self.cell_max_width}px; }}
  th {{ background: #eef1f4; font-weight: 700; }}
</style></head><body><table aria-label="{html.escape(title)}">{''.join(rows)}</table></body></html>"""

    def _estimate(self, table: Table) -> tuple[list[str], int, int]:
        warnings: list[str] = []
        columns = len(table[0])
        max_chars = max((len(cell) for row in table for cell in row), default=0)
        natural_width = sum(min(self.cell_max_width, max(72, max(len(row[col]) for row in table) * 8 + 28)) for col in range(columns))
        width = min(self.max_width, max(240, natural_width + 32))
        if natural_width + 32 > self.max_width:
            warnings.append(f"estimated width {natural_width + 32}px exceeds {self.max_width}px; cells will wrap")

        estimated_height = 40
        for row in table:
            row_height = 44
            for cell in row:
                chars_per_line = max(8, int((self.cell_max_width - 18) / 8))
                line_count = sum(max(1, (len(part) + chars_per_line - 1) // chars_per_line) for part in cell.split("\n"))
                row_height = max(row_height, 20 * line_count + 24)
            estimated_height += row_height
        height = min(self.max_height, max(140, estimated_height + 40))
        if estimated_height + 40 > self.max_height:
            warnings.append(f"estimated height {estimated_height + 40}px exceeds {self.max_height}px; table may need split")
        if max_chars > 500:
            warnings.append(f"long cell detected: {max_chars} characters")
        return warnings, width, height
