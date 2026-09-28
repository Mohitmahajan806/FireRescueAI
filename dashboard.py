"""Small local dashboard for reviewing FireRescueAI CSV detections."""

from __future__ import annotations

import argparse
import csv
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def render_dashboard(csv_path: Path, screenshot_dir: Path) -> str:
    rows = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    count = len(rows)
    people = len({row["track_id"] for row in rows if row.get("track_id")})
    recent = rows[-25:]
    table = "".join(f"<tr><td>{r['timestamp_utc']}</td><td>{r['frame']}</td><td>{r['track_id'] or '?'}</td><td>{r['confidence']}</td><td>({r['x1']}, {r['y1']}) - ({r['x2']}, {r['y2']})</td></tr>" for r in reversed(recent))
    images = "".join(f"<li>{path.name}</li>" for path in sorted(screenshot_dir.glob("*.jpg"), reverse=True)[:20])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>FireRescueAI review</title><style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px;background:#f3f5f4;color:#17221e}}header{{border-bottom:4px solid #f28c28;padding-bottom:1rem}}.metrics{{display:flex;gap:1rem;margin:1rem 0}}.metric{{background:white;border:1px solid #ccd6d1;padding:1rem;min-width:9rem}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border-bottom:1px solid #dde5e1;padding:.5rem;text-align:left;font-size:.9rem}}.warning{{color:#8b2d18;font-weight:700}}section{{margin-top:2rem}}</style></head><body><header><h1>FireRescueAI review dashboard</h1><p class='warning'>All detections are unverified. Human review is required. Research prototype only.</p></header><div class='metrics'><div class='metric'><strong>{count}</strong><br>detections</div><div class='metric'><strong>{people}</strong><br>track IDs</div><div class='metric'><strong>{len(list(screenshot_dir.glob('*.jpg')))}</strong><br>screenshots</div></div><section><h2>Recent detections</h2><table><tr><th>UTC time</th><th>Frame</th><th>Track</th><th>Confidence</th><th>Box</th></tr>{table or '<tr><td colspan="5">No CSV detections yet.</td></tr>'}</table></section><section><h2>Captured screenshots</h2><ul>{images or '<li>No screenshots yet.</li>'}</ul></section></body></html>"""


def serve(csv_path: Path, screenshot_dir: Path, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path != "/":
                self.send_error(404)
                return
            body = render_dashboard(csv_path, screenshot_dir).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = None
    for candidate_port in range(port, port + 10):
        try:
            server = ThreadingHTTPServer((host, candidate_port), Handler)
            break
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    if server is None:
        raise RuntimeError(f"No available port found between {port} and {port + 9}.")
    actual_port = server.server_address[1]
    print(f"Dashboard: http://{host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the FireRescueAI review dashboard")
    parser.add_argument("--csv", type=Path, default=Path("output_recordings/detections.csv"))
    parser.add_argument("--screenshots", type=Path, default=Path("screenshots"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    serve(arguments.csv, arguments.screenshots, arguments.host, arguments.port)
