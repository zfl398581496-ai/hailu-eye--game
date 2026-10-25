from pathlib import Path
import base64
import re
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent
dist = root / "dist"
assets = dist / "assets"
assets.mkdir(parents=True, exist_ok=True)

def decode_chunks(prefix, count, output_name, expected_size=None):
    encoded = "".join((root / "assets-final" / f"{prefix}-{i:03d}.txt").read_text(encoding="utf-8").strip() for i in range(count))
    data = base64.b64decode(encoded, validate=True)
    if len(data) < 24 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise SystemExit(f"{output_name}: invalid WebP signature")
    declared = int.from_bytes(data[4:8], "little") + 8
    if declared != len(data):
        raise SystemExit(f"{output_name}: truncated WebP, header says {declared} bytes, got {len(data)}")
    # The source pictures are encoded as ordinary lossy VP8 WebP. Verify dimensions
    # from the VP8 frame header without relying on non-standard Python imaging packages.
    pos = 12
    width = height = None
    while pos + 8 <= len(data):
        tag = data[pos:pos+4]
        chunk_len = int.from_bytes(data[pos+4:pos+8], "little")
        start = pos + 8
        payload = data[start:start+chunk_len]
        if len(payload) != chunk_len:
            raise SystemExit(f"{output_name}: incomplete {tag!r} chunk")
        if tag == b"VP8 " and chunk_len >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            width = int.from_bytes(payload[6:8], "little") & 0x3fff
            height = int.from_bytes(payload[8:10], "little") & 0x3fff
            break
        pos = start + chunk_len + (chunk_len % 2)
    if expected_size and width is not None and (width, height) != expected_size:
        raise SystemExit(f"{output_name}: expected {expected_size}, got {(width, height)}")
    if expected_size and width is None:
        raise SystemExit(f"{output_name}: VP8 frame dimensions were not found")
    (assets / output_name).write_bytes(data)
    print(f"PASS asset {output_name}: {width}x{height}, {len(data)} bytes")
    return data

decode_chunks("eyes", 7, "eyes.webp", (360, 296))
decode_chunks("birds", 3, "birds.webp", (360, 216))
decode_chunks("bottle", 3, "bottle.webp", (135, 386))

html = (root / "index.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
if len(scripts) != 1:
    raise SystemExit(f"Expected one inline script, found {len(scripts)}")
with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as temp:
    temp.write(scripts[0])
    js_path = Path(temp.name)
try:
    subprocess.run(["node", "--check", str(js_path)], check=True, capture_output=True, text=True)
finally:
    js_path.unlink(missing_ok=True)
for required in ["assets/eyes.webp", "assets/birds.webp", "assets/bottle.webp", "startBtn", "drop-overlay", "YOUR BIRD TYPE"]:
    if required not in html:
        raise SystemExit(f"Missing app feature or asset reference: {required}")
(dist / "index.html").write_text(html, encoding="utf-8")
print("PASS page: JavaScript syntax valid; real eye, six-bird and product images are wired into the page.")
