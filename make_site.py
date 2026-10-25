from pathlib import Path
import base64
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from PIL import Image

root = Path(__file__).resolve().parent
dist = root / "dist"
assets = dist / "assets"
assets.mkdir(parents=True, exist_ok=True)

def decode_chunks(prefix, count, output_name, expected_size=None):
    encoded = "".join((root / "assets-final" / f"{prefix}-{i:03d}.txt").read_text(encoding="utf-8").strip() for i in range(count))
    data = base64.b64decode(encoded, validate=True)
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise SystemExit(f"{output_name}: missing WebP signature")
    declared = int.from_bytes(data[4:8], "little") + 8
    if declared != len(data):
        raise SystemExit(f"{output_name}: truncated WebP, header says {declared} bytes, got {len(data)}")
    with Image.open(BytesIO(data)) as im:
        im.verify()
    with Image.open(BytesIO(data)) as im:
        dims = im.size
    if expected_size and dims != expected_size:
        raise SystemExit(f"{output_name}: expected {expected_size}, got {dims}")
    (assets / output_name).write_bytes(data)
    print(f"PASS asset {output_name}: {dims}, {len(data)} bytes")
    return data

# The sprite grids were cropped directly from the user's supplied pictures.
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
# Verify important user-provided assets are wired into the actual page.
for required in ["assets/eyes.webp", "assets/birds.webp", "assets/bottle.webp", "startBtn", "drop-overlay", "YOUR BIRD TYPE"]:
    if required not in html:
        raise SystemExit(f"Missing app feature or asset reference: {required}")
(dist / "index.html").write_text(html, encoding="utf-8")
print("PASS page: JavaScript syntax valid; references to actual eye, six-bird, and product images present.")
