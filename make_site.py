from pathlib import Path
import base64
import re
import cv2
import numpy as np
from PIL import Image
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

# Build a true layered eye preview from the supplied eye sprite. Each 180x74
# source tile is retained; the original brow is removed with inpainting and
# four paired brow styles are extracted from that same source sheet.
def make_makeup_layers():
    source = cv2.imread(str(assets / "eyes.webp"), cv2.IMREAD_COLOR)
    if source is None or source.shape[:2] != (296, 360):
        raise SystemExit(f"eyes.webp: expected 360x296, got {None if source is None else (source.shape[1], source.shape[0])}")
    clean = source.copy()
    styles = [(0, 0), (0, 1), (2, 0), (3, 1)]
    brow_sheet = np.zeros((74, 180 * len(styles), 4), dtype=np.uint8)

    def get_brow_mask(tile, exclude_blue=False):
        # Only inspect the upper brow zone. Keep actual brush strokes, not pink skin texture.
        roi = tile[:26]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, sat, val = cv2.split(hsv)
        candidate = (val < 214) & ((sat > 26) | (val < 150))
        if exclude_blue:
            blue_decoration = (h >= 95) & (h <= 135) & (sat > 40)
            candidate &= ~blue_decoration
        raw = candidate.astype(np.uint8) * 255
        # Bridge brush-stroke gaps, then keep only the dominant brow shape in each half.
        raw = cv2.morphologyEx(
            raw, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        )
        kept = np.zeros_like(raw)
        for x0, x1 in ((0, 90), (90, 180)):
            half = raw[:, x0:x1]
            count, labels, stats, _ = cv2.connectedComponentsWithStats(half, 8)
            candidates = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                if area >= 8 and width >= 5 and height >= 2:
                    candidates.append((area, label))
            if candidates:
                # Largest coherent component per half removes compression specks and stray marks.
                _, best = max(candidates)
                kept[:, x0:x1][labels == best] = 255
        # Expand only the brow silhouette to cover anti-aliased pixels; keep a soft alpha edge.
        kept = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        full = np.zeros((74, 180), dtype=np.uint8)
        full[:26] = kept
        return full

    masks = {}
    for row in range(4):
        for col in range(2):
            y, x = row * 74, col * 180
            tile = source[y:y + 74, x:x + 180].copy()

            # The red angular eye in row 3, right has no detached brow; don't erase its lash line.
            if (row, col) == (2, 1):
                clean[y:y + 74, x:x + 180] = tile
                continue

            erase_source = get_brow_mask(tile, exclude_blue=False)
            erase = np.zeros((74, 180), dtype=np.uint8)
            erase[:26] = cv2.dilate(
                erase_source[:26],
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5)),
                iterations=1
            )
            # Inpaint directly against the real neighboring skin; reflecting the top edge
            # can mirror the clipped brow back into the fill and create a visible ghost.
            cleaned = cv2.inpaint(tile, erase, 5, cv2.INPAINT_TELEA)
            clean[y:y + 74, x:x + 180] = cleaned

            masks[(row, col)] = get_brow_mask(tile, exclude_blue=True)

    # Only the four selected source cells provide independent brow options.
    for source_cell in styles:
        if source_cell not in masks:
            y, x = source_cell[0] * 74, source_cell[1] * 180
            masks[source_cell] = get_brow_mask(source[y:y + 74, x:x + 180].copy(), exclude_blue=True)

    for style_index, source_cell in enumerate(styles):
        row, col = source_cell
        y, x = row * 74, col * 180
        tile = source[y:y + 74, x:x + 180].copy()
        source_mask = masks[source_cell]
        alpha = cv2.GaussianBlur(source_mask, (3, 3), 0.45)
        rgba = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        brow_sheet[:, style_index * 180:(style_index + 1) * 180] = rgba

    clean_path = assets / "eyes-clean.webp"
    brow_path = assets / "brows.webp"
    Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)).save(clean_path, format="WEBP", quality=95, method=6)
    Image.fromarray(cv2.cvtColor(brow_sheet, cv2.COLOR_BGRA2RGBA)).save(brow_path, format="WEBP", lossless=True, method=6)
    with Image.open(clean_path) as im:
        if im.size != (360, 296):
            raise SystemExit(f"eyes-clean.webp: unexpected dimensions {im.size}")
    with Image.open(brow_path) as im:
        if im.size != (720, 74) or im.mode != "RGBA":
            raise SystemExit(f"brows.webp: expected RGBA 720x74, got {im.mode} {im.size}")
    print(f"PASS asset eyes-clean.webp: 360x296, {clean_path.stat().st_size} bytes")
    print(f"PASS asset brows.webp: RGBA 720x74, {brow_path.stat().st_size} bytes")

make_makeup_layers()

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
for required in ["assets/eyes-clean.webp", "assets/brows.webp", "assets/birds.webp", "assets/bottle.webp", "startBtn", "drop-overlay", "YOUR BIRD TYPE"]:
    if required not in html:
        raise SystemExit(f"Missing app feature or asset reference: {required}")
(dist / "index.html").write_text(html, encoding="utf-8")
print("PASS page: JavaScript syntax valid; real eye, six-bird and product images are wired into the page.")
