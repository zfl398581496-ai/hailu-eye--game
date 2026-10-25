from pathlib import Path
import base64
import shutil

root = Path(__file__).resolve().parent
dist = root / "dist"
dist.mkdir(exist_ok=True)

# Reassemble the optimized 420 x 350 WebP from small text chunks.
encoded = "".join((root / f"eyeasset-{i}.txt").read_text(encoding="utf-8").strip() for i in (1, 2, 3))
image = base64.b64decode(encoded, validate=True)
if len(image) < 1000 or image[:4] != b"RIFF" or image[8:12] != b"WEBP":
    raise SystemExit("Eye reference image failed its WebP signature check.")

(dist / "eyeasset.txt").write_text(encoded, encoding="ascii")
shutil.copy2(root / "index.html", dist / "index.html")
print(f"Prepared site: index.html + {len(image)} byte WebP reference image.")
