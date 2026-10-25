from pathlib import Path
import base64
import re
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent
dist = root / "dist"
dist.mkdir(exist_ok=True)

# Reassemble and validate the reference eye sheet from small repository chunks.
encoded = "".join((root / f"eyeasset-{i}.txt").read_text(encoding="utf-8").strip() for i in (1, 2, 3))
image = base64.b64decode(encoded, validate=True)
if len(image) < 1000 or image[:4] != b"RIFF" or image[8:12] != b"WEBP":
    raise SystemExit("Eye reference image failed its WebP signature check.")
declared_size = int.from_bytes(image[4:8], "little") + 8
if declared_size != len(image):
    raise SystemExit(f"WebP is truncated: header declares {declared_size} bytes; found {len(image)}.")

html = (root / "index.html").read_text(encoding="utf-8")
old_loader = """fetch('eyeasset.txt').then(function(r){if(!r.ok)throw Error('eye asset unavailable');return r.text()}).then(function(s){eyeData=s.trim();eyeImage=new Image();eyeImage.onload=function(){imageReady=true;document.documentElement.style.setProperty('--sheet','url("data:image/webp;base64,'+eyeData+'")');render()};eyeImage.onerror=function(){toast('眼睛素材加载失败，请刷新网页')};eyeImage.src='data:image/webp;base64,'+eyeData}).catch(function(){toast('眼睛素材暂时未加载，请刷新网页')});"""
new_loader = """eyeData='__EYE_IMAGE_BASE64__';eyeImage=new Image();eyeImage.onload=function(){imageReady=true;document.documentElement.style.setProperty('--sheet','url("data:image/webp;base64,'+eyeData+'")');render()};eyeImage.onerror=function(){toast('眼睛素材加载失败，请刷新网页')};eyeImage.src='data:image/webp;base64,'+eyeData;"""
if html.count(old_loader) != 1:
    raise SystemExit(f"Expected exactly one external image loader; found {html.count(old_loader)}.")
html = html.replace(old_loader, new_loader).replace("__EYE_IMAGE_BASE64__", encoded)
if "__EYE_IMAGE_BASE64__" in html or "fetch('eyeasset.txt')" in html:
    raise SystemExit("The final page still contains a non-self-contained image loader.")

scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
if len(scripts) != 1:
    raise SystemExit(f"Expected exactly one inline game script; found {len(scripts)}.")
with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as temp:
    temp.write(scripts[0])
    js_path = Path(temp.name)
try:
    subprocess.run(["node", "--check", str(js_path)], check=True, capture_output=True, text=True)
finally:
    js_path.unlink(missing_ok=True)

# Decode the image with Pillow where available, in addition to checking the container.
try:
    from PIL import Image
    from io import BytesIO
    with Image.open(BytesIO(image)) as im:
        im.verify()
    with Image.open(BytesIO(image)) as im:
        dimensions = im.size
except ImportError:
    dimensions = "decoder check unavailable; validated WebP header and file length"

(dist / "index.html").write_text(html, encoding="utf-8")
print(f"PASS: self-contained page built; JavaScript syntax valid; WebP size={len(image)} bytes; dimensions={dimensions}.")
