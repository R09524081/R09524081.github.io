"""Regenerate publication thumbnails in figures/thumbs/.

Run after adding or replacing a figure in figures/:

    python scripts/make_thumbs.py

Each thumbnail is the figure scaled to fit 540x300 px (2x the 270x150
display box, so it stays sharp on high-DPI screens) on a white background.
Requires Pillow (pip install pillow).
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figures"
DST = SRC / "thumbs"
BOX = (540, 300)

DST.mkdir(exist_ok=True)
names = set()
for src in sorted(SRC.iterdir()):
    if src.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue
    names.add(src.name)
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail(BOX, Image.LANCZOS)
    out = DST / src.name
    if out.suffix.lower() == ".png":
        im.save(out, optimize=True)
    else:
        im.save(out, quality=90, optimize=True)
    print(f"{src.name}: {im.size[0]}x{im.size[1]}, {out.stat().st_size // 1024} KB")

# Remove thumbnails whose figure no longer exists
for old in DST.iterdir():
    if old.name not in names:
        old.unlink()
        print(f"removed stale {old.name}")
