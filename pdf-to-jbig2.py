#!/usr/bin/env python3
"""Convert a scanned (color/gray) image PDF to 1-bit lossless JBIG2.
One full-page image per page; image shared across pages is encoded once.
Global Otsu threshold per image (override with -t N). Page sizes kept.
Usage: pdf-to-jbig2.py [-t THRESH] SOURCE.pdf OUTPUT.pdf"""
import sys, subprocess, tempfile, io
import numpy as np, pikepdf, pymupdf
from PIL import Image
args = sys.argv[1:]; fixed = None
if args[0] == "-t": fixed = int(args[1]); args = args[2:]
src_path, out_path = args
doc = pymupdf.open(src_path); out = pikepdf.new(); tmp = tempfile.mkdtemp()
cache = {}
def otsu(g):
    h = np.bincount(g.ravel(), minlength=256).astype(float); p = h / h.sum()
    w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mt * w - m) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))
for i, page in enumerate(doc):
    imgs = page.get_images(full=True); assert len(imgs) == 1, (i, imgs)
    x = imgs[0][0]
    if x not in cache:
        g = np.asarray(Image.open(io.BytesIO(doc.extract_image(x)["image"])).convert("L"))
        t = fixed if fixed is not None else otsu(g)
        bw = Image.fromarray(((g > t) * 255).astype(np.uint8)).convert("1")
        pp = f"{tmp}/{x}.png"; bw.save(pp)
        jb = subprocess.run(["jbig2", "-p", pp], check=True, capture_output=True).stdout
        im = pikepdf.Stream(out, jb); im.Type = pikepdf.Name.XObject
        im.Subtype = pikepdf.Name.Image; im.Width, im.Height = bw.size
        im.ColorSpace = pikepdf.Name.DeviceGray; im.BitsPerComponent = 1
        im.Filter = pikepdf.Name.JBIG2Decode
        cache[x] = im; print(f"page {i+1}: xref {x} {bw.size} thresh {t} {len(jb)//1024} KiB", flush=True)
    else: print(f"page {i+1}: reuse xref {x}", flush=True)
    r = page.rect; pw, ph = r.width, r.height
    pg = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, pw, ph],
        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=cache[x])),
        Contents=pikepdf.Stream(out, f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()))
    out.pages.append(pikepdf.Page(pg))
out.docinfo["/Producer"] = "pdf-to-jbig2.py"; out.save(out_path)
