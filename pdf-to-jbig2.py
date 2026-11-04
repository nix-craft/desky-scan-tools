#!/usr/bin/env python3
"""Convert a scanned (color/gray) image PDF to 1-bit lossless JBIG2.
One full-page image per page; image shared across pages is encoded once.
Global Otsu threshold per image (override with -t N). Page sizes kept.
--despeckle N removes 8-connected black specks of <= N pixels.
Usage: pdf-to-jbig2.py [-t THRESH] [--despeckle N] SOURCE.pdf OUTPUT.pdf"""
import sys, subprocess, tempfile, io
import numpy as np, pikepdf, pymupdf
from PIL import Image
def despeckle(black, n):
    """black: bool array (True=ink). Remove 8-connected components of area <= n."""
    if n <= 0: return black
    from scipy import ndimage
    lab, k = ndimage.label(black, structure=np.ones((3, 3), bool))
    if k == 0: return black
    sizes = np.bincount(lab.ravel()); keep = sizes > n; keep[0] = False
    return keep[lab]
def otsu(g):
    h = np.bincount(g.ravel(), minlength=256).astype(float); p = h / h.sum()
    w = np.cumsum(p); m = np.cumsum(p * np.arange(256)); mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mt * w - m) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))
def main(argv):
  args = argv[:]; fixed = None; ds = 0
  while args and args[0] in ("-t", "--despeckle"):
    if args[0] == "-t": fixed = int(args[1])
    else: ds = int(args[1])
    args = args[2:]
  src_path, out_path = args
  doc = pymupdf.open(src_path); out = pikepdf.new(); tmp = tempfile.mkdtemp()
  cache = {}
  for i, page in enumerate(doc):
      imgs = page.get_images(full=True); assert len(imgs) == 1, (i, imgs)
      x = imgs[0][0]
      if x not in cache:
          g = np.asarray(Image.open(io.BytesIO(doc.extract_image(x)["image"])).convert("L"))
          t = fixed if fixed is not None else otsu(g)
          ink = despeckle(g <= t, ds)
          bw = Image.fromarray(((~ink) * 255).astype(np.uint8)).convert("1")
          pp = f"{tmp}/{x}.png"; bw.save(pp)
          jb = subprocess.run(["jbig2", "-p", pp], check=True, capture_output=True).stdout
          im = pikepdf.Stream(out, jb); im.Type = pikepdf.Name.XObject
          im.Subtype = pikepdf.Name.Image; im.Width, im.Height = bw.size
          im.ColorSpace = pikepdf.Name.DeviceGray; im.BitsPerComponent = 1
          im.Filter = pikepdf.Name.JBIG2Decode
          cache[x] = im; print(f"page {i+1}: xref {x} {bw.size} thresh {t} ds {ds} {len(jb)//1024} KiB", flush=True)
      else: print(f"page {i+1}: reuse xref {x}", flush=True)
      r = page.rect; pw, ph = r.width, r.height
      b = page.get_image_rects(x)[0]               # keep original placement
      cm = f"{b.width:.3f} 0 0 {b.height:.3f} {b.x0:.3f} {ph - b.y1:.3f} cm"
      pg = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, pw, ph],
          Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=cache[x])),
          Contents=pikepdf.Stream(out, f"q {cm} /Im0 Do Q".encode()))
      out.pages.append(pikepdf.Page(pg))
  out.docinfo["/Producer"] = "pdf-to-jbig2.py"; out.save(out_path)

if __name__ == "__main__": main(sys.argv[1:])
