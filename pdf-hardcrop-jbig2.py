#!/usr/bin/env python3
# NOTE (2026-11-04): superseded by pdf-hardcrop.py, which keeps each image's
# encoding, keeps the page content stream, and handles multiple/shared images.
# This old script binarizes (dithers) and rebuilds pages; kept for now only.
"""Hard-crop 1bpp scan pages: cut hidden pixels out of each image using the
page geometry from a krop/Ghostscript-cropped copy, re-encode as lossless
JBIG2 (generic region, no symbol mode), and size each page to its new image.
Usage: pdf-hardcrop-jbig2.py SOURCE.pdf CROPPED.pdf OUTPUT.pdf"""
import sys, re, subprocess, tempfile, os, io
import pikepdf
from PIL import Image

src_path, crop_path, out_path = sys.argv[1:4]
src = pikepdf.open(src_path); crp = pikepdf.open(crop_path)
assert len(src.pages) == len(crp.pages)
tmp = tempfile.mkdtemp()
subprocess.run(["pdfimages", "-png", src_path, f"{tmp}/p"], check=True)
out = pikepdf.new()
NUM = r"(-?[\d.]+)"
for i, (sp, cp) in enumerate(zip(src.pages, crp.pages)):
    W, H = [float(v) for v in list(sp.obj.MediaBox)[2:4]]
    img = Image.open(f"{tmp}/p-{i:03d}.png").convert("1")
    iw, ih = img.size
    sx, sy = iw / W, ih / H                       # pixels per point
    # cropped page geometry
    cb = [float(v) for v in (cp.obj.get("/CropBox") or cp.obj.MediaBox)]
    c = cp.obj.Contents
    c = b"".join(x.read_bytes() for x in c) if isinstance(c, pikepdf.Array) else c.read_bytes()
    mats = [tuple(map(float, m)) for m in re.findall((NUM + r"\s+") * 6 + r"cm", c.decode("latin1"))]
    a, e, f = 1.0, 0.0, 0.0                        # compose scale/translate cms
    d = 1.0
    for (ma, mb, mc, md, me, mf) in mats:
        e, f = a * me + e, d * mf + f
        a, d = a * ma, d * md
    # image unit square maps to x: e..e+a, y: f..f+d (points, page space)
    pt_x0, pt_y0 = (cb[0] - e) / a * W, (cb[1] - f) / d * H
    pt_x1, pt_y1 = (cb[2] - e) / a * W, (cb[3] - f) / d * H
    L = max(0, round(pt_x0 * sx)); R = min(iw, round(pt_x1 * sx))
    T = max(0, round((H - pt_y1) * sy)); B = min(ih, round((H - pt_y0) * sy))
    crop = img.crop((L, T, R, B)); cw, ch = crop.size
    pp = f"{tmp}/c-{i:03d}.png"; crop.save(pp)
    jb = subprocess.run(["jbig2", "-p", pp], check=True, capture_output=True).stdout
    pw, ph = cw / sx, ch / sy
    im = pikepdf.Stream(out, jb); im.Type = pikepdf.Name.XObject
    im.Subtype = pikepdf.Name.Image; im.Width = cw; im.Height = ch
    im.ColorSpace = pikepdf.Name.DeviceGray; im.BitsPerComponent = 1
    im.Filter = pikepdf.Name.JBIG2Decode
    page = pikepdf.Dictionary(Type=pikepdf.Name.Page,
        MediaBox=[0, 0, round(pw, 3), round(ph, 3)],
        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=im)),
        Contents=pikepdf.Stream(out, f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()))
    out.pages.append(pikepdf.Page(page))
    print(f"page {i+1}: {iw}x{ih} -> {cw}x{ch} px (L{L} T{T}), {pw:.2f}x{ph:.2f} pt, {len(jb)//1024} KiB")
out.docinfo["/Producer"] = "pdf-hardcrop-jbig2.py"
out.save(out_path)
