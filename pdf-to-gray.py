#!/usr/bin/env python3
"""Losslessly convert neutral RGB image XObjects in a PDF to 8-bit DeviceGray.

An image is converted only if it is /DeviceRGB, 8 bpc, has no /SMask, /Mask,
/Decode or /ImageMask, uses a single FlateDecode or DCTDecode filter, and its
decoded pixels are neutral:
  Flate: R==G==B for every pixel (exact) -> one channel, zlib level 9, also
         tries PNG predictor 15 with Sub or Up rows and keeps
         the smallest.
  DCT:   max |R-G|,|G-B| <= --dct-tol (default 2; JPEG chroma rounding) ->
         `jpegtran -grayscale -copy none` (drops chroma coefficients, luma
         coefficients untouched, no requantisation).  Max diff vs the decoded
         original's G channel is reported.
Streams are rewritten in place, each object once, so shared images stay shared.
Content streams, page boxes and matrices are not touched.
--bits 4 (LOSSY, separate mode): neutral RGB or 8-bit DeviceGray images (Flate or
DCT) are quantized to the nearest of 16 levels (multiples of 17) and written as
4-bit DeviceGray Flate (zlib 9, PNG predictor tried, smallest kept); max/mean
error vs the decoded source is reported per image.
Usage: pdf-to-gray.py [--dct-tol N] [--bits 8|4] SOURCE.pdf OUTPUT.pdf
"""
import argparse, io, os, subprocess, sys, zlib
import numpy as np, pikepdf
from PIL import Image
N = pikepdf.Name

def png_predict(g, ftype):
    h, w = g.shape; a = g.astype(np.int16)
    if ftype == 1:   d = a.copy(); d[:, 1:] = a[:, 1:] - a[:, :-1]
    elif ftype == 2: d = a.copy(); d[1:] = a[1:] - a[:-1]
    else:            d = a
    d = (d & 0xFF).astype(np.uint8)
    return np.hstack([np.full((h, 1), ftype, np.uint8), d]).tobytes()

def best_flate(g):
    h, w = g.shape
    cands = [(zlib.compress(g.tobytes(), 9), None)]
    for f in (1, 2):
        cands.append((zlib.compress(png_predict(g, f), 9),
                      pikepdf.Dictionary(Predictor=15, Colors=1, BitsPerComponent=8, Columns=w)))
    return min(cands, key=lambda c: len(c[0]))

def pack4(q):
    h, w = q.shape
    if w % 2: q = np.hstack([q, np.zeros((h, 1), np.uint8)])
    return (q[:, 0::2] << 4) | q[:, 1::2]

def why_skip4(o):
    for k in ("/SMask", "/Mask", "/Decode"):
        if k in o: return f"has {k}"
    if o.get("/ImageMask", False): return "image mask"
    if o.get("/ColorSpace") not in (N.DeviceRGB, N.DeviceGray): return f"colorspace {o.get('/ColorSpace')!r}"[:60]
    if int(o.get("/BitsPerComponent", 0)) != 8: return "bpc != 8"
    f = filters(o)
    if f not in (["/FlateDecode"], ["/DCTDecode"]): return f"filters {f}"
    return None

def convert4(o, dct_tol):
    old = len(o.read_raw_bytes())
    r = why_skip4(o)
    if r: return "skip", "-", old, old, r
    w, h = int(o.Width), int(o.Height); nc = 3 if o.ColorSpace == N.DeviceRGB else 1
    if filters(o) == ["/FlateDecode"]:
        kind = "flate"; a = np.frombuffer(o.read_bytes(), np.uint8)
        if a.size != w * h * nc: return "skip", kind, old, old, "unexpected length"
        a = a.reshape(h, w, nc)
    else:
        kind = "dct"; im = Image.open(io.BytesIO(o.read_raw_bytes()))
        if im.mode not in ("RGB", "L"): return "skip", kind, old, old, f"jpeg mode {im.mode}"
        a = np.asarray(im).reshape(h, w, -1); nc = a.shape[2]
    if nc == 3:
        b = a.astype(np.int16)
        cd = int(max(np.abs(b[..., 0] - b[..., 1]).max(), np.abs(b[..., 1] - b[..., 2]).max()))
        tol = 0 if kind == "flate" else dct_tol
        if cd > tol: return "skip", kind, old, old, f"real color (chroma diff {cd})"
        g = a[..., 1]
    else: g = a[..., 0]
    q = ((g.astype(np.uint16) + 8) // 17).astype(np.uint8)          # nearest of 0..15
    err = np.abs(q.astype(np.int16) * 17 - g)
    pk = np.ascontiguousarray(pack4(q))
    cands = [(zlib.compress(pk.tobytes(), 9), None)]
    for f in (1, 2):
        cands.append((zlib.compress(png_predict(pk, f), 9),
                      pikepdf.Dictionary(Predictor=15, Colors=1, BitsPerComponent=4, Columns=w)))
    data, parms = min(cands, key=lambda c: len(c[0]))
    if "/DecodeParms" in o: del o["/DecodeParms"]
    o.write(data, filter=N.FlateDecode, decode_parms=parms)
    o.ColorSpace = N.DeviceGray; o.BitsPerComponent = 4
    return "ok", kind, old, len(data), f"4-bit, max err {int(err.max())}, mean err {err.mean():.2f}, levels used {len(np.unique(q))}"

def filters(o):
    f = o.get("/Filter")
    if f is None: return []
    return [str(x) for x in f] if isinstance(f, pikepdf.Array) else [str(f)]

def why_skip(o):
    for k in ("/SMask", "/Mask", "/Decode"):
        if k in o: return f"has {k}"
    if o.get("/ImageMask", False): return "image mask"
    if o.get("/ColorSpace") != N.DeviceRGB: return f"colorspace {o.get('/ColorSpace')!r}"[:60]
    if int(o.get("/BitsPerComponent", 0)) != 8: return "bpc != 8"
    f = filters(o)
    if f not in (["/FlateDecode"], ["/DCTDecode"]): return f"filters {f}"
    return None

def convert(o, dct_tol):
    """Returns (status, kind, old_len, new_len, info)."""
    old = len(o.read_raw_bytes())
    r = why_skip(o)
    if r: return "skip", "-", old, old, r
    w, h = int(o.Width), int(o.Height)
    if filters(o) == ["/FlateDecode"]:
        a = np.frombuffer(o.read_bytes(), np.uint8)
        if a.size != w * h * 3: return "skip", "flate", old, old, "unexpected length"
        a = a.reshape(h, w, 3)
        if not ((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()):
            return "skip", "flate", old, old, "real color"
        data, parms = best_flate(np.ascontiguousarray(a[..., 0]))
        if "/DecodeParms" in o: del o["/DecodeParms"]
        o.write(data, filter=N.FlateDecode, decode_parms=parms)
        o.ColorSpace = N.DeviceGray
        return "ok", "flate", old, len(data), "exact" + (" pred" if parms is not None else "")
    raw = o.read_raw_bytes()
    im = Image.open(io.BytesIO(raw))
    if im.mode != "RGB": return "skip", "dct", old, old, f"jpeg mode {im.mode}"
    a = np.asarray(im).astype(np.int16)
    cd = int(max(np.abs(a[..., 0] - a[..., 1]).max(), np.abs(a[..., 1] - a[..., 2]).max()))
    if cd > dct_tol: return "skip", "dct", old, old, f"real color (chroma diff {cd})"
    gj = subprocess.run(["jpegtran", "-grayscale", "-copy", "none"], input=raw,
                        check=True, capture_output=True).stdout
    g = np.asarray(Image.open(io.BytesIO(gj))).astype(np.int16)
    if g.shape != (h, w): return "skip", "dct", old, old, "jpegtran shape mismatch"
    md = int(np.abs(g - a[..., 1]).max())
    if "/DecodeParms" in o: del o["/DecodeParms"]
    o.write(gj, filter=N.DCTDecode)
    o.ColorSpace = N.DeviceGray
    return "ok", "dct", old, len(gj), f"chroma diff {cd}, max diff vs G {md}"

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--dct-tol", type=int, default=2)
    p.add_argument("--bits", type=int, choices=(8, 4), default=8, help="8 = lossless (default); 4 = lossy 16-level")
    a = p.parse_args(argv)
    if os.path.abspath(a.src) == os.path.abspath(a.out): p.error("refusing to overwrite input")
    if os.path.exists(a.out): p.error(f"{a.out} exists; refusing to overwrite")
    pdf = pikepdf.open(a.src)
    tot = {}; skipped = 0
    for o in pdf.objects:
        if not isinstance(o, pikepdf.Stream) or o.get("/Subtype") != N.Image: continue
        st, kind, old, new, info = (convert4 if a.bits == 4 else convert)(o, a.dct_tol)
        print(f"obj {o.objgen[0]}: {int(o.Width)}x{int(o.Height)} {kind} {st} {old} -> {new} ({info})", flush=True)
        if st == "skip": skipped += 1; continue
        t = tot.setdefault(kind, [0, 0, 0]); t[0] += 1; t[1] += old; t[2] += new
    for k, (n, o_, n_) in tot.items():
        print(f"total {k}: {n} images {o_} -> {n_} bytes (saved {o_ - n_}, {100*(o_-n_)/max(o_,1):.1f}%)")
    print(f"skipped: {skipped}")
    pdf.save(a.out, deterministic_id=True)

if __name__ == "__main__": main()
