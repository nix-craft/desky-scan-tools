#!/usr/bin/env python3
"""Losslessly convert neutral RGB image XObjects in a PDF to 8-bit DeviceGray.

An image is converted only if it is /DeviceRGB, 8 bpc, has no /Mask,
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
Images that carry /SMask are converted; the soft-mask stream itself is left
unchanged (still 8-bit) so alpha stays smooth, and if its /Matte has 3
components it is reduced to 1 after the parent becomes DeviceGray.
--bits 4 (LOSSY, separate mode): neutral RGB or 8-bit DeviceGray images (Flate or
DCT) are quantized to the nearest of 16 levels (multiples of 17) and written as
4-bit DeviceGray Flate (zlib 9, PNG predictor tried, smallest kept); max/mean
error vs the decoded source is reported per image.
--palette (with --bits 4, LOSSY): non-neutral RGB images are snapped per channel
to the 16-level grid and written as /Indexed /DeviceRGB, Flate: 4-bit if <= 16
colors, else 8-bit if <= 256.  With more than 256 colors (or to reach <= 16),
colors covering < --min-frac of pixels (default 1e-4) are mapped to the
nearest kept color; pixels changed and max error are reported.
Images are processed in parallel (-j, default = cores); output is
byte-identical to -j 1.  --max-err N (with --bits 4) leaves any image whose
max error would exceed N unchanged.
Usage: pdf-to-gray.py [-j N] [--max-err N] [--dct-tol N] [--bits 8|4] [--palette [--min-frac F]]
                      SOURCE.pdf OUTPUT.pdf
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
    for k in ("/Mask", "/Decode"):
        if k in o: return f"has {k}"
    if o.get("/ImageMask", False): return "image mask"
    if o.get("/ColorSpace") not in (N.DeviceRGB, N.DeviceGray): return f"colorspace {o.get('/ColorSpace')!r}"[:60]
    if int(o.get("/BitsPerComponent", 0)) != 8: return "bpc != 8"
    f = filters(o)
    if f not in (["/FlateDecode"], ["/DCTDecode"]): return f"filters {f}"
    return None

def flate_best(rows, bpc, w):
    cands = [(zlib.compress(rows.tobytes(), 9), None)]
    for f in (1, 2):
        cands.append((zlib.compress(png_predict(rows, f), 9),
                      pikepdf.Dictionary(Predictor=15, Colors=1, BitsPerComponent=bpc, Columns=w)))
    return min(cands, key=lambda c: len(c[0]))

def palettize(rgb, min_frac):
    """rgb uint8 (h,w,3) -> (indices uint8 (h,w), palette list of (r,g,b) 0..255, stats)."""
    q = ((rgb.astype(np.uint16) + 8) // 17)                      # 0..15 per channel
    code = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]       # 12-bit color id
    cnt = np.bincount(code.ravel(), minlength=4096)
    used = np.flatnonzero(cnt); order = used[np.argsort(-cnt[used], kind="stable")]
    total = code.size; ncolors = len(order)
    if ncolors <= 256: keep = order
    else: keep = order[cnt[order] >= min_frac * total][:256]
    if 16 < len(keep):                        # try 4-bit if only rare colors push it over
        k16 = order[cnt[order] >= min_frac * total]
        if len(k16) <= 16: keep = k16
    kc = np.stack([(keep >> 8) & 15, (keep >> 4) & 15, keep & 15], 1).astype(np.int32)
    lut = np.zeros(4096, np.uint8)
    lut[keep] = np.arange(len(keep))
    rest = np.setdiff1d(used, keep)
    if len(rest):
        rc = np.stack([(rest >> 8) & 15, (rest >> 4) & 15, rest & 15], 1).astype(np.int32)
        d = ((rc[:, None, :] - kc[None, :, :]) ** 2).sum(-1)
        lut[rest] = np.argmin(d, 1)
    idx = lut[code]
    pal = kc * 17
    err = np.abs(pal[idx].astype(np.int16) - rgb).max()
    changed = int(cnt[rest].sum()) if len(rest) else 0
    return idx, pal.astype(np.uint8), dict(colors_snapped=ncolors, palette=len(keep),
                                           remapped_px=changed, max_err=int(err))

def convert4(o, dct_tol, palette=False, min_frac=1e-4):
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
        if cd > tol:
            if not palette: return "skip", kind, old, old, f"real color (chroma diff {cd})"
            idx, pal, st = palettize(np.ascontiguousarray(a), min_frac)
            bpc = 4 if len(pal) <= 16 else 8
            rows = np.ascontiguousarray(pack4(idx)) if bpc == 4 else idx
            data, parms = flate_best(rows, bpc, w)
            if "/DecodeParms" in o: del o["/DecodeParms"]
            o.write(data, filter=N.FlateDecode, decode_parms=parms)
            o.ColorSpace = pikepdf.Array([N.Indexed, N.DeviceRGB, len(pal) - 1, pikepdf.String(pal.tobytes())])
            o.BitsPerComponent = bpc
            return "ok", "palette", old, len(data), (f"Indexed {bpc}-bit, {st['palette']} colors "
                f"(of {st['colors_snapped']} snapped), remapped {st['remapped_px']} px, max err {st['max_err']}")
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
    for k in ("/Mask", "/Decode"):
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

_W = {}
def _work(args):
    """Worker: convert one image in a private copy of the PDF; return picklable result."""
    src, og, bits, dct_tol, palette, min_frac, max_err = args
    if _W.get("src") != src: _W.clear(); _W["src"] = src; _W["pdf"] = pikepdf.open(src)
    o = _W["pdf"].get_object(og)
    st, kind, old, new, info = (convert4(o, dct_tol, palette, min_frac) if bits == 4 else convert(o, dct_tol))
    if st == "ok" and max_err is not None and bits == 4:
        import re
        m = re.search(r"max err (\d+)", info)
        if m and int(m.group(1)) > max_err:
            return og, "skip", kind, old, old, f"max err {m.group(1)} > --max-err {max_err}", None
    if st != "ok": return og, st, kind, old, new, info, None
    dp = o.get("/DecodeParms"); cs = o.ColorSpace
    cs = str(cs) if isinstance(cs, pikepdf.Name) else (int(cs[2]), bytes(cs[3]))
    upd = dict(data=o.read_raw_bytes(), filter=str(o.Filter),
               dp=None if dp is None else {k: int(v) for k, v in dp.items()},
               cs=cs, bpc=int(o.BitsPerComponent))
    return og, st, kind, old, new, info, upd

def _apply(o, u):
    if "/DecodeParms" in o: del o["/DecodeParms"]
    dp = None if u["dp"] is None else pikepdf.Dictionary({k: v for k, v in u["dp"].items()})
    o.write(u["data"], filter=N(u["filter"]), decode_parms=dp)
    o.ColorSpace = N(u["cs"]) if isinstance(u["cs"], str) else pikepdf.Array(
        [N.Indexed, N.DeviceRGB, u["cs"][0], pikepdf.String(u["cs"][1])])
    o.BitsPerComponent = u["bpc"]

def fix_smask_matte(o):
    """After parent -> DeviceGray, /Matte on its soft-mask must have 1 component."""
    if o.get("/ColorSpace") != N.DeviceGray: return
    sm = o.get("/SMask")
    if sm is None or not isinstance(sm, pikepdf.Stream) or "/Matte" not in sm: return
    m = [float(x) for x in sm.Matte]
    if len(m) == 3: sm.Matte = [m[1]]
    elif len(m) != 1: sm.Matte = [m[0]]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--dct-tol", type=int, default=2)
    p.add_argument("--palette", action="store_true", help="with --bits 4: Indexed RGB for color images (lossy)")
    p.add_argument("--min-frac", type=float, default=1e-4, help="palette: colors below this pixel fraction are remapped")
    p.add_argument("--bits", type=int, choices=(8, 4), default=8, help="8 = lossless (default); 4 = lossy 16-level")
    p.add_argument("--max-err", type=int, default=None, help="with --bits 4: leave an image unchanged if its max error would exceed N")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (default: cores); output identical to -j 1")
    a = p.parse_args(argv)
    if a.palette and a.bits != 4: p.error("--palette requires --bits 4")
    if os.path.abspath(a.src) == os.path.abspath(a.out): p.error("refusing to overwrite input")
    if os.path.exists(a.out): p.error(f"{a.out} exists; refusing to overwrite")
    pdf = pikepdf.open(a.src)
    imgs = [o for o in pdf.objects if isinstance(o, pikepdf.Stream) and o.get("/Subtype") == N.Image]
    smask_ogs = set()
    for o in imgs:
        sm = o.get("/SMask")
        if isinstance(sm, pikepdf.Stream):
            try: smask_ogs.add(sm.objgen)
            except Exception: pass
    jobs, preset = [], {}
    for o in imgs:
        if o.objgen in smask_ogs:
            old = len(o.read_raw_bytes())
            preset[o.objgen] = (o.objgen, "skip", "-", old, old, "soft-mask stream (left 8-bit)", None)
        else:
            jobs.append((a.src, o.objgen, a.bits, a.dct_tol, a.palette, a.min_frac, a.max_err))
    if a.jobs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.jobs, len(jobs))) as pool: worked = pool.map(_work, jobs, chunksize=1)
    else:
        _W.clear(); worked = [_work(j) for j in jobs]; _W.clear()
    by_og = {r[0]: r for r in worked}; by_og.update(preset)
    tot = {}; skipped = 0
    for o in imgs:
        og, st, kind, old, new, info, upd = by_og[o.objgen]
        print(f"obj {og[0]}: {int(o.Width)}x{int(o.Height)} {kind} {st} {old} -> {new} ({info})", flush=True)
        if st == "skip": skipped += 1; continue
        _apply(o, upd); fix_smask_matte(o)
        t = tot.setdefault(kind, [0, 0, 0]); t[0] += 1; t[1] += old; t[2] += new
    for k, (n, o_, n_) in tot.items():
        print(f"total {k}: {n} images {o_} -> {n_} bytes (saved {o_ - n_}, {100*(o_-n_)/max(o_,1):.1f}%)")
    print(f"skipped: {skipped}")
    pdf.save(a.out, deterministic_id=True)

if __name__ == "__main__": main()
