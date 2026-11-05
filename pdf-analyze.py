#!/usr/bin/env python3
"""Analyze the images in a PDF (read-only): what they are and how they would shrink.

Per unique image object (shared images analysed once): pages, filters,
colorspace, bpc, size, raw bytes; gray vs color (max RGB channel diff);
  gray:  distinct levels, % exactly on the 16-level grid (x17), % within 3,
         4-bit quantization max/mean error;
  color: raw and x17-clustered color counts, top clusters, 16/256-color
         palette max/mean error and pixel cover %.
Also: byte-identical duplicates, /SMask and /Mask images (counted as
part of their parent, not orphans), inline images, vector-only pages, orphan
(unreferenced) images.  Size estimates are real trial conversions: each image
is run through pdf-to-gray.py's own --bits 4 (and --palette) code and the
resulting stream bytes are summed with the file's non-image bytes.
Images are processed in parallel (-j, default = cores).
Usage: pdf-analyze.py SRC.pdf [-j N] [--json OUT] [--progress FILE]
"""
import argparse, hashlib, importlib.util, io, json, os, pathlib, sys, time
import numpy as np, pikepdf
from PIL import Image
N = pikepdf.Name
_spec = importlib.util.spec_from_file_location("p2g", pathlib.Path(__file__).resolve().parent / "pdf-to-gray.py")
p2g = sys.modules.get("p2g") or importlib.util.module_from_spec(_spec)
if "p2g" not in sys.modules: sys.modules["p2g"] = p2g; _spec.loader.exec_module(p2g)

def progress(path, msg):
    if path:
        with open(path, "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def decode(o):
    f = p2g.filters(o)
    if f == ["/DCTDecode"]:
        im = Image.open(io.BytesIO(o.read_raw_bytes()))
    else:
        im = pikepdf.PdfImage(o).as_pil_image()
    if im.mode == "P": im = im.convert("RGB")
    if im.mode == "1": im = im.convert("L")
    if im.mode == "CMYK": im = im.convert("RGB")
    if im.mode not in ("L", "RGB"): im = im.convert("RGB")
    return np.asarray(im)

def gray_stats(g):
    g = g.astype(np.int16)
    cnt = np.bincount(g.ravel(), minlength=256)
    e = np.abs(g - (g + 8) // 17 * 17)
    return dict(levels=int((cnt > 0).sum()), exact17=round(100 * cnt[::17].sum() / g.size, 2),
                within3=round(100 * (e <= 3).mean(), 2), qerr_max=int(e.max()), qerr_mean=round(float(e.mean()), 3))

def color_stats(c):
    c = c.reshape(-1, 3).astype(np.int32)
    raw = len(np.unique((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]))
    q = (c + 8) // 17
    k = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    cnt = np.bincount(k, minlength=4096); order = np.argsort(-cnt, kind="stable"); order = order[cnt[order] > 0]
    top = [("#%02x%02x%02x" % tuple(int(v) * 17 for v in ((j >> 8) & 15, (j >> 4) & 15, j & 15)),
            round(float(100 * cnt[j] / len(k)), 2)) for j in order[:6]]
    s = c[:: max(1, len(c) // 200000)]                    # <=~200k px sample
    res = {}
    for n in (16, 256):
        keep = order[:n]
        pal = np.stack([(keep >> 8) & 15, (keep >> 4) & 15, keep & 15], 1) * 17
        best = np.full(len(s), 999, np.int32)
        for i in range(0, len(pal), 32):
            d = np.abs(s[:, None, :] - pal[None, i:i + 32, :]).max(2).min(1)
            best = np.minimum(best, d)
        res[f"pal{n}"] = dict(maxerr=int(best.max()), meanerr=round(float(best.mean()), 2),
                              cover=round(100 * cnt[keep].sum() / len(k), 2))
    return dict(raw_colors=raw, x17_colors=int(len(order)), top=top, **res)

_W = {}
def work(args):
    src, og = args
    if _W.get("src") != src: _W.clear(); _W["src"] = src; _W["pdf"] = pikepdf.open(src)
    o = _W["pdf"].get_object(og)
    r = dict(obj=og[0], w=int(o.get("/Width", 0)), h=int(o.get("/Height", 0)), filters=p2g.filters(o),
             cs=str(o.get("/ColorSpace"))[:40], bpc=int(o.get("/BitsPerComponent", 0) or 0))
    raw = o.read_raw_bytes(); r["bytes"] = len(raw)
    r["sha"] = hashlib.sha256(raw + repr(sorted((k, repr(v)) for k, v in o.items() if k != "/Length")).encode()).hexdigest()
    try:
        a = decode(o)
    except Exception as e:
        r["kind"] = "undecodable"; r["err"] = str(e)[:80]; r["est4"] = r["estpal"] = len(raw); return r
    if a.ndim == 3 and a.shape[2] == 3:
        b = a.astype(np.int16)
        r["rgbdiff"] = int(max(np.abs(b[..., 0] - b[..., 1]).max(), np.abs(b[..., 1] - b[..., 2]).max()))
    else: r["rgbdiff"] = 0
    gray = r["rgbdiff"] <= 2
    if gray:
        r["kind"] = "gray"; r.update(gray_stats(a[..., 1] if a.ndim == 3 else a))
    else:
        r["kind"] = "color"; r.update(color_stats(a))
    for key, pal in (("est4", False), ("estpal", True)):     # real trial with pdf-to-gray's code
        tmp = pikepdf.open(src); t = tmp.get_object(og)
        st, kind, old, new, info = p2g.convert4(t, 2, pal)
        r[key] = new; r[key + "_info"] = info if st == "ok" else "skip: " + info
        if st == "ok" and "max err" in info:
            r[key + "_maxerr"] = int(info.split("max err ")[1].split(",")[0].split(")")[0])
        tmp.close()
    return r

def xobjects(res, seen=None):
    """Yield image objects reachable from a resources dict (through form XObjects)."""
    seen = set() if seen is None else seen
    xd = res.get("/XObject") if res is not None else None
    if xd is None: return
    for _, o in xd.items():
        if not isinstance(o, pikepdf.Stream) or o.objgen in seen: continue
        seen.add(o.objgen)
        if o.get("/Subtype") == N.Image: yield o
        elif o.get("/Subtype") == N.Form: yield from xobjects(o.get("/Resources"), seen)

def analyze(src, jobs=None, prog=None):
    t0 = time.time(); jobs = jobs or os.cpu_count() or 1
    pdf = pikepdf.open(src); progress(prog, f"analyze start {os.path.basename(src)} ({os.path.getsize(src)} B, {len(pdf.pages)} pages)")
    pages_of = {}; inline = {}; vector_only = []
    for i, pg in enumerate(pdf.pages, 1):
        n = 0
        try:
            ops = pikepdf.parse_content_stream(pg)
            ni = sum(1 for x in ops if isinstance(x, pikepdf.ContentStreamInlineImage))
            used = {str(x.operands[0]) for x in ops if not isinstance(x, pikepdf.ContentStreamInlineImage) and str(x.operator) == "Do"}
        except Exception: ni, used = 0, None
        res = pg.obj.get("/Resources"); xd = res.get("/XObject") if res is not None else None
        for name, o in (xd.items() if xd is not None else []):
            if used is not None and name not in used: continue      # in resources but never drawn
            if not isinstance(o, pikepdf.Stream): continue
            if o.get("/Subtype") == N.Image: imgs = [o]
            elif o.get("/Subtype") == N.Form: imgs = list(xobjects(o.get("/Resources")))
            else: imgs = []
            for im in imgs:
                if i not in pages_of.setdefault(im.objgen, []): pages_of[im.objgen].append(i); n += 1
        if ni: inline[i] = ni
        if not n and not ni: vector_only.append(i)
    allimg = [o.objgen for o in pdf.objects if isinstance(o, pikepdf.Stream) and o.get("/Subtype") == N.Image]
    masks = {}                                   # mask objgen -> parent obj num (/SMask or stream /Mask)
    for og in list(pages_of):
        o = pdf.get_object(og)
        for k in ("/SMask", "/Mask"):
            m = o.get(k)
            if isinstance(m, pikepdf.Stream): masks[m.objgen] = (k[1:], og[0])
    orphans = [og for og in allimg if og not in pages_of and og not in masks]
    progress(prog, f"structure: {len(pages_of)} referenced images, {len(masks)} masks, {len(orphans)} orphans, {len(vector_only)} vector-only pages")
    todo = [(src, og) for og in pages_of]; res = []
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
            for k, r in enumerate(pool.imap(work, todo, chunksize=1), 1):
                res.append(r); progress(prog, f"image {k}/{len(todo)} obj {r['obj']} {r['kind']}")
    else:
        for k, t in enumerate(todo, 1):
            r = work(t); res.append(r); progress(prog, f"image {k}/{len(todo)} obj {r['obj']} {r['kind']}")
    _W.clear()
    first = {}
    for r, (_, og) in zip(res, todo):
        r["pages"] = pages_of[og]
        if r["sha"] in first: r["dup_of"] = first[r["sha"]]
        else: first[r["sha"]] = r["obj"]
    orphan_bytes = sum(len(pdf.get_object(og).read_raw_bytes()) for og in orphans)
    size = os.path.getsize(src); img_bytes = sum(r["bytes"] for r in res)
    overhead = size - img_bytes - orphan_bytes
    uniq = [r for r in res if "dup_of" not in r]
    dedup = overhead + sum(r["bytes"] for r in uniq)
    est_jpeg = overhead + sum(r["est4"] if r["kind"] == "gray" else r["bytes"] for r in uniq)
    est_pal = overhead + sum(r["est4"] if r["kind"] == "gray" else r["estpal"] for r in uniq)
    gr = [r for r in uniq if r["kind"] == "gray"]
    outl = [r["obj"] for r in gr if r["within3"] < 97 or r["qerr_mean"] > 0.3]
    clean = bool(gr) and all(r["qerr_max"] <= 8 for r in gr) and not outl
    summ = dict(file=src, bytes=size, pages=len(pdf.pages), images=len(res), unique=len(uniq),
                duplicates=len(res) - len(uniq), inline=inline, vector_only_pages=len(vector_only),
                orphans=[og[0] for og in orphans], orphan_bytes=orphan_bytes,
                masks={str(og[0]): f"{k} of o{p}" for og, (k, p) in masks.items()},
                gray=len(gr), color=sum(r["kind"] == "color" for r in uniq),
                clean_4bit=clean, outliers=outl, est_dedup=dedup, est_gray4_jpeg_photo=est_jpeg,
                est_gray4_pal256=est_pal, seconds=round(time.time() - t0, 1), jobs=jobs)
    progress(prog, f"analyze done {summ['seconds']}s: est gray4 {est_jpeg} B, pal {est_pal} B")
    return dict(summary=summ, images=res)

def report(d):
    out = []
    for r in d["images"]:
        pg = r["pages"]; pgs = f"p{pg[0]}" + (f"+{len(pg)-1}" if len(pg) > 1 else "")
        if r.get("masks"): pgs += "*"
        head = f"{pgs:8} o{r['obj']:<5} {'/'.join(f[1:4] for f in r['filters']) or '-':8} {r['cs'][1:10]:9} {r['bpc']} {r['w']}x{r['h']:<5} {r['bytes']:>9}"
        if "dup_of" in r: out.append(head + f"  DUP of o{r['dup_of']}"); continue
        if r["kind"] == "gray":
            out.append(head + f"  gray d{r['rgbdiff']} lv{r['levels']} x17 {r['exact17']}% w3 {r['within3']}% q {r['qerr_max']}/{r['qerr_mean']} -> {r['est4']}")
        elif r["kind"] == "color":
            p16, p256 = r["pal16"], r["pal256"]
            out.append(head + f"  COLOR d{r['rgbdiff']} raw {r['raw_colors']} x17 {r['x17_colors']} top {r['top'][:3]} "
                       f"p16 {p16['maxerr']}/{p16['meanerr']}/{p16['cover']}% p256 {p256['maxerr']}/{p256['meanerr']}/{p256['cover']}% -> pal {r['estpal']}")
        else: out.append(head + f"  {r['kind']} {r.get('err','')}")
    s = d["summary"]; mb = lambda b: f"{b/1e6:.2f} MB"
    out.append(f"--- {os.path.basename(s['file'])}: {mb(s['bytes'])}, {s['pages']} pages, {s['images']} images "
               f"({s['unique']} unique, {s['duplicates']} dup), {s['gray']} gray / {s['color']} color, "
               f"{s['vector_only_pages']} vector-only pages, inline {sum(s['inline'].values())}, masks {len(s['masks'])}, orphans {len(s['orphans'])} ({s['orphan_bytes']} B)")
    out.append(f"verdict: {'CLEAN 4-bit' if s['clean_4bit'] else 'NOT clean 4-bit'}" + (f"; outliers obj {s['outliers']}" if s['outliers'] else ""))
    out.append(f"estimate (trial-compressed): dedup {mb(s['est_dedup'])}; gray4 keep photo {mb(s['est_gray4_jpeg_photo'])}; gray4+pal {mb(s['est_gray4_pal256'])}  [{s['seconds']}s, -j {s['jobs']}]")
    return "\n".join(out)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--json"); p.add_argument("--progress")
    a = p.parse_args(argv)
    d = analyze(a.src, a.jobs, a.progress)
    print(report(d))
    if a.json:
        if os.path.abspath(a.json) == os.path.abspath(a.src): p.error("refusing to overwrite source")
        json.dump(d, open(a.json, "w"), indent=1)
    return d

if __name__ == "__main__": main()
