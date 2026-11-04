#!/usr/bin/env python3
"""Fill dither speckles (small white holes) inside solid black areas of 1bpp scans.

A white 4-connected component is filled only if
  area <= --max-hole px (default 16 at 600 dpi, scaled by dpi^2) AND
  the fraction of black pixels in its 1-px 8-neighbour ring >= --surround (0.85)
  AND the local black density in a --window box around it >= --density (0.6).
Real counters/gaps are larger or touch white space, so they are kept; edges
are never grown because only enclosed white pixels change.

Edge stage (default on, --no-edges to skip): on straight horizontal/vertical
edges (>= --edge-min-run px, outside row >=92% white, inside row >=92% black)
fills white notches <= --edge-max-notch px wide on the edge line and removes
isolated 1-px black bumps (--no-bumps to keep).  Never adds black outside the
original edge row; curves/corners/diagonals fail the straight-row test.

PDF in -> PDF out: only image XObject streams are swapped (lossless JBIG2,
`jbig2 -p` generic region); page boxes, content streams, matrices untouched.
PNG/TIFF in -> PNG/TIFF out (TIFF written with Group 4).
Usage: pdf-fill-speckle.py IN OUT [--max-hole N] [--surround R] [--density D]
                           [--window W] [--dpi D] [--dry-run]
                           [--no-edges] [--edge-min-run N] [--edge-max-notch N]
                           [--no-bumps] [--edge-passes N] [-j N]
PDF pages are processed in parallel (-j, default os.cpu_count(); -j 1 serial);
output is identical for any -j.
"""
import argparse, os, subprocess, sys, tempfile
sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def fill(black, max_hole, surround, density, window):
    white = ~black
    lab, n = ndi.label(white)                      # 4-connectivity
    if n == 0: return black.copy(), 0
    area = np.bincount(lab.ravel(), minlength=n + 1)
    cand = area <= max_hole; cand[0] = False
    # ring = 8-neighbour dilation minus the component; black fraction of ring
    small = cand[lab]
    sl = lab * small
    dil = ndi.grey_dilation(sl, size=(3, 3))       # label spreads into ring
    ring = (dil > 0) & (sl == 0)
    tot = np.bincount(dil[ring], minlength=n + 1)
    blk = np.bincount(dil[ring & black], minlength=n + 1)
    # (where two small comps are adjacent the larger label wins; fine)
    frac = np.where(tot > 0, blk / np.maximum(tot, 1), 0)
    loc = ndi.uniform_filter(black.astype(np.float32), size=window)
    locmean = ndi.mean(loc, lab, index=np.arange(n + 1)) if n else []
    ok = cand & (frac >= surround) & (np.asarray(locmean) >= density)
    out = black | ok[lab]
    return out, int(area[ok].sum())


def _edge_top(b, L, max_notch, bumps):
    """Straighten 'top' edges (black below, white above) of array b.
    Row y is an edge line at x if, over a window of L px along the row:
      row y is >= 50% black, row y+1 (inside) >= 75% black, and either
      row y-1 (outside) is >= 70% white (edge row) or row y >= 75% (interior).
    Notches: white runs on row y of length <= max_notch, black on both
    sides along the row and black directly behind (y+1) -> filled.
    Bumps: isolated black px on a row <= 30% black, with a >= 50% black
    edge row behind it and a >= 90% white row outside -> removed.  Windows that do not sit on one straight
    row (curves, corners, diagonals) fail the inside/outside tests."""
    H, W = b.shape
    f = b.astype(np.float32)
    m = ndi.uniform_filter1d(f, size=L, axis=1, mode="constant")
    add = np.zeros_like(b); rem = np.zeros_like(b)
    mo = np.zeros_like(m); mo[1:] = m[:-1]          # outside row mean
    mi = np.zeros_like(m); mi[:-1] = m[1:]          # inside row mean
    bi = np.zeros_like(b); bi[:-1] = b[1:]          # pixel behind
    # edge line: mostly-black row with white outside, or a solid interior row
    line = (m >= 0.5) & (mi >= 0.75) & ((mo <= 0.3) | (m >= 0.75))
    # notch fill: white runs along the row of length <= max_notch
    wh = ~b
    lab, n = ndi.label(wh, structure=[[0,0,0],[1,1,1],[0,0,0]])
    if n:
        ln = np.bincount(lab.ravel(), minlength=n+1)
        lft = np.zeros_like(b); lft[:,1:] = b[:,:-1]
        rgt = np.zeros_like(b); rgt[:,:-1] = b[:,1:]
        # run is bounded by black at both ends; check via min/max column
        short = (ln <= max_notch); short[0] = False
        cand = wh & short[lab] & line & bi
        # every pixel of a run must qualify, and run ends must be black
        okrun = np.ones(n+1, bool); okrun[0] = False
        bad = np.bincount(lab[wh & short[lab] & ~cand], minlength=n+1) > 0
        okrun &= ~bad
        endL = np.bincount(lab[wh & lft], minlength=n+1)
        endR = np.bincount(lab[wh & rgt], minlength=n+1)
        # pixel touching black on left = run start; right = run end
        okrun &= short & (endL > 0) & (endR > 0)
        add = wh & okrun[lab] & cand
    if bumps:
        lft = np.zeros_like(b); lft[:,1:] = b[:,:-1]
        rgt = np.zeros_like(b); rgt[:,:-1] = b[:,1:]
        out = np.zeros_like(b); out[1:] = b[:-1]
        rem = b & ~lft & ~rgt & ~out & bi & (m <= 0.3) & (mi >= 0.5) & (mo <= 0.1)
    return add, rem

def straighten(black, L, max_notch, bumps=True):
    add = np.zeros_like(black); rem = np.zeros_like(black)
    for k in range(4):                     # top, left, bottom, right edges
        r = np.rot90(black, k)
        a_, r_ = _edge_top(np.ascontiguousarray(r), L, max_notch, bumps)
        add |= np.rot90(a_, -k); rem |= np.rot90(r_, -k)
    rem &= ~add
    return (black | add) & ~rem, int(add.sum()), int(rem.sum())

def process(img, a, dpi):
    s = (dpi / 600.0)
    mh = max(1, round(a.max_hole * s * s)); win = max(3, round(a.window * s) | 1)
    black = ~np.array(img.convert("1"), dtype=bool)
    out, changed = fill(black, mh, a.surround, a.density, win)
    a._st = (changed, 0, 0)
    if not a.no_edges:
        L = max(4, round(a.edge_min_run * s))
        for _ in range(a.edge_passes):
            out, ad, rm = straighten(out, L, a.edge_max_notch, not a.no_bumps)
            a._st = (a._st[0], a._st[1] + ad, a._st[2] + rm)
            if ad + rm == 0: break
    return out, black, changed

def to_img(black):
    return Image.fromarray(~black).convert("1")

def _work(job):
    """Worker: decode -> process -> jbig2 encode. Returns (key, stats, jb bytes)."""
    key, mode, size, raw, dpi, a, tmp = job
    im = Image.frombytes(mode, size, raw)
    out, black, ch = process(im, a, dpi)
    jb = None
    if not a.dry_run:
        pp = f"{tmp}/p{key[0]}_{os.getpid()}.png"; to_img(out).save(pp)
        jb = subprocess.run(["jbig2", "-p", pp], check=True, capture_output=True).stdout
        os.remove(pp)
    return key, (im.width, im.height, dpi, ch, a._st[1], a._st[2]), jb

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inp"); p.add_argument("out", nargs="?")
    p.add_argument("--max-hole", type=int, default=16, help="max hole area px at 600 dpi (default 16)")
    p.add_argument("--surround", type=float, default=0.85, help="min black fraction of ring (default 0.85)")
    p.add_argument("--density", type=float, default=0.6, help="min local black density (default 0.6)")
    p.add_argument("--window", type=int, default=15, help="density window px at 600 dpi (default 15)")
    p.add_argument("--dpi", type=float, default=None, help="override dpi (images default 600)")
    p.add_argument("--no-edges", action="store_true", help="skip edge straightening")
    p.add_argument("--edge-min-run", type=int, default=12, help="min straight edge length px at 600 dpi (default 12)")
    p.add_argument("--edge-max-notch", type=int, default=1, help="max notch width px filled along an edge (default 1)")
    p.add_argument("--no-bumps", action="store_true", help="don't remove 1-px black bumps on straight edges")
    p.add_argument("--edge-passes", type=int, default=2, help="edge passes (default 2)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="worker processes (default cpu count; 1 = serial)")
    a = p.parse_args()
    if not a.dry_run and not a.out: p.error("OUT required unless --dry-run")
    if a.out and os.path.abspath(a.out) == os.path.abspath(a.inp): p.error("refusing to overwrite input")
    if a.inp.lower().endswith(".pdf"):
        import pikepdf
        from pikepdf import PdfImage
        pdf = pikepdf.open(a.inp)
        tmp = tempfile.mkdtemp(dir=os.path.expanduser("~"))
        seen = set(); total = 0; tot_e = [0, 0]; jobs = []; xos = {}
        for pn, page in enumerate(pdf.pages, 1):
            W = float(page.mediabox[2]) - float(page.mediabox[0])
            for name, xo in page.images.items():
                if xo.objgen in seen: continue
                seen.add(xo.objgen)
                if int(xo.get("/BitsPerComponent", 0)) != 1:
                    jobs.append(((pn, name), None)); continue
                im = PdfImage(xo).as_pil_image()
                dpi = a.dpi or im.width / (W / 72.0)
                xos[(pn, name)] = xo
                jobs.append(((pn, name), ((pn, name), im.mode, im.size, im.tobytes(), dpi, a, tmp)))
        work = [j for _, j in jobs if j is not None]
        if a.jobs > 1 and len(work) > 1:
            import multiprocessing as mp
            pool = mp.get_context("fork").Pool(min(a.jobs, len(work)))
            results = pool.imap(_work, work)
        else:
            pool = None; results = map(_work, work)
        for key, j in jobs:
            pn, name = key
            if j is None:
                print(f"page {pn} {name}: not 1bpp, skipped"); continue
            k, (w, h, dpi, ch, ad, rm), jb = next(results)
            total += ch; tot_e[0] += ad; tot_e[1] += rm
            print(f"page {pn} {name}: {w}x{h} ~{dpi:.0f}dpi, filled {ch} px, edge +{ad} -{rm} px")
            if jb is None: continue
            xo = xos[k]
            for kk in ("/DecodeParms", "/JBIG2Globals", "/Decode"):
                if kk in xo: del xo[kk]
            xo.write(jb, filter=pikepdf.Name.JBIG2Decode)
            xo.ColorSpace = pikepdf.Name.DeviceGray; xo.BitsPerComponent = 1
        if pool: pool.close(); pool.join()
        print(f"total filled: {total} px, edge added {tot_e[0]} px, edge removed {tot_e[1]} px")
        if not a.dry_run: pdf.save(a.out, deterministic_id=True)
        os.rmdir(tmp)
    else:
        im = Image.open(a.inp)
        dpi = a.dpi or (im.info.get("dpi", (600,))[0] or 600)
        out, black, ch = process(im, a, float(dpi))
        print(f"{im.width}x{im.height} ~{dpi}dpi, filled {ch} px, edge +{a._st[1]} -{a._st[2]} px")
        if not a.dry_run:
            o = to_img(out)
            if a.out.lower().endswith((".tif", ".tiff")): o.save(a.out, compression="group4", dpi=(dpi, dpi))
            else: o.save(a.out, dpi=(dpi, dpi))
if __name__ == "__main__": main()
