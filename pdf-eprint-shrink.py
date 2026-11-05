#!/usr/bin/env python3
"""One-step eprint shrink: analyze -> dedup -> 4-bit gray (orchestrates existing tools).

Steps (each an existing single-purpose tool, run as a subprocess):
  1. pdf-analyze.py        (report + JSON, printed compactly)
  2. pdf-dedup-images.py   (lossless; merges identical images and drops
                            unreferenced/orphaned objects on save)
  3. pdf-to-gray.py --bits 4 --max-err N [--palette]
       --photo jpeg   -> SRC-gray4.pdf         (color images left as they are)
       --photo pal256 -> SRC-gray4-pal256.pdf  (color images as Indexed palette)
       --photo both   -> both files
     Any image whose max error would exceed --max-err is left unchanged.
  4. verify: same page count, same extracted text per page, page sizes equal.
  5. optional --compare-dir: labeled side-by-side PNGs (original | new) of
     page 1 and the page with the worst error.
Never overwrites the source or an existing output.
Usage: pdf-eprint-shrink.py SRC.pdf [--photo jpeg|pal256|both] [--max-err N]
                            [-j N] [--progress FILE] [--compare-dir DIR]
"""
import argparse, json, os, pathlib, subprocess, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parent

def prog(path, msg):
    print(msg, flush=True)
    if path:
        with open(path, "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def run(tool, *args):
    r = subprocess.run([sys.executable, str(HERE / tool), *map(str, args)], capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tool} failed:\n{r.stderr[-2000:]}")
    return r.stdout

def verify(src, out):
    import pymupdf
    a, b = pymupdf.open(src), pymupdf.open(out)
    if len(a) != len(b): return f"page count {len(a)} != {len(b)}"
    for i in range(len(a)):
        if a[i].get_text() != b[i].get_text(): return f"text differs on page {i+1}"
        if tuple(a[i].rect) != tuple(b[i].rect): return f"page size differs on page {i+1}"
    return None

def compare(src, out, pages, d, label):
    import pymupdf
    from PIL import Image, ImageDraw
    a, b = pymupdf.open(src), pymupdf.open(out); files = []
    for p in pages:
        ims = []
        for doc, name in ((a, "original"), (b, label)):
            pm = doc[p - 1].get_pixmap(dpi=100)
            im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            c = Image.new("RGB", (im.width, im.height + 30), "white"); c.paste(im, (0, 30))
            ImageDraw.Draw(c).text((8, 8), f"{name}  p{p}", fill="red"); ims.append(c)
        s = Image.new("RGB", (ims[0].width + ims[1].width + 10, max(i.height for i in ims)), "gray")
        s.paste(ims[0], (0, 0)); s.paste(ims[1], (ims[0].width + 10, 0))
        f = os.path.join(d, f"{pathlib.Path(out).stem}-p{p}-cmp.png"); s.save(f); files.append(f)
    return files

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("--photo", choices=("jpeg", "pal256", "both"), default="jpeg")
    p.add_argument("--max-err", type=int, default=8); p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--progress"); p.add_argument("--compare-dir")
    a = p.parse_args(argv)
    src = os.path.abspath(a.src); stem = src[:-4] if src.lower().endswith(".pdf") else src
    outs = {"jpeg": [(stem + "-gray4.pdf", False)], "pal256": [(stem + "-gray4-pal256.pdf", True)]}
    outs["both"] = outs["jpeg"] + outs["pal256"]
    targets = outs[a.photo]
    for o, _ in targets:
        if os.path.exists(o): p.error(f"{o} exists; refusing to overwrite")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        js = os.path.join(td, "an.json")
        prog(a.progress, "step 1/4 analyze")
        rep = run("pdf-analyze.py", src, "-j", a.jobs, "--json", js, *(["--progress", a.progress] if a.progress else []))
        print("\n".join(rep.strip().splitlines()[-3:]))
        d = json.load(open(js))
        prog(a.progress, "step 2/4 dedup")
        dd = os.path.join(td, "dedup.pdf"); print(run("pdf-dedup-images.py", src, dd).strip().split(": ", 1)[1])
        results = []
        for out, pal in targets:
            prog(a.progress, f"step 3/4 gray4 -> {os.path.basename(out)}")
            tmp = os.path.join(td, "g.pdf")
            if os.path.exists(tmp): os.remove(tmp)
            log = run("pdf-to-gray.py", "--bits", 4, "--max-err", a.max_err, "-j", a.jobs, *(["--palette"] if pal else []), dd, tmp)
            sk = [l for l in log.splitlines() if " skip " in l]
            prog(a.progress, f"step 4/4 verify {os.path.basename(out)}")
            bad = verify(src, tmp)
            if bad: sys.exit(f"verify FAILED ({bad}); nothing written")
            if os.path.exists(out): sys.exit(f"{out} appeared meanwhile; refusing to overwrite")
            os.replace(tmp, out) if os.stat(td).st_dev == os.stat(os.path.dirname(out)).st_dev else __import__("shutil").move(tmp, out)
            results.append((out, sk)); prog(a.progress, f"wrote {os.path.basename(out)} {os.path.getsize(out)} B ({len(sk)} images left unchanged)")
            for l in sk: print("  unchanged:", l.split("(", 1)[-1].rstrip(")"))
        if a.compare_dir:
            os.makedirs(a.compare_dir, exist_ok=True)
            gr = [r for r in d["images"] if r.get("kind") == "gray" and "dup_of" not in r]
            worst = max(gr, key=lambda r: r["qerr_mean"])["pages"][0] if gr else 1
            for out, _ in results:
                for f in compare(src, out, sorted({1, worst}), a.compare_dir, os.path.basename(out)[-20:]):
                    print("compare:", f)
    s = os.path.getsize(src)
    for out, _ in results:
        n = os.path.getsize(out); print(f"{os.path.basename(out)}: {s} -> {n} B (-{100*(s-n)/s:.1f}%)")
    prog(a.progress, f"done in {time.time()-t0:.1f}s")

if __name__ == "__main__": main()
