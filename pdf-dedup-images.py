#!/usr/bin/env python3
"""Merge byte-identical image XObjects into one shared object (lossless).

Two images are merged when their raw (still-encoded) stream bytes and their
dictionaries (except /Length) are identical.  Page /XObject references are
pointed at the first copy; unreferenced copies are dropped on save.  Image
data is copied as-is, never decoded or re-encoded; content streams and page
boxes are untouched.
Usage: pdf-dedup-images.py SOURCE.pdf OUTPUT.pdf
"""
import hashlib, os, sys
import pikepdf

def key(o):
    d = {k: repr(v) for k, v in o.items() if k != "/Length"}
    return hashlib.sha256(repr(sorted(d.items())).encode() + o.read_raw_bytes()).hexdigest()

def dedup(pdf):
    canon = {}; merged = 0
    for page in pdf.pages:
        res = page.obj.get("/Resources"); xd = res.get("/XObject") if res is not None else None
        if xd is None: continue
        for name in list(xd.keys()):
            o = xd[name]
            if o.get("/Subtype") != "/Image": continue
            k = key(o)
            if k in canon:
                if canon[k].objgen != o.objgen: xd[name] = canon[k]; merged += 1
            else: canon[k] = o
    pdf.remove_unreferenced_resources()
    return len(canon), merged

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2: sys.exit(__doc__)
    src, dst = argv
    if os.path.abspath(src) == os.path.abspath(dst) or os.path.exists(dst):
        sys.exit(f"refusing to overwrite {dst}")
    pdf = pikepdf.open(src)
    n, merged = dedup(pdf)
    pdf.save(dst, deterministic_id=True)
    print(f"{dst}: unique images {n}, references merged {merged}")

if __name__ == "__main__": main()
