#!/usr/bin/env python3
"""pdf-regex-strip.py - list or truly redact text matching a Python regex in a PDF.

Usage:
  pdf-regex-strip.py [options] IN.pdf REGEX            # list mode (dry run)
  pdf-regex-strip.py [options] IN.pdf OUT.pdf REGEX    # strip mode
  pdf-regex-strip.py --in-place [options] IN.pdf REGEX # strip, overwrite IN

Matching is per text line by default (lines built from PyMuPDF 'rawdict'
characters, so match offsets map to exact character boxes). --span joins
all lines of a page with '\\n' so a regex can cross lines.
List mode also searches: Info dict, XMP, outline, annotations (contents,
title, subject, link URIs), form fields, page labels, embedded files,
and with --deep every decompressed stream / string object.
Never overwrites IN unless --in-place; never deletes files.
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
import pymupdf

def parse_pages(spec, n):
    if not spec: return list(range(n))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1; b = int(b) if b else n
            out.update(range(a - 1, min(b, n)))
        else:
            out.add(int(part) - 1)
    return sorted(p for p in out if 0 <= p < n)

def page_lines(page):
    """Return list of (text, [char_rect or None], line_rect)."""
    lines = []
    d = page.get_text("rawdict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
    for b in d["blocks"]:
        if b.get("type") != 0: continue
        for ln in b["lines"]:
            txt, boxes = [], []
            for sp in ln["spans"]:
                for ch in sp["chars"]:
                    txt.append(ch["c"]); boxes.append(pymupdf.Rect(ch["bbox"]))
            if txt:
                lines.append(("".join(txt), boxes, pymupdf.Rect(ln["bbox"])))
    return lines

def union(boxes):
    r = pymupdf.Rect()
    for b in boxes:
        if b is not None and not b.is_empty: r |= b
    return r

def find_page_hits(page, rx, span):
    """Yield dicts: text, line, rect(match), linerect."""
    lines = page_lines(page)
    hits = []
    if not span:
        for txt, boxes, lr in lines:
            for m in rx.finditer(txt):
                if m.end() == m.start(): continue
                hits.append(dict(text=m.group(0), line=txt, rects=[union(boxes[m.start():m.end()])], lines=[lr]))
    else:
        joined, boxes, owner = [], [], []
        for i, (txt, bx, lr) in enumerate(lines):
            if i: joined.append("\n"); boxes.append(None); owner.append(None)
            joined.append(txt); boxes.extend(bx); owner.extend([i] * len(bx))
        s = "".join(joined)
        for m in rx.finditer(s):
            if m.end() == m.start(): continue
            idx = sorted({owner[k] for k in range(m.start(), m.end()) if owner[k] is not None})
            rects = [union([boxes[k] for k in range(m.start(), m.end()) if owner[k] == i]) for i in idx]
            hits.append(dict(text=m.group(0), line=" | ".join(lines[i][0] for i in idx),
                             rects=rects, lines=[lines[i][2] for i in idx]))
    return hits

def fmt(r): return "(%.1f,%.1f,%.1f,%.1f)" % tuple(r)

def short(s, n=160):
    s = s.replace("\n", "\\n"); return s if len(s) <= n else s[:n] + "..."

def scan_other(doc, rx, pages, deep):
    """Return list of (where, value, matches)."""
    found = []
    def chk(where, val):
        if not isinstance(val, str) or not val: return
        ms = [m.group(0) for m in rx.finditer(val) if m.group(0)]
        if ms: found.append((where, val, ms))
    for k, v in (doc.metadata or {}).items(): chk("Info/%s" % k, v)
    chk("XMP metadata", doc.get_xml_metadata())
    for i, (lvl, title, pno, *_) in enumerate(doc.get_toc(simple=False)):
        chk("Outline #%d (lvl %d, p%d) title" % (i + 1, lvl, pno), title)
    for pno in pages:
        page = doc[pno]
        for a in page.annots() or []:
            for k in ("content", "title", "subject"):
                chk("p%d annot %s xref %d %s" % (pno + 1, a.type[1], a.xref, k), a.info.get(k))
        for l in page.get_links():
            chk("p%d link URI" % (pno + 1), l.get("uri"))
            chk("p%d link file" % (pno + 1), l.get("file"))
        for w in page.widgets() or []:
            chk("p%d field name" % (pno + 1), w.field_name)
            chk("p%d field '%s' value" % (pno + 1, w.field_name), str(w.field_value) if w.field_value not in (None, "") else "")
            chk("p%d field '%s' label" % (pno + 1, w.field_name), w.field_label)
    try:
        for lab in doc.get_page_labels():
            chk("Page label rule from p%d prefix" % (lab.get("startpage", 0) + 1), lab.get("prefix"))
    except Exception: pass
    for p in pages:
        chk("Page label p%d" % (p + 1), doc[p].get_label())
    for name in doc.embfile_names():
        chk("Embedded file name", name)
        try:
            info = doc.embfile_info(name)
            chk("Embedded file '%s' filename" % name, info.get("filename"))
            chk("Embedded file '%s' desc" % name, info.get("desc"))
        except Exception: pass
    if deep:
        for x in range(1, doc.xref_length()):
            try:
                chk("deep xref %d object" % x, doc.xref_object(x, compressed=False))
                if doc.xref_is_stream(x):
                    raw = doc.xref_stream(x)
                    if raw: chk("deep xref %d stream (decoded)" % x, raw.decode("latin-1"))
            except Exception: pass
    return found

def has_text(doc, pages):
    return [p for p in pages if not doc[p].get_text("text").strip()]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="+", metavar="IN.pdf [OUT.pdf] REGEX")
    ap.add_argument("--list", action="store_true", help="dry run (default when no OUT given)")
    ap.add_argument("--whole-line", action="store_true", help="redact the whole line containing each match")
    ap.add_argument("--fill", action="store_true", help="paint redacted areas white (default: no fill box)")
    ap.add_argument("--images", choices=["none", "pixels", "remove"], default="none",
                    help="images under redactions: none (default), pixels = blank those pixels (scans), remove = drop overlapping images")
    ap.add_argument("--pages", help="page ranges, 1-based, e.g. 1,3-5,8-")
    ap.add_argument("-i", "--ignore-case", action="store_true")
    ap.add_argument("--span", action="store_true", help="match across a page's lines joined with \\n (re.MULTILINE|re.DOTALL off unless in regex)")
    ap.add_argument("--meta", action="store_true", help="strip mode: also remove matches from Info, XMP, outline titles, annotation text")
    ap.add_argument("--deep", action="store_true", help="list mode: also search decompressed streams and every object")
    ap.add_argument("--ocr", action="store_true", help="OCR pages with no text using ocrmypdf first (if installed)")
    ap.add_argument("--in-place", action="store_true", help="allow OUT to be IN (or write back to IN)")
    a = ap.parse_args()
    if len(a.args) == 2: inp, out, pat = a.args[0], None, a.args[1]
    elif len(a.args) == 3: inp, out, pat = a.args
    else: ap.error("give IN.pdf REGEX or IN.pdf OUT.pdf REGEX")
    if a.in_place and out is None: out = inp
    if out and os.path.abspath(out) == os.path.abspath(inp) and not a.in_place:
        ap.error("OUT is the same as IN; use --in-place to allow that")
    listing = a.list or out is None
    flags = re.IGNORECASE if a.ignore_case else 0
    if a.span: flags |= re.MULTILINE
    rx = re.compile(pat, flags)

    src, tmpdir = inp, None
    doc = pymupdf.open(src)
    pages = parse_pages(a.pages, doc.page_count)
    if a.ocr:
        empty = has_text(doc, pages)
        if empty:
            if shutil.which("ocrmypdf"):
                tmpdir = tempfile.mkdtemp(prefix="pdfrx-")
                src = os.path.join(tmpdir, "ocr.pdf")
                print("OCR: %d page(s) without text, running ocrmypdf..." % len(empty), file=sys.stderr)
                subprocess.run(["ocrmypdf", "--skip-text", "-q", inp, src], check=True)
                doc.close(); doc = pymupdf.open(src)
                if a.images == "none":
                    print("OCR: note --images none leaves the scanned pixels visible; use --images pixels", file=sys.stderr)
            else:
                print("WARNING: --ocr requested but ocrmypdf is not installed; image-only pages will have no matches", file=sys.stderr)
    else:
        empty = has_text(doc, pages)
        if empty: print("Note: %d page(s) have no extractable text (scans?) - consider --ocr" % len(empty), file=sys.stderr)

    total = 0
    print("== Page text matches ==")
    page_hits = {}
    for p in pages:
        hits = find_page_hits(doc[p], rx, a.span)
        page_hits[p] = hits
        for h in hits:
            total += 1
            print("p%d  %-30r  bbox=%s\n      line: %s" % (p + 1, short(h["text"], 60),
                  " ".join(fmt(r) for r in h["rects"]), short(h["line"])))
    print("\n== Other places ==")
    other = scan_other(doc, rx, pages, a.deep or listing and a.deep)
    for where, val, ms in other:
        print("%s: %d hit(s) %r\n      value: %s" % (where, len(ms), [short(m, 40) for m in ms[:5]], short(val)))
    if a.deep:
        print("  (deep hits are raw PDF syntax; text in content streams is often hex/kerned so may not match even if visible)")
    nother = sum(len(m) for _, _, m in other)
    print("\nSummary: %d page-text match(es) on %d page(s); %d match(es) in %d other location(s)."
          % (total, sum(1 for h in page_hits.values() if h), nother, len(other)))
    if listing:
        return 0

    # ---- strip mode ----
    img = {"none": pymupdf.PDF_REDACT_IMAGE_NONE, "pixels": pymupdf.PDF_REDACT_IMAGE_PIXELS,
           "remove": pymupdf.PDF_REDACT_IMAGE_REMOVE}[a.images]
    fill = (1, 1, 1) if a.fill else False
    nred = 0
    for p, hits in page_hits.items():
        if not hits: continue
        page = doc[p]
        for h in hits:
            for r in (h["lines"] if a.whole_line else h["rects"]):
                if r.is_empty: continue
                page.add_redact_annot(r, fill=fill); nred += 1
        page.apply_redactions(images=img, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    changed_meta = []
    if a.meta:
        md = doc.metadata or {}
        new = {k: (rx.sub("", v) if isinstance(v, str) else v) for k, v in md.items()}
        if new != md:
            doc.set_metadata({k: v for k, v in new.items() if k not in ("format", "encryption")}); changed_meta.append("Info dict")
        xmp = doc.get_xml_metadata()
        if xmp and rx.search(xmp):
            doc.set_xml_metadata(rx.sub("", xmp)); changed_meta.append("XMP")
        toc = doc.get_toc(simple=False); n = 0
        for e in toc:
            if rx.search(e[1]): e[1] = rx.sub("", e[1]); n += 1
        if n: doc.set_toc(toc); changed_meta.append("%d outline title(s)" % n)
        n = 0
        for p in pages:
            for an in doc[p].annots() or []:
                info = an.info; ch = False
                for k in ("content", "title", "subject"):
                    if info.get(k) and rx.search(info[k]): info[k] = rx.sub("", info[k]); ch = True
                if ch: an.set_info(info); an.update(); n += 1
        if n: changed_meta.append("%d annotation(s)" % n)
    if out == inp:
        tmp = out + ".pdfrx-tmp"
        doc.save(tmp, garbage=3, deflate=True); doc.close(); os.replace(tmp, out)
    else:
        doc.save(out, garbage=3, deflate=True); doc.close()
    print("\nChanged: %d redaction area(s)%s on %d page(s)%s -> %s"
          % (nred, " (whole lines)" if a.whole_line else "", sum(1 for h in page_hits.values() if h),
             ("; meta: " + ", ".join(changed_meta)) if changed_meta else ("; meta: nothing" if a.meta else ""), out))
    if other and not a.meta:
        print("Note: %d non-page-text hit(s) left untouched (use --meta)." % len(other))
    if tmpdir: print("OCR temp file kept at %s" % src)
    return 0

if __name__ == "__main__":
    sys.exit(main())
