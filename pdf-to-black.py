#!/usr/bin/env python3
"""Flatten vector fill/stroke colors in a PDF to DeviceGray black (or keep white).

Walks page content streams, Form XObjects, tiling-pattern paint streams,
annotation appearance streams and Type 3 glyph procs.  Each stream is
rewritten at most once.  Non-white fill/stroke colors become DeviceGray 0
(black); colors whose luminance is >= --white-threshold (default 0.95) stay
white so knockouts and paper-colored fills are not blackened.  Use
--no-keep-white to force every color, including white, to black.

Color-space setters (cs/CS) are replaced with /DeviceGray, and g/G, rg/RG,
k/K, sc/SC, scn/SCN are rewritten to 0 g / 1 g (or G for stroke).
Pattern color spaces and scn/SCN pattern paints are left unchanged and
reported.  Shading paints (sh) and images are left unchanged and counted.

Usage: pdf-to-black.py SOURCE.pdf OUTPUT.pdf [--keep-white]
                         [--no-keep-white] [--white-threshold T]
                         [--no-type3] [-q]
"""
import argparse, os, sys, zlib
import pikepdf
from pikepdf import Name, Operator

N = Name
sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)

FILL_OPS = {"g", "rg", "k", "sc", "scn"}
STROKE_OPS = {"G", "RG", "K", "SC", "SCN"}
GRAY_OP = {"g": "g", "rg": "g", "k": "g", "sc": "g", "scn": "g",
           "G": "G", "RG": "G", "K": "G", "SC": "G", "SCN": "G"}


def _f(x):
    return float(x)

def _nums(operands):
    out = []
    for x in operands:
        if isinstance(x, Name):
            break
        try:
            out.append(_f(x))
        except (TypeError, ValueError):
            break
    return out

def _lum_rgb(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _cmyk_lum(c, m, y, k):
    r, g, b = 1 - min(1, c + k), 1 - min(1, m + k), 1 - min(1, y + k)
    return _lum_rgb(r, g, b)


def _kind_of(cs, stack):
    """Return ('gray',), ('rgb',), ('cmyk',), ('lab',), ('indexed', base, table, hival, n),
    ('pattern',), ('sep',), ('unknown',) after following names and ICC / Alternate."""
    if cs is None:
        return ("gray",)
    if isinstance(cs, Name):
        if cs == N.DeviceGray or cs == N.CalGray:
            return ("gray",)
        if cs == N.DeviceRGB or cs == N.CalRGB:
            return ("rgb",)
        if cs == N.DeviceCMYK:
            return ("cmyk",)
        if cs == N.Lab:
            return ("lab",)
        if cs == N.Pattern:
            return ("pattern",)
        for res in stack:
            if res is None or "/ColorSpace" not in res:
                continue
            d = res.ColorSpace
            if cs in d:
                return _kind_of(d[cs], stack)
        return ("unknown",)
    if isinstance(cs, pikepdf.Array) and len(cs) >= 1:
        tag = cs[0]
        if tag == N.ICCBased:
            icc = cs[1]
            n = int(icc.N) if "/N" in icc else None
            alt = icc.get("/Alternate")
            if n == 1 or alt in (N.DeviceGray, N.CalGray):
                return ("gray",)
            if n == 3 or alt in (N.DeviceRGB, N.CalRGB):
                return ("rgb",)
            if n == 4 or alt == N.DeviceCMYK:
                return ("cmyk",)
            if alt is not None:
                return _kind_of(alt, stack)
            return ("unknown",)
        if tag == N.CalGray:
            return ("gray",)
        if tag == N.CalRGB:
            return ("rgb",)
        if tag == N.Lab:
            return ("lab",)
        if tag == N.Indexed:
            base = cs[1]
            hival = int(cs[2])
            lookup = cs[3]
            table = bytes(lookup.read_bytes()) if isinstance(lookup, pikepdf.Stream) else bytes(lookup)
            bk = _kind_of(base, stack)
            ncomp = {"gray": 1, "rgb": 3, "cmyk": 4, "lab": 3}.get(bk[0], 1)
            return ("indexed", bk, table, hival, ncomp)
        if tag == N.Pattern:
            return ("pattern",)
        if tag in (N.Separation, N.DeviceN):
            return ("sep",)
    return ("unknown",)


def _indexed_comps(kind, index):
    _, base, table, hival, ncomp = kind
    i = max(0, min(int(round(index)), hival))
    off = i * ncomp
    if off + ncomp > len(table):
        return None
    return [table[off + j] / 255.0 for j in range(ncomp)], base


def luminance(kind, operands):
    """Return luminance in 0..1, or None if it cannot be resolved (caller -> black)."""
    vals = _nums(operands)
    k = kind[0]
    if k == "pattern":
        return None
    if k == "indexed":
        if not vals:
            return None
        got = _indexed_comps(kind, vals[0])
        if got is None:
            return None
        comps, base = got
        return luminance(base, comps)
    if k == "sep":
        if not vals or max(abs(v) for v in vals) < 1e-6:
            return 1.0
        return 0.0
    if k == "gray":
        return vals[0] if vals else None
    if k == "rgb":
        return _lum_rgb(*vals[:3]) if len(vals) >= 3 else None
    if k == "cmyk":
        return _cmyk_lum(*vals[:4]) if len(vals) >= 4 else None
    if k == "lab":
        return (vals[0] / 100.0) if vals else None
    return None


def _trailing_pattern(operands):
    return bool(operands) and isinstance(operands[-1], Name)


def _gray_operand(keep_white, thresh, lum):
    if keep_white and lum is not None and lum >= thresh:
        return 1
    return 0


def rewrite_ops(instructions, stack, keep_white, thresh):
    fill_cs, stroke_cs = ("gray",), ("gray",)
    saved = []
    out = []
    st = dict(changed=0, cs=0, fill=0, stroke=0, kept_white=0,
              pattern_left=0, shading_left=0, unresolved=0)
    for operands, operator in instructions:
        op = str(operator)
        if op == "q":
            saved.append((fill_cs, stroke_cs))
            out.append((operands, operator))
            continue
        if op == "Q":
            if saved:
                fill_cs, stroke_cs = saved.pop()
            out.append((operands, operator))
            continue
        if op in ("cs", "CS"):
            kind = _kind_of(operands[0] if operands else N.DeviceGray, stack)
            if kind[0] == "pattern":
                if op == "cs":
                    fill_cs = kind
                else:
                    stroke_cs = kind
                out.append((operands, operator))
                st["pattern_left"] += 1
                continue
            if op == "cs":
                fill_cs = kind
            else:
                stroke_cs = kind
            new = ([N.DeviceGray], Operator(op))
            if not (len(operands) == 1 and operands[0] == N.DeviceGray):
                st["changed"] += 1
                st["cs"] += 1
            out.append(new)
            continue
        if op in FILL_OPS or op in STROKE_OPS:
            is_fill = op in FILL_OPS
            implied = {"g": ("gray",), "G": ("gray",), "rg": ("rgb",), "RG": ("rgb",),
                       "k": ("cmyk",), "K": ("cmyk",)}
            if op in implied:
                kind = implied[op]
                if is_fill:
                    fill_cs = kind
                else:
                    stroke_cs = kind
            else:
                kind = fill_cs if is_fill else stroke_cs
            if kind[0] == "pattern" or (op in ("scn", "SCN") and _trailing_pattern(operands)):
                out.append((operands, operator))
                st["pattern_left"] += 1
                continue
            lum = luminance(kind, operands)
            if lum is None and kind[0] != "gray":
                st["unresolved"] += 1
            gray = _gray_operand(keep_white, thresh, lum)
            new_op = GRAY_OP[op]
            already = (op == new_op and len(operands) == 1 and not isinstance(operands[0], Name)
                       and abs(_f(operands[0]) - gray) < 1e-9)
            if already:
                out.append((operands, operator))
                continue
            out.append(([gray], Operator(new_op)))
            st["changed"] += 1
            st[("fill" if is_fill else "stroke")] += 1
            if gray == 1:
                st["kept_white"] += 1
            continue
        if op == "sh":
            st["shading_left"] += 1
        out.append((operands, operator))
    return out, st


def _write_stream(obj, data):
    if "/DecodeParms" in obj:
        del obj["/DecodeParms"]
    obj.write(zlib.compress(data), filter=N.FlateDecode)


def _page_contents_write(pdf, page, data):
    c = page.get("/Contents")
    if c is None or isinstance(c, pikepdf.Array):
        page.Contents = pdf.make_stream(zlib.compress(data), filter=N.FlateDecode)
    else:
        _write_stream(c, data)


def _res_stack(res, parent):
    if res is None:
        return parent
    return (res,) + tuple(r for r in parent if r is not res)


def _iter_xo_and_patterns(res):
    if res is None:
        return
    if "/XObject" in res:
        for name, xo in res.XObject.items():
            yield "xo", name, xo
    if "/Pattern" in res:
        for name, pat in res.Pattern.items():
            yield "pattern", name, pat
    if "/Font" in res:
        for name, font in res.Font.items():
            yield "font", name, font


def _appearance_streams(annot):
    ap = annot.get("/AP")
    if ap is None:
        return
    for key in ("/N", "/R", "/D"):
        if key not in ap:
            continue
        val = ap[key]
        if isinstance(val, pikepdf.Stream):
            yield f"AP{key}", val
        elif isinstance(val, pikepdf.Dictionary):
            for n, v in val.items():
                if isinstance(v, pikepdf.Stream):
                    yield f"AP{key}{n}", v


def flatten_pdf(pdf, keep_white=True, thresh=0.95, type3=True):
    """Rewrite color operators in pdf (in place). Return (reports, totals)."""
    seen = set()
    reports = []
    images = 0
    shadings = 0

    def do_stream(label, obj, stack, parse_src=None):
        nonlocal images
        try:
            og = obj.objgen
        except Exception:
            og = None
        if og is not None and og in seen:
            return
        if og is not None:
            seen.add(og)
        src = parse_src if parse_src is not None else obj
        try:
            instr = pikepdf.parse_content_stream(src)
        except Exception as e:
            reports.append((label, None, f"parse failed: {e}"))
            return
        new, st = rewrite_ops(instr, stack, keep_white, thresh)
        if st["changed"]:
            data = pikepdf.unparse_content_stream(new)
            if parse_src is not None:
                _page_contents_write(pdf, obj, data)
            else:
                _write_stream(obj, data)
        reports.append((label, st, None))
        # nested resources
        res = None
        if parse_src is not None:
            res = obj.get("/Resources")
        elif isinstance(obj, pikepdf.Stream) or isinstance(obj, pikepdf.Dictionary):
            res = obj.get("/Resources") if "/Resources" in obj else None
        walk_res(res, stack, label)

    def walk_res(res, parent, label):
        nonlocal images, shadings
        if res is None:
            return
        if "/Shading" in res:
            try:
                shadings += len(list(res.Shading.keys()))
            except Exception:
                shadings += 1
        stack = _res_stack(res, parent)
        for kind, name, obj in _iter_xo_and_patterns(res):
            if kind == "xo":
                sub = obj.get("/Subtype")
                if sub == N.Image:
                    try:
                        og = obj.objgen
                    except Exception:
                        og = id(obj)
                    if og not in seen:
                        seen.add(og)
                        images += 1
                    continue
                if sub == N.Form:
                    do_stream(f"{label} form {name}", obj, stack)
            elif kind == "pattern":
                ptype = obj.get("/PatternType")
                if ptype == 1 and isinstance(obj, pikepdf.Stream):
                    do_stream(f"{label} tiling {name}", obj, stack)
                elif ptype == 2:
                    reports.append((f"{label} shading-pattern {name}",
                                    dict(changed=0, cs=0, fill=0, stroke=0, kept_white=0,
                                         pattern_left=0, shading_left=1, unresolved=0), None))
            elif kind == "font" and type3 and obj.get("/Subtype") == N.Type3 and "/CharProcs" in obj:
                fres = obj.get("/Resources")
                fstack = _res_stack(fres, stack) if fres is not None else stack
                for gn, proc in obj.CharProcs.items():
                    if isinstance(proc, pikepdf.Stream):
                        do_stream(f"{label} type3 {name} {gn}", proc, fstack)

    for i, page in enumerate(pdf.pages, 1):
        stack = _res_stack(page.get("/Resources"), ())
        if page.get("/Contents") is not None:
            do_stream(f"page {i}", page, stack, parse_src=page)
        else:
            walk_res(page.get("/Resources"), (), f"page {i}")
        annots = page.get("/Annots")
        if annots:
            for j, annot in enumerate(annots, 1):
                for aname, ap in _appearance_streams(annot):
                    ares = ap.get("/Resources") if "/Resources" in ap else None
                    do_stream(f"page {i} annot {j} {aname}", ap, _res_stack(ares, stack))

    totals = dict(changed=0, cs=0, fill=0, stroke=0, kept_white=0,
                  pattern_left=0, shading_left=0, unresolved=0,
                  images=images, streams_changed=0)
    for label, st, err in reports:
        if not st:
            continue
        for k in ("changed", "cs", "fill", "stroke", "kept_white", "pattern_left",
                  "shading_left", "unresolved"):
            totals[k] += st[k]
        if st["changed"]:
            totals["streams_changed"] += 1
    # resource /Shading counts are in addition to sh operators already tallied
    # (do not double-count; shadings-left in totals already includes 'sh' ops)
    totals["shading_dicts"] = shadings
    return reports, totals


def _fmt(st):
    bits = [f"{st['changed']} ops rewritten (cs {st['cs']}, fill {st['fill']}, stroke {st['stroke']})"]
    if st["kept_white"]:
        bits.append(f"kept white {st['kept_white']}")
    if st["pattern_left"]:
        bits.append(f"patterns left {st['pattern_left']}")
    if st["shading_left"]:
        bits.append(f"shadings left {st['shading_left']}")
    if st["unresolved"]:
        bits.append(f"unresolved->black {st['unresolved']}")
    return "; ".join(bits)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--keep-white", dest="keep_white", action="store_true", default=True,
                   help="keep near-white fills/strokes (default)")
    p.add_argument("--no-keep-white", dest="keep_white", action="store_false",
                   help="turn every color, including white, to black")
    p.add_argument("--white-threshold", type=float, default=0.95,
                   help="luminance at or above this stays white (default 0.95)")
    p.add_argument("--no-type3", action="store_true", help="skip Type 3 glyph procs")
    p.add_argument("-q", action="store_true", help="quiet: one-line summary only")
    a = p.parse_args(argv)
    if os.path.abspath(a.src) == os.path.abspath(a.out):
        p.error("refusing to overwrite input")
    if os.path.exists(a.out):
        p.error(f"{a.out} exists; refusing to overwrite")
    if not 0 <= a.white_threshold <= 1:
        p.error("--white-threshold must be between 0 and 1")
    pdf = pikepdf.open(a.src)
    reports, tot = flatten_pdf(pdf, keep_white=a.keep_white, thresh=a.white_threshold,
                               type3=not a.no_type3)
    if not a.q:
        for label, st, err in reports:
            if err:
                print(f"{label}: {err}")
            elif st and (st["changed"] or st["pattern_left"] or st["shading_left"] or st["unresolved"]):
                print(f"{label}: {_fmt(st)}")
    extra = []
    if tot["images"]:
        extra.append(f"{tot['images']} images left")
    if tot["shading_dicts"]:
        extra.append(f"{tot['shading_dicts']} shading dicts in resources")
    suffix = ("; " + ", ".join(extra)) if extra else ""
    print(f"total: {_fmt(tot)}; {tot['streams_changed']} streams changed{suffix}")
    pdf.save(a.out, deterministic_id=True)


if __name__ == "__main__":
    main()
