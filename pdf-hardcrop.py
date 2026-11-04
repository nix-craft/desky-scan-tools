#!/usr/bin/env python3
"""pdf-hardcrop.py - cut hidden pixels out of the images in a cropped PDF.

A pure cropper/slicer: every image keeps its original encoding, colour
space and bit depth (JBIG2 stays lossless JBIG2, Flate stays Flate, JPEG is
cropped losslessly with jpegtran).  The page's own content stream is kept;
only image XObjects (and the cm that places them) change.  Text and vector
drawing are untouched; boxes (CropBox, MediaBox, /KropFullBox) are kept.

Usage:
  pdf-hardcrop.py CROPPED.pdf OUTPUT.pdf
  pdf-hardcrop.py --boxes-from CROPPED.pdf SOURCE.pdf OUTPUT.pdf
      (take CropBox/MediaBox/KropFullBox from CROPPED, pixels from SOURCE)
  -q  quiet (summary only)
"""
import sys, os, io, math, zlib, subprocess, tempfile, argparse
import pikepdf
from pikepdf import Name, Dictionary, Array, Stream
import numpy as np

EPS = 1e-6

# ---------------------------------------------------------------- matrices
def mul(m, n):
    a, b, c, d, e, f = m; A, B, C, D, E, F = n
    return (a*A + b*C, a*B + b*D, c*A + d*C, c*B + d*D, e*A + f*C + E, e*B + f*D + F)

def inv(m):
    a, b, c, d, e, f = m
    det = a*d - b*c
    if abs(det) < 1e-12:
        return None
    ia, ib, ic, id_ = d/det, -b/det, -c/det, a/det
    return (ia, ib, ic, id_, -(e*ia + f*ic), -(e*ib + f*id_))

def apply(m, x, y):
    a, b, c, d, e, f = m
    return (a*x + c*y + e, b*x + d*y + f)

def rect(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def isect(r, s):
    return (max(r[0], s[0]), max(r[1], s[1]), min(r[2], s[2]), min(r[3], s[3]))

# ---------------------------------------------------------------- images
def filters(st):
    f = st.get("/Filter")
    if f is None: return []
    return [str(x) for x in f] if isinstance(f, Array) else [str(f)]

def ncomp(cs):
    if cs is None: return None
    if isinstance(cs, Name):
        return {"/DeviceGray": 1, "/CalGray": 1, "/G": 1, "/DeviceRGB": 3, "/RGB": 3,
                "/CalRGB": 3, "/Lab": 3, "/DeviceCMYK": 4, "/CMYK": 4}.get(str(cs))
    if isinstance(cs, Array) and len(cs):
        k = str(cs[0])
        if k in ("/Indexed", "/I", "/Separation"): return 1
        if k == "/ICCBased": return int(cs[1].get("/N"))
        if k == "/DeviceN": return len(cs[1])
        return ncomp(cs[0]) if len(cs) == 1 else {"/CalRGB": 3, "/Lab": 3, "/CalGray": 1}.get(k)
    return None

def crop_bits(raw, W, H, bits_per_px, L, T, R, B):
    """Crop packed sample rows (each row padded to a byte) to columns L..R, rows T..B."""
    stride = (W * bits_per_px + 7) // 8
    a = np.frombuffer(raw[:stride*H], np.uint8).reshape(H, stride)[T:B]
    if bits_per_px % 8 == 0:
        k = bits_per_px // 8
        return np.ascontiguousarray(a[:, L*k:R*k]).tobytes()
    bits = np.unpackbits(a, axis=1)[:, L*bits_per_px:R*bits_per_px]
    return np.packbits(bits, axis=1).tobytes()

class Unsupported(Exception):
    pass

def copy_keys(src, dst, skip):
    for k, v in src.items():
        if k not in skip:
            dst[k] = v

SKIP = {"/Filter", "/DecodeParms", "/DP", "/Width", "/Height", "/Length", "/SMask", "/Mask", "/F"}

def crop_image(pdf, st, L, T, R, B, tmp):
    """Return (new_stream, (L,T,R,B) actually used). Encoding is preserved."""
    W, H = int(st.Width), int(st.Height)
    fl = filters(st)
    if len(fl) != 1 and fl:
        raise Unsupported("filter chain " + "+".join(fl))
    f = fl[0] if fl else None
    mask = bool(st.get("/ImageMask", False))
    bpc = 1 if mask else int(st.get("/BitsPerComponent", 8))
    if f in ("/FlateDecode", "/Fl", None):
        n = 1 if mask else ncomp(st.get("/ColorSpace"))
        if n is None: raise Unsupported("colour space %s" % st.get("/ColorSpace"))
        data = crop_bits(st.read_bytes(), W, H, n*bpc, L, T, R, B)
        new = Stream(pdf, zlib.compress(data, 9))
        copy_keys(st, new, SKIP)
        new.Filter = Name.FlateDecode
    elif f in ("/JBIG2Decode",):
        if bpc != 1: raise Unsupported("JBIG2 with BitsPerComponent %d" % bpc)
        im = pikepdf.PdfImage(st).as_pil_image()                    # decodes via jbig2dec (+Globals)
        if im.size != (W, H) or im.mode != "1": raise Unsupported("JBIG2 decode gave %s %s" % (im.mode, im.size))
        a = np.array(im, dtype=bool)[T:B, L:R]                       # True = sample 1
        w, h = R - L, B - T
        pbm = os.path.join(tmp, "c.pbm")                             # PBM: 1 = black = sample 0
        with open(pbm, "wb") as fh:
            fh.write(b"P4\n%d %d\n" % (w, h) + np.packbits(~a, axis=1).tobytes())
        jb = subprocess.run(["jbig2", "-p", pbm], check=True, capture_output=True).stdout
        new = Stream(pdf, jb)
        copy_keys(st, new, SKIP)
        new.Filter = Name.JBIG2Decode
    elif f in ("/DCTDecode", "/DCT"):
        src = os.path.join(tmp, "s.jpg"); dst = os.path.join(tmp, "d.jpg")
        with open(src, "wb") as fh: fh.write(st.read_raw_bytes())
        subprocess.run(["jpegtran", "-copy", "all", "-crop", "%dx%d+%d+%d" % (R-L, B-T, L, T),
                        "-outfile", dst, src], check=True)
        jpg = open(dst, "rb").read()
        nw, nh = jpeg_size(jpg)
        L, T = R - nw, B - nh            # jpegtran moves the top-left corner onto the iMCU grid
        if L < 0 or T < 0: raise Unsupported("unexpected jpegtran geometry")
        new = Stream(pdf, jpg)
        copy_keys(st, new, SKIP)
        new.Filter = Name.DCTDecode
        if "/DecodeParms" in st: new.DecodeParms = st.DecodeParms
    else:
        raise Unsupported("filter %s" % f)
    new.Width, new.Height = R - L, B - T
    return new, (L, T, R, B)

def jpeg_size(b):
    i = 2
    while i < len(b):
        if b[i] != 0xFF: i += 1; continue
        m = b[i+1]
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return int.from_bytes(b[i+7:i+9], "big"), int.from_bytes(b[i+5:i+7], "big")
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7: i += 2; continue
        i += 2 + int.from_bytes(b[i+2:i+4], "big")
    raise Unsupported("no JPEG SOF")

def scale_rect(r, W, H, W2, H2):
    L, T, R, B = r
    if W2 == W and H2 == H: return r
    sx, sy = W2 / W, H2 / H
    q = (L*sx, T*sy, R*sx, B*sy)
    if all(abs(v - round(v)) < 1e-9 for v in q): return tuple(int(round(v)) for v in q)
    return None

# ---------------------------------------------------------------- processor
class HardCropper:
    def __init__(self, pdf, verbose=True):
        self.pdf, self.verbose = pdf, verbose
        self.cache = {}          # (objgen, rect) -> new image stream
        self.tmp = tempfile.mkdtemp(prefix="hardcrop-")
        self.stats = dict(kept=0, cropped=0, removed=0, skipped=0, before=0, after=0)
        self.seen_src, self.seen_new = set(), set()

    def log(self, s):
        if self.verbose: print(s)

    def pixel_rect(self, ctm, vis, W, H):
        """Visible pixel rect (L,T,R,B) of an image drawn with ctm, rounded outward."""
        im = inv(ctm)
        if im is None: return None
        pts = [apply(im, x, y) for x in (vis[0], vis[2]) for y in (vis[1], vis[3])]
        u0 = min(p[0] for p in pts); u1 = max(p[0] for p in pts)
        v0 = min(p[1] for p in pts); v1 = max(p[1] for p in pts)
        L = max(0, math.floor(u0 * W + EPS)); R = min(W, math.ceil(u1 * W - EPS))
        T = max(0, math.floor((1 - v1) * H + EPS)); B = min(H, math.ceil((1 - v0) * H - EPS))
        return (L, T, R, B)

    def new_image(self, st, r):
        key = (st.objgen, r)
        if key in self.cache: return self.cache[key]
        new, r2 = crop_image(self.pdf, st, *r, self.tmp)
        W, H = int(st.Width), int(st.Height)
        for mk in ("/SMask", "/Mask"):
            m = st.get(mk)
            if isinstance(m, Stream):
                mr = scale_rect(r2, W, H, int(m.Width), int(m.Height))
                if mr is None: raise Unsupported(mk[1:] + " grid does not align with image")
                mnew, mr2 = crop_image(self.pdf, m, *mr, self.tmp)
                if mr2 != mr: raise Unsupported(mk[1:] + " crop moved (JPEG grid)")
                new[mk] = self.pdf.make_indirect(mnew)
            elif m is not None:
                new[mk] = m
        new = self.pdf.make_indirect(new)
        self.cache[key] = (new, r2)
        return new, r2

    def process(self, stream_owner, resources, ctm, vis, where, depth=0):
        """Rewrite one content stream (page or form). Returns (instructions, resources, changed)."""
        instrs = pikepdf.parse_content_stream(stream_owner)
        xobjs = resources.get("/XObject") if resources is not None else None
        out, stack, changed = [], [], False
        new_xobj = {}
        for ins in instrs:
            op = str(ins.operator)
            if op == "q": stack.append(ctm)
            elif op == "Q": ctm = stack.pop() if stack else ctm
            elif op == "cm":
                ctm = mul(tuple(float(v) for v in ins.operands), ctm)
            elif op == "INLINE IMAGE":
                ii = ins.iimage
                try:
                    W, H = int(ii.width), int(ii.height)
                    r = self.pixel_rect(ctm, vis, W, H)
                except Exception:
                    r = None
                if r and (r[2] <= r[0] or r[3] <= r[1]):
                    self.log(f"  {where}: inline image {W}x{H}: removed (outside)")
                    self.stats["removed"] += 1; changed = True; continue
                self.log(f"  {where}: inline image: kept (inline images are not sliced)")
                self.stats["kept"] += 1
            elif op == "Do" and xobjs is not None:
                nm = ins.operands[0]
                xo = xobjs.get(nm)
                if isinstance(xo, Stream) and str(xo.get("/Subtype")) == "/Image":
                    res = self.do_image(xo, nm, ctm, vis, where, xobjs, new_xobj)
                    if res == "remove": changed = True; continue
                    if res is not None:
                        out.extend(res); changed = True; continue
                elif isinstance(xo, Stream) and str(xo.get("/Subtype")) == "/Form" and depth < 20:
                    fm = tuple(float(v) for v in xo.get("/Matrix", [1, 0, 0, 1, 0, 0]))
                    fres = xo.get("/Resources", resources)
                    sub, nres, ch = self.process(xo, fres, mul(fm, ctm), vis, f"{where}>{nm}", depth+1)
                    if ch:
                        nf = Stream(self.pdf, pikepdf.unparse_content_stream(sub))
                        copy_keys(xo, nf, {"/Length", "/Filter", "/DecodeParms", "/Resources"})
                        nf.Resources = nres
                        nf = self.pdf.make_indirect(nf)
                        fname = self.fresh(xobjs, new_xobj, "HcF")
                        new_xobj[fname] = nf
                        out.append(pikepdf.ContentStreamInstruction([Name(fname)], pikepdf.Operator("Do")))
                        changed = True; continue
            out.append(ins)
        if changed:
            used = {str(i.operands[0]) for i in out
                    if not isinstance(i, pikepdf.ContentStreamInlineImage) and str(i.operator) == "Do"}
            resources = Dictionary(dict(resources.items()))
            xd = Dictionary()
            for k, v in list(xobjs.items()) + list(new_xobj.items()):
                if k in used: xd[k] = v
            resources.XObject = xd
        return out, resources, changed

    def fresh(self, xobjs, new_xobj, base):
        i = 0
        while True:
            n = f"/{base}{i}"
            if n not in xobjs and n not in new_xobj: return n
            i += 1

    def do_image(self, xo, nm, ctm, vis, where, xobjs, new_xobj):
        W, H = int(xo.Width), int(xo.Height)
        enc = "+".join(f[1:] for f in filters(xo)) or "raw"
        size = len(xo.read_raw_bytes())
        if xo.objgen not in self.seen_src:
            self.seen_src.add(xo.objgen); self.stats["before"] += size
        tag = f"  {where}: {nm} {W}x{H} {enc} {size:,} B"
        a, b, c, d = ctm[:4]
        if not (abs(b) < EPS and abs(c) < EPS) and not (abs(a) < EPS and abs(d) < EPS):
            self.log(f"{tag}: skipped (rotated/skewed placement)"); self.stats["skipped"] += 1
            self.count_after(xo); return None
        r = self.pixel_rect(ctm, vis, W, H)
        if r is None:
            self.log(f"{tag}: skipped (degenerate matrix)"); self.stats["skipped"] += 1; return None
        L, T, R, B = r
        if R <= L or B <= T:
            self.log(f"{tag}: removed (outside visible area)"); self.stats["removed"] += 1
            return "remove"
        if r == (0, 0, W, H):
            self.log(f"{tag}: kept (fully visible)"); self.stats["kept"] += 1
            self.count_after(xo); return None
        try:
            new, (L, T, R, B) = self.new_image(xo, r)
        except (Unsupported, subprocess.CalledProcessError, pikepdf.PdfError) as e:
            self.log(f"{tag}: skipped ({e})"); self.stats["skipped"] += 1
            self.count_after(xo); return None
        nsize = len(new.read_raw_bytes())
        self.count_after(new)
        self.log(f"{tag}: cropped to [{L},{T},{R},{B}] {R-L}x{B-T}, {nsize:,} B")
        self.stats["cropped"] += 1
        name = None
        for k, v in new_xobj.items():
            if v.objgen == new.objgen: name = k
        if name is None:
            name = self.fresh(xobjs, new_xobj, "HcIm"); new_xobj[name] = new
        sx, sy = (R - L) / W, (B - T) / H
        tx, ty = L / W, 1 - B / H
        I, O = pikepdf.ContentStreamInstruction, pikepdf.Operator
        return [I([], O("q")), I([sx, 0, 0, sy, tx, ty], O("cm")),
                I([Name(name)], O("Do")), I([], O("Q"))]

    def count_after(self, st):
        if st.objgen not in self.seen_new:
            self.seen_new.add(st.objgen); self.stats["after"] += len(st.read_raw_bytes())

    def page(self, i, page):
        mb = rect(page.mediabox)
        cb = rect(page.cropbox)
        vis = isect(cb, mb)
        self.log(f"page {i+1}: visible {tuple(round(v, 2) for v in vis)}")
        res = page.obj.get("/Resources") or Dictionary()
        if "/Resources" not in page.obj:
            # inherited resources
            res = page.resources
        instrs, nres, ch = self.process(page.obj, res, (1, 0, 0, 1, 0, 0), vis, f"p{i+1}")
        if ch:
            page.obj.Contents = self.pdf.make_stream(pikepdf.unparse_content_stream(instrs))
            page.obj.Resources = nres

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--boxes-from", metavar="CROPPED")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if os.path.abspath(a.input) == os.path.abspath(a.output):
        sys.exit("refusing to overwrite the input")
    pdf = pikepdf.open(a.input)
    if a.boxes_from:
        crp = pikepdf.open(a.boxes_from)
        if len(crp.pages) != len(pdf.pages): sys.exit("page counts differ")
        for p, q in zip(pdf.pages, crp.pages):
            for k in ("/MediaBox", "/CropBox", "/KropFullBox"):
                if k in q.obj: p.obj[k] = Array([float(v) for v in q.obj[k]])
            if "/CropBox" not in q.obj and "/CropBox" in p.obj: del p.obj["/CropBox"]
    hc = HardCropper(pdf, verbose=not a.quiet)
    for i, p in enumerate(pdf.pages):
        hc.page(i, p)
    pdf.remove_unreferenced_resources()
    pdf.save(a.output, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
    s = hc.stats
    print(f"images: {s['cropped']} cropped, {s['kept']} kept, {s['removed']} removed, "
          f"{s['skipped']} skipped; image bytes {s['before']:,} -> {s['after']:,}; "
          f"file {os.path.getsize(a.input):,} -> {os.path.getsize(a.output):,} B")
    return hc

if __name__ == "__main__":
    main()
