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
import sys, os, io, math, zlib, base64, subprocess, tempfile, argparse
import pikepdf
from pikepdf import Name, Dictionary, Array, Stream
from fractions import Fraction as Fr
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
    if len(raw) < stride * H: raise Unsupported("short image data")
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

# ---------------------------------------------------------------- filters
# "transport" filters are re-encoded losslessly with the same filter (and the
# same predictor); "codec" filters must be last in a chain and are cropped in
# their own format.  Nothing is ever converted to a different encoding.
FULL = {"/Fl": "/FlateDecode", "/LZW": "/LZWDecode", "/RL": "/RunLengthDecode",
        "/AHx": "/ASCIIHexDecode", "/A85": "/ASCII85Decode", "/DCT": "/DCTDecode",
        "/CCF": "/CCITTFaxDecode"}
GEN = {"/FlateDecode", "/LZWDecode", "/RunLengthDecode", "/ASCIIHexDecode", "/ASCII85Decode"}
CODEC = {"/DCTDecode": "dct", "/JBIG2Decode": "jbig2", "/CCITTFaxDecode": "ccitt", "/JPXDecode": "jpx"}

def parms_list(st, n):
    dp = st.get("/DecodeParms", st.get("/DP"))
    if n == 0: return []
    if isinstance(dp, Array): l = [x if isinstance(x, Dictionary) else None for x in dp]
    else: l = [dp if isinstance(dp, Dictionary) else None]
    return (l + [None] * n)[:n]

def pget(p, k, d):
    return d if p is None or k not in p else p[k]

def make_parms(pl):
    if not any(p is not None for p in pl): return None
    return pl[0] if len(pl) == 1 else Array(list(pl))

def pred_encode(data, p):
    pr = int(pget(p, "/Predictor", 1))
    if pr == 1: return data
    colors = int(pget(p, "/Colors", 1)); bpc = int(pget(p, "/BitsPerComponent", 8))
    cols = int(pget(p, "/Columns", 1))
    rowlen = (colors * bpc * cols + 7) // 8
    if len(data) % rowlen: raise Unsupported("predictor row length")
    a = np.frombuffer(data, np.uint8).reshape(-1, rowlen)
    if pr >= 10:                                   # PNG: any per-row type is valid; use Up
        up = a.copy(); up[1:] = a[1:] - a[:-1]
        return np.hstack([np.full((len(a), 1), 2, np.uint8), up]).tobytes()
    if pr == 2 and bpc in (8, 16):                 # TIFF horizontal differencing
        dt = np.uint8 if bpc == 8 else np.dtype(">u2")
        b = a.view(dt).reshape(len(a), cols, colors)
        d = b.copy(); d[:, 1:] = b[:, 1:] - b[:, :-1]
        return d.astype(dt).tobytes()
    raise Unsupported("predictor %d with %d bpc" % (pr, bpc))

def lzw_encode(data, early=1):
    out = bytearray(); acc = 0; nb = 0
    def put(c, w):
        nonlocal acc, nb
        acc = (acc << w) | c; nb += w
        while nb >= 8:
            nb -= 8; out.append((acc >> nb) & 255)
        acc &= (1 << nb) - 1
    tab = {}; nxt = 258; w = 9; put(256, w); code = None
    for b in data:
        if code is None: code = b; continue
        k = (code << 8) | b; c = tab.get(k)
        if c is not None: code = c; continue
        put(code, w); tab[k] = nxt; nxt += 1
        if nxt + early > (1 << w) and w < 12: w += 1
        if nxt >= 4094: put(256, w); tab = {}; nxt = 258; w = 9
        code = b
    if code is not None: put(code, w)
    put(257, w)
    if nb: out.append((acc << (8 - nb)) & 255)
    return bytes(out)

def rl_encode(d):
    out = bytearray(); i = 0; n = len(d)
    while i < n:
        j = i + 1
        while j < n and j - i < 128 and d[j] == d[i]: j += 1
        if j - i >= 2:
            out += bytes([257 - (j - i), d[i]]); i = j; continue
        j = i + 1
        while j < n and j - i < 128 and not (j + 1 < n and d[j] == d[j + 1]): j += 1
        out.append(j - i - 1); out += d[i:j]; i = j
    out.append(128)
    return bytes(out)

def encode_one(f, data, p):
    f = FULL.get(f, f)
    if f == "/FlateDecode": return zlib.compress(pred_encode(data, p), 9)
    if f == "/LZWDecode": return lzw_encode(pred_encode(data, p), int(pget(p, "/EarlyChange", 1)))
    if f == "/RunLengthDecode": return rl_encode(data)
    if f == "/ASCIIHexDecode": return data.hex().upper().encode() + b">"
    if f == "/ASCII85Decode": return base64.a85encode(data, wrapcol=76) + b"~>"
    raise Unsupported("filter " + f)

def transport_decode(pdf, raw, tn, tp):
    if not tn: return raw
    t = Stream(pdf, raw)
    t.Filter = Array([Name(FULL.get(f, f)) for f in tn])
    dp = make_parms(tp)
    if dp is not None: t.DecodeParms = Array(list(tp))
    return t.read_bytes(decode_level=pikepdf.StreamDecodeLevel.specialized)

def fix_pred(p, W, w):
    if p is None or int(pget(p, "/Predictor", 1)) == 1: return p
    if int(pget(p, "/Columns", 1)) != W: raise Unsupported("predictor Columns != Width")
    q = Dictionary(dict(p.items())); q.Columns = w
    return q

def crop_image(pdf, st, L, T, R, B, tmp, ncomp_fn=None):
    """Return (new_stream, (L,T,R,B) actually used). Encoding is preserved."""
    W, H = int(st.Width), int(st.Height)
    fl = filters(st); fln = [FULL.get(f, f) for f in fl]
    pl = parms_list(st, len(fl))
    for f in fln:
        if f not in GEN and f not in CODEC: raise Unsupported("filter %s" % f)
    if any(f in CODEC for f in fln[:-1]): raise Unsupported("image codec not last in filter chain")
    codec = CODEC.get(fln[-1]) if fln else None
    nt = len(fl) - (1 if codec else 0)
    tn, tp = fl[:nt], pl[:nt]
    cp = pl[nt] if codec else None
    payload = transport_decode(pdf, st.read_raw_bytes(), tn, tp)
    mask = bool(st.get("/ImageMask", False))
    bpc = 1 if mask else int(st.get("/BitsPerComponent", 8))
    ncp = cp
    if codec is None:
        n = 1 if mask else (ncomp_fn or ncomp)(st.get("/ColorSpace"))
        if n is None: raise Unsupported("colour space %s" % st.get("/ColorSpace"))
        out = crop_bits(payload, W, H, n*bpc, L, T, R, B)
    elif codec == "jbig2":
        if bpc != 1: raise Unsupported("JBIG2 with BitsPerComponent %d" % bpc)
        t = Stream(pdf, payload); copy_keys(st, t, SKIP); t.Width, t.Height = W, H
        t.Filter = Name.JBIG2Decode
        if cp is not None: t.DecodeParms = cp
        im = pikepdf.PdfImage(t).as_pil_image()                     # decodes via jbig2dec (+Globals)
        if im.size != (W, H) or im.mode != "1": raise Unsupported("JBIG2 decode gave %s %s" % (im.mode, im.size))
        a = np.array(im, dtype=bool)[T:B, L:R]                       # True = sample 1
        w, h = R - L, B - T
        pbm = os.path.join(tmp, "c.pbm")                             # PBM: 1 = black = sample 0
        with open(pbm, "wb") as fh:
            fh.write(b"P4\n%d %d\n" % (w, h) + np.packbits(~a, axis=1).tobytes())
        out = subprocess.run(["jbig2", "-p", pbm], check=True, capture_output=True).stdout
        ncp = None                                                   # generic region, no Globals
    elif codec == "dct":
        src = os.path.join(tmp, "s.jpg"); dst = os.path.join(tmp, "d.jpg")
        with open(src, "wb") as fh: fh.write(payload)
        subprocess.run(["jpegtran", "-copy", "all", "-crop", "%dx%d+%d+%d" % (R-L, B-T, L, T),
                        "-outfile", dst, src], check=True, capture_output=True)
        out = open(dst, "rb").read()
        nw, nh = jpeg_size(out)
        L, T = R - nw, B - nh            # jpegtran moves the top-left corner onto the iMCU grid
        if L < 0 or T < 0: raise Unsupported("unexpected jpegtran geometry")
    elif codec == "ccitt":
        out, ncp = ccitt_crop(pdf, payload, cp, W, H, L, T, R, B, tmp)
    elif codec == "jpx":
        out, (L, T, R, B) = jpx_crop(payload, W, H, L, T, R, B)
        if (L, T, R, B) == (0, 0, W, H): raise Unsupported("visible area touches every JPEG 2000 tile")
    w, h = R - L, B - T
    ntp = [fix_pred(p, W, w) for p in tp]
    enc = out
    for f, p in reversed(list(zip(tn, ntp))): enc = encode_one(f, enc, p)
    if tn and transport_decode(pdf, enc, tn, ntp) != out:
        raise Unsupported("lossless re-encode check failed")
    new = Stream(pdf, enc)
    copy_keys(st, new, SKIP)
    if fl: new.Filter = Name(fln[0]) if len(fl) == 1 else Array([Name(f) for f in fln])
    dp = make_parms(ntp + ([ncp] if codec else []))
    if dp is not None: new.DecodeParms = dp
    new.Width, new.Height = w, h
    return new, (L, T, R, B)

# ---------------------------------------------------------------- CCITT
def ccitt_decode(pdf, data, parms, w, h):
    t = Stream(pdf, data); t.Type, t.Subtype = Name.XObject, Name.Image
    t.Width, t.Height = w, h; t.BitsPerComponent = 1; t.ColorSpace = Name.DeviceGray
    t.Filter = Name.CCITTFaxDecode
    if parms is not None: t.DecodeParms = parms
    im = pikepdf.PdfImage(t).as_pil_image()
    if im.size != (w, h): raise Unsupported("CCITT decode gave %s" % (im.size,))
    return np.array(im.convert("1"), dtype=bool)

def ccitt_encode(arr, K, align, tmp):
    from PIL import Image
    h, w = arr.shape
    path = os.path.join(tmp, "c.tif")
    info = {278: h}
    opt = (1 if K > 0 else 0) | (4 if align else 0)
    if K >= 0 and opt: info[292] = opt
    Image.fromarray(arr).save(path, "TIFF", compression="group4" if K < 0 else "group3", tiffinfo=info)
    t = Image.open(path)
    offs, cnts = t.tag_v2[273], t.tag_v2[279]
    offs = offs if isinstance(offs, tuple) else (offs,)
    cnts = cnts if isinstance(cnts, tuple) else (cnts,)
    if len(offs) != 1: raise Unsupported("libtiff wrote %d strips" % len(offs))
    return open(path, "rb").read()[offs[0]:offs[0] + cnts[0]]

def ccitt_crop(pdf, payload, cp, W, H, L, T, R, B, tmp):
    K = int(pget(cp, "/K", 0)); align = bool(pget(cp, "/EncodedByteAlign", False))
    if K < 0 and align: raise Unsupported("CCITT G4 with EncodedByteAlign (no lossless G4 encoder for it)")
    a = ccitt_decode(pdf, payload, cp, W, H)[T:B, L:R]
    h, w = a.shape
    for inv in (False, True):
        try: data = ccitt_encode(~a if inv else a, K, align, tmp)
        except (OSError, ValueError, KeyError) as e: raise Unsupported("CCITT encode: %s" % e)
        for eol in ([None] if K < 0 else [None, True]):
            q = Dictionary(dict(cp.items())) if cp is not None else Dictionary()
            q.Columns, q.Rows = w, h
            if eol is not None: q.EndOfLine = True
            try:
                if np.array_equal(ccitt_decode(pdf, data, q, w, h), a): return data, q
            except Exception:
                pass
    raise Unsupported("CCITT re-encode did not round-trip")

# ---------------------------------------------------------------- JPEG 2000
def _u(b, i, n): return int.from_bytes(b[i:i+n], "big")
def _p(v, n): return int(v).to_bytes(n, "big")

def jp2_boxes(d, start=0, end=None):
    end = len(d) if end is None else end
    i, out = start, []
    while i + 8 <= end:
        ln, typ, hl = _u(d, i, 4), d[i+4:i+8], 8
        if ln == 1: ln, hl = _u(d, i+8, 8), 16
        elif ln == 0: ln = end - i
        out.append((typ, i, hl, i + ln)); i += ln
    return out

def j2k_crop(cs, L, T, R, B):
    if cs[:4] != b"\xff\x4f\xff\x51": raise Unsupported("not a JPEG 2000 codestream")
    Lsiz = _u(cs, 4, 2)
    X, Y, XO, YO, XT, YT, XTO, YTO = [_u(cs, 8 + 4*k, 4) for k in range(8)]
    nx = -(-(X - XTO) // XT); ny = -(-(Y - YTO) // YT)
    p0 = (XO + L - XTO) // XT; p1 = -(-(XO + R - XTO) // XT)
    q0 = (YO + T - YTO) // YT; q1 = -(-(YO + B - YTO) // YT)
    nXTO, nYTO = XTO + p0*XT, YTO + q0*YT
    nXO, nYO = max(XO, nXTO), max(YO, nYTO)
    nX, nY = min(X, XTO + p1*XT), min(Y, YTO + q1*YT)
    pos = 4 + Lsiz; main = []
    while cs[pos:pos+2] != b"\xff\x90":
        m = _u(cs, pos, 2); ln = _u(cs, pos+2, 2)
        if m == 0xFF60: raise Unsupported("JPEG 2000 PPM marker (packed headers)")
        if m not in (0xFF55, 0xFF57): main.append(cs[pos:pos+2+ln])   # drop TLM/PLM (index lists)
        pos += 2 + ln
        if pos >= len(cs): raise Unsupported("JPEG 2000: no tiles")
    parts = []; ntx = p1 - p0
    while cs[pos:pos+2] == b"\xff\x90":
        Isot, Psot = _u(cs, pos+4, 2), _u(cs, pos+6, 4)
        end = pos + Psot if Psot else len(cs) - 2
        p, q = Isot % nx, Isot // nx
        if p0 <= p < p1 and q0 <= q < q1:
            part = bytearray(cs[pos:end])
            part[4:6] = _p((q - q0)*ntx + (p - p0), 2); part[6:10] = _p(len(part), 4)
            parts.append(bytes(part))
        pos = end
    if cs[pos:pos+2] != b"\xff\xd9": raise Unsupported("JPEG 2000: unexpected data after tiles")
    siz = bytearray(cs[2:4+Lsiz])
    for k, v in enumerate((nX, nY, nXO, nYO, XT, YT, nXTO, nYTO)): siz[6+4*k:10+4*k] = _p(v, 4)
    new = b"\xff\x4f" + bytes(siz) + b"".join(main) + b"".join(parts) + b"\xff\xd9"
    return new, (nXO - XO, nYO - YO, nX - XO, nY - YO)

def jpx_crop(data, W, H, L, T, R, B):
    if data[:4] == b"\xff\x4f\xff\x51":
        new, r = j2k_crop(data, L, T, R, B)
    else:
        boxes = jp2_boxes(data)
        if not boxes or boxes[0][0] != b"jP  ": raise Unsupported("JPX: neither JP2 nor codestream")
        out = bytearray(); r = None
        for typ, s, hl, e in boxes:
            if typ == b"jp2c":
                cs, r = j2k_crop(data[s+hl:e], L, T, R, B)
                out += _p(8 + len(cs), 4) + b"jp2c" + cs
            else:
                out += data[s:e]
        if r is None: raise Unsupported("JPX: no codestream box")
        w, h = r[2] - r[0], r[3] - r[1]
        for typ, s, hl, e in jp2_boxes(out):
            if typ == b"jp2h":
                for t2, s2, hl2, e2 in jp2_boxes(out, s + hl, e):
                    if t2 == b"ihdr": out[s2+hl2:s2+hl2+8] = _p(h, 4) + _p(w, 4)
        new = bytes(out)
    try:                                   # verify: same pixels as the original's region
        from PIL import Image
        A = np.asarray(Image.open(io.BytesIO(data))); Bn = np.asarray(Image.open(io.BytesIO(new)))
        L2, T2, R2, B2 = r
        if A.shape[:2] != (H, W) or not np.array_equal(A[T2:B2, L2:R2], Bn):
            raise Unsupported("JPEG 2000 tile crop did not verify")
    except Unsupported: raise
    except Exception as e: raise Unsupported("JPEG 2000 verify failed: %s" % e)
    return new, r

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

    def flate_stream(self, data):
        st = Stream(self.pdf, zlib.compress(data, 9)); st.Filter = Name.FlateDecode
        return st

    def log(self, s):
        if self.verbose: print(s)

    def pixel_rect(self, ctm, vis, W, H):
        """Visible pixel rect (L,T,R,B) of an image drawn with ctm, rounded outward.
        For rotated/skewed placements this is the bounding box (in image pixels) of the
        visible area mapped back into image space: the tightest lossless rectangle."""
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
        W, H = int(st.Width), int(st.Height)
        masks = [(mk, st.get(mk)) for mk in ("/SMask", "/Mask") if isinstance(st.get(mk), Stream)]
        dims = [(W, H)] + [(int(m.Width), int(m.Height)) for _, m in masks]
        gx = math.gcd(*[d[0] for d in dims]); gy = math.gcd(*[d[1] for d in dims])
        def unit(rr, w, h): return (Fr(rr[0], w), Fr(rr[1], h), Fr(rr[2], w), Fr(rr[3], h))
        def pix(u, w, h):
            # outward onto the grid shared by image and masks (multiples of 1/g)
            x0 = Fr(math.floor(u[0] * gx), gx); x1 = Fr(math.ceil(u[2] * gx), gx)
            y0 = Fr(math.floor(u[1] * gy), gy); y1 = Fr(math.ceil(u[3] * gy), gy)
            return (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        U = unit(r, W, H)
        for _ in range(8):
            new, r2 = crop_image(self.pdf, st, *pix(U, W, H), self.tmp)
            U2 = unit(r2, W, H); mres = []; ok = True
            for mk, m in masks:
                mw, mh = int(m.Width), int(m.Height)
                mr = pix(U2, mw, mh)
                mnew, mr2 = crop_image(self.pdf, m, *mr, self.tmp)
                mres.append((mk, mnew))
                if unit(mr2, mw, mh) != U2:
                    ok = False; mu = unit(mr2, mw, mh)
                    U2 = (min(U2[0], mu[0]), min(U2[1], mu[1]), max(U2[2], mu[2]), max(U2[3], mu[3]))
            if ok: break
            U = U2
        else:
            raise Unsupported("image and mask crops do not line up")
        if r2 == (0, 0, W, H): raise Unsupported("mask pixel grid allows no crop")
        for mk, mnew in mres: new[mk] = self.pdf.make_indirect(mnew)
        for mk in ("/SMask", "/Mask"):
            m = st.get(mk)
            if m is not None and not isinstance(m, Stream): new[mk] = m
        new = self.pdf.make_indirect(new)
        self.cache[key] = (new, r2)
        return new, r2

    def cs_resolver(self, resources):
        def f(cs):
            if isinstance(cs, Name) and ncomp(cs) is None and resources is not None:
                csd = resources.get("/ColorSpace")
                if csd is not None and cs in csd: return ncomp(csd[cs])
            return ncomp(cs)
        return f

    ABBR_K = {"/Width": "/W", "/Height": "/H", "/BitsPerComponent": "/BPC", "/ColorSpace": "/CS",
              "/Filter": "/F", "/DecodeParms": "/DP", "/Decode": "/D", "/ImageMask": "/IM",
              "/Interpolate": "/I"}
    ABBR_V = {"/FlateDecode": "/Fl", "/LZWDecode": "/LZW", "/RunLengthDecode": "/RL",
              "/ASCIIHexDecode": "/AHx", "/ASCII85Decode": "/A85", "/DCTDecode": "/DCT",
              "/CCITTFaxDecode": "/CCF", "/DeviceGray": "/G", "/DeviceRGB": "/RGB",
              "/DeviceCMYK": "/CMYK", "/Indexed": "/I"}

    def do_inline(self, ins, ctm, vis, where, resources):
        ii = ins.iimage; d = ii.obj
        try:
            W, H = int(d.Width), int(d.Height)
            r = self.pixel_rect(ctm, vis, W, H)
        except Exception:
            r = None
        tag = f"  {where}: inline image {W}x{H}" if r else f"  {where}: inline image"
        if r is None:
            self.log(f"{tag}: kept (no geometry)"); self.stats["kept"] += 1; return None
        if r[2] <= r[0] or r[3] <= r[1]:
            self.log(f"{tag}: removed (outside)"); self.stats["removed"] += 1; return "remove"
        if r == (0, 0, W, H):
            self.log(f"{tag}: kept (fully visible)"); self.stats["kept"] += 1; return None
        try:
            raw = ii._data._inline_image_raw_bytes()
            t = Stream(self.pdf, raw); copy_keys(d, t, {"/Length"})
            new, (L, T, R, B) = crop_image(self.pdf, t, *r, self.tmp, self.cs_resolver(resources))
            if (L, T, R, B) == (0, 0, W, H): raise Unsupported("crop grid gives no crop")
            def ab(v):
                if isinstance(v, Name): return Name(self.ABBR_V.get(str(v), str(v)))
                if isinstance(v, Array): return Array([ab(x) for x in v])
                return v
            dd = Dictionary()
            for k, v in new.items():
                if k in ("/Length", "/Type", "/Subtype"): continue
                dd[self.ABBR_K.get(k, k)] = ab(v)
            data = new.read_raw_bytes()
            sx, sy = (R - L) / W, (B - T) / H
            head = b"q %s cm\nBI " % " ".join("%.9g" % v for v in (sx, 0, 0, sy, L / W, 1 - B / H)).encode()
            body = dd.unparse().strip()[2:-2].strip()
            cs = self.pdf.make_stream(head + body + b"\nID\n" + data + b"\nEI\nQ")
            res = list(pikepdf.parse_content_stream(cs))
            got = [i.iimage for i in res if isinstance(i, pikepdf.ContentStreamInlineImage)]
            if len(res) != 4 or len(got) != 1 or got[0]._data._inline_image_raw_bytes().rstrip() != data.rstrip():
                raise Unsupported("re-parsed inline image does not match (binary data looks like EI)")
        except (Unsupported, subprocess.CalledProcessError, pikepdf.PdfError, KeyError, ValueError) as e:
            self.log(f"{tag}: skipped ({e})"); self.stats["skipped"] += 1; return None
        self.log(f"{tag}: cropped to [{L},{T},{R},{B}] {R-L}x{B-T}, {len(data):,} B (was {len(raw):,})")
        self.stats["cropped"] += 1
        return res

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
                res = self.do_inline(ins, ctm, vis, where, resources)
                if res == "remove": changed = True; continue
                if res is not None:
                    out.extend(res); changed = True; continue
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
                        nf = self.flate_stream(pikepdf.unparse_content_stream(sub))
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
            for k, v in (list(xobjs.items()) if xobjs is not None else []) + list(new_xobj.items()):
                if k in used: xd[k] = v
            if xobjs is not None or new_xobj: resources.XObject = xd
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
            page.obj.Contents = self.pdf.make_indirect(self.flate_stream(pikepdf.unparse_content_stream(instrs)))
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
    # never let qpdf transcode existing streams (compress_streams would turn LZW into Flate);
    # the content streams we write are Flate-compressed by flate_stream()
    pdf.save(a.output, compress_streams=False, stream_decode_level=pikepdf.StreamDecodeLevel.none,
             object_stream_mode=pikepdf.ObjectStreamMode.generate)
    s = hc.stats
    print(f"images: {s['cropped']} cropped, {s['kept']} kept, {s['removed']} removed, "
          f"{s['skipped']} skipped; image bytes {s['before']:,} -> {s['after']:,}; "
          f"file {os.path.getsize(a.input):,} -> {os.path.getsize(a.output):,} B")
    return hc

if __name__ == "__main__":
    main()
