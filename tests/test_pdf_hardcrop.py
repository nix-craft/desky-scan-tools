"""Tests for pdf-hardcrop.py: synthetic PDFs built in code."""
import importlib.util, io, os, subprocess, zlib, pathlib
import numpy as np, pikepdf, pymupdf, pytest
from pikepdf import Name, Dictionary, Array, Stream
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hc", HERE.parent / "pdf-hardcrop.py")
hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)

def pattern(w, h, mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h // 32 + 1, w // 32 + 1, 3), dtype=np.uint8)
    a = np.kron(a, np.ones((32, 32, 1), np.uint8))[:h, :w]
    im = Image.fromarray(a, "RGB")
    return im.convert(mode)

def flate_img(pdf, im):
    n = {"RGB": 3, "L": 1}[im.mode]
    st = Stream(pdf, zlib.compress(im.tobytes()))
    st.Type, st.Subtype = Name.XObject, Name.Image
    st.Width, st.Height = im.size; st.BitsPerComponent = 8
    st.ColorSpace = Name.DeviceRGB if n == 3 else Name.DeviceGray
    st.Filter = Name.FlateDecode
    return pdf.make_indirect(st)

def dct_img(pdf, im, **kw):
    b = io.BytesIO(); im.save(b, "JPEG", quality=85, **kw)
    st = Stream(pdf, b.getvalue())
    st.Type, st.Subtype = Name.XObject, Name.Image
    st.Width, st.Height = im.size; st.BitsPerComponent = 8
    st.ColorSpace = Name.DeviceRGB; st.Filter = Name.DCTDecode
    return pdf.make_indirect(st)

def jbig2_img(pdf, im, tmp):
    bw = im.convert("L").point(lambda v: 255 if v > 128 else 0).convert("1")
    p = os.path.join(tmp, "a.png"); bw.save(p)
    jb = subprocess.run(["jbig2", "-p", p], check=True, capture_output=True).stdout
    st = Stream(pdf, jb)
    st.Type, st.Subtype = Name.XObject, Name.Image
    st.Width, st.Height = bw.size; st.BitsPerComponent = 1
    st.ColorSpace = Name.DeviceGray; st.Filter = Name.JBIG2Decode
    return pdf.make_indirect(st)

def add_page(pdf, content, xobjs, crop=None, media=(0, 0, 612, 792), fonts=None):
    res = Dictionary(XObject=Dictionary(xobjs))
    if fonts: res.Font = fonts
    pg = Dictionary(Type=Name.Page, MediaBox=Array(list(media)), Resources=res,
                    Contents=pdf.make_stream(content.encode()))
    if crop: pg.CropBox = Array(list(crop))
    pdf.pages.append(pikepdf.Page(pg))

def run(tmp_path, pdf):
    i, o = tmp_path / "in.pdf", tmp_path / "out.pdf"
    pdf.save(i, stream_decode_level=pikepdf.StreamDecodeLevel.none, compress_streams=False)
    st = hc.main([str(i), str(o), "-q"]).stats
    return i, o, st

def render(path, dpi=100):
    d = pymupdf.open(path)
    out = []
    for p in d:
        pm = p.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), colorspace=pymupdf.csRGB)
        out.append(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3).astype(int))
    return out

def assert_same_render(i, o, tol=0.01, dpi=100):
    """Render both at ~100 dpi (or the image's native dpi, where resampling is 1:1)."""
    for k, (a, b) in enumerate(zip(render(i, dpi), render(o, dpi))):
        assert a.shape == b.shape, k
        # ignore the outer 2px (image edge now coincides with the crop edge) and
        # small resampling-phase differences; real misplacement shows up as large areas
        a, b = a[2:-2, 2:-2], b[2:-2, 2:-2]
        assert np.abs(a - b).mean() < 3, f"page {k+1}: mean diff {np.abs(a - b).mean():.2f}"
        bad = (np.abs(a - b).max(axis=2) > 48).mean()
        assert bad <= tol, f"page {k+1}: {bad:.4%} pixels differ"

def images(o):
    d = pikepdf.open(o); res = []
    for p in d.pages:
        for n, x in p.images.items():
            res.append(x)
    return d, res

CROP = (100, 150, 500, 700)

def test_flate_rgb(tmp_path):
    pdf = pikepdf.new(); im = pattern(850, 1100)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": flate_img(pdf, im)}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, ims = images(o); x = ims[0]
    assert x.Filter == Name.FlateDecode and x.ColorSpace == Name.DeviceRGB and x.BitsPerComponent == 8
    W, H = int(x.Width), int(x.Height)
    assert W < 850 and H < 1100
    sd = pikepdf.open(i)
    A = np.frombuffer(sd.pages[0].images["/Im0"].read_bytes(), np.uint8).reshape(1100, 850, 3)
    Bm = np.frombuffer(x.read_bytes(), np.uint8).reshape(H, W, 3)
    cm = [i_ for i_ in pikepdf.parse_content_stream(d.pages[0]) if str(i_.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * 850); T = 1100 - round((float(cm.operands[5]) + float(cm.operands[3])) * 1100)
    assert (A[T:T + H, L:L + W] == Bm).all()
    assert_same_render(i, o)

def test_jbig2(tmp_path):
    pdf = pikepdf.new(); im = pattern(1275, 1650, seed=3)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": jbig2_img(pdf, im, str(tmp_path))}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, ims = images(o); x = ims[0]
    assert x.Filter == Name.JBIG2Decode and x.BitsPerComponent == 1
    # pixel exact: decoded crop equals original region
    sd = pikepdf.open(i); src = sd.pages[0].images["/Im0"]
    a = np.array(pikepdf.PdfImage(src).as_pil_image())
    b = np.array(pikepdf.PdfImage(x).as_pil_image())
    cm = [i_ for i_ in pikepdf.parse_content_stream(d.pages[0]) if str(i_.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * 1275); T = 1650 - round((float(cm.operands[5]) + float(cm.operands[3])) * 1650)
    assert (a[T:T + b.shape[0], L:L + b.shape[1]] == b).all()
    assert_same_render(i, o, dpi=150)

@pytest.mark.parametrize("sub", [0, 2])
def test_dct_lossless(tmp_path, sub):
    pdf = pikepdf.new(); im = pattern(1000, 1300, seed=5)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": dct_img(pdf, im, subsampling=sub)}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    sd = pikepdf.open(i); src = sd.pages[0].images["/Im0"]
    d, ims = images(o); x = ims[0]
    assert x.Filter == Name.DCTDecode
    a = np.array(Image.open(io.BytesIO(src.read_raw_bytes())).convert("RGB")).astype(int)
    b = np.array(Image.open(io.BytesIO(x.read_raw_bytes())).convert("RGB")).astype(int)
    # recover offset from the placement cm
    cm = [i_ for i_ in pikepdf.parse_content_stream(d.pages[0]) if str(i_.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * 1000); T = 1300 - round((float(cm.operands[5]) + float(cm.operands[3])) * 1300)
    mcu = 8 if sub == 0 else 16
    assert L % mcu == 0 and T % mcu == 0
    region = a[T:T + b.shape[0], L:L + b.shape[1]]
    if sub == 0:
        assert (region == b).all()          # 4:4:4: bit-identical
    else:                                   # 4:2:0: identical away from the new edge (chroma upsampling)
        assert (region[2:, 2:] == b[2:, 2:]).all()
    assert_same_render(i, o, dpi=72 * 1000 / 612)

def test_multiple_images(tmp_path):
    pdf = pikepdf.new()
    xo = {"/A": flate_img(pdf, pattern(100, 100, seed=1)),   # outside
          "/B": flate_img(pdf, pattern(200, 200, seed=2)),   # partial
          "/C": flate_img(pdf, pattern(50, 50, seed=4))}     # inside
    content = ("q 50 0 0 50 10 10 cm /A Do Q "
               "q 200 0 0 200 400 500 cm /B Do Q "
               "q 50 0 0 50 200 300 cm /C Do Q")
    add_page(pdf, content, xo, CROP)
    i, o, st = run(tmp_path, pdf)
    assert (st["removed"], st["cropped"], st["kept"]) == (1, 1, 1)
    d, ims = images(o)
    assert len(ims) == 2
    assert_same_render(i, o)

def test_vector_text_untouched(tmp_path):
    pdf = pikepdf.new()
    font = Dictionary(F1=Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
    content = "0 0 1 rg 50 50 500 600 re f BT /F1 24 Tf 120 400 Td (Hello hardcrop) Tj ET"
    add_page(pdf, content, {}, CROP, fonts=font)
    i, o, st = run(tmp_path, pdf)
    da, db = pikepdf.open(i), pikepdf.open(o); a = da.pages[0]; b = db.pages[0]
    assert b.Contents.read_bytes() == a.Contents.read_bytes()
    assert "Hello hardcrop" in pymupdf.open(o)[0].get_text()
    assert_same_render(i, o)

def test_shared_image(tmp_path):
    pdf = pikepdf.new(); x = flate_img(pdf, pattern(612, 792, seed=7))
    c = "q 612 0 0 792 0 0 cm /Im0 Do Q"
    add_page(pdf, c, {"/Im0": x}, CROP)
    add_page(pdf, c, {"/Im0": x}, CROP)          # same crop -> one copy
    add_page(pdf, c, {"/Im0": x}, (0, 0, 300, 400))  # different crop
    add_page(pdf, c, {"/Im0": x})                # full page -> original kept
    i, o, st = run(tmp_path, pdf)
    d = pikepdf.open(o)
    og = [list(p.images.values())[0].objgen for p in d.pages]
    assert og[0] == og[1] and len(set(og)) == 3
    assert int(list(d.pages[3].images.values())[0].Width) == 612
    assert_same_render(i, o, dpi=72)

def test_flipped(tmp_path):
    pdf = pikepdf.new()
    add_page(pdf, "q -612 0 0 -792 612 792 cm /Im0 Do Q",
             {"/Im0": flate_img(pdf, pattern(612, 792, seed=8))}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    assert_same_render(i, o)

def test_rotated_fully_visible(tmp_path):
    # 45-degree placement whose visible area covers the whole image: kept, not skipped
    pdf = pikepdf.new()
    add_page(pdf, "q 300 300 -300 300 300 100 cm /Im0 Do Q",
             {"/Im0": flate_img(pdf, pattern(300, 300, seed=9))}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["skipped"] == 0 and st["cropped"] == 0 and st["kept"] == 1
    assert_same_render(i, o)

def test_form_nesting(tmp_path):
    pdf = pikepdf.new(); x = flate_img(pdf, pattern(400, 400, seed=10))
    inner = pdf.make_stream(b"q 400 0 0 400 0 0 cm /Im0 Do Q")
    inner.Type, inner.Subtype = Name.XObject, Name.Form
    inner.BBox = Array([0, 0, 400, 400]); inner.Resources = Dictionary(XObject=Dictionary(Im0=x))
    outer = pdf.make_stream(b"q 1 0 0 1 50 50 cm /Fi Do Q")
    outer.Type, outer.Subtype = Name.XObject, Name.Form
    outer.BBox = Array([0, 0, 600, 600]); outer.Matrix = Array([1, 0, 0, 1, 20, 30])
    outer.Resources = Dictionary(XObject=Dictionary(Fi=inner))
    add_page(pdf, "/Fo Do", {"/Fo": outer}, (200, 250, 600, 792))
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d = pikepdf.open(o)
    ws = [int(v.Width) for v in d.objects if isinstance(v, Stream) and v.get("/Subtype") == Name.Image]
    assert ws and max(ws) < 400       # original dropped
    assert_same_render(i, o)

def test_smask(tmp_path):
    pdf = pikepdf.new(); x = flate_img(pdf, pattern(300, 300, seed=11))
    m = flate_img(pdf, pattern(300, 300, "L", seed=12)); del m["/ColorSpace"]; m.ColorSpace = Name.DeviceGray
    x.SMask = m
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": x}, CROP)
    i, o, st = run(tmp_path, pdf)
    d, ims = images(o)
    assert int(ims[0].SMask.Width) == int(ims[0].Width)
    assert_same_render(i, o)


# ------------------------------------------------ extended image types
import base64
def raw_rgb_stream(pdf, im, filt, enc, parms=None):
    st = Stream(pdf, enc)
    st.Type, st.Subtype = Name.XObject, Name.Image
    st.Width, st.Height = im.size; st.BitsPerComponent = 8
    st.ColorSpace = Name.DeviceRGB if im.mode == "RGB" else Name.DeviceGray
    st.Filter = filt
    if parms is not None: st.DecodeParms = parms
    return pdf.make_indirect(st)

def check_pixels(i, o, name="/Im0"):
    """Cropped samples must equal the matching region of the original, bit for bit."""
    sd = pikepdf.open(i); d = pikepdf.open(o)
    src = sd.pages[0].images[name]
    x = list(d.pages[0].images.values())[0]
    W, H = int(src.Width), int(src.Height); w, h = int(x.Width), int(x.Height)
    cm = [k for k in pikepdf.parse_content_stream(d.pages[0]) if str(k.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * W); T = H - round((float(cm.operands[5]) + float(cm.operands[3])) * H)
    n = 3 if src.ColorSpace == Name.DeviceRGB else 1
    S = pikepdf.StreamDecodeLevel.specialized
    A = np.frombuffer(src.read_bytes(decode_level=S), np.uint8).reshape(H, W, n)
    B = np.frombuffer(x.read_bytes(decode_level=S), np.uint8).reshape(h, w, n)
    assert np.array_equal(A[T:T+h, L:L+w], B)
    return d, x

def test_lzw(tmp_path):
    pdf = pikepdf.new(); im = pattern(612, 792, seed=20)
    p = Dictionary(Predictor=12, Colors=3, BitsPerComponent=8, Columns=612)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Name.LZWDecode,
                                     hc.lzw_encode(hc.pred_encode(im.tobytes(), p)), p)}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, x = check_pixels(i, o)
    assert x.Filter == Name.LZWDecode and int(x.DecodeParms.Predictor) == 12
    assert int(x.DecodeParms.Columns) == int(x.Width)
    assert_same_render(i, o, dpi=72)

def test_lzw_early0_tiff_predictor(tmp_path):
    pdf = pikepdf.new(); im = pattern(612, 792, seed=21)
    p = Dictionary(Predictor=2, Colors=3, BitsPerComponent=8, Columns=612, EarlyChange=0)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Name.LZWDecode,
                                     hc.lzw_encode(hc.pred_encode(im.tobytes(), p), 0), p)}, CROP)
    i, o, st = run(tmp_path, pdf)
    d, x = check_pixels(i, o)
    assert st["cropped"] == 1 and int(x.DecodeParms.EarlyChange) == 0

def test_runlength(tmp_path):
    pdf = pikepdf.new(); im = pattern(612, 792, "L", seed=22)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Name.RunLengthDecode, hc.rl_encode(im.tobytes()))}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, x = check_pixels(i, o); assert x.Filter == Name.RunLengthDecode
    assert_same_render(i, o, dpi=72)

def test_filter_chain_raw(tmp_path):
    pdf = pikepdf.new(); im = pattern(612, 792, seed=23)
    p = Dictionary(Predictor=15, Colors=3, BitsPerComponent=8, Columns=612)
    enc = base64.a85encode(zlib.compress(hc.pred_encode(im.tobytes(), p))) + b"~>"
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Array([Name.ASCII85Decode, Name.FlateDecode]),
                                     enc, Array([None, p]))}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, x = check_pixels(i, o)
    assert list(x.Filter) == [Name.ASCII85Decode, Name.FlateDecode]
    assert int(x.DecodeParms[1].Columns) == int(x.Width)

def test_filter_chain_dct(tmp_path):
    pdf = pikepdf.new(); im = pattern(612, 792, seed=24)
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Array([Name.ASCIIHexDecode, Name.DCTDecode]),
                                     b.getvalue().hex().encode() + b">")}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, ims = images(o); x = ims[0]
    assert list(x.Filter) == [Name.ASCIIHexDecode, Name.DCTDecode]
    jpg = bytes.fromhex(x.read_raw_bytes().rstrip(b">").decode())
    assert jpg[:2] == b"\xff\xd8" and hc.jpeg_size(jpg) == (int(x.Width), int(x.Height))
    assert_same_render(i, o, dpi=72)

def ccitt_stream(pdf, bw, tmp, k):
    p = os.path.join(tmp, "g.tif")
    info = {278: bw.size[1]}
    bw.save(p, "TIFF", compression="group4" if k < 0 else "group3", tiffinfo=info)
    t = Image.open(p); off, cnt = t.tag_v2[273], t.tag_v2[279]
    off = off[0] if isinstance(off, tuple) else off; cnt = cnt[0] if isinstance(cnt, tuple) else cnt
    st = Stream(pdf, open(p, "rb").read()[off:off+cnt])
    st.Type, st.Subtype = Name.XObject, Name.Image
    st.Width, st.Height = bw.size; st.BitsPerComponent = 1; st.ColorSpace = Name.DeviceGray
    st.Filter = Name.CCITTFaxDecode
    st.DecodeParms = Dictionary(K=k, Columns=bw.size[0], Rows=bw.size[1], BlackIs1=True)
    return pdf.make_indirect(st)

@pytest.mark.parametrize("k", [-1, 0])
def test_ccitt(tmp_path, k):
    pdf = pikepdf.new()
    bw = pattern(1224, 1584, "L", seed=25).point(lambda v: 255 if v > 128 else 0).convert("1")
    src = ccitt_stream(pdf, bw, str(tmp_path), k)
    ref = hc.ccitt_decode(pdf, src.read_raw_bytes(), src.DecodeParms, 1224, 1584)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": src}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, ims = images(o); x = ims[0]
    assert x.Filter == Name.CCITTFaxDecode and int(x.DecodeParms.K) == k
    w, h = int(x.Width), int(x.Height)
    assert w < 1224 and h < 1584 and int(x.DecodeParms.Columns) == w
    got = hc.ccitt_decode(d, x.read_raw_bytes(), x.DecodeParms, w, h)
    cm = [c for c in pikepdf.parse_content_stream(d.pages[0]) if str(c.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * 1224); T = 1584 - round((float(cm.operands[5]) + float(cm.operands[3])) * 1584)
    assert np.array_equal(ref[T:T+h, L:L+w], got)
    assert_same_render(i, o, dpi=144)

def test_ccitt_g4_byte_aligned_skipped(tmp_path):
    pdf = pikepdf.new()
    bw = pattern(612, 792, "L", seed=26).point(lambda v: 255 if v > 128 else 0).convert("1")
    src = ccitt_stream(pdf, bw, str(tmp_path), -1); src.DecodeParms.EncodedByteAlign = True
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": src}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["skipped"] == 1 and st["cropped"] == 0

@pytest.mark.parametrize("ext", ["j2k", "jp2"])
def test_jpx_tiles(tmp_path, ext):
    pdf = pikepdf.new(); im = pattern(640, 832, seed=27)
    b = io.BytesIO(); im.save(b, "JPEG2000", tile_size=(128, 128), irreversible=True,
                             quality_mode="rates", quality_layers=[20], no_jp2=(ext == "j2k"))
    st = raw_rgb_stream(pdf, im, Name.JPXDecode, b.getvalue())
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": st}, CROP)
    i, o, s = run(tmp_path, pdf)
    assert s["cropped"] == 1
    d, ims = images(o); x = ims[0]
    assert x.Filter == Name.JPXDecode
    w, h = int(x.Width), int(x.Height)
    assert w <= 640 and h < 832
    A = np.asarray(Image.open(io.BytesIO(b.getvalue()))); B = np.asarray(Image.open(io.BytesIO(x.read_raw_bytes())))
    cm = [c for c in pikepdf.parse_content_stream(d.pages[0]) if str(c.operator) == "cm"][-1]
    L = round(float(cm.operands[4]) * 640); T = 832 - round((float(cm.operands[5]) + float(cm.operands[3])) * 832)
    assert L % 128 == 0 and T % 128 == 0           # tile boundaries
    assert np.array_equal(A[T:T+h, L:L+w], B)      # no re-encoding: identical pixels
    assert_same_render(i, o)

def test_jpx_single_tile_skipped(tmp_path):
    pdf = pikepdf.new(); im = pattern(400, 400, seed=28)
    b = io.BytesIO(); im.save(b, "JPEG2000", no_jp2=True)
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q",
             {"/Im0": raw_rgb_stream(pdf, im, Name.JPXDecode, b.getvalue())}, CROP)
    i, o, s = run(tmp_path, pdf)
    assert s["skipped"] == 1 and s["cropped"] == 0

def test_inline_partly_visible(tmp_path):
    pdf = pikepdf.new(); im = pattern(64, 80, seed=29)
    data = zlib.compress(im.tobytes()).hex().encode() + b">"
    content = (b"q 612 0 0 792 0 0 cm BI /W 64 /H 80 /BPC 8 /CS /RGB /F [/AHx /Fl] ID\n" + data + b"\nEI Q")
    pg = Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]), CropBox=Array(list(CROP)),
                    Resources=Dictionary(), Contents=pdf.make_stream(content))
    pdf.pages.append(pikepdf.Page(pg))
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d = pikepdf.open(o)
    ii = [c.iimage for c in pikepdf.parse_content_stream(d.pages[0]) if isinstance(c, pikepdf.ContentStreamInlineImage)]
    assert len(ii) == 1 and ii[0].width < 64 and ii[0].height < 80
    assert [hc.FULL.get(str(f), str(f)) for f in ii[0].filters] == ["/ASCIIHexDecode", "/FlateDecode"]
    assert_same_render(i, o)

def test_inline_named_colorspace(tmp_path):
    pdf = pikepdf.new(); im = pattern(64, 80, "L", seed=30)
    content = (b"q 612 0 0 792 0 0 cm BI /W 64 /H 80 /BPC 8 /CS /Cs0 /F /AHx ID\n" + im.tobytes().hex().encode() + b">\nEI Q")
    pg = Dictionary(Type=Name.Page, MediaBox=Array([0, 0, 612, 792]), CropBox=Array(list(CROP)),
                    Resources=Dictionary(ColorSpace=Dictionary(Cs0=Name.DeviceGray)),
                    Contents=pdf.make_stream(content))
    pdf.pages.append(pikepdf.Page(pg))
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    assert_same_render(i, o)

@pytest.mark.parametrize("cm", ["519.6 300 -300 519.6 250 50",      # rotated 30 degrees
                                "612 0 200 792 -100 0"])           # skewed
def test_rotated_skewed_cropped(tmp_path, cm):
    pdf = pikepdf.new()
    add_page(pdf, f"q {cm} cm /Im0 Do Q", {"/Im0": flate_img(pdf, pattern(600, 600, seed=31))},
             (200, 300, 400, 500))
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1 and st["skipped"] == 0
    d, ims = images(o)
    assert int(ims[0].Width) < 600 and int(ims[0].Height) < 600
    assert_same_render(i, o)

def test_smask_different_grid(tmp_path):
    pdf = pikepdf.new(); x = flate_img(pdf, pattern(612, 792, seed=32))
    m = flate_img(pdf, pattern(306, 396, "L", seed=33))
    x.SMask = m
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": x}, CROP)
    i, o, st = run(tmp_path, pdf)
    assert st["cropped"] == 1
    d, ims = images(o); y = ims[0]
    assert int(y.Width) == 2 * int(y.SMask.Width) and int(y.Height) == 2 * int(y.SMask.Height)
    assert int(y.Width) < 612
    assert_same_render(i, o, dpi=72)

def test_smask_odd_grid(tmp_path):
    # 612 vs 204 (ratio 3) and 792 vs 264: common grid 1/204, 1/264
    pdf = pikepdf.new(); x = flate_img(pdf, pattern(612, 792, seed=34))
    x.SMask = flate_img(pdf, pattern(204, 264, "L", seed=35))
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": x}, (101, 151, 499, 699))
    i, o, st = run(tmp_path, pdf)
    d, ims = images(o); y = ims[0]
    assert st["cropped"] == 1
    assert int(y.Width) == 3 * int(y.SMask.Width) and int(y.Height) == 3 * int(y.SMask.Height)
    assert_same_render(i, o)

def test_smask_jpeg_on_flate(tmp_path):
    pdf = pikepdf.new(); x = dct_img(pdf, pattern(640, 800, seed=36))
    x.SMask = flate_img(pdf, pattern(320, 400, "L", seed=37))
    add_page(pdf, "q 612 0 0 792 0 0 cm /Im0 Do Q", {"/Im0": x}, CROP)
    i, o, st = run(tmp_path, pdf)
    d, ims = images(o); y = ims[0]
    assert st["cropped"] == 1 and y.Filter == Name.DCTDecode
    assert int(y.Width) == 2 * int(y.SMask.Width) and int(y.Height) == 2 * int(y.SMask.Height)
    assert_same_render(i, o)
