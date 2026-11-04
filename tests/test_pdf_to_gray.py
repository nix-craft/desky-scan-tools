import importlib.util, io, pathlib, zlib, numpy as np, pikepdf, pymupdf
from PIL import Image
spec = importlib.util.spec_from_file_location("p2g", pathlib.Path(__file__).parent.parent / "pdf-to-gray.py")
p2g = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2g)
N = pikepdf.Name

def mk(path, rgb, pages=3, dct=False):
    pdf = pikepdf.new(); h, w, _ = rgb.shape
    if dct:
        b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=90)
        xo = pikepdf.Stream(pdf, b.getvalue()); xo.Filter = N.DCTDecode
    else:
        xo = pikepdf.Stream(pdf, zlib.compress(rgb.tobytes())); xo.Filter = N.FlateDecode
    xo.Type = N.XObject; xo.Subtype = N.Image; xo.Width = w; xo.Height = h
    xo.ColorSpace = N.DeviceRGB; xo.BitsPerComponent = 8
    for _ in range(pages):
        pg = pikepdf.Dictionary(Type=N.Page, MediaBox=[0, 0, w, h],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xo)),
            Contents=pikepdf.Stream(pdf, f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode()))
        pdf.pages.append(pikepdf.Page(pg))
    pdf.save(path)

def gray_rgb():
    g = (np.add.outer(np.arange(60), np.arange(80)) * 3 % 256).astype(np.uint8)
    return np.dstack([g, g, g])

def images(path):
    d = pymupdf.open(path); return d, {x[0] for p in d for x in p.get_images()}

def test_flate_roundtrip_and_shared(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    rgb = gray_rgb(); mk(src, rgb); p2g.main([src, out])
    pdf = pikepdf.open(out)
    objs = {pg.Resources.XObject.Im0.objgen for pg in pdf.pages}
    assert len(objs) == 1
    xo = pdf.pages[0].Resources.XObject.Im0
    assert xo.ColorSpace == N.DeviceGray
    g = np.frombuffer(xo.read_bytes(), np.uint8).reshape(60, 80)
    assert (g == rgb[..., 0]).all()
    a, b = pymupdf.open(src), pymupdf.open(out)
    assert a[1].get_pixmap(colorspace=pymupdf.csGRAY).samples == b[1].get_pixmap(colorspace=pymupdf.csGRAY).samples

def test_color_skipped(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    rgb = gray_rgb().copy(); rgb[5, 5, 0] ^= 1           # one off pixel = color
    mk(src, rgb, pages=1); p2g.main([src, out])
    pdf = pikepdf.open(out)
    assert pdf.pages[0].Resources.XObject.Im0.ColorSpace == N.DeviceRGB

def test_dct_gray(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    mk(src, gray_rgb(), pages=2, dct=True); p2g.main([src, out])
    pdf = pikepdf.open(out); xo = pdf.pages[0].Resources.XObject.Im0
    assert xo.ColorSpace == N.DeviceGray and xo.Filter == N.DCTDecode
    assert Image.open(io.BytesIO(xo.read_raw_bytes())).mode == "L"
