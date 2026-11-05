import importlib.util, io, pathlib, zlib, numpy as np, pikepdf, pymupdf
from PIL import Image
spec = importlib.util.spec_from_file_location("p2g", pathlib.Path(__file__).parent.parent / "pdf-to-gray.py")
p2g = importlib.util.module_from_spec(spec); import sys; sys.modules["p2g"] = p2g; spec.loader.exec_module(p2g)
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

def test_bits4_levels_and_shared(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    g = ((np.arange(61 * 81).reshape(61, 81) % 16) * 17).astype(np.uint8)   # odd width
    noisy = np.clip(g.astype(int) + np.random.default_rng(0).integers(-8, 9, g.shape), 0, 255).astype(np.uint8)
    mk(src, np.dstack([noisy] * 3)); p2g.main(["--bits", "4", src, out])
    pdf = pikepdf.open(out)
    assert len({pg.Resources.XObject.Im0.objgen for pg in pdf.pages}) == 1
    xo = pdf.pages[0].Resources.XObject.Im0
    assert int(xo.BitsPerComponent) == 4 and xo.ColorSpace == N.DeviceGray
    pix = pymupdf.open(out)[0].get_pixmap(colorspace=pymupdf.csGRAY)
    r = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    assert (r == g).all()          # noise within +-8 snaps back to the 16 levels

def _color_img(ncol, h=50, w=71):
    rng = np.random.default_rng(1)
    pal = rng.integers(0, 16, (ncol, 3)) * 17
    idx = rng.integers(0, ncol, (h, w))
    return pal[idx].astype(np.uint8)

def _render(path, i=0):
    pix = pymupdf.open(path)[i].get_pixmap(colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)

def test_palette_4bit_exact_shared(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    rgb = _color_img(12); mk(src, rgb, pages=2)
    p2g.main(["--bits", "4", "--palette", src, out])
    pdf = pikepdf.open(out)
    assert len({pg.Resources.XObject.Im0.objgen for pg in pdf.pages}) == 1
    xo = pdf.pages[0].Resources.XObject.Im0
    assert xo.ColorSpace[0] == N.Indexed and int(xo.BitsPerComponent) == 4
    assert (_render(out) == rgb).all() and (_render(out, 1) == rgb).all()

def test_palette_8bit_and_rare_remap(tmp_path):
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    rgb = _color_img(40).copy(); mk(src, rgb, pages=1)
    p2g.main(["--bits", "4", "--palette", src, out])
    pdf = pikepdf.open(out); xo = pdf.pages[0].Resources.XObject.Im0
    assert int(xo.BitsPerComponent) == 8 and (_render(out) == rgb).all()
    idx, pal, st = p2g.palettize(np.dstack([np.full((100, 100), 255, np.uint8)] * 3), 1e-4)
    assert st["palette"] == 1
    big = np.zeros((100, 100, 3), np.uint8); big[0, 0] = (17, 0, 0)  # 1 rare px among 2 colors? -> <=256, kept
    assert p2g.palettize(big, 0.5)[2]["remapped_px"] == 0
    many = (np.arange(300 * 3).reshape(300, 1, 3) % 16 * 17).astype(np.uint8)
    many = np.repeat(np.concatenate([many, np.zeros((5000, 1, 3), np.uint8)]), 2, 1)
    i2, p2, s2 = p2g.palettize(many, 0.01)
    assert s2["palette"] <= 256 and s2["remapped_px"] > 0

def test_palette_requires_bits4(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        p2g.main(["--palette", "a.pdf", str(tmp_path / "b.pdf")])

def _mixed(tmp_path):
    pdf = pikepdf.new(); rng = np.random.default_rng(1)
    for i in range(5):
        g = (rng.integers(0, 16, (40, 50)) * 17).astype(np.uint8)
        rgb = np.dstack([g] * 3) if i % 2 else np.dstack([g, g // 2, g // 3]).astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=92) if i == 4 else None
        xo = pikepdf.Stream(pdf, b.getvalue() if i == 4 else zlib.compress(rgb.tobytes()))
        xo.Filter = N.DCTDecode if i == 4 else N.FlateDecode
        xo.Type = N.XObject; xo.Subtype = N.Image; xo.Width = 50; xo.Height = 40
        xo.ColorSpace = N.DeviceRGB; xo.BitsPerComponent = 8
        pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(Type=N.Page, MediaBox=[0, 0, 50, 40],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xo)),
            Contents=pikepdf.Stream(pdf, b"q 50 0 0 40 0 0 cm /Im0 Do Q"))))
    p = str(tmp_path / "m.pdf"); pdf.save(p); return p

def test_parallel_identical_to_serial(tmp_path):
    src = _mixed(tmp_path)
    for args in (["--bits", "8"], ["--bits", "4"], ["--bits", "4", "--palette"]):
        a, b = str(tmp_path / "s.pdf"), str(tmp_path / "p.pdf")
        p2g.main(args + ["-j", "1", src, a]); p2g.main(args + ["-j", "4", src, b])
        assert open(a, "rb").read() == open(b, "rb").read()
        import os; os.remove(a); os.remove(b)

def test_max_err_skips(tmp_path):
    src = str(tmp_path / "a.pdf"); out = str(tmp_path / "b.pdf")
    g = (np.arange(4000).reshape(40, 100) % 256).astype(np.uint8)   # off-grid levels, err up to 8
    mk(src, np.dstack([g] * 3), pages=1); p2g.main(["--bits", "4", "--max-err", "3", src, out])
    q = pikepdf.open(out); assert q.pages[0].Resources.XObject.Im0.ColorSpace == N.DeviceRGB

def test_smask_parent_converted_mask_kept(tmp_path):
    """Parent with /SMask converts; soft-mask stays 8-bit; Matte shrinks to 1 comp."""
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf")
    pdf = pikepdf.new()
    g = ((np.arange(40 * 50).reshape(40, 50) % 16) * 17).astype(np.uint8)
    rgb = np.dstack([g, g, g])
    # soft mask: mostly opaque (255) with a transparent corner
    alpha = np.full((40, 50), 255, np.uint8); alpha[:5, :5] = 0
    sm = pikepdf.Stream(pdf, zlib.compress(alpha.tobytes()))
    sm.Type = N.XObject; sm.Subtype = N.Image; sm.Width = 50; sm.Height = 40
    sm.ColorSpace = N.DeviceGray; sm.BitsPerComponent = 8; sm.Filter = N.FlateDecode
    sm.Matte = [1, 1, 1]
    xo = pikepdf.Stream(pdf, zlib.compress(rgb.tobytes()))
    xo.Type = N.XObject; xo.Subtype = N.Image; xo.Width = 50; xo.Height = 40
    xo.ColorSpace = N.DeviceRGB; xo.BitsPerComponent = 8; xo.Filter = N.FlateDecode
    xo.SMask = sm
    pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(Type=N.Page, MediaBox=[0, 0, 50, 40],
        Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xo)),
        Contents=pikepdf.Stream(pdf, b"q 50 0 0 40 0 0 cm /Im0 Do Q"))))
    pdf.save(src)
    p2g.main(["--bits", "4", "-j", "1", src, out])
    q = pikepdf.open(out); parent = q.pages[0].Resources.XObject.Im0
    assert parent.ColorSpace == N.DeviceGray and int(parent.BitsPerComponent) == 4
    assert "/SMask" in parent
    sm2 = parent.SMask
    assert int(sm2.BitsPerComponent) == 8 and sm2.ColorSpace == N.DeviceGray
    assert list(sm2.Matte) == [1]
    # soft-mask pixels unchanged
    a2 = np.frombuffer(sm2.read_bytes(), np.uint8).reshape(40, 50)
    assert (a2 == alpha).all()
