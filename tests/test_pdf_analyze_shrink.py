import importlib.util, io, json, os, pathlib, subprocess, sys, zlib
import numpy as np, pikepdf, pymupdf
from PIL import Image
BIN = pathlib.Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("pdfanalyze", BIN / "pdf-analyze.py")
an = importlib.util.module_from_spec(spec); sys.modules["pdfanalyze"] = an; spec.loader.exec_module(an)
N = pikepdf.Name

def img(pdf, rgb, dct=False):
    h, w, _ = rgb.shape
    if dct:
        b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=95); xo = pikepdf.Stream(pdf, b.getvalue()); xo.Filter = N.DCTDecode
    else:
        xo = pikepdf.Stream(pdf, zlib.compress(rgb.tobytes())); xo.Filter = N.FlateDecode
    xo.Type = N.XObject; xo.Subtype = N.Image; xo.Width = w; xo.Height = h
    xo.ColorSpace = N.DeviceRGB; xo.BitsPerComponent = 8
    return xo

def page(pdf, xo=None, text=b""):
    c = (b"q 60 0 0 40 0 0 cm /Im0 Do Q " if xo is not None else b"") + text
    res = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=pikepdf.Dictionary(Type=N.Font, Subtype=N.Type1, BaseFont=N.Helvetica)))
    if xo is not None: res.XObject = pikepdf.Dictionary(Im0=xo)
    pdf.pages.append(pikepdf.Page(pikepdf.Dictionary(Type=N.Page, MediaBox=[0, 0, 60, 40], Resources=res,
                                                     Contents=pikepdf.Stream(pdf, c))))

def make(path):
    pdf = pikepdf.new(); rng = np.random.default_rng(0)
    g = (rng.integers(0, 16, (40, 60)) * 17).astype(np.uint8)
    col = np.zeros((40, 60, 3), np.uint8); col[:20] = (238, 34, 51); col[20:] = 255
    xc = img(pdf, col); xg = img(pdf, np.dstack([g] * 3)); xg2 = img(pdf, np.dstack([g] * 3))   # xg2 = byte dup
    page(pdf, xc, b"BT /F1 8 Tf 2 2 Td (Cover) Tj ET"); page(pdf, xg); page(pdf, xg2); page(pdf, xg)
    page(pdf, None, b"BT /F1 8 Tf 2 2 Td (vector only) Tj ET")
    orphan = img(pdf, np.dstack([g[::-1]] * 3))
    pdf.pages[4].Resources.XObject = pikepdf.Dictionary(Unused=orphan)   # in resources, never drawn
    sm = pikepdf.Stream(pdf, zlib.compress(bytes(2400))); sm.Type = N.XObject; sm.Subtype = N.Image
    sm.Width = 60; sm.Height = 40; sm.ColorSpace = N.DeviceGray; sm.BitsPerComponent = 8; sm.Filter = N.FlateDecode
    xc.SMask = sm
    pdf.save(path)

def test_analyze(tmp_path):
    src = str(tmp_path / "e.pdf"); make(src); prog = str(tmp_path / "p.log")
    d = an.analyze(src, 2, prog); s = d["summary"]
    assert s["pages"] == 5 and s["images"] == 3 and s["unique"] == 2 and s["duplicates"] == 1
    assert s["vector_only_pages"] == 1 and len(s["orphans"]) == 1 and s["gray"] == 1 and s["color"] == 1
    assert s["clean_4bit"] and len(s["masks"]) == 1 and list(s["masks"].values())[0].startswith("SMask")
    g = [r for r in d["images"] if r["kind"] == "gray" and "dup_of" not in r][0]
    assert g["pages"] == [2, 4] and g["exact17"] == 100 and g["qerr_max"] == 0
    c = [r for r in d["images"] if r["kind"] == "color"][0]
    assert c["x17_colors"] == 2 and c["pal16"]["cover"] == 100 and c["pal16"]["maxerr"] <= 8
    assert sum(1 for _ in open(prog)) >= 5
    serial = an.analyze(src, 1)["summary"]
    assert serial["est_gray4_pal256"] == s["est_gray4_pal256"]
    assert "CLEAN" in an.report(d)

def test_shrink_both(tmp_path):
    src = str(tmp_path / "e.pdf"); make(src); before = open(src, "rb").read()
    r = subprocess.run([sys.executable, str(BIN / "pdf-eprint-shrink.py"), src, "--photo", "both",
                        "--compare-dir", str(tmp_path / "cmp"), "--progress", str(tmp_path / "p.log")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert open(src, "rb").read() == before
    j, p = str(tmp_path / "e-gray4.pdf"), str(tmp_path / "e-gray4-pal256.pdf")
    for f in (j, p):
        d = pymupdf.open(f); assert len(d) == 5 and "Cover" in d[0].get_text()
        q = pikepdf.open(f)
        ims = {o.objgen for o in q.objects if isinstance(o, pikepdf.Stream) and o.get("/Subtype") == N.Image}
        assert len(ims) == 3          # cover+SMask+gray: dup merged, unused orphan dropped
    q = pikepdf.open(j); assert q.pages[0].Resources.XObject.Im0.ColorSpace == N.DeviceRGB
    q = pikepdf.open(p); assert q.pages[0].Resources.XObject.Im0.ColorSpace[0] == N.Indexed
    assert q.pages[1].Resources.XObject.Im0.BitsPerComponent == 4
    assert len(os.listdir(tmp_path / "cmp")) >= 2
    r2 = subprocess.run([sys.executable, str(BIN / "pdf-eprint-shrink.py"), src], capture_output=True, text=True)
    assert r2.returncode != 0 and "refusing" in r2.stderr

def test_shrink_max_err(tmp_path):
    pdf = pikepdf.new(); g = (np.arange(2400).reshape(40, 60) % 256).astype(np.uint8)
    page(pdf, img(pdf, np.dstack([g] * 3))); src = str(tmp_path / "x.pdf"); pdf.save(src)
    r = subprocess.run([sys.executable, str(BIN / "pdf-eprint-shrink.py"), src, "--max-err", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr and "unchanged" in r.stdout
    q = pikepdf.open(str(tmp_path / "x-gray4.pdf")); assert q.pages[0].Resources.XObject.Im0.ColorSpace == N.DeviceRGB
