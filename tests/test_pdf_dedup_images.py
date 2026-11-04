import importlib.util, pathlib, zlib, pikepdf, pymupdf
spec = importlib.util.spec_from_file_location("dd", pathlib.Path(__file__).parent.parent / "pdf-dedup-images.py")
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
N = pikepdf.Name

def test_identical_streams_merged(tmp_path):
    pdf = pikepdf.new()
    for k in (0, 1, 0, 1, 0):                      # two distinct images, 5 pages
        data = bytes([k * 200]) * (40 * 30)
        xo = pikepdf.Stream(pdf, zlib.compress(data)); xo.Filter = N.FlateDecode
        xo.Type = N.XObject; xo.Subtype = N.Image; xo.Width = 40; xo.Height = 30
        xo.ColorSpace = N.DeviceGray; xo.BitsPerComponent = 8
        pg = pikepdf.Dictionary(Type=N.Page, MediaBox=[0, 0, 40, 30],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xo)),
            Contents=pikepdf.Stream(pdf, b"q 40 0 0 30 0 0 cm /Im0 Do Q"))
        pdf.pages.append(pikepdf.Page(pg))
    src, out = str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf"); pdf.save(src)
    dd.main([src, out])
    a, b = pymupdf.open(src), pymupdf.open(out)
    assert len({x[0] for p in a for x in p.get_images()}) == 5
    assert len({x[0] for p in b for x in p.get_images()}) == 2
    for i in range(5):
        assert a[i].get_pixmap().samples == b[i].get_pixmap().samples
