import importlib.util, os, pathlib, tempfile
import pikepdf
from pikepdf import Name, Operator, Stream

spec = importlib.util.spec_from_file_location(
    "p2b", pathlib.Path(__file__).parent.parent / "pdf-to-black.py")
p2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2b)

def _ops(path):
    pdf = pikepdf.open(path)
    return [(list(a), str(b)) for a, b in pikepdf.parse_content_stream(pdf.pages[0])]

def _save_page(content, path, resources=None):
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(72, 72))
    page.Contents = Stream(pdf, content)
    if resources:
        page.Resources = resources
    pdf.save(path)

def test_rgb_to_black_keeps_white(tmp_path=None):
    d = tmp_path or pathlib.Path(tempfile.mkdtemp())
    src, out = d / "in.pdf", d / "out.pdf"
    # blue fill, white fill, already-black gray, light-blue stroke
    _save_page(
        b"1 0 0 rg 0 0 20 20 re f "
        b"1 1 1 rg 20 0 20 20 re f "
        b"0 g 40 0 20 20 re f "
        b"0 0.2 0.8 RG 0 0 72 72 re S",
        src)
    p2b.main([str(src), str(out), '-q'])
    ops = _ops(out)
    color = [(a, o) for a, o in ops if o in ("g","G","rg","RG","k","K","cs","CS","sc","SC","scn","SCN")]
    assert all(o in ("g", "G") for _, o in color), color
    # 1 0 0 rg -> 0 g; 1 1 1 rg -> 1 g; 0 g unchanged; 0 0.2 0.8 RG -> 0 G
    assert color == [([0], "g"), ([1], "g"), ([0], "g"), ([0], "G")]
    # refuse overwrite
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            p2b.main([str(src), str(out), "-q"])
            raise AssertionError("should have refused overwrite")
        except SystemExit:
            pass
    assert "refusing to overwrite" in buf.getvalue()

def test_icc_sc_and_form():
    d = pathlib.Path(tempfile.mkdtemp())
    src, out = d / "in.pdf", d / "out.pdf"
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(72, 72))
    icc = Stream(pdf, b"\x00" * 16)  # dummy; only /N and /Alternate are read
    icc.N = 3
    icc.Alternate = Name.DeviceRGB
    cs = pikepdf.Array([Name.ICCBased, icc])
    form = Stream(pdf, b"/Cs1 cs 0 0.198 0.629 sc 0 0 10 10 re f")
    form.Subtype = Name.Form
    form.Resources = pikepdf.Dictionary(ColorSpace=pikepdf.Dictionary(Cs1=cs))
    form.BBox = pikepdf.Array([0, 0, 10, 10])
    page.Resources = pikepdf.Dictionary(
        ColorSpace=pikepdf.Dictionary(Cs1=cs),
        XObject=pikepdf.Dictionary(Fm1=form),
    )
    page.Contents = Stream(pdf, b"/Cs1 cs 0 0.198 0.629 sc 0 0 72 72 re f /Fm1 Do")
    pdf.save(src)
    p2b.main([str(src), str(out), "-q"])
    pdf2 = pikepdf.open(out)
    page_ops = [(list(a), str(b)) for a, b in pikepdf.parse_content_stream(pdf2.pages[0])]
    assert ("cs",) and any(o == "cs" and a == [Name.DeviceGray] for a, o in page_ops)
    assert any(o == "g" and a == [0] for a, o in page_ops)
    assert not any(o in ("rg", "sc", "SC", "scn") for _, o in page_ops)
    form2 = pdf2.pages[0].Resources.XObject["/Fm1"]
    fops = [(list(a), str(b)) for a, b in pikepdf.parse_content_stream(form2)]
    assert any(o == "g" and a == [0] for a, o in fops)
    assert not any(o == "sc" for _, o in fops)

if __name__ == "__main__":
    test_rgb_to_black_keeps_white()
    test_icc_sc_and_form()
    print("ok")
