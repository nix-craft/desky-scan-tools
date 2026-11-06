import subprocess, sys, pathlib, numpy as np, pikepdf
from PIL import Image
TOOL = pathlib.Path(__file__).parent.parent / "pdf-fill-solid.py"
def make_pdf(path, n=3):
    pdf = pikepdf.new()
    for k in range(n):
        a = np.full((300, 300), 255, np.uint8); a[50:250, 50:250] = 0
        a[100 + k, 100 + k] = 255; a[50, 120 + k] = 255   # hole + edge notch
        im = Image.fromarray(a).convert("1")
        xo = pikepdf.Stream(pdf, im.tobytes()); xo.Type = pikepdf.Name.XObject
        xo.Subtype = pikepdf.Name.Image; xo.Width = xo.Height = 300
        xo.ColorSpace = pikepdf.Name.DeviceGray; xo.BitsPerComponent = 1
        pg = pikepdf.Dictionary(Type=pikepdf.Name.Page, MediaBox=[0, 0, 36, 36],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=xo)),
            Contents=pikepdf.Stream(pdf, b"q 36 0 0 36 0 0 cm /Im0 Do Q"))
        pdf.pages.append(pikepdf.Page(pg))
    pdf.save(path)
def test_parallel_matches_serial(tmp_path):
    src = tmp_path / "in.pdf"; make_pdf(src)
    outs = []
    for j in ("1", "3"):
        o = tmp_path / f"out{j}.pdf"
        r = subprocess.run([sys.executable, str(TOOL), "-j", j, str(src), str(o)], capture_output=True, text=True, check=True)
        outs.append((o.read_bytes(), [l for l in r.stdout.splitlines() if l.startswith("page")]))
    assert outs[0] == outs[1]
    assert len(outs[0][1]) == 3 and "filled 0 px" not in outs[0][1][0]
