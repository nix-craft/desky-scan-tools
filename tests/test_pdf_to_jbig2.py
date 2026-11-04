import importlib.util, pathlib, numpy as np
spec = importlib.util.spec_from_file_location("p2j", pathlib.Path(__file__).parent.parent / "pdf-to-jbig2.py")
p2j = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2j)
def test_despeckle():
    a = np.zeros((20, 20), bool)
    a[1, 1] = True                    # 1px speck
    a[5, 5] = a[6, 6] = a[7, 7] = True  # diagonal 3px (8-connected) -> removed at n=3
    a[10:12, 10:12] = True            # 4px blob -> kept at n=3
    a[15, 0:20] = True                # thin line -> kept
    r = p2j.despeckle(a, 3)
    assert not r[1, 1] and not r[6, 6]
    assert r[10:12, 10:12].all() and r[15].all()
    assert (p2j.despeckle(a, 0) == a).all()
