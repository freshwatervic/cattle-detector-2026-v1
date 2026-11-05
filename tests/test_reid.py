import numpy as np
from PIL import Image

from cow_identifier import reid


def _unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


GALLERY = {
    "Bella": {"embedding": _unit(1, 0, 0), "n_photos": 3},
    "Greta": {"embedding": _unit(0, 1, 0), "n_photos": 2},
}


def test_rank_matches_orders_best_first():
    ranked = reid.rank_matches(_unit(0.9, 0.1, 0), GALLERY)
    assert [n for n, _ in ranked] == ["Bella", "Greta"]
    assert ranked[0][1] > ranked[1][1]


def test_margin_and_uncertain_logic():
    clear = [("Bella", 0.81), ("Greta", 0.42)]
    guessing = [("Bella", 0.71), ("Greta", 0.70)]

    assert reid.score_margin(clear) > 0.3
    assert reid.decide_label(clear) == "Bella"
    assert reid.decide_label(guessing) == "Bella"                       # defaults never flag
    assert reid.decide_label(guessing, min_margin=0.05) == reid.UNCERTAIN
    assert reid.decide_label(clear, match_threshold=0.9) == reid.UNCERTAIN
    assert reid.score_margin([("Only", 0.5)]) == float("inf")           # one-cow gallery


def test_identify_pipeline_with_fakes(tmp_path, monkeypatch):
    """Detector -> crop -> match plumbing, without PyTorch or real weights."""
    Image.new("RGB", (200, 100), "white").save(tmp_path / "drone.jpg")

    class Coords(list):
        def tolist(self):          # YOLO returns tensors, which have .tolist()
            return list(self)

    class Box:
        xyxy = [Coords([10, 10, 60, 60])]

    class Detector:
        def predict(self, image, conf, verbose):
            assert isinstance(image, Image.Image)
            return [type("R", (), {"boxes": [Box()]})()]

    monkeypatch.setattr(reid, "embed_pil", lambda model, img: _unit(1, 0.1, 0))
    found = reid.identify_cows_in_image(Detector(), None, GALLERY, tmp_path / "drone.jpg")

    assert len(found) == 1
    assert found[0]["box"] == (10, 10, 60, 60)
    assert found[0]["best_match"] == "Bella"
    assert [n for n, _ in found[0]["ranked"]] == ["Bella", "Greta"]


def test_exif_rotation_is_applied(tmp_path):
    """Phone photos stored sideways must be rotated upright before fingerprinting."""
    img = Image.new("RGB", (40, 20), "white")
    exif = img.getexif()
    exif[0x0112] = 6                       # "rotate 90 degrees clockwise to display"
    img.save(tmp_path / "phone.jpg", exif=exif)
    assert reid.open_rgb(tmp_path / "phone.jpg").size == (20, 40)
