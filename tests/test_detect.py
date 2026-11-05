import numpy as np
import pytest

from cow_identifier import detect


@pytest.mark.parametrize("h,w", [(300, 400), (640, 640), (1000, 1000), (1234, 2345), (3648, 5472)])
def test_tiles_cover_every_pixel(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)
    for tile, x, y in detect.tile_image(img):
        assert tile.shape[0] <= detect.TILE_SIZE and tile.shape[1] <= detect.TILE_SIZE
        covered[y:y + tile.shape[0], x:x + tile.shape[1]] = True
    assert covered.all(), "some pixels (e.g. the bottom-right corner) are not covered by any tile"


def test_iou_and_nms_merge_duplicates():
    assert detect.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert detect.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    dets = [
        {"xyxy": (0, 0, 100, 100), "conf": 0.6, "label": "cow"},
        {"xyxy": (5, 5, 105, 105), "conf": 0.9, "label": "cow"},      # same cow, other tile
        {"xyxy": (500, 500, 600, 600), "conf": 0.4, "label": "cow"},  # different cow
    ]
    kept = detect.nms_merge(dets)
    assert [d["conf"] for d in kept] == [0.9, 0.4]


class _Vec(list):
    """A list with .tolist(), like the tensors YOLO returns."""
    def tolist(self):
        return list(self)


class _Box:
    def __init__(self, cls_id, xyxy, conf):
        self.cls, self.xyxy, self.conf = [cls_id], [_Vec(xyxy)], [conf]


class _FakeModel:
    """Stands in for a YOLO model: reports one box per tile."""
    def __init__(self, names, cls_id):
        self.names, self._cls = names, cls_id

    def predict(self, tile, verbose=False, conf=0.25):
        return [type("R", (), {"boxes": [_Box(self._cls, [10, 10, 50, 50], 0.8)]})()]


def test_custom_single_class_model_keeps_class_zero():
    """The fine-tuned model has one class (cow = id 0); it must not be filtered like COCO."""
    model = _FakeModel({0: "cow"}, cls_id=0)
    dets = detect.detect_on_image(model, np.zeros((300, 300, 3), dtype=np.uint8))
    assert [d["label"] for d in dets] == ["cow"]


def test_coco_model_keeps_only_animals():
    coco_names = {i: f"c{i}" for i in range(80)}
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    assert detect.detect_on_image(_FakeModel(coco_names, cls_id=19), img)[0]["label"] == "cow"
    assert detect.detect_on_image(_FakeModel(coco_names, cls_id=2), img) == []   # "car"
