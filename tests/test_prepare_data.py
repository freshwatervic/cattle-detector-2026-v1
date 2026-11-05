import pytest
from PIL import Image

from cow_identifier import prepare_data


def _make_fake_waid(root):
    """Two train images: one with cattle (class 1) + sheep (class 0), one sheep-only."""
    for split in ("train", "valid", "test"):
        (root / "WAID" / "images" / split).mkdir(parents=True)
        (root / "WAID" / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(root / "WAID" / "images" / split / "a.jpg")
        Image.new("RGB", (10, 10)).save(root / "WAID" / "images" / split / "b.jpg")
        (root / "WAID" / "labels" / split / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
        (root / "WAID" / "labels" / split / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_build_dataset_keeps_only_cattle_as_class_zero(tmp_path):
    _make_fake_waid(tmp_path / "waid")
    out = tmp_path / "dataset"

    counts = prepare_data.build_dataset(tmp_path / "waid", out)

    assert counts == {"train": 1, "valid": 1, "test": 1}       # sheep-only image dropped
    label = (out / "train" / "labels" / "a.txt").read_text().strip().splitlines()
    assert label == ["0 0.500000 0.500000 0.100000 0.100000"]  # cattle -> 0, sheep removed
    assert "names: ['cow']" in (out / "data.yaml").read_text()


def test_rebuild_replaces_a_previous_dataset(tmp_path):
    _make_fake_waid(tmp_path / "waid")
    out = tmp_path / "dataset"
    prepare_data.build_dataset(tmp_path / "waid", out)
    prepare_data.build_dataset(tmp_path / "waid", out)          # second run must not fail
    assert (out / "data.yaml").exists()


def test_refuses_to_delete_an_unrelated_folder(tmp_path):
    _make_fake_waid(tmp_path / "waid")
    precious = tmp_path / "my_project"
    precious.mkdir()
    (precious / "thesis.docx").write_text("do not delete")

    with pytest.raises(SystemExit):
        prepare_data.build_dataset(tmp_path / "waid", precious)
    assert (precious / "thesis.docx").exists()
