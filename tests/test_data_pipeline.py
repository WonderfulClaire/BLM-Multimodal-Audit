import copy
import pytest
import torch
from PIL import Image
from visual_pretrain.pipeline import process_record
from visual_pretrain.dynamic_frames import sample_frames


@pytest.fixture
def record(tmp_path):
    Image.new("RGB", (56, 56), "red").save(tmp_path / "image.png")
    return {
        "id": "x",
        "image": "image.png",
        "caption": "red square",
        "phrases": ["red square"],
        "detections": [{"phrase": "red square", "bbox": [0, 0, 56, 56], "score": 0.9}],
        "negatives": {"red square": {"text": "blue square", "verified": True}},
    }


def test_pipeline_accepts_reviewed_and_rejects_unverified_negative(record, tmp_path):
    result, rejected = process_record(record, tmp_path)
    assert len(result["regions"]) == 1 and not rejected
    record["negatives"]["red square"]["verified"] = False
    result, rejected = process_record(record, tmp_path)
    assert (
        not result["regions"] and rejected[0]["reason"] == "negative_needs_verification"
    )


def test_pipeline_rejects_identical_negative_and_invalid_boxes(record, tmp_path):
    bad = copy.deepcopy(record)
    bad["negatives"]["red square"]["text"] = "red square"
    assert not process_record(bad, tmp_path)[0]["regions"]
    record["detections"][0]["bbox"] = [0, 0, 100, 56]
    assert process_record(record, tmp_path)[1][0]["reason"] == "invalid_detection"


def test_missing_detector_does_not_create_random_boxes(record, tmp_path):
    del record["detections"]
    with pytest.raises(ValueError, match="Missing boxes"):
        process_record(record, tmp_path)


def test_frame_budget_and_temporal_order():
    frames = torch.zeros(20, 3, 4, 4)
    frames[11:] = 1
    indices = sample_frames(frames, token_budget=32, tokens_per_frame=8)
    assert len(indices) <= 4 and list(indices) == sorted(set(indices))
    assert 11 in indices
