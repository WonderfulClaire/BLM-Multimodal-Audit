import pytest
from visual_pretrain.binpack import (
    ImageSample,
    OfflineBinPackerConfig,
    offline_binpack,
    OnlineBinPacker,
    OnlineBinPackerConfig,
)
from visual_pretrain.dynamic_frames import (
    VideoSample,
    VideoBinPackerConfig,
    video_binpack,
)


@pytest.mark.parametrize("online", [False, True])
def test_oversized_image_is_rejected_not_silently_dropped(online):
    sample = ImageSample("large", (280, 280))
    with pytest.raises(ValueError, match="budget"):
        if online:
            OnlineBinPacker(OnlineBinPackerConfig(bin_size=10)).add(sample)
        else:
            offline_binpack([sample], OfflineBinPackerConfig(bin_size=10))


def test_video_truncates_paths_as_well_as_length():
    bins = video_binpack(
        [VideoSample("v", ["a", "b", "c"], 3)],
        VideoBinPackerConfig(bin_size=2, max_frames_per_video=2),
    )
    assert bins[0][0].frame_paths == ["a", "b"]


def test_video_cannot_exceed_bin_capacity():
    with pytest.raises(ValueError, match="budget"):
        video_binpack(
            [VideoSample("v", ["a", "b", "c"], 3)],
            VideoBinPackerConfig(bin_size=2, max_frames_per_video=3),
        )
