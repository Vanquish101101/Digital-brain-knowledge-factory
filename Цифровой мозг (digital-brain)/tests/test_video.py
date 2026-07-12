import subprocess

from kf.video import extract_audio, sample_frames


def _make_test_video(path, duration=3):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=10",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_extract_audio_produces_nonempty_wav(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_test_video(video)

    audio_path = extract_audio(video)

    assert audio_path.exists()
    assert audio_path.stat().st_size > 0


def test_sample_frames_produces_expected_frame_count(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_test_video(video, duration=3)

    frames_dir, frames = sample_frames(video, interval_seconds=1)

    assert frames_dir.exists()
    assert len(frames) >= 2
    assert all(f.exists() for f in frames)
    assert all(f.parent == frames_dir for f in frames)


def test_sample_frames_returns_at_least_one_frame_for_short_video(tmp_path):
    video = tmp_path / "short_clip.mp4"
    _make_test_video(video, duration=2)

    frames_dir, frames = sample_frames(video, interval_seconds=15)

    assert frames_dir.exists()
    assert len(frames) >= 1
    assert all(f.exists() for f in frames)
