import os
import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: Path) -> Path:
    fd, output_path_str = tempfile.mkstemp(prefix=f"{Path(video_path).stem}_", suffix="_audio.wav")
    os.close(fd)
    output_path = Path(output_path_str)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def sample_frames(
    video_path: Path, interval_seconds: int, max_frames: int | None = None
) -> tuple[Path, list[Path]]:
    frames_dir = Path(tempfile.mkdtemp(prefix="kf_frames_"))
    pattern = frames_dir / "frame_%04d.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd.append(str(pattern))
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vframes", "1", str(pattern)],
            check=True,
            capture_output=True,
        )
        frames = sorted(frames_dir.glob("frame_*.png"))
    return frames_dir, frames
