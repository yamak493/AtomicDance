"""Rendering checks for skeleton_render's ffmpeg step."""

import os
import shutil
import tempfile
import unittest

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")

from vis import skeleton_render

SAMPLE_RATE = 30 * 512


def write_audio(path, seconds=1):
    times = np.arange(SAMPLE_RATE * seconds) / SAMPLE_RATE
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * times)).astype(np.float32), SAMPLE_RATE)


def render(directory, audio_name, frames=2):
    audio_path = os.path.join(directory, audio_name)
    write_audio(audio_path)
    output_dir = os.path.join(directory, "renders")
    poses = np.zeros((frames, 24, 3), dtype=np.float32)
    poses[:, :, 2] = np.linspace(0, 1, 24)
    skeleton_render(
        poses,
        epoch="test",
        out=output_dir,
        name=audio_path,
        sound=True,
        contact=np.zeros((frames, 4), dtype=np.float32),
    )
    return os.path.join(output_dir, "test_{}.mp4".format(os.path.splitext(audio_name)[0]))


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
class SkeletonRenderTests(unittest.TestCase):
    def test_writes_video_for_names_a_shell_would_split(self):
        # The ffmpeg call used to interpolate these into a shell string, so any
        # of them silently produced no video at all.
        for audio_name in ("plain.wav", "my song.wav", "track(1).wav", "a&b.wav"):
            with self.subTest(audio_name=audio_name), tempfile.TemporaryDirectory() as directory:
                video_path = render(directory, audio_name)
                self.assertTrue(os.path.isfile(video_path), video_path)
                self.assertGreater(os.path.getsize(video_path), 0)

    def test_reports_a_missing_audio_file(self):
        with tempfile.TemporaryDirectory() as directory:
            poses = np.zeros((2, 24, 3), dtype=np.float32)
            with self.assertRaises(RuntimeError) as caught:
                skeleton_render(
                    poses,
                    epoch="test",
                    out=os.path.join(directory, "renders"),
                    name=os.path.join(directory, "absent.wav"),
                    sound=True,
                    contact=np.zeros((2, 4), dtype=np.float32),
                )
            self.assertIn("ffmpeg", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
