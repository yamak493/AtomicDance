"""Shims that keep the project running on current dependency versions.

The reference environment for the paper was Python 3.7 with PyTorch 1.12 and
librosa 0.9. Hosted notebooks such as Google Colab ship much newer releases
whose defaults differ, so the helpers here paper over the incompatibilities
instead of pinning the old versions, which are not installable there.
"""

import torch

__all__ = ["estimate_tempo", "matrix_sqrtm", "torch_load"]


def torch_load(path, map_location="cpu", mmap=None):
    """``torch.load`` that keeps the pre-2.6 unpickling behaviour.

    PyTorch 2.6 flipped the ``weights_only`` default to ``True``, which refuses
    checkpoints holding anything but tensors and plain containers. This project
    stores argparse namespaces and metadata alongside the weights, so full
    unpickling is required. Checkpoints are trusted local files, either written
    by ``train_atomic.py`` or downloaded from the project's own release.
    """
    kwargs = {"map_location": map_location}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(str(path), weights_only=False, **kwargs)
    except TypeError:
        # PyTorch < 1.13 has no weights_only parameter; < 2.1 has no mmap.
        kwargs.pop("mmap", None)
        return torch.load(str(path), **kwargs)


def estimate_tempo(y, sr):
    """Estimate a starting BPM across librosa versions.

    ``librosa.beat.tempo`` moved to ``librosa.feature.rhythm.tempo`` in 0.10 and
    the alias is dropped in 1.0.
    """
    try:
        from librosa.feature.rhythm import tempo
    except ImportError:  # librosa < 0.10
        from librosa.beat import tempo

    return float(tempo(y=y, sr=sr)[0])


def matrix_sqrtm(matrix):
    """Principal matrix square root, without the removed ``disp`` argument.

    SciPy 1.18 drops ``disp`` and returns the matrix alone instead of a
    ``(matrix, error)`` pair.
    """
    import warnings

    from scipy import linalg

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg.sqrtm(matrix)
    return result[0] if isinstance(result, tuple) else result
