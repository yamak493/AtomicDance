"""Checks for the shims that keep the project runnable on current toolchains."""

import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

from compat import estimate_tempo, matrix_sqrtm, torch_load
from compat.rotation_conversions import (axis_angle_to_matrix,
                                         axis_angle_to_quaternion,
                                         matrix_to_axis_angle,
                                         matrix_to_quaternion,
                                         matrix_to_rotation_6d,
                                         quaternion_apply, quaternion_multiply,
                                         quaternion_to_axis_angle,
                                         quaternion_to_matrix,
                                         rotation_6d_to_matrix)
from compat.smpl import chumpy_free_model_data


def random_rotations(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(count, 3, 3, dtype=torch.float64, generator=generator)
    rotation, upper = torch.linalg.qr(matrix)
    rotation = rotation * torch.sign(torch.diagonal(upper, dim1=-2, dim2=-1)).unsqueeze(-1)
    rotation[torch.det(rotation) < 0] *= -1.0
    return rotation


class RotationConversionTests(unittest.TestCase):
    """The fallbacks replace PyTorch3D, so they must match it numerically."""

    def setUp(self):
        self.rotations = random_rotations(256)

    def test_quaternion_round_trip(self):
        quaternions = matrix_to_quaternion(self.rotations)
        self.assertTrue(torch.allclose(quaternions.norm(dim=-1), torch.ones(256, dtype=torch.float64)))
        self.assertTrue(torch.allclose(quaternion_to_matrix(quaternions), self.rotations, atol=1e-12))

    def test_rotation_6d_round_trip(self):
        six_d = matrix_to_rotation_6d(self.rotations)
        self.assertEqual(tuple(six_d.shape), (256, 6))
        self.assertTrue(torch.allclose(rotation_6d_to_matrix(six_d), self.rotations, atol=1e-12))

    def test_axis_angle_round_trip_including_zero_rotations(self):
        generator = torch.Generator().manual_seed(1)
        axis_angle = torch.randn(64, 3, dtype=torch.float64, generator=generator)
        axis_angle *= 2.0 / axis_angle.norm(dim=-1, keepdim=True)  # stay inside +/- pi
        axis_angle[:4] = 0.0  # the Taylor branch around a zero angle
        self.assertTrue(
            torch.allclose(matrix_to_axis_angle(axis_angle_to_matrix(axis_angle)), axis_angle, atol=1e-12)
        )
        self.assertTrue(
            torch.allclose(quaternion_to_axis_angle(axis_angle_to_quaternion(axis_angle)), axis_angle, atol=1e-12)
        )

    def test_quaternion_apply_matches_matrix_product(self):
        quaternions = matrix_to_quaternion(self.rotations)
        points = torch.randn(256, 3, dtype=torch.float64)
        rotated = quaternion_apply(quaternions, points)
        expected = torch.einsum("bij,bj->bi", self.rotations, points)
        self.assertTrue(torch.allclose(rotated, expected, atol=1e-12))

    def test_quaternion_multiply_composes_rotations(self):
        first = matrix_to_quaternion(self.rotations)
        second = matrix_to_quaternion(random_rotations(256, seed=2))
        composed = quaternion_to_matrix(quaternion_multiply(first, second))
        expected = quaternion_to_matrix(first) @ quaternion_to_matrix(second)
        self.assertTrue(torch.allclose(composed, expected, atol=1e-12))


class TorchLoadTests(unittest.TestCase):
    def test_loads_checkpoint_metadata_rejected_by_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save({"stage": "planner", "args": {"seq_len": 150}, "model": {"w": torch.ones(2)}}, path)
            checkpoint = torch_load(path)
            self.assertEqual(checkpoint["args"]["seq_len"], 150)
            self.assertTrue(torch.equal(torch_load(path, mmap=True)["model"]["w"], torch.ones(2)))


class LibrosaShimTests(unittest.TestCase):
    def test_estimate_tempo_returns_positive_bpm(self):
        sample_rate = 22050
        times = np.arange(sample_rate * 5) / sample_rate
        clicks = (np.sin(2 * np.pi * 440 * times) * (np.sin(2 * np.pi * 2 * times) > 0.9)).astype(np.float32)
        self.assertGreater(estimate_tempo(y=clicks, sr=sample_rate), 0.0)


class ScipyShimTests(unittest.TestCase):
    def test_matrix_sqrtm_returns_a_matrix(self):
        matrix = np.array([[4.0, 0.0], [0.0, 9.0]])
        root = matrix_sqrtm(matrix)
        self.assertEqual(root.shape, (2, 2))
        self.assertTrue(np.allclose(root.real, np.array([[2.0, 0.0], [0.0, 3.0]])))


# Mirrors how chumpy 0.70 pickles a wrapped array: no __reduce__, just a state
# dict that drops the unpicklable weak references.
_FAKE_CHUMPY = """
import weakref


class Ch(object):
    dterms = ['x']

    def __init__(self, x):
        self.x = x
        self._parents = weakref.WeakKeyDictionary()
        self._cache = {'r': None, 'drs': weakref.WeakKeyDictionary()}

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_parents']
        del state['_cache']
        return state

    def __setstate__(self, state):
        state['_parents'] = weakref.WeakKeyDictionary()
        state['_cache'] = {'r': None, 'drs': weakref.WeakKeyDictionary()}
        object.__setattr__(self, '__dict__', state)
"""


def write_chumpy_backed_model(directory, model_builder):
    """Pickle a SMPL-like model with chumpy importable, then uninstall chumpy."""
    package = os.path.join(directory, "chumpy")
    os.makedirs(package)
    with open(os.path.join(package, "__init__.py"), "w") as handle:
        handle.write("from .ch import Ch\n")
    with open(os.path.join(package, "ch.py"), "w") as handle:
        handle.write(_FAKE_CHUMPY)

    path = os.path.join(directory, "SMPL_MALE.pkl")
    sys.path.insert(0, directory)
    try:
        import chumpy

        # The official releases were written by Python 2, i.e. protocol 2.
        with open(path, "wb") as handle:
            pickle.dump(model_builder(chumpy.Ch), handle, protocol=2)
    finally:
        sys.path.remove(directory)
        for module in [name for name in sys.modules if name.split(".")[0] == "chumpy"]:
            del sys.modules[module]
    shutil.rmtree(package)
    return path


class ChumpyFreeSmplTests(unittest.TestCase):
    def test_model_arrays_survive_without_chumpy(self):
        template = np.arange(9, dtype=np.float32).reshape(3, 3)
        weights = np.eye(3, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = write_chumpy_backed_model(
                directory,
                lambda Ch: {
                    "v_template": Ch(template),
                    "weights": Ch(weights),
                    "kintree_table": np.array([[0], [1]], dtype=np.uint32),
                    "bs_style": "lbs",
                },
            )
            with self.assertRaises(ModuleNotFoundError):
                with open(path, "rb") as handle:
                    pickle.load(handle, encoding="latin1")
            data = chumpy_free_model_data(path)

        self.assertTrue(np.array_equal(data["v_template"], template))
        self.assertTrue(np.array_equal(data["weights"], weights))
        self.assertTrue(np.array_equal(data["kintree_table"], np.array([[0], [1]])))
        self.assertEqual(data["bs_style"], "lbs")


if __name__ == "__main__":
    unittest.main()
