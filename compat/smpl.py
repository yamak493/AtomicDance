"""Load the SMPL body model without requiring chumpy.

``smplx`` unpickles ``SMPL_*.pkl`` directly, and the official files store their
arrays as ``chumpy`` objects. chumpy 0.70 is the last release and it imports
neither on Python 3.11+ (it calls the removed ``inspect.getargspec``) nor
against NumPy 2 (it uses the removed ``np.bool``/``np.object`` aliases), so on
Colab the plain ``smplx.SMPL(model_path=...)`` call fails while unpickling.

The model files only ever use chumpy as a container: every value the SMPL model
reads is a plain array wrapped in a ``chumpy.Ch``. Unpickling those wrappers
into a stub and taking their payload therefore reproduces the arrays exactly,
with no autodiff behaviour lost -- ``smplx`` converts them to NumPy and torch
immediately anyway.
"""

import pickle

import numpy as np

__all__ = ["chumpy_free_model_data", "load_smpl"]

_MAX_UNWRAP_DEPTH = 16


class _ChumpyPlaceholder:
    """Stands in for a chumpy object while unpickling a SMPL model file."""

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            # (instance dict, slot dict); SMPL models only populate the former.
            state = state[0] or state[1] or {}
        if isinstance(state, dict):
            self.__dict__.update(state)

    def payload(self):
        """Return the wrapped array (``Ch`` keeps it under the ``x`` term)."""
        if "x" not in self.__dict__:
            raise ValueError("unsupported chumpy object in SMPL model file")
        return self.__dict__["x"]


class _ChumpyFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _ChumpyPlaceholder
        return super().find_class(module, name)


def _unwrap(value):
    for _ in range(_MAX_UNWRAP_DEPTH):
        if not isinstance(value, _ChumpyPlaceholder):
            return value
        value = value.payload()
    raise ValueError("chumpy objects in the SMPL model file are nested too deeply")


def chumpy_free_model_data(model_path):
    """Read a SMPL ``.pkl`` into plain NumPy/SciPy values."""
    with open(str(model_path), "rb") as handle:
        data = _ChumpyFreeUnpickler(handle, encoding="latin1").load()
    unwrapped = {}
    for key, value in data.items():
        value = _unwrap(value)
        if isinstance(value, np.ndarray) and value.dtype == object:
            raise ValueError("unexpected object array for '{}'".format(key))
        unwrapped[key] = value
    return unwrapped


def load_smpl(model_path, gender="MALE", batch_size=1, **kwargs):
    """Build an ``smplx.SMPL`` model, with or without chumpy installed."""
    from smplx import SMPL

    try:
        return SMPL(
            model_path=str(model_path), gender=gender, batch_size=batch_size, **kwargs
        )
    except ModuleNotFoundError as error:
        if error.name != "chumpy":
            raise

    from smplx.utils import Struct

    return SMPL(
        model_path=str(model_path),
        gender=gender,
        batch_size=batch_size,
        data_struct=Struct(**chumpy_free_model_data(model_path)),
        **kwargs
    )
