from typing import Any, Literal, NamedTuple, Optional

import jax
import numpy as np
from torch_dimcheck import A

PyTree = Any
PRNGKey = Any


class DataError(RuntimeError):
    pass


class NaNError(RuntimeError):
    pass


def named_tuple_repr(self) -> str:
    if not (isinstance(self, tuple) and hasattr(self, "_fields")):
        raise TypeError("`self` doesn't look like a NamedTuple")

    def _shape(obj):
        if hasattr(obj, "shape"):
            return tuple(obj.shape)
        else:
            return obj

    fields = []
    for field_name in self._fields:
        field_value = getattr(self, field_name)
        fields.append(f"{field_name}={_shape(field_value)}")

    fields = ", ".join(fields)
    return f"{type(self).__name__}({fields})"


def torch_to(data, target: Literal["np", "jnp", "pmap"]):
    assert target in ("np", "jnp", "pmap"), target

    def _transfer(tensor):
        if hasattr(tensor, "numpy"):
            array: np.ndarray = tensor.numpy()
        else:
            array: np.ndarray = tensor

        if target == "np":
            return array

        if target == "jnp":
            return jax.device_put(array)

        # pmap
        devices = jax.devices()
        n = len(devices)
        B = array.shape[0]
        assert B % n == 0, (B, n)

        shaped = array.reshape(n, -1, *array.shape[1:])
        return jax.device_put_sharded(list(shaped), devices)

    return jax.tree_map(_transfer, data)


class BatchIndexHelper:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        def _index_one(item):
            if not hasattr(item, "__array__"):
                return item
            return item[index]

        return jax.tree_map(_index_one, self.data)

    def __repr__(self):
        return f"<_IndexHelper data={self.data}>"


class Example(NamedTuple):
    points: np.ndarray
    ctx: Optional[Any]
    # We use () instead of None because PyTorch dataloaders don't like None :(
    extras: Any = ()

    __repr__ = named_tuple_repr
    torch_to = torch_to

    @property
    def index(self):
        return BatchIndexHelper(self)

    def discard_extras(self):
        return self._replace(extras=())


class Context3d(NamedTuple):
    image: Optional[np.ndarray]
    K: np.ndarray
    wmat: Optional[np.ndarray] = ()

    __repr__ = named_tuple_repr
    torch_to = torch_to

    @property
    def index(self):
        return BatchIndexHelper(self)


class LogpDetails(NamedTuple):
    logp: A[""]
    prior_logp: A[""]
    delta_reparam: A[""]
    delta_jacobian: A[""]
    trajectory_diff: A["T D*"]
    trajectory_data: A["T D*"]
    latent: A["D*"]

    __repr__ = named_tuple_repr


class SampleDetails(NamedTuple):
    latent: A["X*"]
    sample_diff: A["X*"]
    sample_data: A["X*"]
    trajectory_diff: A["T X*"]
    trajectory_data: A["T X*"]

    __repr__ = named_tuple_repr
