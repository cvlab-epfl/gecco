from typing import List, NamedTuple
from dataclasses import dataclass, fields

import torch
from torch import Tensor

def _raw_repr(obj) -> list[str]:
    lines = []
    lines.append(f'{type(obj).__name__}(')

    for name, value in obj._enumerate_fields():
        if hasattr(value, '_raw_repr'):
            head, *tail, end = value._raw_repr()
            lines.append(f' {name}={head}')
            for line in tail:
                lines.append(f'  {line}')
            lines.append(f' {end}')
        elif torch.is_tensor(value):
            lines.append(f' {name}={tuple(value.shape)},')
        else:
            lines.append(f' {name}={value},')

    lines.append(f')')
    return lines

def apply_to_tensors(obj, f: callable) -> object:
    applied = {}
    for name, value in obj._enumerate_fields():
        if hasattr(value, 'apply_to_tensors'):
            applied[name] = value.apply_to_tensors(f)
        elif torch.is_tensor(value):
            applied[name] = f(value)
        else:
            applied[name] = value
    
    return type(obj)(**applied)

@dataclass(repr=False)
class TensorMixin:
    def _enumerate_fields(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    apply_to_tensors = apply_to_tensors
 
    def to(self, *args, **kwargs):
        return self.apply_to_tensors(lambda t: t.to(*args, **kwargs))
    
    _raw_repr = _raw_repr

    def __repr__(self) -> str:
        return '\n'.join(self._raw_repr())
    
    @classmethod
    def collate_fn(cls, batch):
        return torch.utils.data._utils.collate.collate(
            batch,
            collate_fn_map={cls: cls.stack}
        )
    
class DataError(RuntimeError):
    pass

def _named_tuple_enumerate_fields(obj: NamedTuple):
    yield from obj._asdict().items()

class Context3d(NamedTuple):
    image: Tensor
    K: Tensor

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return '\n'.join(self._raw_repr())

class Example(NamedTuple):
    data: Tensor
    ctx: Context3d | None

    _enumerate_fields = _named_tuple_enumerate_fields
    _raw_repr = _raw_repr
    apply_to_tensors = apply_to_tensors

    def __repr__(self) -> str:
        return '\n'.join(self._raw_repr())