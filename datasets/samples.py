from dataclasses import dataclass, fields, is_dataclass
from typing import Callable, List, NoneType, TypeVar

import torch
from jaxtyping import Float
from torch.utils.data._utils.collate import default_collate, default_collate_fn_map

T = TypeVar("T")

__all__ = ["ImageSample", "collate"]


@dataclass
class ImageSample:
    pixel_values: Float[torch.Tensor, "*batch channels height width"]


def _collate_dataclass_fn(batch: List[T], *, collate_fn_map: dict[type, Callable]) -> T:
    """Collate function for any dataclass."""
    elem = batch[0]
    return type(elem)(
        **{
            field.name: default_collate(
                [getattr(b, field.name) for b in batch], collate_fn_map=collate_fn_map
            )
            for field in fields(elem)
        }
    )


def _collate_nonetype_fn(
    batch: List[NoneType], *, collate_fn_map: dict[type, Callable]
) -> None:
    return None


# Build the collate map once
COLLATE_MAP = {
    **default_collate_fn_map,
    NoneType: _collate_nonetype_fn,
}


def collate(batch: List[T]) -> T:
    elem = batch[0]
    if is_dataclass(elem):
        return _collate_dataclass_fn(batch, collate_fn_map=COLLATE_MAP)
    return default_collate(batch, collate_fn_map=COLLATE_MAP)
