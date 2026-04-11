# Source: type aliases from Older_Imp.ipynb cell 2
from __future__ import annotations
from typing import Literal
import jax


LayerType = Literal['conv', 'pool', 'flat', 'full']
ConnectionType = Literal['random', 'unique', 'flat']
LogicFamily = Literal['full', 'light']

WirePair = tuple[jax.Array, jax.Array]


__all__ = ['ConnectionType', 'LogicFamily', 'WirePair', 'LayerType']
