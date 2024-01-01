from typing import Any, Dict, Generator, List, Tuple


def DictKeyArr(key_iter):
  dict_key_arr: Dict[str, List[Any]] = {key: [] for key in ["i", *key_iter]}
  return dict_key_arr


def batchGen(gen: Generator[Tuple[str, Any], None, None], batch_size: int):
  i_batch = []
  x_batch = []
  l = 0
  for i, x in gen:
    i_batch.append(i)
    x_batch.append(x)
    l += 1
    if l == batch_size:
      yield i_batch, x_batch
      i_batch = []
      x_batch = []
  if l > 0:
    yield i_batch, x_batch
