from ..components import device, infer, toTensor
from .models import attire_model, multitask_model


def batchInferMultitask(batch):
  batch_tensor = toTensor(batch).to(device)
  multitask_pred = infer(multitask_model, batch_tensor)
  return multitask_pred


def batchInferAttire(batch):
  batch_tensor = toTensor(batch).to(device)
  attire_pred = infer(attire_model, batch_tensor)
  return attire_pred[:, 0]
