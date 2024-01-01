import numpy as np

from .models import rfr_bv, rfr_clarity, rfr_pe


def infer(rfr, X):
  y_pred = rfr.predict(np.array([X], dtype=np.float32))
  y_pred = np.clip(y_pred, 1, 10)
  return y_pred[0]


def inferBv(X):
  return infer(rfr_bv, X)


def inferClarity(X):
  return infer(rfr_clarity, X)


def inferPe(X):
  return infer(rfr_pe, X)
