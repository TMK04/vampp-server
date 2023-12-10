import numpy as np
from pathlib import Path
import pickle

wd = Path(__file__).parent


def load(path):
  with open(wd / f"{path}.pkl", "rb") as f:
    return pickle.load(f)


def rfrInfer(rfr, X):
  y_pred = rfr.predict(np.array([X], dtype=np.float32))
  y_pred = np.clip(y_pred, 1, 10)
  return y_pred[0]


rfr_pe = load("pe")
rfr_clarity = load("clarity")
rfr_bv = load("bv")
