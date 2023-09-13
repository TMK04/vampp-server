from config import MODEL_RFR_PATH
import numpy as np
import pickle


def load(suffix):
  with open(f"{MODEL_RFR_PATH}{suffix}.pkl", "rb") as f:
    return pickle.load(f)


def rfrInfer(rfr, X):
  y_pred = rfr.predict(np.array([X], dtype=np.float32))
  y_pred = np.clip(y_pred, 1, 10)
  return y_pred[0]


rfr_pe = load("pe")
rfr_clarity = load("clarity")
rfr_bv = load("bv")