from config import MODEL_RFR_PATH
import numpy as np
import pickle


def load(suffix):
  with open(f"{MODEL_RFR_PATH}{suffix}.pkl", "rb") as f:
    return pickle.load(f)


def infer(rfr, X):
  y_pred = rfr.predict(X)
  y_pred = np.clip(y_pred, 1, 10)
  return y_pred


rfr_pe = load("pe")
rfr_clarity = load("clarity")
rfr_bv = load("bv")