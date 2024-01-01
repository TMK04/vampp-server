import numpy as np

from .models import model_bv, model_clarity, model_pe


def infer(model, X):
  mean = np.mean(X)
  for i in range(len(X)):
    X[i] -= mean
  X.append(mean)
  y_pred = model.predict(np.array([X], dtype=np.float32))
  y_pred = np.clip(y_pred, 1, 10)
  return y_pred[0]


def inferBv(X):
  return infer(model_bv, X)


def inferClarity(X):
  return infer(model_clarity, X)


def inferPe(X):
  return infer(model_pe, X)
