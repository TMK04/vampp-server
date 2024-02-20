import os
import pickle

from server.config import MODELS_DIR

wd = os.path.join(MODELS_DIR, "ridge")


def load(path):
  with open(os.path.join(wd, f"{path}.pkl"), "rb") as f:
    return pickle.load(f)


model_pe = load("pe")
model_clarity = load("clarity")
model_bv = load("bv")
