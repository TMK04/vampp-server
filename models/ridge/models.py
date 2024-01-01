from pathlib import Path
import pickle

wd = Path(__file__).parent


def load(path):
  with open(wd / f"{path}.pkl", "rb") as f:
    return pickle.load(f)


model_pe = load("pe")
model_clarity = load("clarity")
model_bv = load("bv")
