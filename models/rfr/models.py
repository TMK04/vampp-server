from pathlib import Path
import pickle

wd = Path(__file__).parent


def load(path):
  with open(wd / f"{path}.pkl", "rb") as f:
    return pickle.load(f)


rfr_pe = load("pe")
rfr_clarity = load("clarity")
rfr_bv = load("bv")
