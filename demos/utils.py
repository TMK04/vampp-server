import json

X_pe_keys = ("moving", "smiling", "upright", "ec", "pa", "speech_enthusiasm")
X_clarity_keys = ("speech_clarity", "pitch_Clarity")
X_bv_keys = ("pitch_Creativity", "pitch_Feasibility", "pitch_Impact")
X_keys = set(X_pe_keys + X_clarity_keys + X_bv_keys)


def dumpKv(k: str, v):
  return json.dumps({"k": k, "v": v})
