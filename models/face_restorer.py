import os
import subprocess

MODEL_FR_PATH = os.environ.get("MODEL_FR_PATH")
if MODEL_FR_PATH is None:
  raise Exception("MODEL_PL_PATH is not set")

MODEL_FR_OUTPUT_DIR = os.path.join(MODEL_FR_PATH, "nly_center_face")


def restoreFaces(input_dir):
  subprocess.run([
      "python", "inference_gfpgan.py", "-i", input_dir, "-o", MODEL_FR_OUTPUT_DIR, "-v", "1.4",
      "-s", "2", "-only_center_face"
  ],
                 cwd=MODEL_FR_PATH)
