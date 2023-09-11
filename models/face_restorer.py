import os
import subprocess

MODEL_FR_PATH = os.environ.get("MODEL_FR_PATH")
if MODEL_FR_PATH is None:
  raise Exception("MODEL_PL_PATH is not set")


def restoreFaces(input_dir, output_dir):
  subprocess.run([
      "python", "inference_gfpgan.py", "-i",
      os.path.abspath(input_dir), "-o",
      os.path.abspath(output_dir), "-v", "1.4", "-s", "2"
  ],
                 cwd=MODEL_FR_PATH)
