import os
import subprocess

MODEL_FR_PATH = os.environ.get("MODEL_FR_PATH")
if MODEL_FR_PATH is None:
  raise Exception("MODEL_PL_PATH is not set")

OUTPUT_PATH = os.path.join(MODEL_FR_PATH, "nly_center_face")


def restoreFaces(input_dir, output_dir):
  subprocess.run([
      "python", "inference_gfpgan.py", "-i", input_dir, "-v", "1.4", "-s", "2", "-only_center_face"
  ],
                 cwd=MODEL_FR_PATH)
