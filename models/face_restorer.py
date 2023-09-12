from config import MODEL_FR_PATH
import os
import subprocess


def restoreFaces(input_dir, output_dir):
  subprocess.run([
      "python",
      "inference_gfpgan.py",
      "-i",
      os.path.abspath(input_dir),
      "-o",
      os.path.abspath(output_dir),
      "-v",
      "1.4",
      "-s",
      "1",
  ],
                 cwd=MODEL_FR_PATH)
