import subprocess


def compressVideo(input_file, output_file):
  command = [
      "ffmpeg", "-i", input_file, "-vf", "fps=1", "-c:v", "libx265", "-crf", "28", "-acodec",
      "pcm_s16le", "-ar", "16000", "-ac", "1", "--", output_file
  ]
  subprocess.run(command, check=True)


def extractAudio(input_file, output_file):
  command = ["ffmpeg", "-i", input_file, "-vn", "--", output_file]
  subprocess.run(command, check=True)