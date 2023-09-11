import ffmpeg
from ffmpeg import Stream


def compressVideo(input_file: str, output_file: str):
  ffmpeg.input(input_file).output(output_file,
                                  ac=1,
                                  ar=16000,
                                  crf=28,
                                  vcodec="libx265",
                                  vf="fps=1, hue=s=0").run()


def extractAudio(input_file: Stream, output_file: str):
  ffmpeg.input(input_file).audio.output(output_file, acodec="pcm_s16le").run()
