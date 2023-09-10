import ffmpeg

process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(
    1280, 720)).output("out.mp4").overwrite_output().run_async(pipe_stdin=True))
