import ffmpeg
import numpy as np


def export_fig(frames, output_file, fps=5):
    h, w, _ = frames[0].shape
    process = (
        ffmpeg
        .input('pipe:',
               format='rawvideo',
               pix_fmt='rgb24',
               s='{}x{}'.format(w, h),
               filter_complex="[0]split[a][b]; [a]palettegen[palette]; [b][palette]paletteuse",
               r=fps,
               thread_queue_size=1024)
        .output(output_file, r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    for frame in frames:
        process.stdin.write(
            np.asarray(frame, dtype=np.uint8).tobytes()
        )
    process.stdin.close()
    process.wait()