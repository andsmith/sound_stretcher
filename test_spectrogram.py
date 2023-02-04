import cv2
import numpy as np
import logging
from layout import Layout

from sound_tools.sound import Sound
import time
from spectrogram import Spectrogram


def test_spectrogram():
    size = (1000, 800)
    box = {'top': 100, 'left': 10, 'bottom': 790, 'right': 990}
    blank = np.zeros((size[1], size[0], 4), dtype=np.uint8) + 255
    win_name = "test spectrogram"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    sound = Sound("Aphelocoma_californica_-_California_Scrub_Jay_-_XC110976.wav")
    params = Layout.get_value('spectrogram_params')
    spec = Spectrogram(box, sound, params['resolution_hz'], params['resolution_sec'])
    n_frames = 0
    t_start = time.perf_counter()
    controls = {'zoom_t': {'up_key': 't',
                           'down_key': 'g',
                           'value': 1.0},
                'zoom_f': {'up_key': 'f',
                           'down_key': 'v',
                           'value': 1.0},
                'pan_f': {'up_key': 'p',
                          'down_key': 'o',
                          'value': 0.0},
                }

    t = 0.
    draw_times = []
    waits = []

    target_fps = 60.
    fps_delay = 1. / target_fps
    start_time = time.perf_counter()
    while True:

        frame = blank.copy()

        control = {k: controls[k]['value'] for k in controls}

        _ = spec.draw(frame, t=t, contrast=-0.1, **control)

        # scroll
        t += 0.01
        if t > sound.duration_sec:
            t = 0.

        # delay
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed < fps_delay:
            remaining_wait = fps_delay - elapsed
            waits.append(remaining_wait)
            time.sleep(remaining_wait)
        else:
            waits.append(0)
        start_time = time.perf_counter()

        # display
        cv2.imshow(win_name, frame)
        draw_times.append(time.perf_counter())

        # keyboard
        k = cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break
        for control in controls:
            if k & 0xff == ord(controls[control]['up_key']):
                controls[control]['value'] += 0.05
                if controls[control]['value'] > 1.0:
                    controls[control]['value'] = 1.0
                print("Adjusting %s up:  %.3f" % (control, controls[control]['value']))
            elif k & 0xff == ord(controls[control]['down_key']):
                controls[control]['value'] -= 0.05
                if controls[control]['value'] < 0:
                    controls[control]['value'] = 0
                print("Adjusting %s down:  %.3f" % (control, controls[control]['value']))

        # print FPS
        n_frames += 1
        now = time.perf_counter()
        if now - t_start > 3.0:
            print("FPS:  %.4f, mean delay:  %.6f %%" % (n_frames / (now - t_start), np.mean(waits) * 100 / fps_delay))
            waits = []

            t_start, n_frames = now, 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_spectrogram()
    logging.info("Tests complete.")
