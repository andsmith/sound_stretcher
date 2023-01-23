import logging
import matplotlib.pyplot as plt
from sound import Sound, get_encoding_type
from segmentation import SimpleSegmentation
import numpy as np
from scipy.interpolate import interp1d


def _make_test_sound():
    rate = 48000
    max_vol = 32767 * .25
    sound = Sound(framerate=rate, sampwidth=2)
    duration_sec = 4.0
    n_frames = int(rate * duration_sec)
    sample_t = np.linspace(0, duration_sec, n_frames + 1)[:-1]
    data = (np.sin(sample_t * np.pi * 2.0 * 440.0) * max_vol).astype(get_encoding_type(sound.metadata))

    amp_points = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0., 0.]
    amp_point_indices = np.linspace(0, n_frames - 1, len(amp_points)).astype(np.int64)
    amp_points_t = sample_t[amp_point_indices.astype(np.int64)]

    amp_interp = interp1d(amp_points_t, amp_points)
    amp_envelope = amp_interp(sample_t)
    data = data * amp_envelope

    sound.set_data([data])
    sound.write("440_saw.wav")
    return sound


def test_segmentation():
    test_sound = _make_test_sound()
    segs = SimpleSegmentation(test_sound, smoothing_window_sec=0.01)
    dims = {'top': 0, 'bottom': 700, 'left': 0, 'right': 1200}
    image = np.zeros((dims['bottom'], dims['right'], 4), dtype=np.uint8)

    image += np.array((0, 0, 0, 255), dtype=np.uint8).reshape((1, 1, -1))
    segments = segs.get_segmentation(threshold=.04)

    segs.draw_segmented_waveform(image, segments, )

    plt.imshow(image)
    plt.show()


def _segmentation_sandbox():
    logging.basicConfig(level=logging.INFO)
    test_sound = Sound('Aphelocoma_californica_-_California_Scrub_Jay_-_XC110976.wav')
    segs = SimpleSegmentation(test_sound, smoothing_window_sec=0.02)
    seg = segs.get_segmentation(0.03)

    import pprint
    pprint.pprint(seg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _segmentation_sandbox()
    # test_segmentation()
