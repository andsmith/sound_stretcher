import cv2
import numpy as np
import logging
from sound_tools.spectrograms import get_power_spectrum
from util import clip_bounds


class Spectrogram(object):
    """Scrolling spectrogram animation"""

    def __init__(self, bbox, sound, resolution_hz, resolution_sec, max_freq=None):
        self._max_freq = sound.metadata.framerate / 2 if max_freq is None else max_freq
        self._res_hz = resolution_hz
        self._res_sec = resolution_sec
        self._bbox = bbox
        self._width, self._height = self._bbox['right'] - self._bbox['left'], \
                                    self._bbox['bottom'] - self._bbox['top']

        self._data = sound.get_mono_data()
        self._f_rate = sound.metadata.framerate
        z, f, self._t_bin_centers = get_power_spectrum(self._data, self._f_rate,
                                                       resolution_hz=self._res_hz,
                                                       resolution_sec=self._res_sec)
        bin_center_centers = (self._t_bin_centers[:-1] + self._t_bin_centers[1:]) / 2
        self._t_bin_bounds = np.hstack([bin_center_centers, [sound.duration_sec]])

        # transform
        self._f_limits = np.array([0.0, self._max_freq])
        f_valid = np.logical_and(self._f_limits[0] <= f, f <= self._f_limits[1])
        self._f = f[f_valid]
        self._power = np.abs(z[f_valid, :]) ** 2
        logging.info("Created new spectrogram:  %s (F x T) in window(%s), frequencies pruned %i -> %i." % (
            self._power.shape, (self._width, self._height), f_valid.size, f_valid.sum()))

    def draw(self, frame, t, zoom_t, zoom_f, pan_f, contrast):
        """
        Add spectrogram to frame.

        :param frame:  Image to add spectrogram to (drawn within self._bbox)
        :param t:  time index (float) in [0, sound_duration_sec]
        :param zoom_t:  float in (0, inf), stretched_time_t =  normal_time_t / zoom_t (smaller is zoomed in more)
        :param zoom_f:  float in (0, 1], how much of the spectrum to see (vertically)
        :param pan_f:   float in (0, 1], which portion of the spectrum to see
        :param contrast:  float in [-1,10] see _scale_power()

        """

        # get F indices for spectrogram image
        pan_amount = (1.0 - zoom_f) * pan_f
        f_range_rel = np.array((pan_amount, pan_amount + zoom_f))
        freq_range = self._f_limits[1] * f_range_rel
        f_low_i = np.sum(self._f <= freq_range[0])
        f_high_i = np.sum(self._f <= freq_range[1])

        # get T indices for spectrogram image
        n_time_samples = int(np.ceil(self._width * zoom_t))  # pin default zoom to window size with this

        t_ind = np.sum(self._t_bin_bounds <= t)
        t_low_i, t_high_i = clip_bounds(t_ind - int(n_time_samples / 2),
                                        t_ind + int(n_time_samples / 2), n_time_samples, self._t_bin_centers.size - 1)

        # get and scale image
        power = self._power[f_low_i:f_high_i, t_low_i:t_high_i]
        power_scaled = _scale_power(power, contrast)
        spectrogram_img_mono = (power_scaled * 255.0).astype(np.uint8)
        spectrogram_img = cv2.applyColorMap(spectrogram_img_mono, cv2.COLORMAP_HOT)
        spectrogram_img_resized = cv2.resize(spectrogram_img,
                                             (self._width, self._height),
                                             cv2.INTER_NEAREST)[::-1, :, :]

        # put in frame
        frame[self._bbox['top']:self._bbox['bottom'],
        self._bbox['left']:self._bbox['right'], :3] = spectrogram_img_resized

        return spectrogram_img


def _scale_power(p, contrast):
    """
    Scale spectrogram intensities, by 1 of two methods:

        if contrast < 0, return norm(np.log((1-contrast) + p)),  where norm() divides array by max
        else return norm(p)^(-1-contrast)   (alpha correction)

    :param p: |z|^2 values
    :param contrast: float
    :return: scaled values (floats)
    """
    if contrast < 0:
        contrast = (1. - contrast)
        image_f = np.log(contrast + p)
        img = image_f / np.max(image_f)

    else:  # raw or inverse / alpha scaling of normalized values
        alpha = contrast + 1.0
        max_val = np.max(p)
        p = p / max_val
        v = p ** (1.0 / alpha) if contrast != 0. else p
        img = v
    return img
