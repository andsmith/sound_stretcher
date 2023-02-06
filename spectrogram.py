import cv2
import numpy as np
import logging
from sound_tools.spectrograms import get_power_spectrum
from util import clip_bounds, draw_v_line
from layout import Layout
from gui_utils.coordinate_grids import CartesianGrid


def get_bin_bounds(bin_centers, width):
    center_centers = (bin_centers[:-1] + bin_centers[1:]) / 2
    return np.hstack([[center_centers[0] - width], center_centers, [center_centers[-1] + width]])


class Spectrogram(object):
    """Scrolling spectrogram animation"""

    def __init__(self, bbox, sound, resolution_hz, resolution_sec, freq_range=None):
        self._cursor_color = np.array(Layout.get_color('playback_cursor'))
        self._cursor_width = Layout.CURSOR_WIDTH
        self._res_hz = resolution_hz
        self._res_sec = resolution_sec
        self._bbox = bbox
        self._width, self._height = self._bbox['right'] - self._bbox['left'], \
                                    self._bbox['bottom'] - self._bbox['top']

        self._data = sound.get_mono_data()
        self._f_rate = sound.metadata.framerate

        freq_range = (0, sound.metadata.framerate / 2) if freq_range is None else freq_range

        z, f, self._t_bin_centers = get_power_spectrum(self._data, self._f_rate, freq_range=freq_range,
                                                       resolution_hz=self._res_hz, resolution_sec=self._res_sec)

        # throw away DC component & frequencies out of range
        z = z[1:, :]
        f = f[1:]
        f_valid = np.logical_and(freq_range[0] <= f, f <= freq_range[1])
        self._f = f[f_valid]
        self._freq_range = self._f[0], self._f[-1]
        self._power = np.abs(z[f_valid, :]) ** 2

        logging.info(
            "Created new spectrogram:  %s (F x T) in window (%s)." % (self._power.shape, (self._width, self._height)))

        self._grid = CartesianGrid(self._bbox,
                                   adjustability=(False, True),
                                   axis_labels=(None, 'f (Hz)'),
                                   colors={'bkg': None},
                                   draw_props={'user_marker': False, 'draw_bounding_box': False,
                                               'show_ticks': (False, True)})

    def draw(self, frame, t, zoom_t, zoom_f, pan_f, contrast, cursor=False, axes=True):
        """
        Add spectrogram to frame.

        :param frame:  Image to add spectrogram to (drawn within self._bbox)
        :param t:  time index (float) in [0, sound_duration_sec]
        :param zoom_t:  float in (0, inf), stretched_time_t =  normal_time_t / zoom_t (smaller is zoomed in more)
        :param zoom_f:  float in (0, 1], how much of the spectrum to see (vertically)
        :param pan_f:   float in (0, 1], which portion of the spectrum to see
        :param contrast:  float in [-1,10] see _scale_power()
        :param cursor:  Draw vertical line at time t
        :param axes: label axes if True
        """

        # get F indices for spectrogram image
        pan_amount = (1.0 - zoom_f) * pan_f
        f_range_rel = np.array((pan_amount, pan_amount + zoom_f))
        freq_range = self._freq_range[1] * f_range_rel
        f_low_i = np.sum(self._f <= freq_range[0])
        f_high_i = np.sum(self._f <= freq_range[1]) - 1

        # get T indices for spectrogram image
        n_time_samples = int(np.ceil(self._width * zoom_t))  # pin default zoom to window size with this

        t_ind = np.sum(self._t_bin_centers + self._res_sec / 2 < t)
        t_low_i, t_high_i = clip_bounds(t_ind - int(n_time_samples / 2),
                                        t_ind + int(n_time_samples / 2), n_time_samples, self._t_bin_centers.size - 1)
        if t_high_i == t_low_i:
            t_high_i += 1
            if t_high_i > self._t_bin_centers.size - 1:
                t_low_i -= 1
                t_high_i -= 1

        # get and scale image
        power = self._power[f_low_i:f_high_i, t_low_i:t_high_i]
        power_scaled = _scale_power(power, contrast)
        spectrogram_img_mono = (power_scaled * 255.0).astype(np.uint8)
        spectrogram_img = cv2.applyColorMap(spectrogram_img_mono, cv2.COLORMAP_HOT)
        spectrogram_img_resized = cv2.resize(spectrogram_img,
                                             (self._width, self._height),
                                             cv2.INTER_NEAREST)[::-1, :, :]
        # draw cursor
        if cursor:
            t_relative = (t_ind - t_low_i) / (t_high_i - t_low_i)
            cursor_x = int(t_relative * self._width + self._bbox['left'] - self._cursor_width / 2)
            draw_v_line(spectrogram_img_resized, cursor_x, self._cursor_width, self._cursor_color[:3])
        # put in frame
        frame[self._bbox['top']:self._bbox['bottom'],
        self._bbox['left']:self._bbox['right'], :3] = spectrogram_img_resized

        if axes:
            f_bounds = self._f[f_low_i] - self._res_hz / 2, \
                       self._f[f_high_i] + self._res_hz / 2
            t_bounds = self._t_bin_centers[t_low_i] - self._res_sec / 2, \
                       self._t_bin_centers[t_high_i] + self._res_sec / 2

            self._grid.set_param_range(0, t_bounds[0], t_bounds[1])
            self._grid.set_param_range(1, f_bounds[0], f_bounds[1])
            self._grid.draw(frame)


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
