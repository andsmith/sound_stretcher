"""
Separate audio foreground / noise
"""
import logging
import numpy as np
from sound import Sound
from util import compact_intervals, get_interval_compliment
from spectrograms import get_power_spectrum
from scipy.ndimage import gaussian_filter1d


class SimpleSegmentation(object):
    """
    Threshold on sound energy in 1000-10000 hz band
    """
    FILTER_RES_HZ = 150.0
    FILTER_RES_SEC = 0.001
    FILTER_FREQS = (800.0, 10000.0)

    def __init__(self, sound, smoothing_window_sec=0.05):
        logging.info("Creating SimpleSegmentation(smoothing_window_sec=%.4f sec, sound has %i samples)." %
                     (smoothing_window_sec, sound.metadata.nframes))
        self._sound = sound
        self._window_sec = smoothing_window_sec
        self._analyze()

    def _analyze(self):
        """
        Calc integral of power spectrum over relevant frequencies to get power(t)
        """
        # partition data
        data = self._sound.get_mono_data()
        spectrum, f, t = get_power_spectrum(data, self._sound.metadata.framerate,
                                            SimpleSegmentation.FILTER_RES_HZ,
                                            SimpleSegmentation.FILTER_RES_SEC,
                                            SimpleSegmentation.FILTER_FREQS)
        self._dt = np.mean(np.diff(t))
        self._smoothing_width = int(self._window_sec / self._dt)

        power = np.sum(np.abs(spectrum) ** 2., axis=0)
        logging.info("Smoothing power(t) at width %.4f sec (%i samples)." % (self._window_sec,
                                                                             self._smoothing_width))

        power_smoothed = gaussian_filter1d(power, self._smoothing_width)
        power_normalized = power_smoothed / np.max(power_smoothed)
        self._power = power_normalized
        self._timestamps = t

    def get_segmentation(self, threshold, margin_duration_sec=0.0):
        """
        Segments are portions of peaks of smoothed power(t) above threshold (and margin).
        :param threshold: number in [0,1]
        :param margin_duration_sec:  Add this much to the ends of each segment (buggy?)
        :return:  dict('start_times': start timestamps of each segment of sound,
                       'stop_times': stop timestamps of each segment of sound,
                       'starts':  numpy array of the first index of each segment
                       'stop': numpy array of the last index of each segment}
        """
        valid = np.int64(self._power >= threshold)
        if np.sum(valid) == 0:
            return []
        segment_starts = np.where((valid[1:] - valid[:-1]) == 1)[0] + 1
        segment_stops = np.where((valid[1:] - valid[:-1]) == -1)[0] + 1
        margin_samples = int(margin_duration_sec * self._sound.metadata.framerate)
        if valid[0]:
            segment_starts = np.hstack([[0], segment_starts])
        if valid[-1]:
            segment_stops = np.hstack([segment_stops, [valid.size - 1]])
        segment_starts -= margin_samples
        segment_stops += margin_samples
        intervals = [(segment_starts[i], segment_stops[i]) for i in range(len(segment_starts))]
        intervals = compact_intervals(intervals, valid.size - 1)
        starts, stops = [seg[0] for seg in intervals], [seg[1] for seg in intervals]
        start_times = self._timestamps[starts]
        stop_times = self._timestamps[stops]
        start_indices = (start_times * self._sound.metadata.framerate).astype(np.int64)
        stop_indices = (stop_times * self._sound.metadata.framerate).astype(np.int64)
        logging.info("Segmentation at threshold %.5f:  sound has %i segments." % (threshold, len(intervals),))

        return {'starts': np.array(start_indices),
                'stops': np.array(stop_indices),
                'start_times': start_times,
                'stop_times': stop_times}

    def draw_segmented_waveform(self, image, segmentation, bbox=None, sound_color=(255, 255, 255, 255),
                                noise_color=(128, 128, 128, 255)):
        """
        Draw waveform on an image, in 2 colors.
        :param image:  draw on this image
        :param segmentation:  output of get_segmentation
        :param bbox:  dict with 'top', 'bottom','left','right', bounds within image to draw (scaled to max amplitude)
        :param sound_color: draw waveform of sound segments in this color
        :param noise_color: draw waveform between segments in this color
        """
        if bbox is None:
            bbox = {'top': 0, 'bottom': image.shape[0], 'left': 0, 'right': image.shape[1]}

        data = self._sound.get_mono_data()
        audio_mean = np.mean(data)
        # bin audio into number of horizontal pixels, get max & min for each one
        width = bbox['right'] - bbox['left']

        bin_size = int(data.size / width)
        partitions = data[:bin_size * width].reshape(width, bin_size)
        max_vals, min_vals = np.max(partitions - audio_mean, axis=1), np.min(partitions - audio_mean, axis=1)
        audio_max, audio_min = np.max(max_vals), np.min(min_vals)

        y_center = int((bbox['bottom'] + bbox['top']) / 2)
        y_height_limit = (y_center - bbox['top']) * .95
        y_values_high = y_center + np.int64(max_vals / audio_max * y_height_limit)
        y_values_low = y_center - np.int64(min_vals / audio_min * y_height_limit)

        def _color_seg(intervals, color):
            """
            Draw a segments of one color
            :param intervals: [(start, stop), ...]
            :param color: numpy array (r,g,b,a)
            """
            for inter_low, inter_high in intervals:
                for x in range(inter_low, inter_high):
                    image[y_values_low[x]:y_values_high[x] - 1, x, :] = color

        # scale intervals from sound samples to pixels
        factor = float(width) / data.size
        start_indices, stop_indices = segmentation['starts'], segmentation['stops']
        sound_segs = [(int(factor * start_i), int(factor * stop_i)) for start_i, stop_i in
                      zip(start_indices, stop_indices)]
        noise_segs = get_interval_compliment(sound_segs, max_vals.size)
        _color_seg(sound_segs, sound_color)
        _color_seg(noise_segs, noise_color)
