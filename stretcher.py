import cv2
from enum import IntEnum
import time
import numpy as np
import logging
from tkinter import filedialog
import tkinter as tk
from scipy.interpolate import interp1d
import os

from sound import read_sound
from segmentation import SimpleSegmentation
from util import get_interval_compliment
from sound import SoundPlayer

CURSOR_ALPHA = 200

COLORS = {'slate': (0x30, 0x36, 0x3d, 255),
          'off white': (0xf6, 0xee, 0xe5, 255),
          'sky blue': (85, 206, 255, 255),
          'gray': (200, 200, 200, 255),
          'brown': (78, 53, 36, 255),
          'cursor_green': (0x54, 0x8f, 0x66, 255)}
CURSOR_WIDTH = 4


class Layout(object):
    COLOR_SCHEME = {'background': COLORS['slate'],
                    'wave_sound': COLORS['sky blue'],
                    'wave_noise': COLORS['brown'],
                    'playback_cursor': COLORS['cursor_green'][:3] + (CURSOR_ALPHA,),
                    'mouse_cursor': COLORS['gray'][:3] + (CURSOR_ALPHA,)}

    @staticmethod
    def get_color(name):
        return np.array(Layout.COLOR_SCHEME[name], dtype=np.uint8)


class StretchAppStates(IntEnum):
    init = 0
    playing = 1
    idle = 2


SOUND_BUFFER_SIZE = 1024 * 2


class StretchApp(object):

    def __init__(self, ):
        self._state = StretchAppStates.init
        self._filename = None
        self._win_name = "Sound Stretcher / Player"
        self._refresh_delay = 1. / 30.

        self._size = 1000, 600
        self._shutdown = False
        self._mouse_pos = None

        self._samples = []

        # controls
        self._noise_threshold = 5.0
        self._stretch_factor = 1.0
        self._margin_dur_sec = 0.2

        self._data = None  # waveform
        self._metadata = None  # wave params
        self._duration_sec = None  # unstretched
        self._segmentor = None
        self._segments = {'intervals': [],  # (start, stop) pairs of indices into self._data
                          'starts': None,  # numpy array of start indices
                          'stops': None}  # numpy array of stop indices, intervals = zip(starts, stops)
        self._next_sample = None  # where in self._data will the next sample buffer start

        self._image = None  # waveform background
        self._audio = None  # audio player, initialized with sound file parameters after loading.
        self._stretched_timestamps = None  # timestamps of samples, for interpolation
        # self._stretched_dt = None
        self._interp = None

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()  # use cv2 window as main window

        self._run()

    def _mouse(self, event, x, y, flags, param):
        """
        CV2 mouse callback.  App state changes happen here & when sound finishes.
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = x, y
            return  # no dragging-type interaction

        if self._state == StretchAppStates.init:

            if event in [cv2.EVENT_LBUTTONDOWN,
                         cv2.EVENT_RBUTTONDOWN]:
                self._load_file()

        else:
            if event == cv2.EVENT_RBUTTONUP:
                if self._state == StretchAppStates.playing:
                    self._stop_playback()
                    self._state = StretchAppStates.idle

                self._load_file()

            elif event == cv2.EVENT_LBUTTONUP:
                if self._state == StretchAppStates.idle:
                    self._state = StretchAppStates.playing
                    x_frac = float(x) / self._size[0]
                    self._start_playback(x_frac)
                else:
                    self._state = StretchAppStates.idle
                    self._stop_playback()

    def _stop_playback(self):
        self._audio.stop()

    def _load_file(self):
        # get file
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename is None or len(filename) == 0:
            logging.info("Not loading new sound.")
            return
        self._state = StretchAppStates.idle
        logging.info("Reading sound file:  %s" % (filename,))

        # read file
        self._data, self._metadata = read_sound(filename)
        self._duration_sec = self._metadata.nframes / float(self._metadata.framerate)
        if self._metadata.nchannels > 1:
            logging.info("Sound contains multi-channel data, converted to mono.")
            self._data = np.mean(self._data, axis=0)
        else:
            self._data = self._data[0]  # de-list

        # init data
        self._segmentor = SimpleSegmentation(self._data, self._metadata)
        self._resegment()
        self._set_stretch()
        self._image = self._get_background()
        time_indices = np.linspace(0, self._duration_sec, self._metadata.nframes + 1)[:-1]
        self._interp = interp1d(time_indices, self._data)

        # init audio
        if self._audio is not None:
            self._audio.shutdown()
        self._audio = SoundPlayer(self._metadata.sampwidth,
                                  self._metadata.framerate,
                                  self._metadata.nchannels,
                                  self._get_playback_samples,
                                  frames_per_buffer=SOUND_BUFFER_SIZE)

    def _set_stretch(self, new_factor=None):
        """
        Change stretch factor for UI.
        Call whenever these change:  stretch factor, sound file
        """
        self._stretch_factor = new_factor if new_factor is not None else self._stretch_factor
        self._buffer_duration = float(SOUND_BUFFER_SIZE) / self._metadata.framerate / self._stretch_factor

        # generic timestamps for the next buffer of sound, spaced for interpolation
        self._stretched_timestamps = np.linspace(0.0, self._buffer_duration, SOUND_BUFFER_SIZE + 1)[:-1]
        self._stretched_dt = self._stretched_timestamps[1] - self._stretched_timestamps[0]

    def _resegment(self):
        margin_samples = int(self._metadata.framerate * self._margin_dur_sec)
        self._segments = self._segmentor.get_partitioning(self._noise_threshold, margin_samples)

    def _get_background(self):
        audio_mean = np.mean(self._data)

        # bin audio into number of horizontal pixels, get max & min for each one
        bin_size = int(self._data.size / self._size[0])
        partitions = self._data[:bin_size * self._size[0]].reshape(self._size[0], bin_size)
        max_vals, min_vals = np.max(partitions - audio_mean, axis=1), \
                             np.min(partitions - audio_mean, axis=1)
        audio_max, audio_min = np.max(max_vals), np.min(min_vals)
        y_center = int(self._size[1] / 2)
        y_height_limit = self._size[1] / 2.05
        y_values_high = y_center + np.int64((max_vals) / audio_max * y_height_limit)
        y_values_low = y_center - np.int64(min_vals / audio_min * y_height_limit)
        image = np.zeros((self._size[1], self._size[0], 4), dtype=np.uint8) + Layout.get_color('background')

        def _color_seg(intervals, color):
            """
            Draw a segments of one color
            :param intervals: [(start, stop), ...]
            :param color: (r,g,b,a)
            """
            for inter_low, inter_high in intervals:
                for x in range(inter_low, inter_high):
                    image[y_values_low[x]:y_values_high[x] - 1, x, :] = color

        # scale intervals from sound samples to pixels
        factor = float(self._size[0]) / self._data.size
        sound_segs = [(int(factor * seg[0]), int(factor * seg[1])) for seg in self._segments['intervals']]
        noise_segs = get_interval_compliment(sound_segs, self._size[0])
        _color_seg(sound_segs, Layout.get_color('wave_sound'))
        _color_seg(noise_segs, Layout.get_color('wave_noise'))
        return image

    def _start_playback(self, begin_pos_rel=0.):
        """
        User prompted to start playing
        :param begin_pos_rel:  Where to start, float in [0, 1]
        """
        self._next_frame_index = int(begin_pos_rel * self._data.size)
        begin_time = begin_pos_rel * self._duration_sec
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." % (begin_time, self._stretch_factor))

        self._audio.start()

    def _get_playback_samples(self, n_samples):
        """
        Generate next sound buffer.
        If not currently in a sound segment, skip to the start of the next one
        :param n_samples:  Interpolated/spliced sound data
        """

        endpoint = self._next_frame_index + n_samples
        if endpoint > self._data.size:
            endpoint = self._data.size
            logging.info("Sound finished, outside sound segment.")
            self._state = StretchAppStates.idle
        samples = self._data[self._next_frame_index:endpoint]
        self._next_frame_index = endpoint
        self._samples.append(samples)

        return samples

        if n_samples != SOUND_BUFFER_SIZE:
            logging.warn("Getting %i samples instead of %i." % (n_samples, SOUND_BUFFER_SIZE))
        # import ipdb; ipdb.set_trace()
        n_segs_starting_before = np.sum(self._segments['starts'] < self._next_frame_index)
        n_segs_stopping_before = np.sum(self._segments['stops'] < self._next_frame_index)
        if n_segs_starting_before == n_segs_stopping_before:
            seg_ind = n_segs_stopping_before
            if seg_ind == self._segments['starts'].size:  # out of data, outside segment
                logging.info("Sound finished, outside sound segment.")
                self._state = StretchAppStates.idle
                return self._data[:2] * 0  # signal end
            else:
                logging.info("Skipping ahead to segment %i, (%i -> %i)" % (seg_ind, self._next_frame_index,
                                                                           self._segments['starts'][seg_ind]))
                self._next_frame_index = self._segments['starts'][seg_ind]

        start_timestamp = float(self._next_frame_index) / self._metadata.framerate
        stop_timestamp = start_timestamp + self._buffer_duration
        if stop_timestamp > self._duration_sec:
            logging.info("Sound finished, inside sound segment.")

            stop_timestamp = self._duration_sec  # out of data, inside segment
            buffer_size = (stop_timestamp - start_timestamp) * self._metadata.framerate * self._stretch_factor
        else:
            buffer_size = n_samples

        samples = self._interp(self._stretched_timestamps[:buffer_size] + start_timestamp)
        self._next_frame_index += buffer_size
        self._samples.append(samples)
        return samples

    def _run(self):
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name, self._size)
        cv2.setMouseCallback(self._win_name, self._mouse)

        t_start = time.perf_counter()

        while not self._shutdown:

            frame = self._make_frame()
            now = time.perf_counter()
            elapsed = now - t_start
            remaining_wait = self._refresh_delay - elapsed
            if remaining_wait > 0:
                time.sleep(remaining_wait)

            cv2.imshow(self._win_name, frame[:, :, 2::-1].copy())
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                self._shutdown = True

        samples = np.concatenate(self._samples)
        print(samples.shape)
        import matplotlib.pyplot as plt
        plt.plot(samples);
        plt.show()

    def _make_frame(self):

        if self._state == StretchAppStates.init:
            frame = np.zeros((self._size[1], self._size[0], 4), dtype=np.uint8) + Layout.get_color('background')
        else:
            frame = self._image.copy()
        if self._mouse_pos is not None:
            mouse_line_x = self._mouse_pos[0]
            _draw_v_line(frame, mouse_line_x, CURSOR_WIDTH, Layout.get_color('mouse_cursor'))

        if self._state == StretchAppStates.playing:
            playback_line_x = int(float(self._next_frame_index) / self._metadata.nframes * self._size[0])
            # print("LINE %i" % (playback_line_x,))
            _draw_v_line(frame, playback_line_x, CURSOR_WIDTH, Layout.get_color('playback_cursor'))
        return frame


def _draw_v_line(image, x, width, color):
    """
    Draw vertical line on image.
    :param image: to draw on
    :param x: x coordinate of line
    :param width: of line in pixels (should be even?)
    :param color: of line to draw
    """
    x_coords = np.array([x - width / 2, x + width / 2])
    if x_coords[0] < 0:
        x_coords += x_coords[0]
    if x_coords[1] > image.shape[1] - 1:
        x_coords -= x_coords[1] - image.shape[1] + 1

    x_coords = np.int64(x_coords)
    if len(color) == 4 and color[3] < 255:
        line = image[:, x_coords[0]:x_coords[1], :]
        alpha = float(color[3]) / 255.
        new_line = alpha * color + (1.0 - alpha) * line
        image[:, x_coords[0]:x_coords[1], :] = np.uint8(new_line)
    else:
        image[:, x_coords[0]:x_coords[1], :] = color


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
