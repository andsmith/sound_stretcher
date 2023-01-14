import cv2
from enum import IntEnum
import time
import numpy as np
import logging
from tkinter import filedialog
import tkinter as tk
from sound import read_sound
import os

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


class StretchApp(object):

    def __init__(self, ):
        self._state = StretchAppStates.init
        self._cur_playback_time_sec = 0.0
        self._filename = None
        self._win_name = "Sound Stretcher / Player"
        self._refresh_delay = 1. / 30.
        self._s = 1.0
        self._size = 1000, 600
        self._shutdown = False
        self._mouse_pos = None

        self._data = None
        self._metadata = None  # wave params
        self._duration_sec = None  # unstretched
        self._segments = None  # (start, stop) pairs of indices into self._data
        self._image = None

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()  # use cv2 window as main window

        self._run()

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = x, y
            return  # no dragging-type interaction

        if self._state == StretchAppStates.init:

            if event in [cv2.EVENT_LBUTTONDOWN,
                         cv2.EVENT_RBUTTONDOWN]:
                self._load_file()

        elif self._state == StretchAppStates.idle or self._state == StretchAppStates.playing:
            if event == cv2.EVENT_RBUTTONUP:
                self._load_file()
            elif event == cv2.EVENT_LBUTTONUP:
                x_frac = float(x) / self._size[0]
                self._start_playback(x_frac)

    def _load_file(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename is None or len(filename) == 0:
            logging.info("Not loading new sound.")
            return
        self._state = StretchAppStates.idle
        logging.info("Reading sound file:  %s" % (filename,))

        self._data, self._metadata = read_sound(filename)
        self._duration_sec = self._metadata.nframes / float(self._metadata.framerate)
        if self._metadata.nchannels > 1:
            logging.info("Sound contains multi-channel data, converted to mono.")
            self._data = np.mean(self._data, axis=0)
        else:
            self._data = self._data[0]  # de-list
        self._segments = self._get_segmentation()
        self._image = self._get_background()

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
            for inter_low, inter_high in intervals:
                for x in range(inter_low, inter_high):
                    image[y_values_low[x]:y_values_high[x] - 1, x, :] = color

        # scale intervals from sound samples to pixels
        factor = float(self._size[0]) / self._data.size
        sound_segs = [(int(factor * seg[0]), int(factor * seg[1])) for seg in self._segments]
        noise_segs = get_interval_compliment(sound_segs, self._size[0])
        _color_seg(sound_segs, Layout.get_color('wave_sound'))
        _color_seg(noise_segs, Layout.get_color('wave_noise'))
        return image

    def _get_segmentation(self):
        return [(0, int(self._data.size / 2))]  # first half, stub

    def _start_playback(self, begin_pos_rel=0.):
        begin_index = int(begin_pos_rel * self._data.size)
        begin_time = begin_pos_rel * self._duration_sec
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." % (begin_time, self._s))
        # see if in a segment or between them

        # starts_before = np.array([seg[0] <= begin_index for seg in self._segments])
        # contains = np.array([seg[0] <= begin_index < seg[1] for seg in self._segments])

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

            cv2.imshow(self._win_name, frame[:, :, 2::-1])
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                self._shutdown = True

    def _make_frame(self):

        if self._state == StretchAppStates.init:
            frame = np.zeros((self._size[1], self._size[0], 4), dtype=np.uint8) + Layout.get_color('background')
        else:
            frame = self._image.copy()
        if self._mouse_pos is not None:
            mouse_line_x = self._mouse_pos[0]
            _draw_v_line(frame, mouse_line_x, CURSOR_WIDTH, Layout.get_color('mouse_cursor'))

        if self._state == StretchAppStates.playing:
            playback_line_x = int(self._cur_playback_time_sec / self._duration_sec * self._size[0])
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


def get_interval_compliment(intervals, max_val):
    """
    Get minimal intervals whose union with input is whole range
    (For ints)
    :param intervals:  list of pairs (low, high) of intervals, in order,
    :return:  list of anti_interval pairs, so union of interval list with anti_interval list is (0, max_val)
    """
    if len(intervals) == 0:
        anti_intervals = [(0, max_val)]
    else:
        anti_intervals = []
        if intervals[0][0] > 0:
            anti_intervals = [(0, intervals[0][0])]
        for seg_i, segment in enumerate(intervals):
            if seg_i < len(intervals) - 1:
                #  anti-interval starts at end of this interval, ends at beginning of next
                anti_intervals.append((segment[1], intervals[seg_i + 1][0]))

            else:
                anti_intervals.append((segment[1], max_val))
    return anti_intervals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
