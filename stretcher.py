import cv2
from enum import IntEnum
import time
import numpy as np
import logging
from tkinter import filedialog
import tkinter as tk
from sound import read_sound
import os

COLORS = {'off white': (0xf6, 0xee, 0xe5),
          'sky blue': (0, 0, 254),
          'gray': (200, 200, 200),
          'brown': (78, 53, 36), }


class StretchAppStates(IntEnum):
    init = 0
    playing = 1
    idle = 2


class StretchApp(object):

    def __init__(self, ):
        self._state = StretchAppStates.init
        self._filename = None
        self._win_name = "Sound Stretcher / Player"
        self._refresh_delay = 1. / 30.
        self._s = 1.0
        self._size = 1000, 600
        self._shutdown = False
        self._mouse_pos = None
        self._data = None
        self._metadata = None  # wave params

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()

        self._run()

    def _mouse(self, event, x, y, flags, param):
        if self._state == StretchAppStates.init:

            if event in [cv2.EVENT_LBUTTONDOWN,
                         cv2.EVENT_RBUTTONDOWN]:
                self._load_file()
        elif self._state == StretchAppStates.init or self._state == StretchAppStates.playing:
            if event == cv2.EVENT_RBUTTONUP:
                self._load_file()
            elif event == cv2.EVENT_LBUTTONUP:
                x_frac = float(x) / self._size[0]
                self._start_playback(x_frac)

    def _load_file(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename is None:
            return
        logging.info("Reading sound file:  %s" % (filename,))
        self._data, wav_info = read_sound(filename)
        if wav_info.nchannels > 1:
            logging.info("Sound contains multi-channel data, converted to mono.")
            self._data = np.mean(self._data, axis=0)

    def _start_playback(self, begin_pos_rel=0.):
        pass

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

            cv2.imshow(self._win_name, frame)
            k = cv2.waitKey(1)
            if k & 0xff == ord('q'):
                self._shutdown = True

    def _make_frame(self):
        return np.zeros((self._size[1], self._size[0], 3), dtype=np.uint8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
