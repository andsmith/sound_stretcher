"""
Main app
"""
import cv2
from enum import IntEnum
import time
import numpy as np
import logging
from tkinter import filedialog
import tkinter as tk
from scipy.interpolate import interp1d
import os
from threading import Thread, Lock
from version import VERSION
from sound_tools.sound import Sound
from sound_tools.sound_player import SoundPlayer
from util import make_unique_filename, in_area, draw_v_line, exp_fact_from_control_value
from layout import Layout
from help import HelpDisplay
from controls import ControlPanel
from spectrogram import Spectrogram
from text_box import TextBox


class StretchAppStates(IntEnum):
    init = 0
    playing = 1
    idle = 2
    busy = 3


# For playback, make larger if app lags, but this will hurt UI responsiveness
SOUND_BUFFER_SIZE = 1024 * 2


class StretchApp(object):

    def __init__(self, ):
        self._state = StretchAppStates.init
        self._filename = None
        self._win_name = "Sound Stretcher %s" % (VERSION,)
        self._shutdown = False  # flag for main loop
        self._show_axes = False
        self._showing_help = False

        self._playback_position_t = None  # where in sound data will the next sample buffer come from (during playback)
        self._mouse_cursor_t = 0.0  # where where user last left the playback line in the wave area

        # fixed params
        self._refresh_delay = 1. / 30.  # FPS

        # window/area sizes
        self._window_size = Layout.get_value('window_size')
        self._msg_area = Layout.get_value("msg_area")
        self._waveform_area = Layout.get_value('wave_area')
        self._spectrogram_area = Layout.get_value('spectrum_area')
        self._control_area = Layout.get_value('control_area')
        self._spectrogram = None  # create on load

        self._controls = ControlPanel(self._control_update, self._control_area)
        self._help = HelpDisplay(self._window_size[::-1])

        self._audio = None  # SoundPlayer(), initialized with sound file parameters after loading.
        self._sound = None  # Sound() object

        self._interps = None  # list of interp1d objects to interpolate each channel
        self._bkg = None  # app background image
        self._last_frame = None  # draw message box on this

        # for dropped audio frames
        self._last_encoded_samples = None  # send these if necessary
        self._audio_lock = Lock()  # dropped frame detected when (not acquire(timeout=0)) is true

        self._tkroot = tk.Tk()  # for open/save file dialogs
        self._tkroot.withdraw()  # use cv2 window as main window

        msg_font_info = {'font': Layout.get_value('msg_font'),
                         'bkg_color': Layout.get_color('msg_bkg_color'),
                         'text_color': Layout.get_color('msg_text_color')}

        self._messages = {'loading': TextBox(box_dims=self._msg_area,
                                             text_lines=['loading & analyzing ...'], centered=True,
                                             **msg_font_info),

                          'saving': TextBox(box_dims=self._msg_area,
                                            text_lines=['saving ...'], centered=True,
                                            **msg_font_info)}

        self._cur_msg_key = None

        # FPS info
        self._run_stats = {'buffer_count': 0,
                           'frame_count': 0,
                           'start_time': time.perf_counter(),
                           'update_interval_sec': 5.0,
                           'idle_t': 0.0,
                           'dropped_audio_frames': 0}

        self._user_params = dict(stretch_factor=self._controls.get_value('stretch_factor'),
                                 spectrogram_contrast=self._controls.get_value('spectrogram_contrast'),
                                 zoom_f=self._controls.get_value('zoom_f'),
                                 zoom_t=self._controls.get_value('zoom_t'),
                                 pan_f=self._controls.get_value('pan_f'))
        # interpolate for smoother transitions
        self._last_stretch_factor = self._user_params['stretch_factor']

        self._run()

    def _control_update(self, name, value):
        """
        User is updating our params (control panel callback).

        :param name:  name of param
        :param value: new value
        """
        self._user_params[name] = value
        # Special actions triggered by user control changes should start here.

    def _get_time_from_pos(self, mouse_x):
        """
        Assume in waveform_area
        :param mouse_x: int pixel location
        :return: float t, in [0.0, self._sound.duration_sec]
        """
        pos_rel = (mouse_x - self._waveform_area['left']) / float(
            self._waveform_area['right'] - self._waveform_area['left'])
        return pos_rel * self._sound.duration_sec

    def _get_pos_from_time(self, t):
        """
        inverse of _get_time_from_pos
        """
        pos_rel = t / self._sound.duration_sec
        return int(self._waveform_area['left'] + pos_rel * (self._waveform_area['right'] - self._waveform_area['left']))

    def _mouse(self, event, x, y, flags, param):
        """
        CV2 mouse callback.  App state changes happen here & when sound finishes & hotkey handler.
        """
        # let control panel know about the event
        if self._state != StretchAppStates.init:
            self._controls.mouse(event, x, y)

        if self._state == StretchAppStates.init:
            if event == cv2.EVENT_LBUTTONUP:
                self._load_file()

        elif self._state in [StretchAppStates.idle, StretchAppStates.playing]:

            if not in_area((x, y), self._waveform_area):
                return

            # update mouse cursor position
            if event == cv2.EVENT_MOUSEMOVE:
                if self._sound is not None:
                    self._mouse_cursor_t = self._get_time_from_pos(x)

            # check for clicks
            if event == cv2.EVENT_LBUTTONUP:
                if self._state == StretchAppStates.playing:
                    self._stop_playback()
                self._state = StretchAppStates.playing
                self._start_playback(self._mouse_cursor_t)

    def _stop_playback(self):
        logging.info("Stopping playback.")
        self._audio.stop()

    def _load_file(self):

        if self._state == StretchAppStates.playing:
            self._stop_playback()
            self._state = StretchAppStates.idle

        # ask user for filename
        all_patterns = Sound.NATIVE_FORMATS + Sound.OTHER_FORMATS
        global_pattern = [("All sound formats", [extension for format in all_patterns for extension in format[1]])]
        filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                              title="Select sound to stretch...",
                                              filetypes=global_pattern + all_patterns)

        if filename is None or len(filename) == 0:
            logging.info("Not loading new sound.")
            return

        # destroy audio (invalid sound params)
        if self._audio is not None:
            self._audio.shutdown()
            self._audio = None

        self._cur_msg_key = 'loading'
        self._state = StretchAppStates.busy

        def finish_loading():
            logging.info("Reading sound file:  %s" % (filename,))
            self._filename = filename
            self._sound = Sound(self._filename)

            # create background image w/waveform
            self._bkg = self._get_background()

            # do analysis for spectrogram animation
            params = Layout.get_value('spectrogram_params')
            self._spectrogram = Spectrogram(bbox=self._spectrogram_area,
                                            sound=self._sound,
                                            **params)
            # create interpolation objects for stretching sound samples.
            time_indices = np.linspace(0,
                                       (self._sound.metadata.nframes - 1) / self._sound.metadata.framerate,
                                       self._sound.metadata.nframes)
            self._interps = [interp1d(time_indices, chan_samples) for chan_samples in self._sound.data]

            # init audio w/sound file params
            self._audio = SoundPlayer(self._sound.metadata.sampwidth,
                                      self._sound.metadata.framerate,
                                      self._sound.metadata.nchannels,
                                      self._get_playback_samples,
                                      frames_per_buffer=SOUND_BUFFER_SIZE)

            self._state = StretchAppStates.idle
            logging.info("Sound loaded.")

        Thread(target=finish_loading).start()  # finish slow things in thread so UI keeps working

    def _get_background(self):
        """
        Get as much of the window that is static as possible
        """
        logging.info("Generating new app background...")
        image = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8) + np.array(
            Layout.get_color('bkg'), dtype=np.uint8)
        self._sound.draw_waveform(image, bbox=self._waveform_area, color=Layout.get_color('wave_sound'))
        logging.info("\t... generated.")
        return image

    def _start_playback(self, begin_t=0.):
        """
        User prompted to start playing
        :param begin_t:  Where to start, float in [0.0, duration]
        """
        if not 0 <= begin_t < self._sound.duration_sec:
            raise Exception("Start playback after end?")
        self._playback_position_t = begin_t
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." %
                     (self._playback_position_t, self._user_params['stretch_factor']))
        self._audio.start()

    def _get_playback_samples(self, n_samples):
        """
        Generate next sound buffer.
        :param n_samples:  number of samples
        :return:  N-element numpy array of sound samples (data type specified in pyaudio init)
        """
        stretch_factor = self._user_params['stretch_factor']
        if not self._audio_lock.acquire(blocking=False):
            samples = self._last_encoded_samples[:n_samples]
            self._run_stats['dropped_audio_frames'] += 1
            return samples

        # how far ahead to get buffer data
        if self._last_stretch_factor == stretch_factor:
            # evenly placed samples, made closer than 1/frame_rate by self._stretch factor
            t_end = self._playback_position_t + n_samples / self._sound.metadata.framerate / stretch_factor
            timestamps = np.linspace(self._playback_position_t, t_end, n_samples + 1)  # cache this?

        else:
            # Stretch factor at beginning should be old value, new value at end
            dt_first = 1.0 / self._sound.metadata.framerate / self._last_stretch_factor
            dt_last = 1.0 / self._sound.metadata.framerate / stretch_factor
            dt = np.linspace(dt_first, dt_last, n_samples)
            timestamps = np.hstack([[0.0], np.cumsum(dt)]) + self._playback_position_t

        # final timestamp is not used, it is the next buffer's first value
        if timestamps[-2] > self._sound.duration_sec:
            logging.info("Sound finished.")
            n_samps = np.sum(timestamps < self._sound.duration_sec)

            timestamps = timestamps[:n_samps]
            self._state = StretchAppStates.idle
            if timestamps.size == 0:
                empty_samples = [chan_interp(0) for chan_interp in self._interps]
                empty_bytes = self._sound.encode_samples(empty_samples)
                return bytes(empty_bytes)  # end of sound

        # interpolate
        samples = [chan_interp(timestamps[:-1]) for chan_interp in self._interps]
        samples_encoded = self._sound.encode_samples(samples)

        self._playback_position_t = timestamps[-1]
        self._run_stats['buffer_count'] += 1
        self._last_stretch_factor = stretch_factor
        self._last_encoded_samples = samples_encoded

        self._audio_lock.release()
        return samples_encoded

    def _run(self):
        """
        Main loop.
        """
        logging.info("Stretcher app starting.")
        cv2.namedWindow(self._win_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self._win_name, self._window_size)
        cv2.setMouseCallback(self._win_name, self._mouse)

        t_start = time.perf_counter()  # when last frame shown

        while not self._shutdown:

            # generate the next frame, wait until time, then show it
            self._last_frame = self._make_frame()
            now = time.perf_counter()
            elapsed = now - t_start
            remaining_wait = self._refresh_delay - elapsed
            if remaining_wait > 0:
                self._run_stats['idle_t'] += remaining_wait
                time.sleep(remaining_wait)

            t_start = now
            cv2.imshow(self._win_name, self._last_frame)
            k = cv2.waitKey(1)
            self._keyboard(k)

            # timing stats
            self._run_stats['frame_count'] += 1
            now = time.perf_counter()

            duration = now - self._run_stats['start_time']
            if duration > self._run_stats['update_interval_sec']:
                frame_rate = self._run_stats['frame_count'] / duration
                buffer_rate = self._run_stats['buffer_count'] / duration
                logging.info("display FPS:  %.2f,  audio buffers/sec:  %.2f (%i dropped requests)" % (
                    frame_rate, buffer_rate, self._run_stats['dropped_audio_frames']))
                self._run_stats['start_time'] = now
                self._run_stats['idle_t'] = 0.0
                self._run_stats['frame_count'] = 0
                self._run_stats['dropped_audio_frames'] = 0
                self._run_stats['buffer_count'] = 0

        logging.info("Stretcher app stopping.")

    def _keyboard(self, k):
        if k & 0xff == ord('q'):
            self._shutdown = True
        elif k & 0xff == ord('a'):
            self._show_axes = not self._show_axes
        elif k & 0xff == ord('h'):
            self._showing_help = not self._showing_help
            logging.info("Toggle help:  %s" % (self._showing_help,))
        elif k & 0xff == ord('s'):
            self._save()
        elif k & 0xff == ord('d'):
            logging.info("\n\n\n==========================\nDebug console:")
            import ipdb
            ipdb.set_trace()

        elif k & 0xff == ord(' '):
            if self._state == StretchAppStates.idle:
                # pause, restart at same position (if mouse not on spectrogram)
                self._playback_position_t = self._mouse_cursor_t if self._mouse_cursor_t is not None else 0.0

                self._start_playback(self._playback_position_t)
                self._state = StretchAppStates.playing
            elif self._state == StretchAppStates.playing:
                self._mouse_cursor_t = self._playback_position_t  # so spectrogram doesn't jump
                self._stop_playback()
                self._state = StretchAppStates.idle
        elif k & 0xff == ord('l'):
            self._load_file()

    def _make_frame(self):

        if self._state == StretchAppStates.init:
            # Before anything is loaded, just show help
            frame = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8)
            self._help.add_help(frame)

        elif self._state == StretchAppStates.busy:
            # a thread is finishing a task so just show previous frame with message about task
            frame = self._last_frame.copy()
            text = self._messages[self._cur_msg_key]
            text.write_text(frame)

        else:
            mouse_cursor_color = np.array(Layout.get_color('mouse_cursor'), dtype=np.uint8)
            playback_cursor_color = np.array(Layout.get_color('playback_cursor'), dtype=np.uint8)
            frame = self._bkg.copy()

            # draw in wave area
            mouse_line_x = self._get_pos_from_time(self._mouse_cursor_t)
            draw_v_line(frame, mouse_line_x, Layout.CURSOR_WIDTH, mouse_cursor_color,
                        y_range=self._waveform_area)
            spectrum_t = self._mouse_cursor_t

            if self._state == StretchAppStates.playing:
                playback_line_x = self._get_pos_from_time(self._playback_position_t)
                draw_v_line(frame, playback_line_x, Layout.CURSOR_WIDTH, playback_cursor_color,
                            y_range=self._waveform_area)
                spectrum_t = self._playback_position_t

            # draw the rest
            if self._spectrogram is not None:
                zoom = self._user_params['zoom_t'] / self._user_params['stretch_factor']
                zoom_f, pan_f = self._user_params['zoom_f'], self._user_params['pan_f']
                contrast = self._user_params['spectrogram_contrast']
                self._spectrogram.draw(frame, spectrum_t, zoom, zoom_f, pan_f, contrast,
                                       cursor=self._state == StretchAppStates.playing,
                                       axes=self._show_axes)

            self._controls.draw(frame)

            if self._showing_help:
                self._help.add_help(frame)

        return frame

    def _save(self):
        """
        Write stretched file.
        """
        if self._sound is None:
            logging.info("no sound loaded, not saving ")
            return
        file_name = os.path.splitext(self._filename)[0]
        out_filename = make_unique_filename("%s_x%.4f.wav" % (file_name, self._user_params['stretch_factor']))
        out_filename = filedialog.asksaveasfilename(initialfile=out_filename)

        self._cur_msg_key = 'saving'
        old_state, self._state = self._state, StretchAppStates.busy

        def finish_saving():
            logging.info("Stretching whole sound to save...")
            new_time_indices = np.linspace(0, self._sound.duration_sec,
                                           int(self._sound.metadata.nframes * self._user_params['stretch_factor']))[:-1]
            new_chan_data = [interpolator(new_time_indices) for interpolator in self._interps]
            logging.info("\t... done.")

            self._sound.write_data(out_filename, data=new_chan_data)
            self._state = old_state

        Thread(target=finish_saving()).start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
