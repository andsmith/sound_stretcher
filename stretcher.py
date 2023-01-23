"""
Main app.
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
from threading import Thread

from sound import Sound, SoundPlayer
from util import make_unique_filename, in_area, draw_v_line
from layout import Layout
from help import HelpDisplay
from controls import ControlPanel

from spectrograms import get_power_spectrum

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
        self._win_name = "Sound Stretcher"
        self._shutdown = False
        self._mouse_pos = None
        self._showing_help = False
        self._playback_position_t = None  # where in sound data will the next sample buffer come from (during playback)

        # fixed params
        self._refresh_delay = 1. / 30.  # FPS

        # window/area sizes
        self._window_size = Layout.get_value('window_size')

        def _get_region_dims_abs(dims_rel):
            return {'top': int(dims_rel['top'] * self._window_size[1]),
                    'bottom': int(dims_rel['bottom'] * self._window_size[1]),
                    'left': int(dims_rel['left'] * self._window_size[0]),
                    'right': int(dims_rel['right'] * self._window_size[0])}

        self._msg_area = _get_region_dims_abs(Layout.get_value("msg_area_rel"))
        self._waveform_area = _get_region_dims_abs(Layout.get_value('wave_area_rel'))
        self._spectrum_area = _get_region_dims_abs(Layout.get_value('spectrum_area_rel'))
        self._control_area = _get_region_dims_abs(Layout.get_value('control_area_rel'))
        self._interaction_area = _get_region_dims_abs(Layout.get_value('interaction_area_rel'))

        self._controls = ControlPanel(self._control_update, self._control_area)
        self._help = HelpDisplay(self._window_size[::-1])

        self._sound = None  # waveform
        self._interps = None  # interp1d objects to interpolate each channel
        self._bkg = None  # app background image
        self._audio = None  # audio player, initialized with sound file parameters after loading.
        self._last_frame = None

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()  # use cv2 window as main window
        self._messages = {'loading': TextBox(box_dims=self._msg_area, text_lines=['Loading /', 'Analyzing ..'],
                                             **Layout.get_value('msg_text_params')),
                          'saving': TextBox(box_dims=self._msg_area, text_lines=['Saving..'],
                                            **Layout.get_value('msg_text_params'))}

        self._cur_msg = None  # should be one of the keys in self._messages

        # FPS info
        self._run_stats = {'buffer_count': 0,
                           'frame_count': 0,
                           'start_time': time.perf_counter(),
                           'update_interval_sec': 5.0,
                           'idle_t': 0.0}
        # user params
        self._stretch_factor = self._controls.get_value('stretch_factor')

        # start app
        self._run()

    def _control_update(self, name, value):
        """
        Control panel is updating our params (callback), take other appropriate action for each, etc.
        :param name:  name of param
        :param value: new value
        """
        if name == 'stretch_factor':
            self._set_stretch(value)

    def _mouse(self, event, x, y, flags, param):
        """
        CV2 mouse callback.  App state changes happen here & when sound finishes & hotkey handler.
        """
        if self._state != StretchAppStates.init:
            self._controls.mouse(event, x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = x, y

        if self._state == StretchAppStates.init:
            if event == cv2.EVENT_LBUTTONUP:
                self._load_file()
        elif self._state in [StretchAppStates.idle, StretchAppStates.playing]:
            # just wave stuff after this
            if not in_area((x, y), self._interaction_area):
                return

            if event == cv2.EVENT_LBUTTONUP:
                if self._state == StretchAppStates.playing:
                    self._stop_playback()

                self._state = StretchAppStates.playing
                x_frac = float(x) / self._window_size[0]
                self._start_playback(x_frac)

    def _stop_playback(self):
        logging.info("Stopping playback.")
        self._playback_position_t = None
        self._audio.stop()

    def _load_file(self):

        if self._state == StretchAppStates.playing:
            self._stop_playback()
            self._state = StretchAppStates.idle

        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if filename is None or len(filename) == 0:
            logging.info("Not loading new sound.")
            return

        # destroy audio (old sound params)
        if self._audio is not None:
            self._audio.shutdown()
            self._audio = None

        self._cur_msg = 'loading'
        self._state = StretchAppStates.busy

        def finish_loading():

            logging.info("Reading sound file:  %s" % (filename,))
            self._filename = filename
            self._sound = Sound(self._filename)

            # init data
            self._bkg = self._get_background()
            time_indices = np.linspace(0, self._sound.duration_sec, self._sound.metadata.nframes + 1)[:-1]
            self._interps = [interp1d(time_indices, chan_samples) for chan_samples in self._sound.data]

            # init audio
            self._audio = SoundPlayer(self._sound.metadata.sampwidth,
                                      self._sound.metadata.framerate,
                                      self._sound.metadata.nchannels,
                                      self._get_playback_samples,
                                      frames_per_buffer=SOUND_BUFFER_SIZE)

            self._state = StretchAppStates.idle
            logging.info("Sound loaded.")

        Thread(target=finish_loading).start()  # return so UI keeps working

    def _set_stretch(self, new_factor=None):
        """
        Change stretch factor for UI.
        """
        self._stretch_factor = new_factor if new_factor is not None else self._stretch_factor

    def _get_background(self):
        # generate image without sliders (just slider background color
        logging.info("Generating new app background...")
        image = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8) + np.array(
            Layout.get_color('wave_bkg'), dtype=np.uint8)
        self._sound.draw_waveform(image, bbox=self._waveform_area, color=Layout.get_color('wave_sound'))
        self._add_spectrogram_background(image)
        logging.info("\t... complete.")
        return image



    def _add_spectrogram_background(self, frame):
        params = Layout.get_value('spectrogram_params')
        t_res = params['time_resolution_sec']
        hz_res = params['frequency_resolution_hz']

        z, freqs, times = get_power_spectrum(self._sound.get_mono_data(),
                                             self._sound.metadata.framerate,
                                             resolution_hz=hz_res,
                                             resolution_sec=t_res)

        freq_range = params['plot_freq_range_hz']
        min_freq_ind, max_freq_ind = np.sum(freqs < freq_range[0]), np.sum(freqs < freq_range[1])
        power = np.log(3 + np.abs(z[min_freq_ind:max_freq_ind, :]))
        power = (power / np.max(power) * 255.0).astype(np.uint8)

        image = cv2.applyColorMap(power, cv2.COLORMAP_HOT)
        dest_dims = self._spectrum_area['right'] - self._spectrum_area['left'], self._spectrum_area['bottom'] - \
                    self._spectrum_area['top']

        spectrum_image = cv2.resize(image, dest_dims, cv2.INTER_CUBIC)[::-1, :, ::-1]

        frame[self._spectrum_area['top']:  self._spectrum_area['bottom'],
        self._spectrum_area['left']:  self._spectrum_area['right'], :3] = spectrum_image

        frame[self._spectrum_area['top']:  self._spectrum_area['bottom'],
        self._spectrum_area['left']:  self._spectrum_area['right'], 3] = 255

    def _start_playback(self, begin_pos_rel=0.):
        """
        User prompted to start playing
        :param begin_pos_rel:  Where to start, float in [0, 1]
        """
        self._playback_position_t = begin_pos_rel * self._sound.duration_sec
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." %
                     (self._playback_position_t, self._stretch_factor))
        self._audio.start()

    def _get_playback_samples(self, n_samples):
        """
        Generate next sound buffer.
        :param n_samples:  Interpolated/spliced sound data
        """
        # how far ahead to get buffer data
        t_end = self._playback_position_t + n_samples / self._sound.metadata.framerate / self._stretch_factor
        timestamps = np.linspace(self._playback_position_t, t_end, n_samples + 1)  # cache this?

        if timestamps[-1] > self._sound.duration_sec:
            logging.info("Sound finished.")
            n_samps = np.sum(timestamps < self._sound.duration_sec)
            timestamps = timestamps[:n_samps]
            self._state = StretchAppStates.idle

        # interpolate
        samples = [chan_interp(timestamps[:-1]) for chan_interp in self._interps]
        self._playback_position_t = timestamps[-1]

        samples_encoded = self._sound.encode_samples(samples)
        self._run_stats['buffer_count'] += 1
        return samples_encoded

    def _run(self):
        cv2.namedWindow(self._win_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self._win_name, self._window_size)
        cv2.setMouseCallback(self._win_name, self._mouse)

        t_start = time.perf_counter()  # when last frame shown

        while not self._shutdown:

            self._last_frame = self._make_frame()
            now = time.perf_counter()
            elapsed = now - t_start
            remaining_wait = self._refresh_delay - elapsed
            if remaining_wait > 0:
                self._run_stats['idle_t'] += remaining_wait
                time.sleep(remaining_wait)

            cv2.imshow(self._win_name, self._last_frame[:, :, 2::-1].copy())
            k = cv2.waitKey(1)
            t_start = now

            self._keyboard(k)

            # stats
            self._run_stats['frame_count'] += 1
            now = time.perf_counter()

            duration = now - self._run_stats['start_time']
            if duration > self._run_stats['update_interval_sec']:
                frame_rate = self._run_stats['frame_count'] / duration
                buffer_rate = self._run_stats['buffer_count'] / duration
                logging.info("Frame rate:  %.2f FPS,  idle:  %.2f %%, audio buffers/sec:  %.2f" % (
                    frame_rate, 100. * self._run_stats['idle_t'] / duration, buffer_rate))
                self._run_stats['start_time'] = now
                self._run_stats['idle_t'] = 0.0
                self._run_stats['frame_count'] = 0
                self._run_stats['buffer_count'] = 0

    def _keyboard(self, k):
        if k & 0xff == ord('q'):
            self._shutdown = True
        elif k & 0xff == ord('h'):
            self._showing_help = not self._showing_help
            logging.info("Toggle help:  %s" % (self._showing_help,))
        elif k & 0xff == ord('s'):
            self._save()
        elif k & 0xff == ord('d'):
            print("\n\n\n==========================\nDebug console:")
            import ipdb
            ipdb.set_trace()
        elif k & 0xff == ord(' '):
            if self._state == StretchAppStates.idle:
                self._start_playback(0.0)
                self._state = StretchAppStates.playing
            elif self._state == StretchAppStates.playing:
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
            frame = self._last_frame.copy()
            text = self._messages[self._cur_msg]
            text.write_text(frame)
        else:
            frame = self._bkg.copy()

        mouse_cursor_color = np.array(Layout.get_color('mouse_cursor'), dtype=np.uint8)
        playback_cursor_color = np.array(Layout.get_color('playback_cursor'), dtype=np.uint8)

        if self._state in [StretchAppStates.playing, StretchAppStates.idle]:
            # draw mouse
            if self._mouse_pos is not None and in_area(self._mouse_pos, self._interaction_area):
                mouse_line_x = self._mouse_pos[0]
                draw_v_line(frame, mouse_line_x, Layout.CURSOR_WIDTH, mouse_cursor_color,
                            y_range=self._interaction_area)
            # draw controls
            self._controls.draw(frame)

        # draw playback time marker
        if self._state == StretchAppStates.playing:
            playback_line_x = int(self._playback_position_t / self._sound.duration_sec * self._window_size[0])

            draw_v_line(frame, playback_line_x, Layout.CURSOR_WIDTH, playback_cursor_color,
                        y_range=self._interaction_area)

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
        out_filename = make_unique_filename("%s_x%.4f.wav" % (file_name, self._stretch_factor))
        out_filename = filedialog.asksaveasfilename(initialfile=out_filename)

        self._cur_msg = 'saving'
        old_state, self._state = self._state, StretchAppStates.busy

        def finish_saving():
            logging.info("Stretching whole sound to save...")
            new_time_indices = np.linspace(0, self._sound.duration_sec,
                                           int(self._sound.metadata.nframes * self._stretch_factor))[:-1]
            new_chan_data = [interpolator(new_time_indices) for interpolator in self._interps]
            logging.info("\t... done.")

            self._sound.write_data(out_filename, data=new_chan_data)
            self._state = old_state

        Thread(target=finish_saving()).start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
