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
from threading import Thread, Lock
from version import VERSION
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
        self._win_name = "Sound Stretcher %s" % (VERSION,)
        self._shutdown = False

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
        self._spectrogram_size = self._spectrogram_area['right'] - self._spectrogram_area['left'], \
                                 self._spectrogram_area['bottom'] - self._spectrogram_area['top']
        self._controls = ControlPanel(self._control_update, self._control_area)
        self._help = HelpDisplay(self._window_size[::-1])
        self._valid_freqs = None  # stored for spectrogram
        self._sound = None  # Sound() object
        self._interps = None  # list of interp1d objects to interpolate each channel
        self._bkg = None  # app background image
        self._audio = None  # SoundPlayer(), initialized with sound file parameters after loading.
        self._last_frame = None  # draw message box on this

        # for dropped audio frames
        self._last_encoded_samples = None  # send these if necessary
        self._audio_lock = Lock()  # dropped frame detected if not acquire(timeout=0)

        # for zoom/panning of spectrogram
        self._spectrogram_raw = None  # dict("z", "f","t") # (z-values, frequency list, timestamp list)
        self._spectrogram_master_image = None  # subset of this is displayed

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()  # use cv2 window as main window
        msg_font_info = {'font': Layout.get_value('msg_font'),
                         'bkg_color': Layout.get_color('msg_bkg_color'),
                         'text_color': Layout.get_color('msg_text_color')}
        self._messages = {'loading': TextBox(box_dims=self._msg_area,
                                             text_lines=['Loading /', 'Analyzing ..'],
                                             **msg_font_info),
                          'saving': TextBox(box_dims=self._msg_area,
                                            text_lines=['Saving..'],
                                            **msg_font_info)}

        self._cur_msg = None  # should be one of the keys in self._messages

        # FPS info
        self._run_stats = {'buffer_count': 0,
                           'frame_count': 0,
                           'start_time': time.perf_counter(),
                           'update_interval_sec': 5.0,
                           'idle_t': 0.0,
                           'dropped_audio_frames': 0}

        # user params
        self._stretch_factor = self._controls.get_value('stretch_factor')
        self._spectrogram_contrast = self._controls.get_value('spectrogram_contrast')
        self._spectrogram_limit = self._controls.get_value('spectrogram_limit')
        self._spectrogram_shift = self._controls.get_value('spectrogram_shift')

        # start app
        self._last_stretch_factor = self._stretch_factor  # for smoother transitions, interpolate
        logging.info("Stretcher init complete.")
        self._run()

    def _control_update(self, name, value):
        """
        User/control panel is updating our params (callback), take other appropriate action for each, etc.
        :param name:  name of param
        :param value: new value
        """
        logging.info("Changing %s:  %.2f" % (name, value,))

        if name == 'stretch_factor':
            self._set_stretch(value)
        if name == 'spectrogram_contrast':
            self._spectrogram_contrast = value
            self._rebuild_spectrogram_image()
        if name == 'spectrogram_limit':
            self._spectrogram_limit = value
            self._recompute_valid_spectrogram_frquencies()
        if name == 'spectrogram_shift':
            self._spectrogram_shift = value
            self._recompute_valid_spectrogram_frquencies()

    def _recompute_valid_spectrogram_frquencies(self):
        shift = self._spectrogram_shift * self._spectrogram_limit
        self._valid_freqs = np.logical_and(1.0 <= self._spectrogram_raw['f'] - shift,
                                           self._spectrogram_raw['f'] - shift <= self._spectrogram_limit)
        if np.sum(self._valid_freqs) == 0:
            self._valid_freqs[0] = True

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
        if self._state != StretchAppStates.init:
            self._controls.mouse(event, x, y)

        if self._state == StretchAppStates.init:
            if event == cv2.EVENT_LBUTTONUP:
                self._load_file()

        elif self._state in [StretchAppStates.idle, StretchAppStates.playing]:
            # just wave stuff after this
            if not in_area((x, y), self._waveform_area):
                return

            # update cursor position
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

            # create background image
            self._bkg = self._get_background()

            # do analysis for spectrogram & create master image

            self._spectrogram_raw = self._get_power_spectrum()
            freq_max_subset = np.logical_and(1.0 <= self._spectrogram_raw['f'],
                                             self._spectrogram_raw['f'] <= Layout.MAX_SPECTROGRAM_FREQ)
            self._spectrogram_raw['power'] = np.abs(self._spectrogram_raw['z'][freq_max_subset, :])
            logging.info("Created new raw spectrogram:  %s (F x T)" % (self._spectrogram_raw['power'].shape[:2],))
            self._recompute_valid_spectrogram_frquencies()

            self._rebuild_spectrogram_image()

            # create interpolation objects
            time_indices = np.linspace(0,
                                       (self._sound.metadata.nframes - 1) / self._sound.metadata.framerate,
                                       self._sound.metadata.nframes)
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
        """
        Get as much of the window that is static as possible
        """
        logging.info("Generating new app background...")
        image = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8) + np.array(
            Layout.get_color('bkg'), dtype=np.uint8)
        self._sound.draw_waveform(image, bbox=self._waveform_area, color=Layout.get_color('wave_sound'))
        # self._add_spectrogram_background(image)  now done live
        logging.info("\t... complete.")
        return image

    def _get_power_spectrum(self):
        params = Layout.get_value('spectrogram_params')
        t_res = params['time_resolution_sec']
        hz_res = params['frequency_resolution_hz']

        z, freqs, times = get_power_spectrum(self._sound.get_mono_data(),
                                             self._sound.metadata.framerate,
                                             resolution_hz=hz_res,
                                             resolution_sec=t_res,
                                             freq_range=(1.0, Layout.MAX_SPECTROGRAM_FREQ))
        z = np.abs(z)
        return {'z': z, 'f': freqs, 't': times}

    def _rebuild_spectrogram_image(self):
        """
        visualize the STFT result
        """
        contrast_range = self._controls.get_control('spectrogram_contrast').get_prop('range')
        def _scale_z2n_values(z2n, contrast):
            """
            Scale unit intensities, by 1 of two methods:

                if contrast < 0, return np.log(-contrast + z2n)
                else return z2n^(-1-contrast)   (alpha correction)

            :param z2n: |z|^2 values scaled to [0, 1]
            :param contrast: float
            :return: scaled values (floats)
            """
            if contrast < 0:
                contrast = (1.-contrast)
                # padded log scaling, then normalize
                print(contrast,np.max(z2n),np.min(z2n))
                image_f = np.log(contrast + z2n)
                return image_f / np.max(image_f)

            else:  # raw / alpha scaling of normalized values
                # reverse range, so looks better next to log
                contrast = contrast_range[1]-contrast
                image_f = z2n / np.max(z2n)
                alpha = 1.0/(contrast + 1.0)
                return image_f ** alpha

        logging.info("Generating new spectrogram master image, contrast=%.2f" % (self._spectrogram_contrast,))
        image_float = _scale_z2n_values(self._spectrogram_raw['power'], self._spectrogram_contrast)
        image = (image_float * 255.0).astype(np.uint8)
        self._spectrogram_master_image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
        logging.info("\tdone generating master image.")

    def _start_playback(self, begin_t=0.):
        """
        User prompted to start playing
        :param begin_t:  Where to start, float in [0.0, duration]
        """
        if not 0 <= begin_t < self._sound.duration_sec:
            raise Exception("Start playback after end?")
        self._playback_position_t = begin_t
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." %
                     (self._playback_position_t, self._stretch_factor))
        self._audio.start()

    def _get_playback_samples(self, n_samples):
        """
        Generate next sound buffer.
        :param n_samples:  number of samples
        :return:  N-element numpy array of sound samples (data type specified in pyaudio init)
        """

        if not self._audio_lock.acquire(blocking=False):
            samples = self._last_encoded_samples[:n_samples]
            self._run_stats['dropped_audio_frames'] += 1
            return samples

        # how far ahead to get buffer data
        if self._last_stretch_factor == self._stretch_factor:
            # evenly placed samples, made closer by self._stretch factor
            t_end = self._playback_position_t + n_samples / self._sound.metadata.framerate / self._stretch_factor
            timestamps = np.linspace(self._playback_position_t, t_end, n_samples + 1)  # cache this?

        else:
            # Stretch factor at beginning should be old value, new value at end
            dt_first = 1.0 / self._sound.metadata.framerate / self._last_stretch_factor
            dt_last = 1.0 / self._sound.metadata.framerate / self._stretch_factor
            dt = np.linspace(dt_first, dt_last, n_samples)
            timestamps = np.hstack([[0.0], np.cumsum(dt)]) + self._playback_position_t

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

        self._last_encoded_samples = samples
        self._playback_position_t = timestamps[-1]

        samples_encoded = self._sound.encode_samples(samples)
        self._run_stats['buffer_count'] += 1
        self._last_stretch_factor = self._stretch_factor
        self._audio_lock.release()
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
                logging.info("Frame rate:  %.2f FPS,  audio buffers/sec:  %.2f (%i dropped)" % (
                    frame_rate, buffer_rate, self._run_stats['dropped_audio_frames']))
                self._run_stats['start_time'] = now
                self._run_stats['idle_t'] = 0.0
                self._run_stats['frame_count'] = 0
                self._run_stats['dropped_audio_frames'] = 0
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
        if self._state in [StretchAppStates.playing, StretchAppStates.idle]:

            # Draw spectrogram
            if self._spectrogram_master_image is not None:
                spectrogram_img = self._spectrogram_master_image[self._valid_freqs, :]
                spectrogram_img = cv2.resize(spectrogram_img, self._spectrogram_size, cv2.INTER_CUBIC)[::-1, :, ::-1]

                frame[self._spectrogram_area['top']:self._spectrogram_area['bottom'],
                self._spectrogram_area['left']:self._spectrogram_area['right'], :3] = spectrogram_img
                frame[self._spectrogram_area['top']:self._spectrogram_area['bottom'],
                self._spectrogram_area['left']:self._spectrogram_area['right'], 3] = 255

            mouse_cursor_color = np.array(Layout.get_color('mouse_cursor'), dtype=np.uint8)
            playback_cursor_color = np.array(Layout.get_color('playback_cursor'), dtype=np.uint8)

            # draw mouse cursor
            if self._mouse_cursor_t is not None:
                mouse_line_x = self._get_pos_from_time(self._mouse_cursor_t)

                draw_v_line(frame, mouse_line_x, Layout.CURSOR_WIDTH, mouse_cursor_color,
                            y_range=self._waveform_area)
            # draw controls
            self._controls.draw(frame)

            # draw playback time marker
            if self._state == StretchAppStates.playing:
                playback_line_x = self._get_pos_from_time(self._playback_position_t)
                draw_v_line(frame, playback_line_x, Layout.CURSOR_WIDTH, playback_cursor_color,
                            y_range=self._waveform_area)

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
