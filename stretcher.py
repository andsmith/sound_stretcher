import cv2
from enum import IntEnum
import time
import numpy as np
import logging
from tkinter import filedialog
import tkinter as tk
from scipy.interpolate import interp1d
import os

from sound import Sound, SoundPlayer
from segmentation import SimpleSegmentation
from util import get_interval_compliment, make_unique_filename, in_area
from layout import Layout
from help import HelpDisplay
from controls import ControlPanel

from spectrograms import get_power_spectrum


class StretchAppStates(IntEnum):
    init = 0
    playing = 1
    idle = 2


SOUND_BUFFER_SIZE = 1024 * 2


class StretchApp(object):

    def __init__(self, ):
        # state
        self._state = StretchAppStates.init
        self._filename = None
        self._win_name = "Sound Stretcher / Player"
        self._shutdown = False
        self._mouse_pos = None
        self._showing_help = False
        self._next_sample = None  # where in sound data will the next sample buffer start

        # fixed params
        self._refresh_delay = 1. / 30.

        # window/area sizes
        self._window_size = Layout.get_value('window_size')

        def _get_region_dims_abs(dims_rel):
            return {'top': int(dims_rel['top'] * self._window_size[1]),
                    'bottom': int(dims_rel['bottom'] * self._window_size[1]),
                    'left': int(dims_rel['left'] * self._window_size[0]),
                    'right': int(dims_rel['right'] * self._window_size[0])}

        self._waveform_area = _get_region_dims_abs(Layout.get_value('wave_area_rel'))
        self._spectrum_area = _get_region_dims_abs(Layout.get_value('spectrum_area_rel'))
        self._control_area = _get_region_dims_abs(Layout.get_value('control_area_rel'))
        self._interaction_area = _get_region_dims_abs(Layout.get_value('interaction_area_rel'))

        # data
        self._help = HelpDisplay(self._window_size[::-1])
        self._sound = None  # waveform
        self._segmentor = None
        self._interps = None
        self._image = None  # app background image
        self._spectrum_image = None
        self._audio = None  # audio player, initialized with sound file parameters after loading.

        self._segments = {'intervals': [],  # (start, stop) pairs of indices into self._sound.data
                          'starts': None,  # numpy array of start indices
                          'stops': None}  # numpy array of stop indices, intervals = zip(starts, stops)
        self._stretched_timestamps = None  # timestamps of samples, for interpolation
        # self._stretched_dt = None

        self._tkroot = tk.Tk()
        self._tkroot.withdraw()  # use cv2 window as main window

        self._controls = ControlPanel(self._control_update, self._control_area)

        # user params
        self._noise_threshold = self._controls.get_value('noise_threshold')
        self._stretch_factor = self._controls.get_value('stretch_factor')

        self._start_inds = []
        self._buffer_sizes = []
        self._start_times = []
        self._stop_times = []
        self._run_stats = {'frame_count': 0,
                           'start_time': time.perf_counter(),
                           'update_interval_sec': 5.0,
                           'idle_t': 0.0}

        self._run()

    def _control_update(self, name, value):
        """
        Control panel is updating our params (callback), take other appropriate action for each, etc.
        :param name:  name of param
        :param value: new value
        """
        if name == 'stretch_factor':
            self._set_stretch(value)
        if name == 'noise_threshold':
            self._noise_threshold = value
            self._resegment()  # need to re-run

    def _mouse(self, event, x, y, flags, param):
        """
        CV2 mouse callback.  App state changes happen here & when sound finishes.  (in future: hotkeys as well)
        """
        if self._state != StretchAppStates.init:
            self._controls.mouse(event, x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_pos = x, y

        if self._state == StretchAppStates.init:

            if event == cv2.EVENT_LBUTTONUP:
                self._load_file()

        else:
            # just wave stuff after this
            if not in_area((x, y), self._interaction_area):
                return

            if event == cv2.EVENT_LBUTTONUP:

                if self._state == StretchAppStates.idle:
                    self._state = StretchAppStates.playing
                    x_frac = float(x) / self._window_size[0]
                    self._start_playback(x_frac)
                else:
                    self._state = StretchAppStates.idle
                    self._stop_playback()

    def _stop_playback(self):
        logging.info("Stopping playback.")
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
        self._filename = filename
        self._sound = Sound(self._filename)

        # init data
        self._seg_params = Layout.get_value('segmentation_params')
        self._segmentor = SimpleSegmentation(self._sound, self._seg_params['smoothing_kern_width_sec'])
        self._make_spectrogram_image()
        self._resegment()
        self._set_stretch()
        time_indices = np.linspace(0, self._sound.duration_sec, self._sound.metadata.nframes + 1)[:-1]
        self._interps = [interp1d(time_indices, chan_samples) for chan_samples in self._sound.data]

        # init audio
        if self._audio is not None:
            self._audio.shutdown()
        self._audio = SoundPlayer(self._sound.metadata.sampwidth,
                                  self._sound.metadata.framerate,
                                  self._sound.metadata.nchannels,
                                  self._get_playback_samples,
                                  frames_per_buffer=SOUND_BUFFER_SIZE)
        logging.info("Sound loaded.")

    def _set_stretch(self, new_factor=None):
        """
        Change stretch factor for UI.
        Call whenever these change:  stretch factor, sound file
        """

        self._stretch_factor = new_factor if new_factor is not None else self._stretch_factor
        self._stretched_buffer_duration = \
            float(SOUND_BUFFER_SIZE) / self._sound.metadata.framerate / self._stretch_factor

        # generic timestamps for the next buffer of sound, spaced for interpolation
        self._stretched_timestamps = np.linspace(0.0, self._stretched_buffer_duration, SOUND_BUFFER_SIZE + 1)[:-1]
        self._stretched_dt = self._stretched_timestamps[1] - self._stretched_timestamps[0]

    def _resegment(self):
        self._segments = self._segmentor.get_segmentation(threshold=self._noise_threshold,
                                                          margin_duration_sec=self._seg_params['margin_sec'])
        self._image = self._get_background()

    def _get_background(self):
        # generate image without sliders (just slider background color
        image = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8) + np.array(
            Layout.get_color('wave_bkg'), dtype=np.uint8)
        self._add_wave_background(image)
        self._add_spectrogram_background(image)
        return image

    def _make_spectrogram_image(self):
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

        self._spectrum_image = cv2.resize(image, dest_dims, cv2.INTER_CUBIC)[::-1, :, ::-1]

    def _add_spectrogram_background(self, frame):
        frame[self._spectrum_area['top']:  self._spectrum_area['bottom'],
        self._spectrum_area['left']:  self._spectrum_area['right'], :3] = self._spectrum_image

        frame[self._spectrum_area['top']:  self._spectrum_area['bottom'],
        self._spectrum_area['left']:  self._spectrum_area['right'], 3] = 255

    def _add_wave_background(self, image):
        self._segmentor.draw_segmented_waveform(image,
                                                self._segments,
                                                bbox=self._waveform_area,
                                                sound_color=Layout.get_color('wave_sound'),
                                                noise_color=Layout.get_color('wave_noise'))

    def _start_playback(self, begin_pos_rel=0.):
        """
        User prompted to start playing
        :param begin_pos_rel:  Where to start, float in [0, 1]
        """
        self._next_frame_index = int(begin_pos_rel * self._sound.metadata.nframes)
        self._start_ind = self._next_frame_index
        begin_time = begin_pos_rel * self._sound.duration_sec
        logging.info("Beginning playback at %.2f seconds, at stretch factor %.2f." % (begin_time, self._stretch_factor))

        self._audio.start()

    def _get_playback_samples(self, n_samples):
        """
        Generate next sound buffer.
        If not currently in a sound segment, skip to the start of the next one
        :param n_samples:  Interpolated/spliced sound data
        """
        '''
        endpoint = self._next_frame_index + n_samples
        if endpoint > self._sound.metadata.nframes:
            endpoint = self._sound.metadata.nframes
            self._state = StretchAppStates.idle
        nc = self._sound.metadata.sampwidth
        reference_samples = self._sound.data_raw[nc * self._next_frame_index:nc * (self._next_frame_index + endpoint)]
        self._next_frame_index = endpoint
        '''
        n_segs_starting_before = np.sum(self._segments['starts'] < self._next_frame_index)
        n_segs_stopping_before = np.sum(self._segments['stops'] < self._next_frame_index)
        if n_segs_starting_before == n_segs_stopping_before:
            seg_ind = n_segs_stopping_before
            if seg_ind == self._segments['starts'].size:  # out of data, outside segment
                logging.info("Sound finished.")
                self._state = StretchAppStates.idle
                return self._sound.data_raw[:2] * 0  # signal end
            else:
                logging.info("Skipping ahead to segment %i, (%i -> %i)" % (seg_ind, self._next_frame_index,
                                                                           self._segments['starts'][seg_ind]))
                self._next_frame_index = self._segments['starts'][seg_ind]

        start_timestamp = float(self._next_frame_index) / self._sound.metadata.framerate
        stop_timestamp = start_timestamp + self._stretched_buffer_duration

        self._start_inds.append(self._next_frame_index)
        self._start_times.append(start_timestamp)
        self._stop_times.append(stop_timestamp)
        if stop_timestamp > self._sound.duration_sec:
            logging.info("Sound finished, inside sound segment.")
            stop_timestamp = self._sound.duration_sec  # out of data, inside segment
            buffer_size = int((stop_timestamp - start_timestamp) / self._stretched_dt)
            self._state = StretchAppStates.idle
        else:
            buffer_size = n_samples
        self._buffer_sizes.append(buffer_size)

        samples = [chan_interp(self._stretched_timestamps[:buffer_size] + start_timestamp)
                   for chan_interp in self._interps]
        self._next_frame_index = int(stop_timestamp * self._sound.metadata.framerate) + self._stretched_dt

        samples_encoded = self._sound.encode_samples(samples)

        return samples_encoded

    def _run(self):
        cv2.namedWindow(self._win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win_name, self._window_size)
        cv2.setMouseCallback(self._win_name, self._mouse)

        t_start = time.perf_counter()  # when last frame shown

        while not self._shutdown:

            frame = self._make_frame()
            now = time.perf_counter()
            elapsed = now - t_start
            remaining_wait = self._refresh_delay - elapsed
            if remaining_wait > 0:
                self._run_stats['idle_t'] += remaining_wait
                time.sleep(remaining_wait)

            cv2.imshow(self._win_name, frame[:, :, 2::-1].copy())
            k = cv2.waitKey(1)
            t_start = now

            self._keyboard(k)

            # stats
            self._run_stats['frame_count'] += 1
            now = time.perf_counter()

            duration = now - self._run_stats['start_time']
            if duration > self._run_stats['update_interval_sec']:
                frame_rate = self._run_stats['frame_count'] / duration
                logging.info("Frame rate:  %.2f FPS,  idle:  %.2f %%" % (
                    frame_rate, 100. * self._run_stats['idle_t'] / duration))
                self._run_stats['start_time'] = now
                self._run_stats['idle_t'] = 0.0
                self._run_stats['frame_count'] = 0

    def _keyboard(self, k):
        if k & 0xff == ord('q'):
            self._shutdown = True
        elif k & 0xff == ord('h'):
            self._showing_help = not self._showing_help
            logging.info("Toggle help:  %s" % (self._showing_help,))
        elif k & 0xff == ord('s'):
            self._save()
        elif k & 0xff == ord('d'):
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
            if self._state == StretchAppStates.playing:
                self._stop_playback()
                self._state = StretchAppStates.idle
            self._load_file()

    def _make_frame(self):
        if self._state == StretchAppStates.init or self._image is None:
            # Before anything is loaded, just show help
            frame = np.zeros((self._window_size[1], self._window_size[0], 4), dtype=np.uint8)
            self._help.add_help(frame)

        else:
            frame = self._image.copy()

        mouse_cursor_color = np.array(Layout.get_color('mouse_cursor'), dtype=np.uint8)
        playback_cursor_color = np.array(Layout.get_color('playback_cursor'), dtype=np.uint8)

        if self._state in [StretchAppStates.playing, StretchAppStates.idle]:
            # draw mouse
            if self._mouse_pos is not None and in_area(self._mouse_pos, self._interaction_area):
                mouse_line_x = self._mouse_pos[0]
                _draw_v_line(frame, mouse_line_x, Layout.CURSOR_WIDTH, mouse_cursor_color,
                             y_range=self._interaction_area)
            # draw controls
            self._controls.draw(frame)

        # draw playback time marker
        if self._state == StretchAppStates.playing:
            playback_line_x = int(float(self._next_frame_index) / self._sound.metadata.nframes * self._window_size[0])
            # print("LINE %i" % (playback_line_x,))
            _draw_v_line(frame, playback_line_x, Layout.CURSOR_WIDTH, playback_cursor_color,
                         y_range=self._interaction_area)

        if self._showing_help:
            self._help.add_help(frame)

        return frame

    def _save(self):
        """
        Write stretched file.
        """
        logging.info("Stretching whole sound...")
        new_time_indices = np.linspace(0, self._sound.duration_sec,
                                       int(self._sound.metadata.nframes * self._stretch_factor))
        new_chan_data = [interpolator(new_time_indices) for interpolator in self._interps]
        logging.info("\t... done.")

        file_name = os.path.splitext(self._filename)[0]
        out_filename = make_unique_filename("%s_x%.4f.wav" % (file_name, self._stretch_factor))
        self._sound.write_file(new_chan_data, out_filename)


def _draw_v_line(image, x, width, color, y_range=None):
    """
    Draw vertical line on image.
    :param image: to draw on
    :param x: x coordinate of line
    :param width: of line in pixels (should be even?)
    :param y_range:  dict with 'top' and 'bottom' or None for whole image
    :param color: of line to draw
    """
    x_coords = np.array([x - width / 2, x + width / 2])
    if x_coords[0] < 0:
        x_coords += x_coords[0]
    if x_coords[1] > image.shape[1] - 1:
        x_coords -= x_coords[1] - image.shape[1] + 1

    if y_range is None:
        y_low, y_high = 0, image.shape[0]
    else:
        y_low, y_high = y_range['top'], y_range['bottom']

    x_coords = np.int64(x_coords)
    if len(color) == 4 and color[3] < 255:
        line = image[y_low: y_high, x_coords[0]:x_coords[1], :]
        alpha = float(color[3]) / 255.
        new_line = alpha * color + (1.0 - alpha) * line
        image[y_low: y_high, x_coords[0]:x_coords[1], :] = np.uint8(new_line)
    else:
        image[y_low: y_high, x_coords[0]:x_coords[1], :] = color


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    StretchApp()
