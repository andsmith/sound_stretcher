import cv2
from util import exp_fact_from_control_value


def _rgba_to_bgra(r, g, b, a):
    return b, g, r, a


COLORS = {'slate': _rgba_to_bgra(0x30, 0x36, 0x3d, 255),
          'off white': _rgba_to_bgra(0xf6, 0xee, 0xe5, 255),
          'sky blue': _rgba_to_bgra(85, 206, 255, 255),
          'light_gray': _rgba_to_bgra(200, 200, 200, 255),
          'dark_gray': _rgba_to_bgra(60, 60, 60, 255),
          'neutral': _rgba_to_bgra(127, 127, 127, 255),
          'brown': _rgba_to_bgra(78, 53, 36, 255),
          'cursor_green': _rgba_to_bgra(0x54, 0x8f, 0x66, 255),
          'dark_green': _rgba_to_bgra(0, 32, 15, 255)}


class Layout(object):
    # to not bog down display, make spectrogram width the window width:
    MAX_SPECTROGRAM_TIME_SLICES = 1400

    # window dimensions
    WINDOW_SIZE = MAX_SPECTROGRAM_TIME_SLICES, 800  # W x H
    WAVE_HEIGHT = 150  # region of window for waveform image
    CONTROL_HEIGHT = 120  # region of window for control panel
    MIN_WINDOW_SIZE = 700, 400  # W x H, render at least this big (window scales smaller images down)

    CURSOR_WIDTH = 4  # pixels
    CURSOR_ALPHA = 200
    HELP_TEXT_ALPHA = 255
    HELP_BKG_ALPHA = 235

    _COLOR_SCHEME = {'bkg': COLORS['slate'],
                     'control_axis': COLORS['light_gray'],
                     'control_slider': COLORS['off white'],
                     'control_text': COLORS['off white'],
                     'wave_sound': COLORS['sky blue'],
                     'help_text': COLORS['slate'][:3] + (HELP_TEXT_ALPHA,),
                     'help_bkg': COLORS['light_gray'][:3] + (HELP_BKG_ALPHA,),
                     'msg_text_color': COLORS['slate'][:3] + (HELP_TEXT_ALPHA,),
                     'msg_bkg_color': COLORS['off white'][:3] + (HELP_BKG_ALPHA,),
                     'playback_cursor': COLORS['cursor_green'][:3] + (CURSOR_ALPHA,),
                     'mouse_cursor': COLORS['light_gray'][:3] + (CURSOR_ALPHA,)}

    # misc key-value look-up
    _LUT = {'window_size': WINDOW_SIZE,
            'help_font': cv2.FONT_HERSHEY_DUPLEX,
            'msg_font': cv2.FONT_HERSHEY_DUPLEX,
            'control_panel_font': cv2.FONT_HERSHEY_DUPLEX,
            'control_text_spacing': 5,  # pixels, line spacing in a label
            'control_h_spacing': 5,  # pixels between controls
            'control_v_spacing': 5,  # pixels between controls

            'slider_dims': {'axis_thickness': 6,
                            'marker_thickness': 16,
                            'h_indent': 15,  # scale from edges
                            'text_indent_h_v': (10, 4),  # label from edges
                            'height': 15},  # of scale parts

            # Spectrogram analysis settings, cycled with the 'r' hotkey (first is the default).
            #   resolution_hz:  STFT window length is 1 / resolution_hz seconds
            #   resolution_sec:  STFT hop size (one spectrogram column per screen pixel at default zoom)
            'spectrogram_ranges': [{'name': 'birdsong',
                                    'resolution_sec': 0.001,  # birdsong is fast, fine time steps
                                    'resolution_hz': 100.0,  # 10 ms window
                                    'freq_range': [0, 20000]},
                                   {'name': 'human',
                                    # 20 ms window:  usual compromise for speech, short enough to stay within
                                    # one phoneme, long enough to span 2 pitch periods of a ~100 Hz voice,
                                    # so harmonics (pitch) and formant bands are both visible.
                                    'resolution_sec': 0.002,  # speech is slower, ~3 sec visible at default zoom
                                    'resolution_hz': 50.0,
                                    # formants F1-F4 are below ~5 kHz, fricative (s, sh, f) energy up to ~8 kHz
                                    'freq_range': [0, 8000]}], }

    # list of lists for rows/columns
    CONTROLS = [[{'name': 'stretch_factor',  # ################  ROW 1
                  'label': lambda x: 'Stretch\n%.2f x' % (x,),
                  'range': (1.0, 15.0),
                  'resolution': 0.005,
                  'init': 1.0,
                  'sample_value': 10.0,  # large test value for text fitting
                  'text_width': 145},
                 {'name': 'zoom_t',
                  'label': lambda x: 'zoom T\n1 / %4.1f' % (1. / x,),
                  'range': (0.005, 5.),
                  'resolution': .005,
                  'init': 1.0,
                  'sample_value': 1 / 7.,
                  'text_width': 100,
                  'total_width': 400}],  # optional, fitting not checked!

                [{'name': 'spectrogram_contrast',  # ################  ROW 2
                  'label': lambda x: 'contrast\n%.2f' % (x,),
                  'range': (-.05, 6.0),  # negative for log-scaling, positive for alpha correction
                  'resolution': 0.05,
                  'init': 3.,
                  'sample_value': 6.0,
                  'text_width': 110},

                 {'name': 'pan_f',
                  'label': lambda x: 'pan F\n%i %%' % (int(100 * x),),
                  'range': (0.0, 1.0),
                  'resolution': 0.01,
                  'init': 0.0,
                  'sample_value': .77,
                  'text_width': 100},

                 {'name': 'zoom_f',
                  'label': lambda x: 'zoom F\n%i %%' % (int(100 * x),),
                  'range': (0.01, 1.0),
                  'resolution': 0.01,
                  'init': 1.0,
                  'sample_value': 100.77,
                  'text_width': 100,
                  'total_width': 400},

                 ]]

    @staticmethod
    def get_scale(window_size):
        """
        How much bigger/smaller than the default (WINDOW_SIZE) a window is.
        :param window_size: (width, height)
        :return: (scale_x, scale_y)
        """
        return window_size[0] / Layout.WINDOW_SIZE[0], window_size[1] / Layout.WINDOW_SIZE[1]

    @staticmethod
    def get_areas(window_size):
        """
        Regions of the window, proportional to the window size.
        :param window_size: (width, height)
        :return: dict with 'msg_area', 'wave_area', 'spectrum_area', 'control_area', each a dict
            with 'top', 'bottom', 'left', 'right'
        """
        w, h = window_size
        wave_bottom = int(h * Layout.WAVE_HEIGHT / Layout.WINDOW_SIZE[1])
        control_top = h - int(h * Layout.CONTROL_HEIGHT / Layout.WINDOW_SIZE[1])
        return {'msg_area': {'top': int(h / 3), 'bottom': int(h / 3 * 2),
                             'left': int(w / 3), 'right': int(w / 3 * 2)},  # centered box
                'wave_area': {'top': 0, 'bottom': wave_bottom, 'left': 0, 'right': w},  # top band of window
                'spectrum_area': {'top': wave_bottom, 'bottom': control_top, 'left': 0, 'right': w},  # middle band
                'control_area': {'top': control_top, 'bottom': h, 'left': 0, 'right': w}}  # bottom band

    @staticmethod
    def get_color(name):
        return Layout._COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout._LUT[name]
