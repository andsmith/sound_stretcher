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
            'msg_area': {'top': int(WINDOW_SIZE[1] / 3), 'bottom': int(WINDOW_SIZE[1] / 3 * 2),
                         'left': int(WINDOW_SIZE[0] / 3), 'right': int(WINDOW_SIZE[0] / 3 * 2)},  # centered box

            'wave_area': {'top': 0, 'bottom': WAVE_HEIGHT, 'left': 0, 'right': WINDOW_SIZE[0]},  # top band of window
            'spectrum_area': {'top': WAVE_HEIGHT, 'bottom': WINDOW_SIZE[1] - CONTROL_HEIGHT, 'left': 0,
                              'right': WINDOW_SIZE[0]},  # middle band
            'control_area': {'top': WINDOW_SIZE[1] - CONTROL_HEIGHT, 'bottom': WINDOW_SIZE[1], 'left': 0,
                             'right': WINDOW_SIZE[0]},  # bottom band
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

            'spectrogram_params': {'resolution_sec': 0.001,
                                   'resolution_hz': 100.0,
                                   'freq_range': [0, 20000]}, }

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
    def get_color(name):
        return Layout._COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout._LUT[name]
