import cv2

COLORS = {'slate': (0x30, 0x36, 0x3d, 255),
          'off white': (0xf6, 0xee, 0xe5, 255),
          'sky blue': (85, 206, 255, 255),
          'light_gray': (200, 200, 200, 255),
          'dark_gray': (60, 60, 60, 255),
          'brown': (78, 53, 36, 255),
          'cursor_green': (0x54, 0x8f, 0x66, 255),
          'dark_green': (0, 32, 15, 255)}


class Layout(object):
    # window dimensions
    WINDOW_SIZE = 1400,800
    WAVE_HEIGHT = 150  # region of window for waveform image
    CONTROL_HEIGHT = 100  # region of window for control panel

    # other things needed in definitions below
    CURSOR_WIDTH = 4  # pixels
    CURSOR_ALPHA = 200
    HELP_TEXT_ALPHA = 255
    HELP_BKG_ALPHA = 235
    MAX_SPECTROGRAM_FREQ = 16383.0  # for display

    COLOR_SCHEME = {'bkg': COLORS['slate'],
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
    LUT = {'help_font': cv2.FONT_HERSHEY_TRIPLEX,
           'msg_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_panel_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_text_spacing': 5,  # pixels between lines
           'control_h_spacing': 20,  # pixels between controls
           'window_size':WINDOW_SIZE,
           'msg_area': {'top': int(WINDOW_SIZE[1]/4), 'bottom': int(WINDOW_SIZE[1]/4*3),
                            'left': int(WINDOW_SIZE[0]/4), 'right': int(WINDOW_SIZE[0]/4*3)},  # centered box

           'wave_area': {'top': 0, 'bottom': WAVE_HEIGHT, 'left': 0, 'right': WINDOW_SIZE[0]},  # top band of window
           'spectrum_area': {'top': WAVE_HEIGHT, 'bottom': WINDOW_SIZE[1]-CONTROL_HEIGHT, 'left': 0, 'right': WINDOW_SIZE[0]},  # middle band
           'control_area': {'top': WINDOW_SIZE[1]-CONTROL_HEIGHT, 'bottom': WINDOW_SIZE[1], 'left': 0, 'right': WINDOW_SIZE[0]},  # bottom band
           'spectrogram_cursor_affinity': 0.4,  # how fast the spectrogram follows the mouse
           'slider_dims': {'axis_thickness': 6,
                           'marker_thickness': 16,
                           'h_indent': 15,  # scale from edges
                           'text_indent_h_v': (10, 4),  # label from edges
                           'height': 15},  # of scale parts
           'spectrogram_params': {'time_resolution_sec': 0.001,
                                  'frequency_resolution_hz': 110.0}, }

    # list of lists for rows/columns
    CONTROLS = [[{'name': 'stretch_factor',
                  'label': 'Stretch\n%.2f x',
                  'range': (1.0, 10.0),
                  'resolution': 0.01,
                  'init': 1.0,
                  'sample_value': 10.0,  # large test value for text fitting
                  'text_width': 145}],

                [{'name': 'spectrogram_contrast',
                  'label': 'contrast\n%.2f',
                  'range': (-1.0, 4.0),
                  'resolution': 0.05,
                  'init': 0,
                  'sample_value': -10.0,
                  'text_width': 110},

                 {'name': 'spectrogram_limit',
                  'label': 'display range\n1 - %i Hz',
                  'range': (1.0, MAX_SPECTROGRAM_FREQ),
                  'resolution': 1.,
                  'init': 10000.,
                  'sample_value': MAX_SPECTROGRAM_FREQ,
                  'text_width': 150},

                 {'name': 'spectrogram_shift',
                  'label': 'display shift\n%.2f',
                  'range': (0.,1.),  # maximum shift is 1 row left of the image?
                  'resolution': .01,
                  'init': 0.,
                  'sample_value': .77,
                  'text_width': 150},
                 ]]

    @staticmethod
    def get_color(name):
        return Layout.COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout.LUT[name]
