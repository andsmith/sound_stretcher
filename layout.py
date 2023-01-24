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
    CURSOR_WIDTH = 4  # pixels
    CURSOR_ALPHA = 200
    HELP_TEXT_ALPHA = 255
    HELP_BKG_ALPHA = 215
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
    LUT = {'help_font': cv2.FONT_HERSHEY_COMPLEX,
           'msg_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_panel_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_text_spacing': 5,  # pixels between lines
           'control_h_spacing': 20,  # pixels between controls
           'window_size': (1200, 600),
           'msg_area_rel': {'top': .3, 'bottom': .7, 'left': 0.3, 'right': .7},  # centered box

           'wave_area_rel': {'top': 0., 'bottom': .7 / 3, 'left': 0., 'right': 1.},  # top band of window
           'spectrum_area_rel': {'top': .7 / 3, 'bottom': .7, 'left': 0., 'right': 1.},  # middle band
           'control_area_rel': {'top': 0.7, 'bottom': 1.0, 'left': 0., 'right': 1.},  # bottom band
           'interaction_area_rel': {'top': 0., 'bottom': 0.7, 'left': 0., 'right': 1.0},  # top two bands
           'spectrogram_cursor_affinity': 0.4,  # how fast the spectrogram follows the mouse
           'slider_dims': {'axis_thickness': 6,
                           'marker_thickness': 16,
                           'h_indent': 15,  # scale from edges
                           'text_indent_h_v': (10,4),  # label from edges
                           'height': 15},  # of scale parts
           'spectrogram_params': {'time_resolution_sec': 0.001,
                                  'frequency_resolution_hz': 110.0},}

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
                  'range': (-10.0, 10.0),
                  # [-10 ,0) = log-scaled, 0 = raw, (0, 10] alpha corrected w/contrast=(alpha-1)
                  'resolution': 0.05,
                  'init': 0,
                  'sample_value': -10.0,  # large test value for text fitting
                  'text_width': 110},
                 {'name': 'spectrogram_limit',
                  'label': 'freq. range\n1 - %i Hz',
                  'range': (1.0, 16384.0),
                  'resolution': 10.,
                  'init': 10000.,
                  'sample_value': 16384.0,  # large test value for text fitting
                  'text_width': 150},
                 ]]

    @staticmethod
    def get_color(name):
        return Layout.COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout.LUT[name]
