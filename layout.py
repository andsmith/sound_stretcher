import cv2

COLORS = {'slate': (0x30, 0x36, 0x3d, 255),
          'off white': (0xf6, 0xee, 0xe5, 255),
          'sky blue': (85, 206, 255, 255),
          'gray': (200, 200, 200, 255),
          'dark_gray': (60, 60, 60, 255),
          'brown': (78, 53, 36, 255),
          'cursor_green': (0x54, 0x8f, 0x66, 255),
          'dark_green': (0, 32, 15, 255)}


class Layout(object):
    CURSOR_WIDTH = 4
    CURSOR_ALPHA = 200
    HELP_TEXT_ALPHA = 255
    HELP_BKG_ALPHA = 200
    COLOR_SCHEME = {'wave_bkg': COLORS['slate'],
                    'control_bkg': COLORS['dark_gray'],
                    'control_axis': COLORS['gray'],
                    'control_slider': COLORS['off white'],
                    'control_text': COLORS['sky blue'],
                    'wave_sound': COLORS['sky blue'],
                    'wave_noise': COLORS['brown'],
                    'help_text': COLORS['dark_green'][:3] + (HELP_TEXT_ALPHA,),
                    'help_bkg': COLORS['gray'][:3] + (HELP_BKG_ALPHA,),
                    'playback_cursor': COLORS['cursor_green'][:3] + (CURSOR_ALPHA,),
                    'mouse_cursor': COLORS['gray'][:3] + (CURSOR_ALPHA,),
                    'slider_text_bkg': (64, 64, 64, 64),
                    'slider_text': COLORS['off white']}

    # misc key-value look-up
    LUT = {'help_font': cv2.FONT_HERSHEY_COMPLEX,
           'control_panel_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_panel_font_scale': .6,

           'window_size': (1200, 600),
           'wave_area_rel': {'top': 0., 'bottom': .75, 'left': 0., 'right': 0.},
           'control_area_rel': {'top': 0.8, 'bottom': 1.0, 'left': 0., 'right': 0.},
           'slider_dims': {'axis_thickness': 6,
                           'marker_thickness': 16,
                           'h_indent': 20,
                           'height': 15}}

    CONTROLS = [{'name': 'stretch_factor',
                 'label': 'stretch factor: %.2f',
                 'range': (1.0, 10.0),
                 'resolution': 0.25,
                 'init': 1.5,
                 'sample_value': 10.0,  # large test value for text fitting
                 'text_width': 250},

                {'name': 'noise_threshold',
                 'label': 'sound threshold: %.1f',
                 'range': (0.0, 40.0),
                 'sample_value': 100.0,
                 'resolution': .5,
                 'init': 5.0,
                 'text_width': 250}]

    @staticmethod
    def get_color(name):
        return Layout.COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout.LUT[name]
