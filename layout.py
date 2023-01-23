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
    CURSOR_WIDTH = 4  # pixels
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
                    'msg_text_color': COLORS['dark_green'][:3] + (HELP_TEXT_ALPHA,),
                    'msg_bkg_color': COLORS['off white'][:3] + (HELP_BKG_ALPHA,),
                    'playback_cursor': COLORS['cursor_green'][:3] + (CURSOR_ALPHA,),
                    'mouse_cursor': COLORS['gray'][:3] + (CURSOR_ALPHA,),
                    'slider_text_bkg': (64, 64, 64, 64),
                    'slider_text': COLORS['off white']}

    # misc key-value look-up
    LUT = {'help_font': cv2.FONT_HERSHEY_COMPLEX,
           'control_panel_font': cv2.FONT_HERSHEY_DUPLEX,
           'control_panel_font_scale': .9,

           'window_size': (1200, 500),
           'wave_area_rel': {'top': 0., 'bottom': .85/2, 'left': 0., 'right': 1.},
           'msg_area_rel': {'top': .3, 'bottom': .7, 'left': 0.3, 'right': .7},
           'spectrum_area_rel': {'top': .85/2, 'bottom': .85, 'left': 0., 'right': 1.},
           'control_area_rel': {'top': 0.85, 'bottom': 1.0, 'left': 0., 'right': 1.},
           'interaction_area_rel': {'top': 0., 'bottom': 0.85,'left': 0., 'right': 1.0},
           'slider_dims': {'axis_thickness': 6,
                           'marker_thickness': 16,
                           'h_indent': 20,
                           'height': 15},
           'spectrogram_params': {'plot_freq_range_hz': (0., 13750.),
                                  'time_resolution_sec': 0.001,
                                  'frequency_resolution_hz': 110.0},
           'msg_text_params' : {'font': cv2.FONT_HERSHEY_DUPLEX,
                                'bkg_color': COLOR_SCHEME['msg_bkg_color'],
                                'text_color': COLOR_SCHEME['msg_text_color']}}

    CONTROLS = [{'name': 'stretch_factor',
                 'label': 'Stretch: %.2f x',
                 'range': (1.0, 10.0),
                 'resolution': 0.01,
                 'init': 1.0,
                 'sample_value': 10.0,  # large test value for text fitting
                 'text_width': 250}]
    '''
    {'name': 'noise_threshold',
     'label': 'sound threshold: %.4f',
     'range': (0.0, 1.),
     'sample_value': .777,
     'resolution': .01,
     'smoothing_sec': 0.01,  # sd of gaussian smoothing kernel
     'init': .00,
     'text_width': 250}
     '''

    @staticmethod
    def get_color(name):
        return Layout.COLOR_SCHEME[name]

    @staticmethod
    def get_value(name):
        return Layout.LUT[name]
