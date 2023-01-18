import numpy  as np

COLORS = {'slate': (0x30, 0x36, 0x3d, 255),
          'off white': (0xf6, 0xee, 0xe5, 255),
          'sky blue': (85, 206, 255, 255),
          'gray': (200, 200, 200, 255),
          'brown': (78, 53, 36, 255),
          'cursor_green': (0x54, 0x8f, 0x66, 255),
          'dark_green': (0,32,15, 255)}


class Layout(object):
    CURSOR_WIDTH = 4
    CURSOR_ALPHA = 200
    HELP_TEXT_ALPHA=.85
    HELP_BKG_ALPHA = 0.66
    COLOR_SCHEME = {'background': COLORS['slate'],
                    'wave_sound': COLORS['sky blue'],
                    'wave_noise': COLORS['brown'],
                    'help_text': COLORS['dark_green'][:3] + (HELP_TEXT_ALPHA,),
                    'help_bkg': COLORS['gray'][:3] + (HELP_BKG_ALPHA,),
                    'playback_cursor': COLORS['cursor_green'][:3] + (CURSOR_ALPHA,),
                    'mouse_cursor': COLORS['gray'][:3] + (CURSOR_ALPHA,)}

    @staticmethod
    def get_color(name):
        return np.array(Layout.COLOR_SCHEME[name], dtype=np.uint8)
