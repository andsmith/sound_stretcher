import numpy as np
import cv2
import logging
from text_box import TextBox
from layout import Layout


class HelpDisplay(object):
    HELP = ('Sound Stretcher ',
            '',
            '   * Click anywhere to load file.',
            '   * Click wave/spectrum to start playback at that points.',
            '   * Click again to stop.',
            '   * Adjust noise threshold & stretch in slider window.',
            '',
            '   Hotkeys:',
            '     (space) - Play / Stop',
            '           q - quit',
            '           h - toggle this help',
            '           l - load new sound file',
            '           s - save  (input.wav -> input_x3.00.wav, etc.)',
            '',
            '',
            'file types: .wav (w/ffmpeg:  .mp3, .m4a, and .ogg)')

    def __init__(self, image_shape):
        self._shape = image_shape[:2]
        margin_px = 40
        box_dims = {'top': margin_px, 'bottom': image_shape[0] - margin_px,
                    'left': margin_px, 'right': image_shape[1] - margin_px}
        self._font = Layout.get_value('help_font')
        self._bkg_col = Layout.get_color('help_bkg')
        self._text_col = Layout.get_color('help_text')
        self._text_box = TextBox(box_dims, HelpDisplay.HELP,
                                 bkg_color=self._bkg_col, text_color=self._text_col,
                                 font=self._font)

    def add_help(self, image):
        if image.shape[:2] != self._shape:
            raise Exception("Image shape changed (%s -> %s)!" % (image.shape, self._shape))
        self._text_box.write_text(image)
