import numpy as np
import cv2
import logging
from text_box import TextBox
from layout import Layout
from version import VERSION

class HelpDisplay(object):
    HELP = ('Sound Stretcher - %s' % (VERSION,),
            '',
            '   * Click this screen to open file...',
            '',
            '   * Hotkeys:',
            '',
            '       (space) - Pause / Play',
            '       h - show this help',
            '       l - load new sound file',
            '       s - save sound w/current stretch',
            '       q - quit',
            '',
            '   * File types: .wav (w/ffmpeg .mp3, .m4a, and .ogg/.oga)')

    def __init__(self, frame_shape):
        """
        :param frame_shape:  of wnidow
        """
        self._shape = frame_shape[:2]
        margin_px = 100
        box_dims = {'top': margin_px, 'bottom': frame_shape[0] - margin_px,
                    'left': margin_px, 'right': frame_shape[1] - margin_px}
        self._font = Layout.get_value('help_font')
        self._bkg_col = Layout.get_color('help_bkg')
        self._text_col = Layout.get_color('help_text')
        self._text_box = TextBox(box_dims, HelpDisplay.HELP,
                                 bkg_color=self._bkg_col, text_color=self._text_col,
                                 font=self._font, font_scale=2.0,thickness=1,line_style=cv2.LINE_AA)

    def add_help(self, image):
        if image.shape[:2] != self._shape:
            raise Exception("Image shape changed (%s -> %s)!" % (image.shape, self._shape))
        self._text_box.write_text(image)
