import numpy as np
import cv2
import logging

from layout import Layout

HELP = ('Sound Stretcher ',
        '',
        '   * Click anywhere to load file.',
        '   * Click on waveform to start playback at that point.',
        '   * Click again to stop.',
        '   * Adjust noise threshold & stretch in slider window.',
        '',
        '   Hotkeys:',
        '      q - quit',
        '      h - toggle this help',
        '      l - load new sound file',
        '      s - save  (input.wav -> input_x3.00.wav, etc.)',
        '',
        '',
        'file types: .wav (w/ffmpeg:  .mp3, .m4a, and .ogg)')


class HelpDisplay(object):
    """
    Write help text on a box over everything in frame
    """

    def __init__(self, image_shape):
        self._text_sizes = None  # ((width, height) baseline) for each line of text
        self._box_poly_coords = None  # for bkg box}
        self._image_shape = None

        self._font = cv2.FONT_HERSHEY_COMPLEX
        self._font_thickness = 1
        self._text_color = Layout.get_color('help_text').tolist()
        self._bkg_color = Layout.get_color("help_bkg").tolist()

        self._spacing = dict(
            v_spacing_pixels=15,
            outside_indent=25,
            inside_v_indent=15,
            left_indent=15,
            right_indent=15 * 2)


    def _calc_font_scale(self):
        font_scale = 1.0

        too_big = True
        for _ in range(100):
            sizes = [cv2.getTextSize(line, self._font, font_scale, self._font_thickness) for line in HELP]

            # background_rect
            text_width = np.max([size[0][0] for size in sizes])
            text_height = np.sum([size[0][1] for size in sizes]) + (len(sizes) - 1) * self._spacing['v_spacing_pixels']

            box = {'left': self._spacing['outside_indent'],
                   'right': self._spacing['outside_indent'] + self._spacing['left_indent'] + self._spacing[
                       'right_indent'] + text_width,
                   'top': self._spacing['outside_indent'],
                   'bottom': self._spacing['outside_indent'] + self._spacing['inside_v_indent'] * 2 + text_height}

            if box['bottom'] + self._spacing['outside_indent'] >= self._image_shape[0] or \
                    box['right'] + self._spacing['outside_indent'] >= self._image_shape[1]:
                font_scale *= 0.95
                logging.info("Text too big for image, shrinking:  %.3f" % (font_scale,))
            else:
                too_big = False
                break
        if too_big:
            raise Exception("Could not fit text in image.")

        box_coords = np.array([[box['left'], box['top']],
                               [box['right'], box['top']],
                               [box['right'], box['bottom']],
                               [box['left'], box['bottom']]], dtype=np.int32)
        self._box_poly_coords = box_coords
        self._text_sizes = sizes
        self._font_scale = font_scale

    def add_help(self, image):
        """
        Display hotkey info, etc.
        :param image: input image, for info to be drawn on.
        """
        if self._image_shape is None or image.shape != self._image_shape:
            self._image_shape = image.shape
            self._calc_font_scale()

        image = cv2.fillPoly(image, [self._box_poly_coords], self._bkg_color, cv2.LINE_AA)
        # text lines

        x = self._spacing['outside_indent'] + self._spacing['left_indent']
        y = self._spacing['outside_indent'] + self._spacing['inside_v_indent'] + self._text_sizes[0][0][1]
        for line_ind, line in enumerate(HELP):
            cv2.putText(image, line, (x, y), self._font, self._font_scale, self._text_color, self._font_thickness,
                        cv2.LINE_AA)
            y += self._text_sizes[line_ind][0][1] + self._spacing['v_spacing_pixels']

