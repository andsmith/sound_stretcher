import cv2
from layout import Layout
import numpy as np
import logging


class TextBox(object):
    """
    Write text on a box over image.

    Initialize with text & params, then call write_text(image) to add to the image.
    """

    def __init__(self, box_dims, text_lines, bkg_color=(64, 64, 64, 200), text_color=(255, 255, 255, 255),
                 font=cv2.FONT_HERSHEY_COMPLEX, thickness=1, line_style=cv2.LINE_AA, font_scale=1.0, centered=False):
        """
        Create a text box in a specific place on the image.  (can change text/font)
        :param box_dims:  dict('top','bottom','left','right') in pixels, where text will go

        """
        self._box_dims = box_dims
        self._box_width = box_dims['right'] - box_dims['left']
        self._box_height = box_dims['bottom'] - box_dims['top']
        self._font = {'font': font, 'thickness': thickness}
        self._line_style = line_style

        logging.info("Creating text box:  %s x %i  \"%s..." % (self._box_width, self._box_height, text_lines[0][:10]))

        self._spacing = dict(
            v_spacing_pixels=10,  # between lines
            v_indent=15,  # top & bottom
            left_indent=15,
            right_indent=15)

        self._font_scale, self._sizes = get_font_scale(text_lines,
                                                       width=self._box_width - self._spacing['left_indent'] -
                                                             self._spacing['right_indent'],
                                                       height=self._box_height - 2.0 * self._spacing['v_indent'],
                                                       v_spacing_pixels=self._spacing['v_spacing_pixels'],
                                                       font_scale=font_scale,
                                                       **self._font)
        self._bkg_color, self._text_color = bkg_color, text_color
        self._overlay_weighted, self._bkg_weights = self._draw_overlay_box(text_lines, centered=centered)

    def _draw_overlay_box(self, text_lines, centered=False):
        img = np.zeros((self._box_height, self._box_width, 4)) + np.array(self._bkg_color, dtype=np.uint8)

        # text lines
        if centered:
            x, y = get_centered_offset(self._box_dims, self._sizes, self._spacing['v_spacing_pixels'])
        else:
            x, y = self._spacing['left_indent'], self._spacing['v_indent']
        for line_ind, line in enumerate(text_lines):
            # add text line height
            y += self._sizes[line_ind][0][1]
            cv2.putText(img, line, (x, y), self._font['font'],
                        self._font_scale,
                        self._text_color,
                        self._font['thickness'],
                        self._line_style)
            # add baseline & extra spacing
            y += self._sizes[line_ind][1] + self._spacing['v_spacing_pixels']

        alpha = np.expand_dims(img[:, :, 3] / 255., 2)
        img_weighted = img[:, :, :3] * alpha
        bkg_weights = 1.0 - alpha

        return img_weighted, bkg_weights

    def write_text(self, image):
        """
        Display hotkey info, etc.
        :param image: input image, for info to be drawn on.
        """
        # draw box
        # image[self._box_dims['top']: self._box_dims['bottom'],
        #      self._box_dims['left']:self._box_dims['right'],:] = self._overlay
        # draw box blended
        old = image[self._box_dims['top']: self._box_dims['bottom'],
              self._box_dims['left']:self._box_dims['right'], :3]
        new = old * self._bkg_weights + self._overlay_weighted
        new[new > 255] = 255
        image[self._box_dims['top']: self._box_dims['bottom'],
        self._box_dims['left']:self._box_dims['right'], :3] = new
        image[self._box_dims['top']: self._box_dims['bottom'],
        self._box_dims['left']:self._box_dims['right'], 3] = 255


def get_centered_offset(bbox, sizes, v_spacing):
    """
    Get the top-left coordinate for cv2.putText() so it's centered.
    :param bbox: dict('top','left','right','bottom')
    :param sizes: list of [((width, height) baseline), ... for each line of text]
    :param v_spacing:  this many pixels between lines of text
    :return: (x, y)
    """
    height = np.sum([size[0][1] + size[1] for size in sizes]) + v_spacing * (len(sizes) - 1)  # height + baseline
    width = np.max([size[0][0] for size in sizes])
    return int((bbox['right'] - bbox['left'] - width) / 2.), int((bbox['bottom'] - bbox['top'] - height) / 2.),


def get_font_scale(lines, width, height, v_spacing_pixels, font, thickness, font_scale=1.0):
    """
    Automatically resize font if it won't fit in the box.

    :param lines:  list of strings
    :param width:  pixels to fit text horizontally
    :param height:  pixels to fit text vertically
    :param v_spacing_pixels:  pixels between each text line
    :param font:  cv2 puttext param
    :param thickness:   cv2 puttext param
    :param font_scale:   cv2 puttext param
    :return: font_scale, [((width, height), baseline), ... for each line of text]
    """

    too_big = True
    for _ in range(100):
        sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]

        # background_rect
        text_width = np.max([size[0][0] for size in sizes])
        text_height = np.sum([size[0][1] + size[1] for size in sizes]) + (len(sizes) - 1) * v_spacing_pixels

        if text_width > width or text_height > height:
            font_scale *= 0.95
            # logging.info("Text (%i x %i) too big for box (%i x %i), shrinking font scale to:  %.3f" % (
            #    text_width, text_height, width, height, font_scale,))
        else:
            too_big = False
            break
    if too_big:
        raise Exception("Could not fit text in image.")

    return font_scale, sizes
