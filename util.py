import logging
import os
import numpy as np


def exp_fact_from_control_value(x, n_powers_max, precision = 1):
    """
    For zooming in at exponential scales.

    :param x: float in (0, 1.]
    :param n_powers_max:  maximum zoom is 2 ** n_powers_max
    :return:  zoom scale corresponding to control value x
    """
    factor = 2.0 ** (n_powers_max * (1.0 - x))
    return factor


def in_area(pos, bbox):
    """
    :returns: True if position is inside bounding box.
    """
    return bbox['left'] <= pos[0] < bbox['right'] and bbox['top'] <= pos[1] < bbox['bottom']


def make_unique_filename(unversioned):
    if not os.path.exists(unversioned):
        return unversioned
    version = 0
    file_part, ext = os.path.splitext(unversioned)
    filename = "%s_%i%s" % (file_part, version, ext)
    while os.path.exists(filename):
        version += 1
        filename = "%s_%i%s" % (file_part, version, ext)

    return filename


def draw_v_line(image, x, width, color, y_range=None):
    """
    Draw vertical line on image.
    :param image: to draw on
    :param x: x coordinate of line
    :param width: of line in pixels (should be even?)
    :param y_range:  dict with 'top' and 'bottom' or None for whole image
    :param color: of line to draw
    """
    x_coords = np.array([x - width / 2, x + width / 2])
    if x_coords[0] < 0:
        x_coords += x_coords[0]
    if x_coords[1] > image.shape[1] - 1:
        x_coords -= x_coords[1] - image.shape[1] + 1

    if y_range is None:
        y_low, y_high = 0, image.shape[0]
    else:
        y_low, y_high = y_range['top'], y_range['bottom']

    x_coords = np.int64(x_coords)
    if len(color) == 4 and color[3] < 255:
        line = image[y_low: y_high, x_coords[0]:x_coords[1], :]
        alpha = float(color[3]) / 255.
        new_line = alpha * color + (1.0 - alpha) * line
        image[y_low: y_high, x_coords[0]:x_coords[1], :] = np.uint8(new_line)
    else:
        image[y_low: y_high, x_coords[0]:x_coords[1], :] = color
