from copy import deepcopy
import numpy as np
import cv2
from layout import Layout
from text_box import TextBox
from util import in_area


def indent(bbox, margin):
    return {'top': bbox['top'] + margin, 'bottom': bbox['bottom'] - margin,
            'left': bbox['left'] + margin, 'right': bbox['right'] - margin}


class Slider(object):

    def __init__(self, bbox, props):
        self.name = props['name']
        self._font = Layout.get_value('control_panel_font')
        self._font_scale = Layout.get_value('control_panel_font_scale')
        self._text_color = Layout.get_color('control_text')
        self._axis_color = Layout.get_color('control_axis')
        self._slider_color = Layout.get_color('control_slider')
        self._dims = Layout.get_value('slider_dims')

        self._props = props
        self._value = props['init']

        text_box = bbox.copy()
        text_box['right'] = bbox['left'] + props['text_width']
        slider_box = bbox.copy()
        slider_box['left'] = text_box['right'] + 1
        self._text_box = text_box
        self._slider_box = slider_box

        self._x_axis = (self._slider_box['left'] + self._dims['h_indent'],
                        self._slider_box['right'] - self._dims['h_indent'],)

        self._slide_left, self._slide_right = (self._slider_box['left'] + self._dims['h_indent'],
                                               self._slider_box['right'] - self._dims['h_indent'])

        self._box = bbox

        self._clicked = False
        self._tb = None

    def get_value(self):
        return self._value

    def mouse(self, event, x, y):
        """
        Process user moving mouse.
        (params same as cv2 callback)
        returns:  Value, if it changed, else None
        """
        if event == cv2.EVENT_MOUSEMOVE:
            if self._clicked:
                return self._move_to(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._clicked = False

        elif event == cv2.EVENT_LBUTTONDOWN and in_area((x, y), self._slider_box):
            self._clicked = True
            return self._move_to(x, y)

    def _move_to(self, x, y):
        if self._x_axis[0] <= x <= self._x_axis[1]:
            new_value = self._get_value_from_pos(x)
            if new_value != self._value:
                self._value = new_value
                self._image = None
                return self._value

        return None

    def _get_value_from_pos(self, x):
        """
        :param x: mouse position pixels (x-coord)
        :return:  value on slider scale
        """
        rel_pos = (x - self._slide_left) / float(self._slide_right - self._slide_left)
        value = rel_pos * (self._props['range'][1] - self._props['range'][0]) + self._props['range'][0]
        return np.round(value / self._props['resolution']) * self._props['resolution']

    def _get_pos_from_value(self, value):

        rel_pos = (value - self._props['range'][0]) / (self._props['range'][1] - self._props['range'][0])
        x = self._slide_left + rel_pos * (self._slide_right - self._slide_left)
        return int(x)

    def draw(self, image):
        """
        Draw current state of slider
        :return:  image
        """
        indent = 15
        image[self._box['top']:self._box['bottom'],
        self._box['right']:self._box['left'], :] = Layout.get_color('control_bkg')
        center_y = int((self._slider_box['bottom'] + self._slider_box['top']) / 2)

        # write text
        string = self._props['label'] % (self._value,)
        (width, height), _ = cv2.getTextSize(string, self._font, self._font_scale, 1)
        text_xy = (self._text_box['right'] - width,
                   center_y + int(height / 2))

        cv2.putText(image, string, text_xy, self._font, self._font_scale, self._text_color, 1, cv2.LINE_AA)

        # draw slider
        h = self._dims['height']
        a_t = int(self._dims['axis_thickness'] / 2)
        m_t = int(self._dims['marker_thickness'] / 2)

        image[center_y - a_t:center_y + a_t, self._slide_left:self._slide_right, :] = self._axis_color  # axis
        image[center_y - h:center_y + h, self._slide_left - a_t:self._slide_left + a_t, :] = self._axis_color  # left
        image[center_y - h:center_y + h, self._slide_right - a_t:self._slide_right + a_t, :] = self._axis_color  # right
        marker_x = self._get_pos_from_value(self._value)
        image[center_y - h:center_y + h, marker_x - a_t:marker_x + m_t, :] = self._slider_color  # slider


class ControlPanel(object):
    def __init__(self, update_callback, region):
        self._callback = update_callback
        self._box = region
        self._clicked_ind = 0
        self._controls = self._make_controls()

    def _make_controls(self):
        """
        For now, arrange vertically
        """
        n = len(Layout.CONTROLS)
        y_div_lines = np.linspace(self._box['top'], self._box['bottom'], n + 1).astype(np.int64)
        ctrl = []
        for i, control in enumerate(Layout.CONTROLS):
            box = {'top': y_div_lines[i],
                   'bottom': y_div_lines[i + 1],
                   'left': self._box['left'],
                   'right': self._box['right']}
            ctrl.append(Slider(box, control))

        return ctrl

    def get_value(self, name):
        ctrl = [c for c in self._controls if c.name == name]
        if len(ctrl) != 1:
            raise Exception("Control not found:  %s" % (name,))
        return ctrl[0].get_value()

    def mouse(self, event, x, y):
        for ctrl_i, control in enumerate(self._controls):
            updated_value = control.mouse(event, x, y)
            if updated_value is not None:
                self._callback(control.name, updated_value)

    def draw(self, frame):
        """
        Draw on current frame
        """
        for control in self._controls:
            control.draw(frame)
