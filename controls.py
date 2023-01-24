from copy import deepcopy
import numpy as np
import cv2
from layout import Layout
from util import in_area
import logging
from text_box import get_font_scale, get_centered_offset


def indent(bbox, margin):
    return {'top': bbox['top'] + margin, 'bottom': bbox['bottom'] - margin,
            'left': bbox['left'] + margin, 'right': bbox['right'] - margin}


class Slider(object):
    """
    Standard "slider" ui element, with value label on the left
    """

    def __init__(self, bbox, props):
        """
        Initialize a slider
        :param bbox: dict with 'top','bottom','left','right' bounding slider area
        :param props: dict, imported from Layout.CONTROLS
        """
        self._box = bbox
        self.name = props['name']  # for app to use
        self._font = Layout.get_value('control_panel_font')
        self._text_color = Layout.get_color('control_text')
        self._axis_color = Layout.get_color('control_axis')
        self._slider_color = Layout.get_color('control_slider')

        self._props = props

        # state
        self._clicked = False  # user has button down
        self._value = props['init']

        # resize text as needed
        self._text_bbox = bbox.copy()
        self._text_bbox['right'] = self._text_bbox['left'] + props['text_width']
        assert self._text_bbox['right'] <= self._box['right']
        test_string = props['label'] % props['sample_value']
        logging.info("Fitting text box for slider %s with sample lines:  %s" % (self._props['name'], test_string))
        # if self._props['name']=='spectrogram_contrast':
        #    import ipdb; ipdb.set_trace()
        self._set_font_scale(test_string=test_string)

        # pre-calc some geometry
        dims = Layout.get_value('slider_dims')
        slider_box = bbox.copy()
        slider_box['left'] = self._text_bbox['right'] + 1
        self._slider_bbox = slider_box
        self._slide_left, self._slide_right = (self._slider_bbox['left'] + dims['h_indent'],
                                               self._slider_bbox['right'] - dims['h_indent'])

        # self._text_x = self._text_bbox['left'] + dims['text_indent_h_v'][0]  # left justify
        self._text_x = int((self._text_bbox['left'] + self._text_bbox['right']) / 2)  # center

        s = dims['h_indent']
        h = dims['height']  # height of slider marker & axis ends
        center_y = int((self._box['bottom'] + self._slider_bbox['top']) / 2)
        a_t = int(dims['axis_thickness'] / 2)
        m_t = int(dims['marker_thickness'] / 2)
        self._marker = {'top': center_y - h, 'bottom': center_y + h, 'left_rel': -m_t, 'right_rel': m_t}
        self._scale_left = self._slider_bbox['left'] + s
        self._scale_right = self._slider_bbox['right'] - s
        self._scale_parts = [
            {'top': center_y - a_t, 'bottom': center_y + a_t, 'left': self._scale_left, 'right': self._scale_right,
             'color': self._axis_color},  # axis
            {'top': center_y - h, 'bottom': center_y + h, 'left': self._scale_left - a_t,
             'right': self._scale_left + a_t,
             'color': self._axis_color},  # left end
            {'top': center_y - h, 'bottom': center_y + h, 'left': self._scale_right - a_t,
             'right': self._scale_right + a_t,
             'color': self._axis_color}]  # right end

    def _set_font_scale(self, test_string):
        """
        Given size of the label's bounding box, and a test string ideally as big as possible,
        determine the font scale, and get the vertical spacing.
        :param test_string:  label%(value,)
        """
        lines = test_string.split('\n')
        v_spacing = Layout.get_value('control_text_spacing')
        text_indent_h_v = Layout.get_value('slider_dims')['text_indent_h_v']
        width = self._text_bbox['right'] - self._text_bbox['left']
        height = self._text_bbox['bottom'] - self._text_bbox['top']
        self._font_scale, self._spacing = get_font_scale(lines, width - text_indent_h_v[0] * 2,
                                                         height - text_indent_h_v[1] * 2, v_spacing,
                                                         self._font, 1)

        self._y_offset = get_centered_offset(self._text_bbox, self._spacing, v_spacing)[1]

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

        elif event == cv2.EVENT_LBUTTONDOWN and in_area((x, y), self._slider_bbox):
            self._clicked = True
            return self._move_to(x, y)

    def _move_to(self, x, y):
        if self._scale_left <= x <= self._scale_right:
            new_value = self._get_value_from_pos(x)
            if new_value != self._value:
                self._value = new_value
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

        # draw text_box
        # part = self._text_bbox
        # image[part['top']: part['bottom'], part['left']: part['right'], :] = (255, 0, 0, 255)
        # draw text_box
        # part = self._slider_bbox
        # image[part['top']: part['bottom'], part['left']: part['right'], :] = (0, 255, 0, 255)

        # write text
        string = self._props['label'] % (self._value,)

        y = self._text_bbox['top'] + self._y_offset
        lines = string.split('\n')
        widths = [cv2.getTextSize(line, self._font, self._font_scale, 1)[0][0] for line in lines]

        v_spacing = Layout.get_value('control_text_spacing')

        for line_no, ((_, height), baseline) in enumerate(self._spacing):

            y += height
            x = self._text_x - int(widths[line_no] / 2)
            cv2.putText(image, lines[line_no], (x, y), self._font, self._font_scale, self._text_color, 1, cv2.LINE_AA)
            y += v_spacing + baseline

        # draw scale
        for part in self._scale_parts:
            image[part['top']: part['bottom'], part['left']: part['right'], :] = part['color']

        # draw marker
        marker_x = self._get_pos_from_value(self._value)

        image[self._marker['top']: self._marker['bottom'],
        marker_x + self._marker['left_rel']:marker_x + self._marker['right_rel'], :] = self._slider_color  # slider


class ControlPanel(object):
    """
    Read list of controls from layout.py, initialize where they will go in the window, etc.
    """

    def __init__(self, update_callback, region):
        """
        Initialize all controls.
        :param update_callback: Call this function when a user interacts and changes a value.
        :param region:  dict with 'top','bottom','left','right', where the controls go in the output frames
        """
        self._callback = update_callback
        self._bbox = region
        self._clicked_ind = 0
        self._controls = self._make_controls()

    def _make_controls(self):
        """
        Space everything out evenly for now.
        layout defined by list of lists (rows, columns) in Layout.CONTROLS
        """
        ctrl = []
        n_rows = len(Layout.CONTROLS)
        y_div_lines = np.linspace(self._bbox['top'], self._bbox['bottom'], n_rows + 1).astype(np.int64)
        h_spacing = Layout.get_value('control_h_spacing')
        for row_i, control_row in enumerate(Layout.CONTROLS):
            n_cols = len(control_row)
            width = self._bbox['right'] - self._bbox['left']
            control_width = int((width - (n_cols - 1) * h_spacing)/n_cols)

            for col_i, control in enumerate(control_row):
                x_left = col_i * (control_width + h_spacing)
                box = {'top': y_div_lines[row_i],
                       'bottom': y_div_lines[row_i + 1],
                       'left': x_left,
                       'right': x_left + control_width}
                logging.info("\tControl (%i, %i):  %s @%s" % (row_i, col_i, control['name'], box))

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
