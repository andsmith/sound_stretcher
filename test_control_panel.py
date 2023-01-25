from controls import ControlPanel
import numpy as np
import cv2
import logging
from layout import Layout
def test_control_panel():
    bkg=np.array(Layout.get_color('bkg'),dtype=np.uint8)
    img = np.zeros((Layout.WINDOW_SIZE[1],Layout.WINDOW_SIZE[0], 4), dtype=np.uint8) + bkg

    def _update(name, value):
        print("Updated:  %s = %s" % (name, value))

    c_box = Layout.get_value('control_area')
    print(c_box)
    panel = ControlPanel(_update, c_box)

    def _mouse( event, x, y, flags, param):
        panel.mouse(event, x, y)

    cv2.namedWindow('control_test', cv2.WINDOW_AUTOSIZE)

    cv2.setMouseCallback('control_test', _mouse)
    while True:
        frame = img.copy()

        panel.draw(frame)

        cv2.imshow("control_test", frame[:, :, 2::-1].copy())
        k=cv2.waitKey(1)
        if k & 0xff == ord('q'):
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_control_panel()
