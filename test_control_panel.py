from controls import ControlPanel
import numpy as np
import cv2

def test_control_panel():

    img = np.zeros((600, 1000, 4), dtype=np.uint8)
    img[:, :, 3] = 255

    def _update(name, value):
        print("Updated:  %s = %s" % (name, value))

    panel = ControlPanel(_update, {'top': 500, 'bottom': img.shape[0], 'left': 0, 'right': img.shape[1]})

    def _mouse( event, x, y, flags, param):
        panel.mouse(event, x, y)

    cv2.namedWindow('control_test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('control_test', img.shape[:2][::-1])
    cv2.setMouseCallback('control_test', _mouse)
    while True:
        frame = img.copy()
        panel.draw(frame)
        cv2.imshow("control_test", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    test_control_panel()
