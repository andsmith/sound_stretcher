import numpy as np
import matplotlib.pyplot as plt
from text_box import TextBox

def test_text_box():
    def mk_bbox(img):
        return{'top': 20, 'bottom': img.shape[0] - 20,
            'left': 20, 'right': img.shape[1] - 20}
    img_short = np.zeros((200, 800, 4), dtype=np.uint8)
    img_short_narrow= np.zeros((200, 300, 4), dtype=np.uint8)
    img_short[:, :, 2:] = 255
    img_short_narrow[:, :, 2:] = 255

    lines = ['line 1', 'Line 2 is a lot longer than line 1.','line3','line4','line5']
    box = TextBox(mk_bbox(img_short), lines)
    box.write_text(img_short)

    box = TextBox(mk_bbox(img_short_narrow), lines)
    box.write_text(img_short_narrow)

    plt.subplot(2,1,1)
    plt.imshow(img_short)
    plt.subplot(2,1,2)
    plt.imshow(img_short_narrow)
    plt.show()


if __name__ == "__main__":
    test_text_box()
