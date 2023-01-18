import matplotlib.pyplot as plt
import logging
import numpy as np

from help import HelpDisplay
from layout import Layout

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    size = (1000, 800)
    frame = np.zeros((size[1], size[0], 4), dtype=np.uint8) + Layout.get_color('background')
    helper = HelpDisplay(frame.shape)
    helper.add_help(frame)
    plt.imshow(frame)
    plt.show()
