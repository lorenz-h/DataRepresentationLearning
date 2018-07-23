import scipy.misc
import pywt
import numpy as np
import matplotlib.pyplot as plt

for feature in range(20, 21):
    file_string = "Dataset2/Training/sample" + str(feature) + ".png"
    image = scipy.misc.imread(file_string).astype(float)
    image = image[..., 0]
    assert image.ndim == 2
    wp = pywt.wavedec2(image, 'db1', mode='symmetric', level=2)
    print(wp[0].shape)
    level1 = np.vstack((np.hstack((wp[0], wp[1][0])), np.hstack((wp[1][1], wp[1][2]))))
    level2 = np.vstack((np.hstack((level1, wp[2][0])), np.hstack((wp[2][1], wp[2][2]))))
    plt.imshow(level2)
    plt.show()
