import numpy as np


class RandomOrientation90(object):
    def __call__(self, img):
        degrees = 90*np.random.randint(0, 4)
        img.rotate(degrees)
        return img
