import scipy.misc
import numpy as np
import tensorflow as tf

from models.configuration import StyleConfig
from models.styler import StyleGraph


class StyleNet(object):
    def __init__(self, config=None):
        self.config = config or StyleConfig()
        self.width = None
        self.height = None

    @property
    def shape(self):
        return self.height, self.width

    def set_img_shape(self, height, width):
        self.width = width
        self.height = height

    def prepare_styles(self):
        scaled_styles = [self.read_img(style) for style in self.config.styles]
        for i, style in enumerate(scaled_styles):
            style_scale = self.config.style_scale
            if self.config.style_scales:
                style_scale = self.config.style_scales[i]
            scaled_styles[i] = (scipy.misc.imresize(style, style_scale * self.width / style.shape[1]) -
                                self.config.model.pixel_mean)
        return scaled_styles

    def prepare_initial(self, img):
        if self.config.initial_img is None:
            return tf.random_normal((1,) + img.shape) * 0.256
        else:
            initial_img = self.read_img(self.config.initial_img)
            initial_img = scipy.misc.imresize(initial_img, self.shape)
            return initial_img - self.config.model.pixel_mean

    def prepare_img(self, img):
        img = self.read_img(img)
        if not any([self.height, self.width]):
            shape = (img.shape[0], img.shape[1])
        elif all([self.height, self.width]):
            shape = (self.height, self.width)
        elif self.width is not None:
            shape = (img.shape[0] * self.width / img.shape[1], self.width)
        elif self.height is not None:
            shape = (self.height, img.shape[1] * self.height / img.shape[0])

        self.set_img_shape(*shape)
        return scipy.misc.imresize(img, shape) - self.config.model.pixel_mean

    def read_img(self, path):
        return scipy.misc.imread(path).astype(np.float)

    def prepare_styled_img(self, img):
        return scipy.misc.imresize(img, self.shape) + self.config.model.pixel_mean

    def style(self, img, steps):
        img = self.prepare_img(img)
        styles = self.prepare_styles()
        initial = self.prepare_initial(img)
        styler = StyleGraph(self.config, img, styles, initial, steps)
        results = styler.fit()
        return results
