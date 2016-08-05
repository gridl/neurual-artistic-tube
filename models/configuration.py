from models.vgg import Vgg19


class StyleConfig(object):
    def __init__(self, styles=None, model=None, initial_img=None, learning_rate=1e1, weight_regularization=1e2,
                 every_n_steps=None, content_weight=5e0, style_scale=1.0, style_scales=None, content_layer=None,
                 style_layers=None, style_weight=None, style_blend_weights=None):
        self.styles = styles or ['./styles/style1.jpg']
        self.model = model or Vgg19()
        self.initial_img = initial_img
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.style_scale = style_scale
        self.style_scales = style_scales or []
        self.style_blend_weights = self.set_style_blend_weights(style_blend_weights)
        self.learning_rate = learning_rate
        self.weight_regularization = weight_regularization
        self.every_n_steps = every_n_steps
        self.content_layer_name = content_layer or self.model.CONTENT_LAYER_NAME
        self.style_layer_names = style_layers or self.model.STYLE_LAYER_NAMES

    def set_style_blend_weights(self, style_weights=None):
        if not style_weights:  # default behavior: all styles have same weights
            styles_len = len(self.styles)
            return [1.0 / styles_len] * styles_len

        return [w / sum(self.style_blend_weights) for w in self.style_blend_weights]
