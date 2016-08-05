import tensorflow as tf


def _tensor_size(tensor):
    return tf.reduce_prod((d.value for d in tensor.get_shape()), 1)


class StyleGraph(object):
    def __init__(self, config, img, styles, initial, steps,
                 forward_graph=None, train_graph=None):
        tf.reset_default_graph()
        self.forward_graph = forward_graph or tf.Graph()
        self.train_graph = train_graph or tf.Graph()
        self.forward_session = tf.Session(graph=self.forward_graph)
        self.train_session = tf.Session(graph=self.train_graph)
        self.config = config
        self.img = img
        self.shape = (1,) + img.shape
        self.styles = styles
        self.style_shapes = [(1,) + style.shape for style in styles]
        self.initial = initial
        self.steps = steps

    def build_content_op(self):
        self.content_placeholder = tf.placeholder('float', shape=self.shape)
        self.content_op = self.config.model.build(self.content_placeholder)[self.config.content_layer_name]

    def evaluate_content_op(self):
        return self.forward_session.run([self.content_op], {self.content_placeholder: [self.img]})[0]

    def build_style_ops(self):
        # todo optimize this to only create one placeholder, they all should have the same shape
        self.style_placeholders = [tf.placeholder('float', shape=style_shape) for
                                   style_shape in self.style_shapes]

        self.style_ops = [{} for _ in self.style_shapes]
        for i, style_placeholder in enumerate(self.style_placeholders):
            # todo should create only one network, the network should work for both
            net = self.config.model.build(style_placeholder)
            for layer in self.config.style_layer_names:
                self.style_ops[i][layer] = net[layer]

    @staticmethod
    def gram(m):
        m_shape = m.get_shape().as_list()
        m = tf.reshape(m, (-1, m_shape[3]))
        return tf.matmul(m, m, transpose_b=True)
        if m_shape[1] * m_shape[2] < m_shape[3]:
            return tf.matmul(m, m, transpose_b=True)
        else:
            return tf.matmul(m, m, transpose_a=True)

    def evaluate_style_ops(self):
        style_values = [{} for _ in self.styles]
        for i, (style, style_ops_by_layers) in enumerate(zip(self.styles, self.style_ops)):
            # todo put all gram ops in one session call
            for layer, op in style_ops_by_layers.items():
                gram = self.gram(op)
                layer_val, = self.forward_session.run([gram], {self.style_placeholders[i]: [style]})
                style_values[i][layer] = layer_val/layer_val.size
        return style_values

    def build_train_op(self, content_value, style_values):
        def content_loss():
            return self.config.content_weight * (2 * tf.nn.l2_loss(
                net[self.config.model.CONTENT_LAYER_NAME] - content_value) / content_value.size)

        def style_loss():
            style_loss = 0
            for i, style_value in enumerate(style_values):
                style_losses = []
                for layer_name in self.config.style_layer_names:
                    layer = net[layer_name]
                    _, height, width, number = map(lambda i: i.value, layer.get_shape())  # as_list() instead
                    size = height * width * number
                    gram = self.gram(layer) / size
                    style_gram = style_value[layer_name]
                    style_losses.append(2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
                style_loss += self.config.style_weight * self.config.style_blend_weights[i] * tf.reduce_sum(style_losses)

            return style_loss

        def total_variation_loss():
            tv_y_size = _tensor_size(self.trainable[:, 1:, :, :])
            tv_x_size = _tensor_size(self.trainable[:, :, 1:, :])
            return self.config.weight_regularization * 2 * (
                (tf.nn.l2_loss(self.trainable[:, 1:, :, :] - self.trainable[:, :self.config.content_shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(self.trainable[:, :, 1:, :] - self.trainable[:, :, :self.config.content_shape[2] - 1, :]) /
                 tv_x_size))

        def total_loss():
            self.content_loss = content_loss()
            self.style_loss = style_loss()
            self.total_variation_loss = total_variation_loss()
            return self.content_loss + self.style_loss + self.total_variation_loss

        self.trainable = tf.Variable(initial_value=self.initial, name='img_var')
        net = self.config.model.build(self.trainable)

        self.loss = total_loss()
        self.train = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

    def build_forward_graph(self):
        with self.forward_graph.as_default():
            self.build_content_op()
            self.build_style_ops()

    def build_train_graph(self):
        # in order to build this graph we need to evaluate the forward_graph
        with self.forward_graph.as_default():
            content_value = self.evaluate_content_op()
            style_values = self.evaluate_style_ops()

        with self.train_graph.as_default():
            self.build_train_op(content_value, style_values)
            self.train_session.run(tf.initialize_all_variables())

    def train(self, step):
        def print_progress():
            print('step n: {}'.format(step))

            message = """
            content_loss: {}
            style_loss: {}
            total_variation_loss: {}
            total_loss: {}
            """.format(content_loss, style_loss, total_variation_loss, loss)
            print(message)

        _, loss, content_loss, style_loss, total_variation_loss = self.session.run(
            [self.train, self.loss, self.content_loss, self.style_loss, self.total_variation_loss])

        if step % self.config.every_n_steps == 0:
            print_progress()
            return self.train_session.run([self.trainable])[0]

    def fit(self):
        self.build_forward_graph()
        self.build_train_graph()
        results = []
        for step in range(1, self.steps):
            result = self.train(step)
            if result:
                results.append(result)

        return results
