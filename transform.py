# Most code in this file was borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py

import tensorflow as tf


class Transform:
    def __init__(self, n_styles, mode='train'):
        self.n_styles = n_styles
        if mode == 'train':
            self.reuse = None
        else:
            self.reuse = True

    def net(self, image, style_index):
        conv1 = self._conv_layer(image, 16, 9, 1, name='conv1')
        cinst1 = self._conditional_instance_norm(conv1, style_index, 'cinst1')

        conv2 = self._conv_layer(cinst1, 32, 3, 2, name='conv2')
        cinst2 = self._conditional_instance_norm(conv2, style_index, 'cinst2')

        conv3 = self._conv_layer(cinst2, 32, 3, 2, name='conv3')
        cinst3 = self._conditional_instance_norm(conv3, style_index, 'cinst3')

        resid1 = self._residual_block(cinst3, 3, name='resid1')
        resid2 = self._residual_block(resid1, 3, name='resid2')
        resid3 = self._residual_block(resid2, 3, name='resid3')
        resid4 = self._residual_block(resid3, 3, name='resid4')
        resid5 = self._residual_block(resid4, 3, name='resid5')

        conv_t1 = self._conv_tranpose_layer(resid5, 32, 3, 1, name='convt1')
        cinst4 = self._conditional_instance_norm(
            conv_t1, style_index, 'cinst4')

        conv_t2 = self._conv_tranpose_layer(cinst4, 16, 3, 1, name='convt2')

        conv_t3 = self._conv_layer(conv_t2, 3, 9, 1, relu=False, name='convt3')
        return (tf.nn.tanh(conv_t3)+1)*127.5

    def _reflection_padding(self, net, padding):
        return tf.pad(net, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "REFLECT")

    def _conv_layer(self, net, num_filters, filter_size, strides, padding='VALID', relu=True, name=None):
        weights_init = self._conv_init_vars(
            net, num_filters, filter_size, name=name)
        strides_shape = [1, strides, strides, 1]
        net = self._reflection_padding(net, (filter_size//2, filter_size//2))
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)
        net = self._instance_norm(net, name=name)
        if relu:
            net = tf.nn.relu(net)
        return net

    def _conv_tranpose_layer(self, net, num_filters, filter_size, strides, name=None):
        _, rows, cols, _ = [i.value for i in net.get_shape()]
        # Upsample
        net = tf.image.resize_images(
            net, (rows*2, cols*2), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return self._conv_layer(net, num_filters, filter_size, strides, name=name)

    def _residual_block(self, net, style_index, filter_size=3, name=None):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        tmp = self._conv_layer(net, 32, filter_size, 1,
                               padding='VALID', relu=True, name=name + '_1')
        return self._conv_layer(tmp, 32, filter_size, 1, padding='VALID', relu=False, name=name + '_2') + net

    def _conditional_instance_norm(self, x, style_index, scope_bn):
        with tf.variable_scope(scope_bn, reuse=self.reuse):
            shift = tf.get_variable(name=scope_bn+'shift', shape=[
                                    self.n_styles, x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            scale = tf.get_variable(name=scope_bn+'scale', shape=[
                                    self.n_styles, x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        shift = tf.gather(shift, style_index)
        scale = tf.gather(scale, style_index)
        x = self._instance_norm(x, name=scope_bn, shift=shift, scale=scale)
        return x

    def _instance_norm(self, net, name=None, shift=None, scale=None):
        _, _, _, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
        if shift == None or scale == None:
            with tf.variable_scope(name, reuse=self.reuse):
                shift = tf.get_variable(
                    'shift', initializer=tf.zeros(var_shape), dtype=tf.float32)
                scale = tf.get_variable(
                    'scale', initializer=tf.ones(var_shape), dtype=tf.float32)
        epsilon = 1e-3
        normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
        return scale * normalized + shift

    def _conv_init_vars(self, net, out_channels, filter_size, transpose=False, name=None):
        _, _, _, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size,
                             in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size,
                             out_channels, in_channels]
        with tf.variable_scope(name, reuse=self.reuse):
            weights_init = tf.get_variable('weight', shape=weights_shape,
                                           initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           dtype=tf.float32)
        return weights_init

    def _depthwise_conv_layer(self, net, num_filters, filter_size, strides, padding='SAME', relu=True, channel_mul=1,
                              name=None):
        depthwise_weights_init, pointwise_weights_init = self._depthwiseconv_init_vars(net, num_filters, channel_mul,
                                                                                       filter_size, name=name)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.separable_conv2d(net, depthwise_weights_init, pointwise_weights_init, strides_shape,
                                     padding=padding)
        net = self._instance_norm(net, name=name)
        if relu:
            net = tf.nn.relu(net)
        return net

    def _depthwiseconv_init_vars(self, net, out_channels, channel_multiplier, filter_size, name=None):
        _, _, _, in_channels = [i.value for i in net.get_shape()]
        depthwise_weights_shape = [filter_size,
                                   filter_size, in_channels, channel_multiplier]
        pointwise_weights_shape = [
            1, 1, in_channels * channel_multiplier, out_channels]

        with tf.variable_scope(name, reuse=self.reuse):
            depthwise_weights = tf.get_variable('depthwise_weight', shape=depthwise_weights_shape,
                                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                dtype=tf.float32)
            pointwise_weights = tf.get_variable('pointwise_weight', shape=pointwise_weights_shape,
                                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                dtype=tf.float32)
        return depthwise_weights, pointwise_weights
