import tensorflow as tf
from transform import Transform
import numpy as np
import os

class StyleTransferTester:

    def __init__(self, session, content_image, model_path, n_styles, style_index=0):
        # session
        self.sess = session
        self.n_styles = n_styles
        # input images
        self.x0 = content_image
        self.style_index0 = np.array([style_index], dtype=np.int32)
        
        # input model
        self.model_path = model_path

        # image transform network
        self.transform = Transform(n_styles)

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):
        # graph input
        self.x = tf.placeholder(tf.float32, shape=self.x0.shape, name='input')
        self.style_index = tf.placeholder(tf.int32, shape=(1,), name='style_index')
        self.style_index_batch = tf.expand_dims(self.style_index, 0)

        self.xi = tf.expand_dims(self.x, 0)  # add one dim for batch
        # result image from transform-net
        self.y_hat = self.transform.net(self.xi / 255.0, self.style_index_batch)
        self.y_hat = tf.squeeze(self.y_hat)  # remove one dim for batch
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)
        self.y_hat = tf.reshape(self.y_hat, np.array((-1,) + self.x0.shape, dtype=np.int32))

    def test(self):
        # initialize parameters
        self.sess.run(tf.global_variables_initializer())

        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)
        
        # get transformed image
        output = self.sess.run(self.y_hat, feed_dict={self.x: self.x0, self.style_index: self.style_index0})

        return output

    def save_as_tflite(self, model_name):
        self.sess.run(tf.global_variables_initializer())

        # load pre-trained model
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

        converter = tf.lite.TFLiteConverter.from_session(self.sess, [self.x, self.style_index], [self.y_hat])
        
        tflite_model = converter.convert()

        if not os.path.exists('tflite_models_final/'):
            os.mkdir('tflite_models_final')

        with open('tflite_models_final/{}.tflite'.format(model_name), 'wb') as f:
            f.write(tflite_model)

    def save_as_saved_model(self, model_name, max_size):
        self.sess.run(tf.global_variables_initializer())
        tf.saved_model.simple_save(self.sess, '{}/{}/'.format(max_size, model_name), {'input': self.xi},
                                   {'output': self.y_hat})





