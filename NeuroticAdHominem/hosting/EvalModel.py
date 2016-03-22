import NeuralAdHominem as nah
from NeuralAdHominem import TextCNN
from NeuralAdHominem import Options as opts
from NeuralAdHominem.training import preprocess

import tensorflow as tf
import numpy as np

class EvalModel(object):
    def __init__(self):
        graph = tf.Graph()

        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        session = ft.Session(session_conf)

        saver = tf.train.Saver(tf.all_variables())

        with graph.as_default():
            with session as sess:
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=2,
                    vocab_size=len(vocabulary),
                    embedding_size=opts.embedding_dim,
                    filter_sizes=map(int, opts.filter_sizes.split(",")),
                    num_filters=opts.num_filters,
                    l2_reg_lambda=opts.l2_reg_lambda)

                sess.run(tf.initialize_all_variables())

                saver = tf.train.Saver(tf.all_variables())
                ckpt = tf.train.get_checkpoint_state(opts.model_location)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

        self.graph = graph
        self.session = session
        self.cnn = cnn

    def eval(self, tensor):
        with model.graph.as_default():
            with model.session as sess:
                feed_dict = {
                  cnn.input_x: tensor,
                  cnn.input_y: np.array([[1,0]]),
                  cnn.dropout_keep_prob: 1.0
                }
                accuracy = sess.run([cnn.accuracy], feed_dict)

                if(accuracy is 1):
                    return "nonexample"
                elif(accuracy is 0):
                    return "nonexample"

