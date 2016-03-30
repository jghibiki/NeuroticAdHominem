import NeuroticAdHominem as nah
from NeuroticAdHominem import TextCNN
from NeuroticAdHominem import Options as opts
from NeuroticAdHominem.training import preprocess

import tensorflow as tf
import numpy as np

class EvalModel(object):
    def __init__(self):
        graph = tf.Graph()

        session_conf = tf.ConfigProto(
          allow_soft_placement=opts.allow_soft_placement,
          log_device_placement=opts.log_device_placement)
        session = tf.Session(config=session_conf)


        cnn = TextCNN(
            sequence_length=opts.sentence_padding,
            num_classes=2,
            vocab_size=len(nah.vocabulary) + opts.vocab_oversizing,
            embedding_size=opts.embedding_dim,
            filter_sizes=map(int, opts.filter_sizes.split(",")),
            num_filters=opts.num_filters,
            l2_reg_lambda=opts.l2_reg_lambda)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(opts.model_location)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)

        self.graph = graph
        self.session = session
        self.cnn = cnn

    def eval(self, tensor):
        feed_dict = {
          self.cnn.input_x: [tensor],
          self.cnn.input_y: np.array([[1,0]]),
          self.cnn.dropout_keep_prob: 1.0
        }

        accuracy = self.session.run([self.cnn.accuracy], feed_dict)[0]

        if(int(accuracy) is 1):
            return "nonexample"
        elif(int(accuracy) is 0):
            return "example"
        else:
            return accuracy
