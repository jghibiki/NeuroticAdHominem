from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell
import itertools
from collections import Counter
import csv
import re
import numpy as np
import string
import tensorflow as tf
import os
import time
import datetime
from sklearn.cross_validation import StratifiedShuffleSplit
from TextCNN import TextCNN
from store import options as opts
from store import vocab
import preprocess

training_process = None
training_conn = None

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def train():

    sentences = []
    labels = []
    x = []
    y = []
    _y = []

    with open('data.csv', 'rb') as f:
        reader = csv.reader(f, delimiter=',')

        for row in reader:
            words = preprocess.clean(row[1])
            sentences.append(words)
            labels.append(([0, 1] if row[0] == "example" else [1, 0]))
            _y.append(1 if row[0] == "example" else 0)

    padded_sentences = [ preprocess.pad(sentence) for sentence in sentences ]

    word_counts = Counter(itertools.chain(*padded_sentences))

    # Mapping from index to word
    vocab.vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocab.vocabulary = {x: i for i, x in enumerate(vocab.vocabulary_inv)}


    x = np.array([[vocab.vocabulary[word] for word in sentence] for sentence in padded_sentences])
    y = np.array(labels)


# Split Dataset
# ==================================================

# Load data
    print("Loading data...")
# Randomly shuffle data
    sss = StratifiedShuffleSplit(_y, 1, test_size=0.1, random_state=0)
    for train, test in sss:
        x_train = x[train]
        y_train = y[train]

        x_dev = x[test]
        y_dev = y[test]


# Training
# ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=opts["allow_soft_placement"],
          log_device_placement=opts["log_device_placement"])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocab.vocabulary),
                embedding_size=opts["embedding_dim"],
                filter_sizes=map(int, opts["filter_sizes"].split(",")),
                num_filters=opts["num_filters"],
                l2_reg_lambda=opts["l2_reg_lambda"])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: opts["dropout_keep_prob"]
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            batches = batch_iter(
                zip(x_train, y_train), opts["batch_size"], opts["num_epochs"])
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % opts["evaluate_every"] == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")

            saver.save(sess, opts["model_location"] + "model.chpt")

