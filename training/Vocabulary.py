import preprocess
import pickle
import  itertools
import collections
from collections import Counter
import random
import math
import csv
from store import options as opts
import store
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell

class Vocabulary(object):
    def __init__(self):

        self.wordSet = set()
        self.vocabGrowth = 0
        self.vocabulary = {}
        self.vocabulary_inv = []

        # Build Vocab
        with open('vocab.csv', 'rb') as f:
            reader = csv.reader(f.read().splitlines())

            for row in reader:
                if len(row) > 0:
                    words = preprocess.clean(row[0])
                    for word in words:
                        self.addWord(word)

        self.addWord(opts["sentence_padding_token"])
        self.vocabulary_size = len(self.wordSet)
        store.log("Vocabulary Size: %s" % self.vocabulary_size)


        self.embeddings = None
        self.data_index = 0
        self.data = []

    def __len__(self):
        return len(self.wordSet)

    def load(self):
        with open("word_embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

    def train(self):
        count = ['UNK']
        count.extend(self.vocabulary_inv)
        dictionary = dict()
        for word in count:
            dictionary[word] = len(dictionary)
        unk_count = 0

        self.data = [ idx for word, idx in self.vocabulary.iteritems() ]

        store.log('Sample data %s' % self.data[:10])

        def generate_batch(batch_size, num_skips, skip_window):
            assert batch_size % num_skips == 0
            assert num_skips <= 2 * skip_window
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            span = 2 * skip_window + 1 # [ skip_window target skip_window ]
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(self.data[self.data_index])
                self.data_index = (self.data_index + 1) % len(self.data)
            for i in range(batch_size // num_skips):
                target = skip_window  # target label at the center of the buffer
                targets_to_avoid = [ skip_window ]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
            return batch, labels

        batch, labels = generate_batch(batch_size=10, num_skips=10, skip_window=5)

        for i in range(10):
            store.log('%s -> %s' % (batch[i], labels[i, 0]))
            store.log('%s -> %s' % (self.vocabulary_inv[batch[i]], self.vocabulary_inv[labels[i, 0]]))

        batch_size = 20
        embedding_size = 128  # Dimension of the embedding vector.
        skip_window = 10       # How many words to consider left and right.
        num_skips = 20         # How many times to reuse an input to generate a label.
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_size = 16     # Random set of words to evaluate similarity on.
        valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
        num_sampled = 64    # Number of negative examples to sample.

        graph = tf.Graph()
        with graph.as_default():
            # Input da 4ta.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                # Construct the variables for the NCE loss
                with tf.name_scope("nce_weights") as scope:
                    nce_weights = tf.Variable(
                        tf.truncated_normal([self.vocabulary_size, embedding_size],
                                            stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
                nce_biases_hist = tf.histogram_summary("nce_biases", nce_biases)

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope("loss") as scope:
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                                 num_sampled, self.vocabulary_size))
            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope("train") as scope:
                optimizer = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)


        # Step 5: Begin training.
        num_steps = 1001
        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("/tmp/tensor_logs/expiriment_1", session.graph_def)

            #Adds an op to initialize all variables
            init_op = tf.initialize_all_variables()

            # Begins running the init opp
            init_op.run()

            store.log("Initialized")
            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(
                    batch_size, num_skips, skip_window)
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                summary_str, _, loss_val = session.run([merged, optimizer, loss], feed_dict=feed_dict)
                writer.add_summary(summary_str, step)
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    store.log("Average loss at step %s: %s" % (step, average_loss))
                    average_loss = 0
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 5000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = self.vocabulary_inv[valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = self.vocabulary_inv[nearest[k]]
                        log_str = "%s %s" % (log_str, close_word)
                    store.log(log_str)

            final_embeddings = normalized_embeddings.eval()

            # save final embeddings for Expiriment #2
            with open("word_embeddings.pkl", "wb") as f:
                pickle.dump(final_embeddings, f)

            self.embeddings = final_embeddings





    def addWord(self, word):
        word = word.encode('ascii', 'replace')
        if(word not in self.wordSet):
            self.vocabulary_inv.append(word)
            self.vocabulary[word] = self.vocabulary_inv.index(word)
            self.wordSet.add(word)
            self.vocabGrowth += 1

    def getIdFromWord(self, word):
        word = word.encode('ascii', 'replace')
        return self.vocabulary[word]

    def getWordFromId(self, id):
        return self.vocabulary_inv[id]

    def getGrowth(self):
        return self.vocabGrowth

    def resetGrowth(self):
        self.vocabGrowth = 0
