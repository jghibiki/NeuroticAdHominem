import NeuralAdHominem as nah
from NeuralAdHominem import TextCNN
from NeuralAdHominem import Options as opts
from NeuralAdHominem.training import preprocess
from NeuralAdHominem.hosting.EvalModel import EvalModel

import tensorflow as tf
import numpy as np
from multiprocessing import Pipe, Process

host_process = None


def fork_host():
    """
        Begin the model eval process
    """
    parent_conn, child_conn = Pipe()
    host_process = Process(target=host, args=(child_conn,))
    host_proces.start()

def defork_host():
    """
        Wait for model eval process to end
    """
    if(host_process):
        host_process.join()

def host(conn):
    """
        Model eval process
    """
    # load the trained model
    model = EvalModel()

    # begin eval listener loop
    while True:

        # recieve a sentence to eval
        to_eval = conn.recv()
        sentence = preprocess.clean(to_eval)
        padded_sentence = preprocess.pad(sentence)
        word_ids = []

        # get word id's
        for word in padded_sentence:
            if(word not in nah.vocabulary_inv.keys()):
                nah.vocabulary.append(word)
                nah.vocabulary_inv[word] = nah.vocabulary.index(word)

            word_ids.append(nah.vocabulary_inv[word])

        # run evaluation
        result = model.eval(np.array(word_ids))

        # return evaluation result (either "example" or "nonexample")
        conn.send(result)


def eval(sentence):
    """
        Send a sentence to the eval process and wait for the result.
    """
    parent_conn.send(sentence)
    result = parent_conn.recv()
    return result

