import NeuroticAdHominem as nah
from NeuroticAdHominem import TextCNN
from NeuroticAdHominem import Options as opts
from NeuroticAdHominem.training import preprocess
from NeuroticAdHominem.hosting.EvalModel import EvalModel
from flask_socketio import emit

import tensorflow as tf
import numpy as np
from multiprocessing import Pipe, Process

host_process = None
parent_conn = None


def launch():
    """
        Begin the model eval process
    """
    global parent_conn
    global host_process

    parent_conn, child_conn = Pipe()
    host_process = Process(target=host, args=(child_conn,))
    host_process.start()


def kill():
    """
        Wait for model eval process to end
    """
    global host_process
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
            word = word.encode('ascii', 'replace')
            if(word not in nah.vocabulary.keys()):
                print("New word: %s" % word)
                nah.vocabulary_inv.append(word)
                nah.vocabulary[word] = nah.vocabulary_inv.index(word)

            word_ids.append(nah.vocabulary[word])

        # run evaluation
        result = model.eval(np.array(word_ids))

        # return evaluation result (either "example" or "nonexample")
        conn.send(result)


def eval(sentence):
    """
        Send a sentence to the eval process and wait for the result.
    """
    global parent_conn
    parent_conn.send(sentence)
    result = parent_conn.recv()
    return result


def emitEval(sentence, classification):
    emit("stream:eval", {"sentence": sentence, "classification": classification})

