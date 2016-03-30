import NeuroticAdHominem as nah
from NeuroticAdHominem import TextCNN
from NeuroticAdHominem import Options as opts
from NeuroticAdHominem.training import preprocess
from NeuroticAdHominem.hosting.EvalModel import EvalModel
from flask_socketio import emit
import sys

import tensorflow as tf
import numpy as np
from multiprocessing import Pipe, Process

host_process = None
eval_conn = None
model_conn = None


def launch():
    """
        Begin the model eval process
    """
    global eval_conn
    global model_conn
    global host_process

    eval_conn, child_eval_conn = Pipe()
    model_conn, child_model_conn = Pipe()
    host_process = Process(target=host, args=(child_eval_conn, child_model_conn,))
    host_process.start()



def host(eval_conn, model_conn):
    """
        Model eval process
    """
    # load the trained model
    model = EvalModel()

    # begin eval listener loop
    while True:

        # recieve a sentence to eval
        if eval_conn.poll():
            to_eval = eval_conn.recv()
            sentence = preprocess.clean(to_eval)
            padded_sentence = preprocess.pad(sentence)
            word_ids = []

            # get word id's
            for word in padded_sentence:
                nah.vocabulary.addWord(word)
                id = nah.vocabulary.getIdFromWord(word)
                word_ids.append(id)

            # run evaluation
            result = model.eval(np.array(word_ids))

            # return evaluation result (either "example" or "nonexample")
            eval_conn.send(result)

        if model_conn.poll():
            # load the newly trained model
            to_load = model_conn.recv() # for now just clears the queue
            model = EvalModel()
            model_conn.send(True)

def reload():
    global model_conn
    model_conn.send(True)
    eval_conn.recv() # block until we confirm the model was reloaded
    nah.log("Model Reloaded")
    sys.stdout.flush()


def eval(sentence):
    """
        Send a sentence to the eval process and wait for the result.
    """
    global eval_conn
    eval_conn.send(sentence)
    result = eval_conn.recv()
    return result


def emitEval(sentence, classification):
    emit("stream:eval", {"sentence": sentence, "classification": classification})

