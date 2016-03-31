from TextCNN import TextCNN
from store import vocab
import store
from store import options as opts
import preprocess
from EvalModel import EvalModel
import sys

import tensorflow as tf
import numpy as np
import redis
import threading
import json



def launch():
    """
        Begin the model eval process
    """

    store.log("Launching evaluator")
    r = redis.Redis(host="redis")
    client = ModelListener(r)
    client.start()

class ModelListener(threading.Thread):

    def __init__(self, r):
        threading.Thread.__init__(self)
        # load the trained model
        self.model = EvalModel()
        self.redis = r
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(["model"])

    def eval(self, item):
        sentence = preprocess.clean(item)
        padded_sentence = preprocess.pad(sentence)
        word_ids = []

        # get word id's
        for word in padded_sentence:
            vocab.addWord(word)
            id = vocab.getIdFromWord(word)
            word_ids.append(id)

        # run evaluation
        result = self.model.eval(np.array(word_ids))

        self.redis.publish("server", json.dumps({"sentence": item, "classification": result}))


    def reload(self):
        model = EvalModel()


    def run(self):
        for item in self.pubsub.listen():
            if item["data"] == "KILL":
                self.pubsub.unsubscribe()
                store.log(self, "unsubscribe and finished")
                break
            elif isinstance(item["data"], str):
                data = json.loads(item["data"])
                if(data[0] == "eval"):
                    store.log("eval: %s" % data[1])
                    self.eval(data[1])
                elif(data[0] == "reload"):
                    store.log("reload")
                    self.reload()
            else:
                store.log("bad data: %s" % item["data"])
